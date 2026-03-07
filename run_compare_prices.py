import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from chronos import Chronos2Pipeline

try:
    import chinese_calendar as cc
except ImportError as exc:
    raise ImportError(
        "Missing dependency `chinese_calendar`. Install it with: py -m pip install chinese_calendar"
    ) from exc


SEGMENTS = ["valley", "flat", "peak"]
EXTRA_LAGS = [2, 4, 192]
ROLL_WINDOWS = [4, 16, 96, 672]
EPS = np.finfo(np.float64).eps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare price forecasting models with segmented weekly retraining."
    )
    parser.add_argument("--train_csv", type=str, default="guangdong_2025_complete.csv")
    parser.add_argument("--valid_csv", type=str, default="guangdong_2026_complete.csv")
    parser.add_argument(
        "--targets", type=str, default="day_ahead_price,real_time_price"
    )
    parser.add_argument("--chronos_path", type=str, default="models/chronos-2")
    parser.add_argument("--week_steps", type=int, default=672)
    parser.add_argument("--lags", type=str, default="1,96,672")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--xgb_seed", type=int, default=42)
    parser.add_argument("--xgb_n_estimators", type=int, default=800)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.03)
    parser.add_argument("--xgb_subsample", type=float, default=0.9)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.9)
    parser.add_argument("--xgb_reg_lambda", type=float, default=1.0)
    parser.add_argument("--xgb_min_train_rows", type=int, default=300)
    parser.add_argument(
        "--xgb_history_mode",
        type=str,
        default="walk_forward",
        choices=["walk_forward", "strict_recursive"],
        help="walk_forward uses observed past values as soon as they are available; strict_recursive uses model predictions within each weekly block.",
    )
    parser.add_argument(
        "--max_valid_steps",
        type=int,
        default=None,
        help="Optional debug mode: only forecast first N validation steps.",
    )
    parser.add_argument(
        "--hybrid_select_metric",
        type=str,
        default="sMAPE_image",
        choices=["MAE", "sMAPE_image"],
        help="Metric used to select best base model per segment for the hybrid model.",
    )
    parser.add_argument(
        "--rt_peak_retrain_steps",
        type=int,
        default=96,
        help="For real_time_price only: retrain peak-segment XGBoost every N steps.",
    )
    parser.add_argument(
        "--rt_peak_n_estimators_mult",
        type=float,
        default=1.5,
        help="For real_time_price only: multiplier on n_estimators for peak-segment XGBoost.",
    )
    parser.add_argument(
        "--rt_peak_max_depth_add",
        type=int,
        default=2,
        help="For real_time_price only: additive max_depth boost for peak-segment XGBoost.",
    )
    parser.add_argument(
        "--rt_peak_learning_rate_mult",
        type=float,
        default=0.8,
        help="For real_time_price only: multiplier on learning_rate for peak-segment XGBoost.",
    )
    return parser.parse_args()


def parse_lags(lags_text: str) -> List[int]:
    lag_values = [int(x.strip()) for x in lags_text.split(",") if x.strip()]
    for required in (1, 96, 672):
        if required not in lag_values:
            lag_values.append(required)
    return sorted(set(lag_values + EXTRA_LAGS))


def ensure_15min_contiguous(index: pd.DatetimeIndex) -> None:
    diffs = index.to_series().diff().dropna()
    if not (diffs == pd.Timedelta(minutes=15)).all():
        raise ValueError("Timeline is not contiguous at 15-minute frequency.")


def get_segments(index: pd.DatetimeIndex) -> pd.Series:
    minute_of_day = index.hour * 60 + index.minute
    valley = minute_of_day < 8 * 60
    flat = (
        ((minute_of_day >= 8 * 60) & (minute_of_day < 10 * 60))
        | ((minute_of_day >= 12 * 60) & (minute_of_day < 14 * 60))
        | ((minute_of_day >= 19 * 60) & (minute_of_day < 24 * 60))
    )
    seg = np.where(valley, "valley", np.where(flat, "flat", "peak"))
    return pd.Series(seg, index=index, name="segment")


def build_calendar_daily_features(dates: List[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for ts in dates:
        d = ts.date()
        is_holiday = bool(cc.is_holiday(d))
        is_workday = bool(cc.is_workday(d))
        is_weekend = d.weekday() >= 5
        _, holiday_name_raw = cc.get_holiday_detail(d)
        holiday_name = str(holiday_name_raw) if is_holiday and holiday_name_raw else "None"
        rows.append(
            {
                "date": pd.Timestamp(d),
                "is_holiday": int(is_holiday),
                "is_workday": int(is_workday),
                "is_weekend": int(is_weekend),
                "is_makeup_workday": int(is_workday and is_weekend),
                "holiday_name": holiday_name,
            }
        )
    daily = pd.DataFrame(rows).set_index("date").sort_index()
    next_holiday = daily["is_holiday"].shift(-1).fillna(0).astype(int)
    prev_holiday = daily["is_holiday"].shift(1).fillna(0).astype(int)
    daily["before_holiday_1d"] = next_holiday
    daily["after_holiday_1d"] = prev_holiday
    return daily


def build_static_features(index: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.Series]:
    segment = get_segments(index)
    minute_of_day = index.hour * 60 + index.minute
    slot96 = (minute_of_day // 15).astype(int)

    static = pd.DataFrame(index=index)
    static["hour"] = index.hour.astype(float)
    static["minute"] = index.minute.astype(float)
    static["dayofweek"] = index.dayofweek.astype(float)
    static["slot96"] = slot96.astype(float)
    static["sin_slot96"] = np.sin(2 * np.pi * slot96 / 96.0)
    static["cos_slot96"] = np.cos(2 * np.pi * slot96 / 96.0)

    unique_dates = sorted(set(pd.Timestamp(d) for d in index.date))
    daily = build_calendar_daily_features(unique_dates)
    mapped = daily.reindex(pd.to_datetime(index.date)).set_index(index)
    static = pd.concat(
        [
            static,
            mapped[
                [
                    "is_holiday",
                    "is_workday",
                    "is_weekend",
                    "is_makeup_workday",
                    "before_holiday_1d",
                    "after_holiday_1d",
                ]
            ],
        ],
        axis=1,
    )

    holiday_name_series = mapped["holiday_name"].astype(str)
    holiday_dummies = pd.get_dummies(holiday_name_series, prefix="holiday_name")
    static = pd.concat([static, holiday_dummies], axis=1)
    static = static.astype(float)
    return static, segment


def build_dynamic_features(series: pd.Series, lags: List[int]) -> pd.DataFrame:
    dynamic = pd.DataFrame(index=series.index)
    for lag in lags:
        dynamic[f"lag_{lag}"] = series.shift(lag)
    shifted = series.shift(1)
    for window in ROLL_WINDOWS:
        dynamic[f"roll_mean_{window}"] = shifted.rolling(window, min_periods=1).mean()
    dynamic["roll_std_96"] = shifted.rolling(96, min_periods=2).std()
    dynamic["diff_1"] = dynamic["lag_1"] - dynamic["lag_2"]
    dynamic["diff_96"] = dynamic["lag_1"] - dynamic["lag_96"]
    return dynamic


def calc_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE_pct": np.nan,
            "sMAPE_pct": np.nan,
            "sMAPE_image": np.nan,
        }

    yt = y_true[mask].to_numpy()
    yp = y_pred[mask].to_numpy()
    err = yp - yt
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mape = np.mean(np.abs(err) / np.maximum(np.abs(yt), EPS)) * 100.0
    smape = np.mean(2.0 * np.abs(err) / (np.abs(yt) + np.abs(yp) + EPS)) * 100.0
    # Custom metric from the provided screenshot:
    # mape_i = min(|y-ŷ| / max(|y|, eps), 1), score = 1 - mean(mape_i)
    mape_i = np.abs(err) / np.maximum(np.abs(yt), EPS)
    mape_i = np.minimum(mape_i, 1.0)
    smape_image = 1.0 - np.mean(mape_i)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_pct": float(mape),
        "sMAPE_pct": float(smape),
        "sMAPE_image": float(smape_image),
    }


def build_feature_row(
    pos: int,
    known_values: np.ndarray,
    static_row: pd.Series,
    feature_cols: List[str],
    lags: List[int],
) -> pd.DataFrame:
    row = {}
    for lag in lags:
        idx = pos - lag
        row[f"lag_{lag}"] = known_values[idx] if idx >= 0 else np.nan

    for window in ROLL_WINDOWS:
        start = max(0, pos - window)
        hist = known_values[start:pos]
        if hist.size == 0 or np.all(np.isnan(hist)):
            row[f"roll_mean_{window}"] = np.nan
        else:
            row[f"roll_mean_{window}"] = float(np.nanmean(hist))

    hist96 = known_values[max(0, pos - 96) : pos]
    finite96 = hist96[~np.isnan(hist96)]
    row["roll_std_96"] = float(np.std(finite96, ddof=1)) if finite96.size >= 2 else np.nan
    row["diff_1"] = row.get("lag_1", np.nan) - row.get("lag_2", np.nan)
    row["diff_96"] = row.get("lag_1", np.nan) - row.get("lag_96", np.nan)

    for c in static_row.index:
        row[c] = static_row[c]

    row_df = pd.DataFrame([row])
    for c in feature_cols:
        if c not in row_df.columns:
            row_df[c] = np.nan
    return row_df[feature_cols]


def fit_single_segment_model(
    x_df: pd.DataFrame,
    y: pd.Series,
    segments: pd.Series,
    cutoff_pos: int,
    segment: str,
    xgb_params: Dict,
    min_train_rows: int,
) -> Tuple[Optional[xgb.XGBRegressor], int]:
    pos_arr = np.arange(len(y))
    seg_mask = (
        (pos_arr < cutoff_pos)
        & y.notna().to_numpy()
        & (segments.to_numpy() == segment)
    )
    seg_count = int(seg_mask.sum())
    if seg_count < min_train_rows:
        return None, seg_count
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(x_df.loc[seg_mask], y.loc[seg_mask], verbose=False)
    return model, seg_count


def fit_segment_models(
    x_df: pd.DataFrame,
    y: pd.Series,
    segments: pd.Series,
    cutoff_pos: int,
    xgb_params: Dict,
    xgb_params_by_segment: Dict[str, Dict],
    min_train_rows: int,
) -> Tuple[Dict[str, xgb.XGBRegressor], xgb.XGBRegressor, Dict[str, int]]:
    pos_arr = np.arange(len(y))
    train_base = (pos_arr < cutoff_pos) & y.notna().to_numpy()
    models: Dict[str, xgb.XGBRegressor] = {}
    segment_counts: Dict[str, int] = {}

    global_model = None
    global_count = int(train_base.sum())
    if global_count >= min_train_rows:
        global_model = xgb.XGBRegressor(**xgb_params)
        global_model.fit(x_df.loc[train_base], y.loc[train_base], verbose=False)

    for seg in SEGMENTS:
        seg_params = xgb_params_by_segment.get(seg, xgb_params)
        model, seg_count = fit_single_segment_model(
            x_df=x_df,
            y=y,
            segments=segments,
            cutoff_pos=cutoff_pos,
            segment=seg,
            xgb_params=seg_params,
            min_train_rows=min_train_rows,
        )
        segment_counts[seg] = seg_count
        if model is not None:
            models[seg] = model
    return models, global_model, segment_counts


def rolling_segmented_xgb_predict(
    value_series: pd.Series,
    static_df: pd.DataFrame,
    segments: pd.Series,
    forecast_positions: np.ndarray,
    week_steps: int,
    lags: List[int],
    xgb_params: Dict,
    xgb_params_by_segment: Dict[str, Dict],
    min_train_rows: int,
    scheme_name: str,
    history_mode: str,
    peak_retrain_steps: Optional[int] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    dynamic_df = build_dynamic_features(value_series, lags)
    x_df = pd.concat([dynamic_df, static_df], axis=1)
    feature_cols = list(x_df.columns)

    values = value_series.to_numpy(dtype=float)
    known_values = values.copy()
    preds = np.full(len(values), np.nan)
    logs: List[Dict] = []

    use_observed_history = history_mode == "walk_forward"

    for i in range(0, len(forecast_positions), week_steps):
        block = forecast_positions[i : i + week_steps]
        if len(block) == 0:
            continue
        cutoff_pos = int(block[0])

        # For new week, history before cutoff uses observed values.
        known_values[:cutoff_pos] = values[:cutoff_pos]

        models, global_model, seg_counts = fit_segment_models(
            x_df=x_df,
            y=value_series,
            segments=segments,
            cutoff_pos=cutoff_pos,
            xgb_params=xgb_params,
            xgb_params_by_segment=xgb_params_by_segment,
            min_train_rows=min_train_rows,
        )

        cutoff_time = value_series.index[cutoff_pos]
        block_end_time = value_series.index[int(block[-1])]
        for seg in SEGMENTS:
            logs.append(
                {
                    "scheme": scheme_name,
                    "week_cutoff_time": cutoff_time,
                    "week_end_time": block_end_time,
                    "segment": seg,
                    "train_rows": seg_counts.get(seg, 0),
                    "event_type": "weekly",
                }
            )

        for pos in block:
            seg = segments.iat[pos]
            if (
                peak_retrain_steps is not None
                and seg == "peak"
                and pos > cutoff_pos
                and ((pos - cutoff_pos) % peak_retrain_steps == 0)
            ):
                peak_model, peak_rows = fit_single_segment_model(
                    x_df=x_df,
                    y=value_series,
                    segments=segments,
                    cutoff_pos=pos,
                    segment="peak",
                    xgb_params=xgb_params_by_segment.get("peak", xgb_params),
                    min_train_rows=min_train_rows,
                )
                if peak_model is not None:
                    models["peak"] = peak_model
                logs.append(
                    {
                        "scheme": scheme_name,
                        "week_cutoff_time": value_series.index[pos],
                        "week_end_time": block_end_time,
                        "segment": "peak",
                        "train_rows": peak_rows,
                        "event_type": "peak_intraday",
                    }
                )
            static_row = static_df.iloc[pos]
            row_df = build_feature_row(pos, known_values, static_row, feature_cols, lags)
            model = models.get(seg, None)
            if model is None:
                model = global_model

            if model is None:
                pred = row_df["lag_1"].iloc[0]
                pred = 0.0 if pd.isna(pred) else float(pred)
            else:
                pred = float(model.predict(row_df)[0])

            preds[pos] = pred
            known_values[pos] = pred
            if use_observed_history and not np.isnan(values[pos]):
                known_values[pos] = values[pos]

    pred_series = pd.Series(preds, index=value_series.index, name=f"{scheme_name}_pred")
    log_df = pd.DataFrame(logs)
    return pred_series, log_df


def rolling_chronos_predict(
    value_series: pd.Series,
    forecast_positions: np.ndarray,
    week_steps: int,
    pipeline: Chronos2Pipeline,
    scheme_name: str,
) -> pd.Series:
    index = value_series.index
    preds = np.full(len(value_series), np.nan)

    for i in range(0, len(forecast_positions), week_steps):
        block = forecast_positions[i : i + week_steps]
        if len(block) == 0:
            continue
        cutoff_pos = int(block[0])
        block_len = int(len(block))

        context = value_series.iloc[:cutoff_pos].copy()
        if len(context) == 0:
            continue
        context = context.ffill()
        if context.isna().all():
            context = pd.Series(np.zeros(len(context)), index=context.index, dtype=float)
        else:
            first_valid = context.dropna().iloc[0]
            context = context.fillna(float(first_valid))

        context_df = pd.DataFrame(
            {"item_id": "series_0", "timestamp": context.index, "target": context.values}
        )
        pred_df = pipeline.predict_df(
            context_df,
            prediction_length=block_len,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            quantile_levels=[0.5],
            batch_size=128,
        )
        pred_df = pred_df.sort_values("timestamp")
        if "predictions" in pred_df.columns:
            block_pred = pred_df["predictions"].to_numpy(dtype=float)
        elif "0.5" in pred_df.columns:
            block_pred = pred_df["0.5"].to_numpy(dtype=float)
        else:
            raise RuntimeError(f"{scheme_name}: cannot find point prediction column in Chronos output.")

        preds[block] = block_pred[:block_len]

    return pd.Series(preds, index=index, name=f"{scheme_name}_pred")


def choose_segment_best_models(
    actual_valid: pd.Series,
    pred_map: Dict[str, pd.Series],
    segment_valid: pd.Series,
    selection_metric: str,
    hybrid_name: str,
) -> Tuple[pd.Series, pd.DataFrame]:
    rows = []
    selected = {}
    for seg in SEGMENTS:
        seg_mask = segment_valid == seg
        best_name = None
        best_metric = -math.inf if selection_metric == "sMAPE_image" else math.inf
        best_rmse = math.inf
        for model_name, pred in pred_map.items():
            m = calc_metrics(actual_valid[seg_mask], pred[seg_mask])
            rows.append(
                {
                    "segment": seg,
                    "model": model_name,
                    "MAE": m["MAE"],
                    "RMSE": m["RMSE"],
                    "MAPE_pct": m["MAPE_pct"],
                    "sMAPE_pct": m["sMAPE_pct"],
                    "sMAPE_image": m["sMAPE_image"],
                }
            )
            metric_val = m[selection_metric]
            rmse = m["RMSE"]
            if np.isnan(metric_val):
                continue
            if selection_metric == "sMAPE_image":
                better = (metric_val > best_metric) or (
                    math.isclose(metric_val, best_metric) and rmse < best_rmse
                )
            else:
                better = (metric_val < best_metric) or (
                    math.isclose(metric_val, best_metric) and rmse < best_rmse
                )
            if better:
                best_metric = metric_val
                best_rmse = rmse
                best_name = model_name
        selected[seg] = best_name

    detail_df = pd.DataFrame(rows)
    chosen_rows = []
    hybrid = pd.Series(index=actual_valid.index, dtype=float, name=hybrid_name)
    for seg in SEGMENTS:
        name = selected[seg]
        mask = segment_valid == seg
        if name is None:
            hybrid.loc[mask] = np.nan
            chosen_rows.append(
                {
                    "segment": seg,
                    "selected_model": None,
                    "selected_metric": selection_metric,
                    "selected_metric_value": np.nan,
                    "selected_RMSE": np.nan,
                }
            )
            continue
        hybrid.loc[mask] = pred_map[name].loc[mask]
        metrics_row = detail_df[(detail_df["segment"] == seg) & (detail_df["model"] == name)].iloc[0]
        chosen_rows.append(
            {
                "segment": seg,
                "selected_model": name,
                "selected_metric": selection_metric,
                "selected_metric_value": metrics_row[selection_metric],
                "selected_RMSE": metrics_row["RMSE"],
            }
        )

    selection_df = pd.DataFrame(chosen_rows).merge(
        detail_df, how="left", left_on=["segment", "selected_model"], right_on=["segment", "model"]
    )
    selection_df = selection_df.drop(columns=["model"])
    return hybrid, selection_df


def plot_forecasts(
    out_file: Path,
    valid_index: pd.DatetimeIndex,
    actual: pd.Series,
    preds: Dict[str, pd.Series],
    target: str,
) -> None:
    plt.figure(figsize=(16, 6))
    plt.plot(valid_index, actual, label="actual", linewidth=1.8, color="black")
    for model_name, pred in preds.items():
        plt.plot(valid_index, pred, label=model_name, linewidth=1.0, alpha=0.85)
    plt.title(f"{target} Forecast Comparison (Validation)")
    plt.xlabel("Time")
    plt.ylabel(target)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()


def run_for_target(
    target: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    args: argparse.Namespace,
    chronos_pipe: Chronos2Pipeline,
    out_dir: Path,
) -> None:
    print(f"\n=== Running target: {target} ===")
    if target not in train_df.columns or target not in valid_df.columns:
        raise ValueError(f"Target `{target}` not found in both train/valid CSV.")

    train_part = train_df[["times", target]].copy()
    valid_part = valid_df[["times", target]].copy()
    train_part["split"] = "train"
    valid_part["split"] = "valid"
    all_df = pd.concat([train_part, valid_part], axis=0, ignore_index=True)
    all_df["times"] = pd.to_datetime(all_df["times"], errors="coerce", format="mixed")
    if all_df["times"].isna().any():
        raise ValueError(f"Found invalid timestamp in target {target}.")
    all_df = all_df.sort_values("times").reset_index(drop=True)
    all_df = all_df.drop_duplicates(subset=["times"], keep="last")
    all_df = all_df.set_index("times")
    ensure_15min_contiguous(all_df.index)

    static_df, segments = build_static_features(all_df.index)

    value_series = all_df[target].astype(float)
    train_mask = all_df["split"] == "train"
    valid_mask = all_df["split"] == "valid"
    train_positions = np.where(train_mask.to_numpy())[0]
    valid_positions = np.where(valid_mask.to_numpy())[0]
    if args.max_valid_steps is not None:
        valid_positions = valid_positions[: args.max_valid_steps]
    if len(valid_positions) == 0:
        raise ValueError("No validation points selected.")

    valid_index = all_df.index[valid_positions]
    segment_valid = segments.loc[valid_index]
    actual_valid = value_series.loc[valid_index]

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae", "mape"],
        "n_estimators": args.xgb_n_estimators,
        "max_depth": args.xgb_max_depth,
        "learning_rate": args.xgb_learning_rate,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample_bytree,
        "reg_lambda": args.xgb_reg_lambda,
        "random_state": args.xgb_seed,
        "tree_method": "hist",
        "n_jobs": -1,
    }
    xgb_params_by_segment = {seg: dict(xgb_params) for seg in SEGMENTS}
    peak_retrain_steps = None
    if target == "real_time_price":
        peak_params = dict(xgb_params)
        peak_params["n_estimators"] = max(
            50, int(round(xgb_params["n_estimators"] * args.rt_peak_n_estimators_mult))
        )
        peak_params["max_depth"] = max(
            3, xgb_params["max_depth"] + args.rt_peak_max_depth_add
        )
        peak_params["learning_rate"] = max(
            0.005, xgb_params["learning_rate"] * args.rt_peak_learning_rate_mult
        )
        xgb_params_by_segment["peak"] = peak_params
        peak_retrain_steps = args.rt_peak_retrain_steps
    lags = parse_lags(args.lags)

    # A) Pure XGBoost
    pred_xgb_full, log_xgb = rolling_segmented_xgb_predict(
        value_series=value_series,
        static_df=static_df,
        segments=segments,
        forecast_positions=valid_positions,
        week_steps=args.week_steps,
        lags=lags,
        xgb_params=xgb_params,
        xgb_params_by_segment=xgb_params_by_segment,
        min_train_rows=args.xgb_min_train_rows,
        scheme_name="xgb_only",
        history_mode=args.xgb_history_mode,
        peak_retrain_steps=peak_retrain_steps,
    )
    pred_xgb = pred_xgb_full.loc[valid_index]

    # B) Pure Chronos-2
    pred_chronos_full = rolling_chronos_predict(
        value_series=value_series,
        forecast_positions=valid_positions,
        week_steps=args.week_steps,
        pipeline=chronos_pipe,
        scheme_name="chronos_only",
    )
    pred_chronos = pred_chronos_full.loc[valid_index]

    # OOF windows on train for residual training
    warmup = max(672, max(lags))
    oof_positions = train_positions[train_positions >= warmup]

    # C) Chronos base + XGB residual
    chrono_oof_train_full = rolling_chronos_predict(
        value_series=value_series,
        forecast_positions=oof_positions,
        week_steps=args.week_steps,
        pipeline=chronos_pipe,
        scheme_name="chronos_oof_train",
    )
    residual_for_xgb = pd.Series(np.nan, index=value_series.index, dtype=float)
    residual_for_xgb.iloc[oof_positions] = (
        value_series.iloc[oof_positions] - chrono_oof_train_full.iloc[oof_positions]
    ).to_numpy()
    residual_for_xgb.loc[valid_index] = (value_series.loc[valid_index] - pred_chronos).to_numpy()

    pred_resid_xgb_full, log_c = rolling_segmented_xgb_predict(
        value_series=residual_for_xgb,
        static_df=static_df,
        segments=segments,
        forecast_positions=valid_positions,
        week_steps=args.week_steps,
        lags=lags,
        xgb_params=xgb_params,
        xgb_params_by_segment=xgb_params_by_segment,
        min_train_rows=args.xgb_min_train_rows,
        scheme_name="chronos_xgb_residual_xgb_part",
        history_mode=args.xgb_history_mode,
        peak_retrain_steps=peak_retrain_steps,
    )
    pred_chronos_xgb = pred_chronos + pred_resid_xgb_full.loc[valid_index]

    # D) XGB base + Chronos residual
    xgb_oof_train_full, log_xgb_oof = rolling_segmented_xgb_predict(
        value_series=value_series,
        static_df=static_df,
        segments=segments,
        forecast_positions=oof_positions,
        week_steps=args.week_steps,
        lags=lags,
        xgb_params=xgb_params,
        xgb_params_by_segment=xgb_params_by_segment,
        min_train_rows=args.xgb_min_train_rows,
        scheme_name="xgb_oof_train",
        history_mode=args.xgb_history_mode,
        peak_retrain_steps=peak_retrain_steps,
    )
    residual_for_chronos = pd.Series(np.nan, index=value_series.index, dtype=float)
    residual_for_chronos.iloc[oof_positions] = (
        value_series.iloc[oof_positions] - xgb_oof_train_full.iloc[oof_positions]
    ).to_numpy()
    residual_for_chronos.loc[valid_index] = (value_series.loc[valid_index] - pred_xgb).to_numpy()

    pred_resid_chronos_full = rolling_chronos_predict(
        value_series=residual_for_chronos,
        forecast_positions=valid_positions,
        week_steps=args.week_steps,
        pipeline=chronos_pipe,
        scheme_name="xgb_chronos_residual_chronos_part",
    )
    pred_xgb_chronos = pred_xgb + pred_resid_chronos_full.loc[valid_index]

    base_pred_map = {
        "xgb_only": pred_xgb,
        "chronos_only": pred_chronos,
        "chronos_xgb_residual": pred_chronos_xgb,
        "xgb_chronos_residual": pred_xgb_chronos,
    }

    hybrid_name = (
        "hybrid_segment_best_smape_image"
        if args.hybrid_select_metric == "sMAPE_image"
        else "hybrid_segment_best_mae"
    )
    hybrid_pred, segment_selection = choose_segment_best_models(
        actual_valid=actual_valid,
        pred_map=base_pred_map,
        segment_valid=segment_valid,
        selection_metric=args.hybrid_select_metric,
        hybrid_name=hybrid_name,
    )

    all_pred_map = dict(base_pred_map)
    all_pred_map[hybrid_name] = hybrid_pred

    # Save per-model predictions
    for model_name, pred in all_pred_map.items():
        out_pred = pd.DataFrame(
            {
                "times": valid_index,
                "target": target,
                "actual": actual_valid.to_numpy(),
                "segment": segment_valid.to_numpy(),
                "prediction": pred.to_numpy(),
            }
        )
        out_pred.to_csv(out_dir / f"predictions_{target}_{model_name}.csv", index=False)

    # Save metrics
    metric_rows = []
    for model_name, pred in all_pred_map.items():
        metric = calc_metrics(actual_valid, pred)
        metric_rows.append({"model": model_name, **metric})
    metrics_df = pd.DataFrame(metric_rows).sort_values("MAE")
    metrics_df.to_csv(out_dir / f"metrics_comparison_{target}.csv", index=False)

    # Save hybrid segment selection details
    segment_selection.to_csv(out_dir / f"segment_model_selection_{target}.csv", index=False)

    # Save retraining logs
    retrain_logs = pd.concat([log_xgb, log_c, log_xgb_oof], ignore_index=True)
    retrain_logs.insert(0, "target", target)
    retrain_logs.to_csv(out_dir / f"retrain_log_{target}.csv", index=False)

    # Plot
    plot_forecasts(
        out_file=out_dir / f"forecast_compare_{target}.png",
        valid_index=valid_index,
        actual=actual_valid,
        preds=all_pred_map,
        target=target,
    )

    print(metrics_df.to_string(index=False))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    valid_df = pd.read_csv(args.valid_csv)
    if "times" not in train_df.columns or "times" not in valid_df.columns:
        raise ValueError("Both CSV files must include `times` column.")

    target_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    if len(target_list) == 0:
        raise ValueError("No target specified.")

    print(f"Loading Chronos-2 model from: {args.chronos_path}")
    chronos_pipe = Chronos2Pipeline.from_pretrained(args.chronos_path, device_map="cpu")

    for target in target_list:
        run_for_target(
            target=target,
            train_df=train_df,
            valid_df=valid_df,
            args=args,
            chronos_pipe=chronos_pipe,
            out_dir=out_dir,
        )

    print(f"\nDone. Outputs are in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
