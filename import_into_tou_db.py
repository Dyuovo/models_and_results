import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class RuleWindow:
    effective_from: Optional[pd.Timestamp]
    effective_to: Optional[pd.Timestamp]
    segments: List[Dict[str, int | str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import Guangdong CSV data into SQLite tou.db"
    )
    parser.add_argument("--db", type=str, default="tou.db")
    parser.add_argument("--province_code", type=str, default="GD")
    parser.add_argument(
        "--csv_files",
        type=str,
        default="guangdong_2025_complete.csv,guangdong_2026_complete.csv",
    )
    return parser.parse_args()


def get_province_id(conn: sqlite3.Connection, province_code: str) -> int:
    row = conn.execute(
        "SELECT id FROM province WHERE code = ?",
        (province_code,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Province code `{province_code}` not found in table `province`.")
    return int(row[0])


def load_rule_windows(conn: sqlite3.Connection, province_id: int) -> List[RuleWindow]:
    rules = conn.execute(
        """
        SELECT id, effective_from, effective_to
        FROM tou_rule
        WHERE province_id = ?
        ORDER BY effective_from
        """,
        (province_id,),
    ).fetchall()
    if not rules:
        raise ValueError("No TOU rules found for this province.")

    windows: List[RuleWindow] = []
    for rule_id, effective_from, effective_to in rules:
        segments = conn.execute(
            """
            SELECT period, start_minute, end_minute
            FROM tou_rule_segment
            WHERE rule_id = ?
            ORDER BY start_minute
            """,
            (rule_id,),
        ).fetchall()
        seg_list = [
            {"period": p, "start_minute": int(s), "end_minute": int(e)}
            for p, s, e in segments
        ]
        windows.append(
            RuleWindow(
                effective_from=pd.to_datetime(effective_from, errors="coerce"),
                effective_to=(
                    pd.to_datetime(effective_to, errors="coerce")
                    if effective_to is not None
                    else None
                ),
                segments=seg_list,
            )
        )
    return windows


def pick_period(ts: pd.Timestamp, windows: List[RuleWindow]) -> Optional[str]:
    d = ts.normalize()
    minute = ts.hour * 60 + ts.minute
    selected: Optional[RuleWindow] = None

    for w in windows:
        left_ok = True if w.effective_from is None else d >= w.effective_from
        right_ok = True if w.effective_to is None else d <= w.effective_to
        if left_ok and right_ok:
            selected = w
    if selected is None:
        return None

    for seg in selected.segments:
        if seg["start_minute"] <= minute < seg["end_minute"]:
            return str(seg["period"])
    return None


def prepare_long_rows(
    df: pd.DataFrame,
    source_name: str,
    windows: List[RuleWindow],
) -> pd.DataFrame:
    metric_to_col = {
        "day_ahead_price": "day_ahead_price",
        "real_time_price": "real_time_price",
        "load": "load_actual",
    }

    pieces = []
    for metric, col in metric_to_col.items():
        if col not in df.columns:
            continue
        one = df[["times", col]].copy()
        one = one.rename(columns={col: "value"})
        one["metric"] = metric
        one = one[one["value"].notna()]
        pieces.append(one)

    if not pieces:
        return pd.DataFrame(columns=["ts", "metric", "value", "period", "source"])

    long_df = pd.concat(pieces, ignore_index=True)
    long_df["period"] = long_df["times"].apply(lambda x: pick_period(x, windows))
    long_df["ts"] = long_df["times"].dt.strftime("%Y-%m-%d %H:%M:%S")
    long_df["source"] = source_name
    long_df = long_df[["ts", "metric", "value", "period", "source"]]
    return long_df


def main() -> None:
    args = parse_args()
    csv_paths = [Path(x.strip()) for x in args.csv_files.split(",") if x.strip()]
    if not csv_paths:
        raise ValueError("No CSV files provided.")

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON;")

    province_id = get_province_id(conn, args.province_code)
    windows = load_rule_windows(conn, province_id)

    total_rows = 0
    total_by_metric: Dict[str, int] = {}

    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        df["times"] = pd.to_datetime(df["times"], errors="coerce", format="mixed")
        df = df[df["times"].notna()].copy()
        long_df = prepare_long_rows(df, path.name, windows)
        if long_df.empty:
            print(f"{path.name}: no rows to import")
            continue

        records = [
            (
                province_id,
                row.ts,
                row.metric,
                float(row.value),
                row.period,
                row.source,
            )
            for row in long_df.itertuples(index=False)
        ]

        conn.executemany(
            """
            INSERT INTO ts_value (province_id, ts, metric, value, period, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(province_id, ts, metric) DO UPDATE SET
              value = excluded.value,
              period = excluded.period,
              source = excluded.source
            """,
            records,
        )
        conn.commit()

        total_rows += len(records)
        for metric, cnt in long_df["metric"].value_counts().to_dict().items():
            total_by_metric[metric] = total_by_metric.get(metric, 0) + int(cnt)
        print(f"{path.name}: imported {len(records)} rows")

    db_total = conn.execute("SELECT COUNT(*) FROM ts_value").fetchone()[0]
    print(f"import_attempt_rows={total_rows}")
    print(f"import_by_metric={total_by_metric}")
    print(f"ts_value_total_rows={db_total}")
    conn.close()


if __name__ == "__main__":
    main()
