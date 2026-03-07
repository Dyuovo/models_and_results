import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze XGBoost feature importance files and generate visualizations."
    )
    parser.add_argument("--input_dir", type=str, default="results_fi_all_quick")
    parser.add_argument("--output_dir", type=str, default="results_fi_all_quick_analysis")
    parser.add_argument("--top_n", type=int, default=10)
    return parser.parse_args()


def feature_family(feature: str) -> str:
    if feature.startswith("lag_"):
        return "lag"
    if feature.startswith("roll_mean_"):
        return "roll_mean"
    if feature.startswith("roll_std_"):
        return "roll_std"
    if feature.startswith("diff_"):
        return "diff"
    if feature.startswith("holiday_name_"):
        return "holiday_name"
    if feature in {
        "is_holiday",
        "is_workday",
        "is_makeup_workday",
        "before_holiday_1d",
        "after_holiday_1d",
    }:
        return "holiday_binary"
    if feature in {
        "hour",
        "minute",
        "slot96",
        "sin_slot96",
        "cos_slot96",
        "dayofweek",
        "is_weekend",
        "month",
    }:
        return "calendar_time"
    return "other"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_inputs(input_dir: Path) -> Dict[str, pd.DataFrame]:
    summary_files = sorted(
        [
            p
            for p in input_dir.glob("feature_importance_*.csv")
            if not p.name.startswith("feature_importance_detail_")
        ]
    )
    detail_files = sorted(input_dir.glob("feature_importance_detail_*.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No summary files found in {input_dir}")
    if not detail_files:
        raise FileNotFoundError(f"No detail files found in {input_dir}")

    summary_frames: List[pd.DataFrame] = []
    for p in summary_files:
        df = pd.read_csv(p)
        df["source_file"] = p.name
        summary_frames.append(df)
    summary_all = pd.concat(summary_frames, ignore_index=True)

    detail_frames: List[pd.DataFrame] = []
    for p in detail_files:
        df = pd.read_csv(p)
        df["source_file"] = p.name
        detail_frames.append(df)
    detail_all = pd.concat(detail_frames, ignore_index=True)

    return {"summary": summary_all, "detail": detail_all}


def build_tables(summary_all: pd.DataFrame, detail_all: pd.DataFrame, top_n: int) -> Dict[str, pd.DataFrame]:
    s = summary_all.copy()
    d = detail_all.copy()
    s["feature_family"] = s["feature"].map(feature_family)
    d["feature_family"] = d["feature"].map(feature_family)

    target_feature = (
        s.groupby(["target", "feature", "feature_family"], as_index=False)["importance_mean"]
        .mean()
        .sort_values(["target", "importance_mean"], ascending=[True, False])
    )
    target_feature["rank_in_target"] = (
        target_feature.groupby("target")["importance_mean"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    top_target_feature = target_feature[target_feature["rank_in_target"] <= top_n].copy()

    scheme_segment_top = (
        s.sort_values(["target", "scheme", "segment", "importance_mean"], ascending=[True, True, True, False])
        .groupby(["target", "scheme", "segment"], as_index=False)
        .head(top_n)
        .copy()
    )

    family_share = (
        s.groupby(["target", "feature_family"], as_index=False)["importance_mean"]
        .mean()
        .sort_values(["target", "importance_mean"], ascending=[True, False])
    )
    family_share["family_share_pct"] = (
        family_share["importance_mean"]
        / family_share.groupby("target")["importance_mean"].transform("sum")
        * 100.0
    )

    stability = (
        d.groupby(["target", "scheme", "segment", "feature", "feature_family"], as_index=False)["importance"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "importance_mean", "std": "importance_std", "median": "importance_median", "count": "retrain_count"})
    )
    stability["importance_std"] = stability["importance_std"].fillna(0.0)
    stability["importance_cv"] = np.where(
        stability["importance_mean"].abs() > 1e-12,
        stability["importance_std"] / stability["importance_mean"].abs(),
        np.nan,
    )
    stability = stability.sort_values(["target", "importance_mean"], ascending=[True, False])

    stability_top = (
        stability[stability["importance_mean"] > 0]
        .sort_values(["target", "importance_cv"], ascending=[True, False])
        .groupby("target", as_index=False)
        .head(top_n)
    )

    return {
        "summary_all": s,
        "detail_all": d,
        "target_feature": target_feature,
        "top_target_feature": top_target_feature,
        "scheme_segment_top": scheme_segment_top,
        "family_share": family_share,
        "stability": stability,
        "stability_top": stability_top,
    }


def plot_top_features_by_target(top_target_feature: pd.DataFrame, output_path: Path, top_n: int) -> None:
    targets = sorted(top_target_feature["target"].unique().tolist())
    n = len(targets)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.2 * rows))
    axes = np.array(axes).reshape(-1)

    for i, target in enumerate(targets):
        ax = axes[i]
        sub = top_target_feature[top_target_feature["target"] == target].copy()
        sub = sub.sort_values("importance_mean", ascending=True).tail(top_n)
        ax.barh(sub["feature"], sub["importance_mean"], color="#2b7bba")
        ax.set_title(f"{target} Top-{top_n} Features")
        ax.set_xlabel("Mean Importance")
        ax.set_ylabel("Feature")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_family_share(family_share: pd.DataFrame, output_path: Path) -> None:
    pivot = family_share.pivot_table(
        index="target",
        columns="feature_family",
        values="family_share_pct",
        aggfunc="sum",
        fill_value=0.0,
    )
    pivot = pivot.sort_index()
    families = pivot.columns.tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(pivot))
    colors = plt.cm.tab20(np.linspace(0, 1, len(families)))

    for i, fam in enumerate(families):
        vals = pivot[fam].to_numpy()
        ax.bar(pivot.index, vals, bottom=bottom, label=fam, color=colors[i])
        bottom += vals

    ax.set_ylabel("Share of Importance (%)")
    ax.set_title("Feature Family Share by Target")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_heatmaps(summary_all: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    targets = sorted(summary_all["target"].unique().tolist())
    for target in targets:
        sub = summary_all[summary_all["target"] == target].copy()
        top_feats = (
            sub.groupby("feature", as_index=False)["importance_mean"]
            .mean()
            .sort_values("importance_mean", ascending=False)
            .head(top_n)["feature"]
            .tolist()
        )
        heat = (
            sub[sub["feature"].isin(top_feats)]
            .pivot_table(
                index="feature",
                columns="scheme",
                values="importance_mean",
                aggfunc="mean",
                fill_value=0.0,
            )
            .reindex(index=top_feats)
        )
        if heat.empty:
            continue

        heat_values = heat.to_numpy()
        vmin = np.nanpercentile(heat_values, 5)
        vmax = np.nanpercentile(heat_values, 95)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(heat))))
        im = ax.imshow(
            heat_values,
            aspect="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{target} Feature-Scheme Heatmap (Top-{top_n})")
        ax.set_xticks(np.arange(len(heat.columns)))
        ax.set_xticklabels(heat.columns, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(heat.index)))
        ax.set_yticklabels(heat.index)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Mean Importance")
        fig.tight_layout()
        fig.savefig(output_dir / f"heatmap_feature_scheme_{target}.png", dpi=180)
        plt.close(fig)


def write_report(tables: Dict[str, pd.DataFrame], output_path: Path, top_n: int) -> None:
    summary_all = tables["summary_all"]
    detail_all = tables["detail_all"]
    top_target_feature = tables["top_target_feature"]
    family_share = tables["family_share"]
    stability = tables["stability"]

    lines: List[str] = []
    lines.append("# Feature Importance Analysis Report")
    lines.append("")
    lines.append("## Dataset Overview")
    lines.append(f"- Summary rows: {len(summary_all):,}")
    lines.append(f"- Detail rows: {len(detail_all):,}")
    lines.append(f"- Targets: {', '.join(sorted(summary_all['target'].unique().tolist()))}")
    lines.append(f"- Schemes: {', '.join(sorted(summary_all['scheme'].unique().tolist()))}")
    lines.append("")
    lines.append("## Visual Outputs")
    lines.append("- top_features_by_target.png: Top-N features per target (mean importance).")
    lines.append("- feature_family_share_by_target.png: Family share stacked bars.")
    lines.append("- heatmap_feature_scheme_<target>.png: Feature vs scheme heatmaps (Top-N).")
    lines.append("")

    lines.append("## Top Features by Target")
    for target in sorted(top_target_feature["target"].unique().tolist()):
        lines.append(f"### {target}")
        sub = top_target_feature[top_target_feature["target"] == target].copy()
        sub = sub.sort_values("importance_mean", ascending=False).head(top_n)
        for _, r in sub.iterrows():
            lines.append(
                f"- {r['feature']} ({r['feature_family']}): mean_importance={r['importance_mean']:.6f}"
            )
        lines.append("")

    lines.append("## Feature Family Share (Top 5 each target)")
    for target in sorted(family_share["target"].unique().tolist()):
        lines.append(f"### {target}")
        sub = family_share[family_share["target"] == target].copy()
        sub = sub.sort_values("family_share_pct", ascending=False).head(5)
        for _, r in sub.iterrows():
            lines.append(f"- {r['feature_family']}: {r['family_share_pct']:.2f}%")
        lines.append("")

    lines.append("## Stability (By Retrain CV)")
    lines.append("- `importance_cv = std / |mean|`; larger means less stable across retrains.")
    for target in sorted(stability["target"].unique().tolist()):
        lines.append(f"### {target}")
        sub = stability[(stability["target"] == target) & (stability["importance_mean"] > 0)].copy()
        sub = sub.sort_values("importance_cv", ascending=False).head(5)
        for _, r in sub.iterrows():
            lines.append(
                f"- {r['feature']} ({r['scheme']}/{r['segment']}): cv={r['importance_cv']:.3f}, mean={r['importance_mean']:.6f}, retrains={int(r['retrain_count'])}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    raw = read_inputs(input_dir)
    tables = build_tables(raw["summary"], raw["detail"], top_n=args.top_n)

    # CSV outputs
    tables["summary_all"].to_csv(output_dir / "feature_importance_summary_all.csv", index=False)
    tables["detail_all"].to_csv(output_dir / "feature_importance_detail_all.csv", index=False)
    tables["target_feature"].to_csv(output_dir / "feature_importance_target_feature.csv", index=False)
    tables["top_target_feature"].to_csv(output_dir / "feature_importance_top_target_feature.csv", index=False)
    tables["scheme_segment_top"].to_csv(output_dir / "feature_importance_top_by_scheme_segment.csv", index=False)
    tables["family_share"].to_csv(output_dir / "feature_importance_family_share.csv", index=False)
    tables["stability"].to_csv(output_dir / "feature_importance_stability.csv", index=False)
    tables["stability_top"].to_csv(output_dir / "feature_importance_stability_top.csv", index=False)

    # Visual outputs
    plot_top_features_by_target(
        tables["top_target_feature"],
        output_dir / "top_features_by_target.png",
        top_n=args.top_n,
    )
    plot_family_share(
        tables["family_share"],
        output_dir / "feature_family_share_by_target.png",
    )
    plot_heatmaps(
        tables["summary_all"],
        output_dir,
        top_n=min(args.top_n, 12),
    )

    # Markdown report
    write_report(
        tables=tables,
        output_path=output_dir / "analysis_report.md",
        top_n=args.top_n,
    )
    print(f"Analysis outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
