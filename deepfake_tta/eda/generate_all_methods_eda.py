from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "results_data" / "all_methods_results.csv"
OUT_DIR = ROOT / "figures"
SUMMARY_DIR = ROOT / "summary"
METRIC_COLS = ["acc", "f1", "auc", "ap", "eer"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    px.defaults.template = "plotly_white"
    px.defaults.width = 1200
    px.defaults.height = 620

    df = pd.read_csv(CSV_PATH)
    df["level_raw"] = df["level"].astype(str)
    df["level_num"] = pd.to_numeric(df["level"], errors="coerce")
    df["is_train"] = df["level_raw"].str.lower().eq("train") | df["corruption"].eq("train_ffpp")
    df_eval = df[~df["is_train"]].copy()

    summary = df_eval.groupby("method", as_index=False)[METRIC_COLS].mean()
    summary["rank_acc"] = summary["acc"].rank(ascending=False, method="min")
    summary["rank_f1"] = summary["f1"].rank(ascending=False, method="min")
    summary["rank_auc"] = summary["auc"].rank(ascending=False, method="min")
    summary["rank_ap"] = summary["ap"].rank(ascending=False, method="min")
    summary["rank_eer"] = summary["eer"].rank(ascending=True, method="min")
    summary["mean_rank"] = summary[["rank_acc", "rank_f1", "rank_auc", "rank_ap", "rank_eer"]].mean(axis=1)
    summary = summary.sort_values("mean_rank")
    method_order = summary["method"].tolist()
    summary.to_csv(SUMMARY_DIR / "method_metric_summary.csv", index=False)

    summary_long = summary.melt(id_vars="method", value_vars=METRIC_COLS, var_name="metric", value_name="value")
    fig = px.bar(
        summary_long,
        x="method",
        y="value",
        color="method",
        facet_col="metric",
        facet_col_wrap=3,
        category_orders={"method": method_order, "metric": METRIC_COLS},
        title="Average Metrics By Method",
        text_auto=".3f",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    fig.write_html(OUT_DIR / "average_metrics_by_method.html")

    trend_long = (
        df_eval.groupby(["level_num", "method"], as_index=False)[METRIC_COLS]
        .mean()
        .melt(id_vars=["level_num", "method"], value_vars=METRIC_COLS, var_name="metric", value_name="value")
    )
    fig = px.line(
        trend_long,
        x="level_num",
        y="value",
        color="method",
        facet_col="metric",
        facet_col_wrap=3,
        markers=True,
        category_orders={"method": method_order, "metric": METRIC_COLS},
        title="Metric Trends Across Corruption Levels",
    )
    fig.update_layout(xaxis_title="Corruption level", yaxis_title="Metric value")
    fig.write_html(OUT_DIR / "metric_trends_by_level.html")

    corruption_long = (
        df_eval.groupby(["corruption", "method"], as_index=False)[METRIC_COLS]
        .mean()
        .melt(id_vars=["corruption", "method"], value_vars=METRIC_COLS, var_name="metric", value_name="value")
    )
    fig = px.bar(
        corruption_long,
        x="method",
        y="value",
        color="method",
        facet_row="metric",
        facet_col="corruption",
        category_orders={"method": method_order, "metric": METRIC_COLS},
        title="Mean Metrics By Corruption And Method",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-30, height=1300)
    fig.write_html(OUT_DIR / "metrics_by_corruption.html")

    heat = df_eval.copy()
    heat["condition"] = "L" + heat["level_num"].astype(int).astype(str) + " | " + heat["corruption"]
    for metric in METRIC_COLS:
        pivot = heat.pivot_table(index="method", columns="condition", values=metric, aggfunc="mean")
        pivot = pivot.reindex(method_order)
        fig = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale="RdYlGn_r" if metric == "eer" else "RdYlGn",
            title=f"{metric.upper()} Heatmap By Method And Condition",
            text_auto=".3f",
        )
        fig.update_layout(xaxis_title="Condition", yaxis_title="Method")
        fig.write_html(OUT_DIR / f"heatmap_{metric}.html")

    key_cols = ["level_num", "corruption"]
    baseline = (
        df_eval[df_eval["method"].eq("linear_probe")][key_cols + METRIC_COLS]
        .rename(columns={c: f"baseline_{c}" for c in METRIC_COLS})
    )
    delta = df_eval.merge(baseline, on=key_cols, how="left")
    for metric in METRIC_COLS:
        delta[f"delta_{metric}"] = delta[metric] - delta[f"baseline_{metric}"]

    delta_cols = [f"delta_{c}" for c in METRIC_COLS]
    delta_summary = (
        delta[~delta["method"].eq("linear_probe")]
        .groupby("method", as_index=False)[delta_cols]
        .mean()
    )
    delta_summary["mean_positive_delta_rank"] = (
        delta_summary["delta_acc"].rank(ascending=False)
        + delta_summary["delta_f1"].rank(ascending=False)
        + delta_summary["delta_auc"].rank(ascending=False)
        + delta_summary["delta_ap"].rank(ascending=False)
        + delta_summary["delta_eer"].rank(ascending=True)
    ) / 5
    delta_summary = delta_summary.sort_values("mean_positive_delta_rank")
    delta_summary.to_csv(SUMMARY_DIR / "delta_vs_linear_probe.csv", index=False)

    delta_long = delta_summary.melt(id_vars="method", value_vars=delta_cols, var_name="metric", value_name="delta")
    delta_long["metric"] = delta_long["metric"].str.replace("delta_", "", regex=False)
    fig = px.bar(
        delta_long,
        x="method",
        y="delta",
        color="delta",
        facet_col="metric",
        facet_col_wrap=3,
        color_continuous_scale="RdYlGn",
        category_orders={"method": delta_summary["method"].tolist(), "metric": METRIC_COLS},
        title="Mean Metric Delta vs Linear Probe",
        text_auto="+.3f",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(xaxis_tickangle=-30)
    fig.write_html(OUT_DIR / "delta_vs_linear_probe.html")

    best_rows = []
    for metric in METRIC_COLS:
        best_metric = (
            df_eval.sort_values(metric, ascending=(metric == "eer"))
            .groupby(["level_num", "corruption"], as_index=False)
            .first()[["level_num", "corruption", "method", metric]]
            .rename(columns={metric: "value"})
        )
        best_metric["metric"] = metric
        best_rows.append(best_metric)
    best_all = pd.concat(best_rows, ignore_index=True)
    best_all.to_csv(SUMMARY_DIR / "best_method_per_condition.csv", index=False)

    win_counts = best_all.value_counts(["metric", "method"]).rename("wins").reset_index()
    fig = px.bar(
        win_counts,
        x="method",
        y="wins",
        color="method",
        facet_col="metric",
        facet_col_wrap=3,
        category_orders={"method": method_order, "metric": METRIC_COLS},
        title="Best Method Count Per Metric",
        text_auto=True,
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    fig.write_html(OUT_DIR / "best_method_counts.html")

    print("saved summaries to:", SUMMARY_DIR)
    print("saved figures to:", OUT_DIR)
    print(summary[["method"] + METRIC_COLS + ["mean_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
