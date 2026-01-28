#!/usr/bin/env python3

"""Visualize DARP solution metrics produced by solve_and_extract_metrics.py."""

from __future__ import annotations

import argparse
import csv
import json
import os
import glob
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _find_latest_metrics(metrics_dir: str) -> str:
    patterns = [
        os.path.join(metrics_dir, "*_darp_metrics_*.csv"),
        os.path.join(metrics_dir, "*_darp_metrics_*.json"),
    ]
    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No metrics files found under {metrics_dir}")
    return max(candidates, key=os.path.getmtime)


def _load_metrics(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".csv"):
        records: List[Dict[str, Any]] = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        return records
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Metrics JSON must be a list of per-company results")
    return data


def _split_ok_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ok = [r for r in records if r.get("solve_status") == "ok"]
    not_ok = [r for r in records if r.get("solve_status") != "ok"]
    return ok, not_ok


def _plot_hist(values: List[float], title: str, xlabel: str, output_path: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=25, color="#4c78a8", edgecolor="black", alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_scatter(x: List[float], y: List[float], title: str, xlabel: str, ylabel: str, output_path: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7, color="#59a14f", s=40, edgecolor="black", linewidth=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_status_bar(ok_count: int, fail_count: int, output_path: str) -> None:
    plt.figure(figsize=(8, 5))
    labels = ["ok", "not_ok"]
    values = [ok_count, fail_count]
    colors = ["#4c78a8", "#e15759"]
    bars = plt.bar(labels, values, color=colors, edgecolor="black")
    plt.title("Solve Status Count")
    plt.xlabel("Status")
    plt.ylabel("Count")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f"{int(height)}",
                 ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_delay_box(total_delays: List[float], output_path: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.boxplot([total_delays], labels=["Total"])
    plt.title("Delay Distribution (minutes)")
    plt.ylabel("Delay")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _color_scale(values: List[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return [0.5 for _ in arr]
    vmin = np.min(finite)
    vmax = np.max(finite)
    if np.isclose(vmin, vmax):
        return [0.5 for _ in arr]
    scaled = (arr - vmin) / (vmax - vmin)
    scaled = np.where(np.isfinite(scaled), scaled, 0.5)
    return list(scaled)


def _natural_key(text: str) -> List[object]:
    parts = re.split(r"(\d+)", str(text))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _plot_all_distributions(df: pd.DataFrame, exclude_cols: List[str], output_path: str) -> bool:
    numeric_cols = [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_cols:
        return False

    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, numeric_cols):
        values = df[col].dropna()
        ax.hist(values, bins=15, color="#457b9d", alpha=0.85)
        ax.set_title(col, fontsize=9)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(numeric_cols):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def _save_case_summary_table(case_df: pd.DataFrame, output_path: str) -> bool:
    if case_df.empty:
        return False

    cols = [
        "case_name",
        "company_count",
        "solve_ok_rate",
        "solution_cost_mean",
        "total_time_mean",
        "vehicles_used_mean",
        "served_clients_mean",
        "avg_total_delay_mean",
        "max_total_delay_mean",
    ]
    cols = [c for c in cols if c in case_df.columns]
    if not cols:
        return False

    table_df = case_df[cols].copy()
    numeric_cols = [c for c in cols if c != "case_name"]

    max_rows = 40
    if len(table_df) > max_rows:
        table_df = table_df.head(max_rows)

    fig, ax = plt.subplots(figsize=(12, max(3, 0.35 * len(table_df))))
    ax.axis("off")
    cell_text = table_df.values.tolist()
    table = ax.table(cellText=cell_text, colLabels=cols, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.1)

    for j, col in enumerate(numeric_cols, start=1):
        colors = _color_scale(table_df[col].values)
        for i, color in enumerate(colors, start=1):
            table[(i, j)].set_facecolor((1.0 - color, 1.0, 1.0 - color))

    for j in range(len(cols)):
        table[(0, j)].set_facecolor("#2a9d8f")
        table[(0, j)].set_text_props(color="white", weight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def _build_case_summary(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["solve_status"] = df.get("solve_status", "")
    df["case_name"] = df.get("case_name", "")
    df["company_name"] = df.get("company_name", "")

    numeric_cols = [
        "num_vehicles",
        "num_clients",
        "solve_time_sec",
        "solution_cost",
        "total_time",
        "total_waiting_time",
        "vehicles_used",
        "vehicle_utilization",
        "served_clients",
        "total_delays",
        "avg_total_delay",
        "max_total_delay",
        "max_route_time",
        "avg_route_time",
        "capacity_utilization",
        "time_utilization",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    group = df.groupby("case_name")
    status = group["solve_status"].agg(
        solve_ok_rate=lambda s: (s == "ok").mean(),
        solve_infeasible_rate=lambda s: (s == "infeasible").mean(),
    )
    counts = group["company_name"].nunique().rename("company_count")

    ok_df = df[df["solve_status"] == "ok"]
    if ok_df.empty:
        numeric_summary = pd.DataFrame(index=status.index)
    else:
        ok_group = ok_df.groupby("case_name")
        available_numeric = [col for col in numeric_cols if col in ok_df.columns]
        numeric_summary = ok_group[available_numeric].mean().add_suffix("_mean")

    summary = pd.concat([counts, status, numeric_summary], axis=1).reset_index()
    summary = summary.sort_values("case_name", key=lambda s: s.map(_natural_key)).reset_index(drop=True)
    return summary


def visualize_metrics(metrics_path: str, output_dir: str) -> List[str]:
    records = _load_metrics(metrics_path)
    ok_records, not_ok_records = _split_ok_records(records)

    os.makedirs(output_dir, exist_ok=True)
    outputs: List[str] = []

    _plot_status_bar(len(ok_records), len(not_ok_records), os.path.join(output_dir, "solve_status.png"))
    outputs.append("solve_status.png")

    case_summary = _build_case_summary(records)
    if not case_summary.empty:
        case_summary_path = os.path.join(output_dir, "case_summary.csv")
        case_summary.to_csv(case_summary_path, index=False)
        outputs.append("case_summary.csv")

        case_table_path = os.path.join(output_dir, "case_summary_table.png")
        if _save_case_summary_table(case_summary, case_table_path):
            outputs.append("case_summary_table.png")

    if not ok_records:
        return outputs

    costs = [_to_float(r.get("solution_cost", 0)) for r in ok_records]
    total_times = [_to_float(r.get("total_time", 0)) for r in ok_records]
    num_clients = [_to_float(r.get("num_clients", 0)) for r in ok_records]
    vehicles_used = [_to_float(r.get("vehicles_used", 0)) for r in ok_records]
    avg_total_delay = [_to_float(r.get("avg_total_delay", 0)) for r in ok_records]

    _plot_hist(costs, "Solution Cost Distribution", "Solution Cost", os.path.join(output_dir, "cost_distribution.png"))
    outputs.append("cost_distribution.png")

    _plot_hist(total_times, "Total Route Time Distribution", "Total Time (minutes)",
               os.path.join(output_dir, "total_time_distribution.png"))
    outputs.append("total_time_distribution.png")

    _plot_scatter(num_clients, costs, "Cost vs Clients", "Number of Clients", "Solution Cost",
                  os.path.join(output_dir, "cost_vs_clients.png"))
    outputs.append("cost_vs_clients.png")

    _plot_scatter(num_clients, total_times, "Total Time vs Clients", "Number of Clients",
                  "Total Time (minutes)", os.path.join(output_dir, "time_vs_clients.png"))
    outputs.append("time_vs_clients.png")

    _plot_scatter(vehicles_used, costs, "Cost vs Vehicles Used", "Vehicles Used", "Solution Cost",
                  os.path.join(output_dir, "cost_vs_vehicles.png"))
    outputs.append("cost_vs_vehicles.png")

    _plot_delay_box(avg_total_delay, os.path.join(output_dir, "avg_delay_boxplot.png"))
    outputs.append("avg_delay_boxplot.png")

    ok_df = pd.DataFrame(ok_records)
    exclude_cols = ["case_name", "company_name", "solve_status"]
    dist_path = os.path.join(output_dir, "darp_metric_distributions.png")
    if _plot_all_distributions(ok_df, exclude_cols, dist_path):
        outputs.append("darp_metric_distributions.png")

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DARP metrics output")
    parser.add_argument("--input", help="Path to *_darp_metrics_*.csv or .json (defaults to latest)")
    parser.add_argument("--metrics-dir", default="metrics_out", help="Directory to search for latest metrics file")
    parser.add_argument("--output-dir", default="metrics_out/visuals", help="Directory to write plots")
    args = parser.parse_args()

    metrics_path = args.input or _find_latest_metrics(args.metrics_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"plots_{timestamp}")

    outputs = visualize_metrics(metrics_path, output_dir)
    print(f"Plots written to: {output_dir}")
    for name in outputs:
        print(f"- {name}")


if __name__ == "__main__":
    main()
