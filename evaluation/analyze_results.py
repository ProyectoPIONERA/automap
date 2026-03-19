"""
evaluation/analyze_results.py
==============================
Load experiment CSV logs and print research-paper-ready summary tables.

Usage:
    python -m evaluation.analyze_results data/experiments/experiment_log_*.csv
    python -m evaluation.analyze_results data/experiments/experiment_log_20260316_143000.csv
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_logs(paths: list[str]) -> pd.DataFrame:
    """Read one or more experiment CSV logs into a single DataFrame."""
    frames = []
    for p in paths:
        for f in glob.glob(p):
            frames.append(pd.read_csv(f))
    if not frames:
        print("[ERROR] No log files found.")
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def summarise(df: pd.DataFrame) -> None:
    """Print grouped summary tables for all available metrics."""

    group_cols = ["dataset", "llm"]

    print("\n" + "=" * 70)
    print(" EXPERIMENT SUMMARY")
    print("=" * 70)

    # ── Level 1 ──
    l1_cols = [c for c in df.columns if c.startswith("L1_")]
    if l1_cols:
        print("\n── Level 1: Pipeline Success Metrics ──\n")
        agg = {}
        if "L1_pipeline_success" in df.columns:
            # Success rate: mean of boolean gives fraction
            df["_L1_success_numeric"] = df["L1_pipeline_success"].astype(float)
            agg["_L1_success_numeric"] = ["mean", "sum", "count"]
        if "L1_retry_count" in df.columns:
            agg["L1_retry_count"] = ["mean", "std", "min", "max"]
        if "L1_total_triples" in df.columns:
            agg["L1_total_triples"] = ["mean", "std", "min", "max"]
        if "L1_total_latency_sec" in df.columns:
            agg["L1_total_latency_sec"] = ["mean", "std", "min", "max"]

        if agg:
            table = df.groupby(group_cols, dropna=False).agg(agg).round(3)
            # Flatten MultiIndex columns
            table.columns = ["_".join(c).strip("_") for c in table.columns]
            if "_L1_success_numeric_mean" in table.columns:
                table = table.rename(columns={
                    "_L1_success_numeric_mean": "success_rate",
                    "_L1_success_numeric_sum": "successes",
                    "_L1_success_numeric_count": "total_runs",
                })
            print(table.to_string())

    # ── Level 2 ──
    l2_cols = [c for c in df.columns if c.startswith("L2_") and c != "L2_skipped" and c != "L2_skip_reason"]
    if l2_cols:
        print("\n\n── Level 2: Gold-Standard KG Comparison ──\n")
        # Only rows where L2 was not skipped
        df_l2 = df[df.get("L2_skipped", True) == False].copy() if "L2_skipped" in df.columns else df.copy()
        if len(df_l2) == 0:
            print("  [WARN] No Level 2 results available (all skipped).")
        else:
            agg = {}
            for col in ["L2_norm_triple_precision", "L2_norm_triple_recall", "L2_norm_triple_f1",
                         "L2_predicate_precision", "L2_predicate_recall", "L2_predicate_f1",
                         "L2_class_precision", "L2_class_recall", "L2_class_f1"]:
                if col in df_l2.columns:
                    agg[col] = ["mean", "std"]
            if agg:
                table = df_l2.groupby(group_cols, dropna=False).agg(agg).round(4)
                table.columns = ["_".join(c).strip("_") for c in table.columns]
                print(table.to_string())

    # ── Level 3 ──
    l3_cols = [c for c in df.columns if c.startswith("L3_") and c != "L3_skipped" and c != "L3_skip_reason"]
    if l3_cols:
        print("\n\n── Level 3: Column Coverage ──\n")
        df_l3 = df[df.get("L3_skipped", True) == False].copy() if "L3_skipped" in df.columns else df.copy()
        if len(df_l3) == 0:
            print("  [WARN] No Level 3 results available (all skipped).")
        else:
            agg = {}
            for col in ["L3_column_coverage_by_yarrrml", "L3_columns_mapped_yarrrml",
                         "L3_column_coverage_by_value",
                         "L3_columns_total",
                         "L3_columns_mapped_value"]:
                if col in df_l3.columns:
                    agg[col] = ["mean", "std"]
            if agg:
                table = df_l3.groupby(group_cols, dropna=False).agg(agg).round(4)
                table.columns = ["_".join(c).strip("_") for c in table.columns]
                print(table.to_string())

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse experiment logs and print summary tables.",
    )
    parser.add_argument(
        "logs", nargs="+",
        help="Path(s) or glob(s) to experiment_log_*.csv files.",
    )
    args = parser.parse_args()

    df = load_logs(args.logs)
    print(f"Loaded {len(df)} experiment rows from {len(args.logs)} path(s).")
    summarise(df)


if __name__ == "__main__":
    main()

