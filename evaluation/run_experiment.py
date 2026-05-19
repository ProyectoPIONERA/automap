"""
evaluation/run_experiment.py
============================
Batch experiment runner for the Agentic RML Pipeline.

Runs every combination of (dataset × LLM × repetition), evaluates at
the requested level(s), and logs everything to a single CSV file that
you can load in pandas / Excel for your research paper.

Usage examples
--------------
    # Run with all 3 evaluation levels
    python -m evaluation.run_experiment --levels 1 2 3

    # Quick smoke-test: pipeline metrics only
    python -m evaluation.run_experiment --levels 1

    # Gold comparison + column coverage, 5 repetitions
    python -m evaluation.run_experiment --levels 2 3 --repeat 5

    # Only a specific dataset/LLM combo
    python -m evaluation.run_experiment --levels 1 2 3 --dataset bikeshare_weather --llm gemma3-12b
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from datetime import datetime
from itertools import product
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path so imports work when invoked with -m
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from evaluation.metrics import evaluate, print_metrics
from config.settings import get_llm_metadata


# ──────────────────────────────────────────────────────────────
# Configuration registry
# ──────────────────────────────────────────────────────────────

DATASETS: dict[str, dict] = {
    "bikeshare_weather": {
        "csv_path": "data/input/weather_cleaned.csv",
        "ontology_path": "data/input/ontology.ttl",
        "gold_kg_path": "data/gold/bikeshare_gold.nt",
        "base_uri": "http://example.com/",
    },
    # ── Add more datasets here ──────────────────────────────
    # "sensor_readings": {
    #     "csv_path": "data/input/sensors.csv",
    #     "ontology_path": "data/input/sensor_ontology.ttl",
    #     "gold_kg_path": "data/gold/sensor_gold.nt",
    #     "base_uri": "http://example.com/sensor/",
    # },
}

LLM_CONFIGS: dict[str, dict] = {
    "gemma3-12b": {
        "LM_STUDIO_MODEL": "google/gemma-3-12b",
        "LLM_MODEL_DEFAULT": "google/gemma-3-12b",
    },
    # ── Add more LLMs here ──────────────────────────────────
    # "mistral-nemo": {
    #     "LM_STUDIO_MODEL": "mistralai/mistral-nemo-instruct-2407",
    #     "LLM_MODEL_DEFAULT": "mistralai/mistral-nemo-instruct-2407",
    # },
    # "qwen2.5-14b": {
    #     "LM_STUDIO_MODEL": "qwen/qwen-2.5-14b-instruct",
    #     "LLM_MODEL_DEFAULT": "qwen/qwen-2.5-14b-instruct",
    # },
}


# ──────────────────────────────────────────────────────────────
# Single experiment run
# ──────────────────────────────────────────────────────────────

def run_single(
    dataset_name: str,
    dataset_cfg: dict,
    llm_name: str,
    llm_env: dict,
    levels: list[int],
    run_tag: str,
) -> dict:
    """Execute one full pipeline run, evaluate, and return metrics dict."""

    # Apply LLM env overrides
    for key, val in llm_env.items():
        os.environ[key] = val

    # Force fresh import so env changes take effect
    # (settings.py reads env at import-time via DEFAULT_MODEL)
    import importlib
    import config.settings as _cs
    importlib.reload(_cs)

    from graph.workflow import build_rml_graph

    app = build_rml_graph()

    run_dir = f"data/experiments/{run_tag}"
    os.makedirs(run_dir, exist_ok=True)

    initial_state = {
        "csv_path": dataset_cfg["csv_path"],
        "ontology_path": dataset_cfg["ontology_path"],
        "base_uri": dataset_cfg.get("base_uri", "http://example.org/"),
        "schema_info": {},
        "ontology_info": {},
        "mapping_plan": {},
        "prefixes_output": "",
        "entity_yarrrml": "",
        "yarrrml_output": "",
        "rdf_output": "",
        "feedback": "",
        "retry_count": 0,
        "cq_sparql_retry_count": 0,
        "generated_cqs": [],
        "user_sparql_queries": [],
        "sparql_validation_results": [],
        "messages": [],
        "run_dir": run_dir,
        "predicate_conflict_cols": [],
    }

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # ---------- Execute pipeline ----------
    t0 = time.time()
    for event in app.stream(initial_state, config):
        for node_name, output in event.items():
            print(f"  [{node_name}]", end=" ")
            if "messages" in output:
                print(output["messages"][-1][:80])
            else:
                print()
    elapsed = time.time() - t0

    result = app.get_state(config).values

    # Save YARRRML for inspection
    yarrrml = result.get("yarrrml_output", "")
    if yarrrml:
        with open(os.path.join(run_dir, "final_mapping.yaml"), "w") as f:
            f.write(yarrrml)

    # ---------- Evaluate ----------
    gold_path = dataset_cfg.get("gold_kg_path")
    metrics = evaluate(
        levels=levels,
        pipeline_result=result,
        gold_kg_path=gold_path,
        elapsed_time=elapsed,
    )

    # Attach experiment metadata
    metrics["run_id"] = run_tag
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["dataset"] = dataset_name
    metrics["llm"] = llm_name
    metrics["levels_evaluated"] = ",".join(str(l) for l in sorted(levels))

    # Stamp per-agent model & temperature configuration
    metrics.update(get_llm_metadata())

    # Save per-run JSON
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print_metrics(metrics, levels)

    return metrics


# ──────────────────────────────────────────────────────────────
# Batch runner
# ──────────────────────────────────────────────────────────────

def run_all(
    levels: list[int],
    repeat: int = 1,
    filter_dataset: str | None = None,
    filter_llm: str | None = None,
) -> list[dict]:
    """Run the full (dataset × LLM × repeat) experiment matrix."""

    datasets = {k: v for k, v in DATASETS.items()
                if filter_dataset is None or k == filter_dataset}
    llms = {k: v for k, v in LLM_CONFIGS.items()
            if filter_llm is None or k == filter_llm}

    if not datasets:
        print(f"[ERROR] Unknown dataset '{filter_dataset}'. Available: {list(DATASETS.keys())}")
        return []
    if not llms:
        print(f"[ERROR] Unknown LLM '{filter_llm}'. Available: {list(LLM_CONFIGS.keys())}")
        return []

    total_runs = len(datasets) * len(llms) * repeat
    all_metrics: list[dict] = []

    log_dir = "data/experiments"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )

    current = 0
    for (ds_name, ds_cfg), (llm_name, llm_env) in product(
        datasets.items(), llms.items()
    ):
        for run_num in range(repeat):
            current += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_tag = f"{ds_name}__{llm_name}__run{run_num}__{ts}"

            print(f"\n{'─' * 60}")
            print(f" EXPERIMENT {current}/{total_runs}")
            print(f" Dataset: {ds_name}  |  LLM: {llm_name}  |  Run #{run_num + 1}")
            print(f"{'─' * 60}")

            try:
                m = run_single(ds_name, ds_cfg, llm_name, llm_env, levels, run_tag)
                all_metrics.append(m)
            except Exception as exc:
                err = {
                    "run_id": run_tag,
                    "dataset": ds_name,
                    "llm": llm_name,
                    "L1_pipeline_success": False,
                    "error": str(exc),
                }
                all_metrics.append(err)
                print(f"  [ERROR] Crashed: {exc}")

            # Append-safe CSV write after every run
            _write_csv(log_path, all_metrics)

    print(f"\n[DONE] All experiments complete.  Log -> {log_path}")
    return all_metrics


def _write_csv(path: str, rows: list[dict]) -> None:
    """Write list-of-dicts to CSV (overwrites each time, crash-safe)."""
    if not rows:
        return
    keys = sorted(set().union(*(m.keys() for m in rows)))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            # Convert non-serializable values to strings
            safe = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                    for k, v in row.items()}
            w.writerow(safe)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation experiments for the Agentic RML Pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluation.run_experiment --levels 1
  python -m evaluation.run_experiment --levels 1 2 3 --repeat 3
  python -m evaluation.run_experiment --levels 2 3 --dataset bikeshare_weather --llm gemma3-12b
  python -m evaluation.run_experiment --levels 1 3 --repeat 5
        """,
    )
    parser.add_argument(
        "--levels", nargs="+", type=int, required=True,
        choices=[1, 2, 3],
        help="Evaluation levels to run (any combination of 1, 2, 3).",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Number of repetitions per (dataset, LLM) pair (default: 1).",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help=f"Run only this dataset. Available: {list(DATASETS.keys())}",
    )
    parser.add_argument(
        "--llm", type=str, default=None,
        help=f"Run only this LLM config. Available: {list(LLM_CONFIGS.keys())}",
    )

    args = parser.parse_args()

    print(f"\nEvaluation levels: {sorted(args.levels)}")
    print(f"Repetitions:       {args.repeat}")
    if args.dataset:
        print(f"Dataset filter:    {args.dataset}")
    if args.llm:
        print(f"LLM filter:        {args.llm}")
    print()

    run_all(
        levels=sorted(args.levels),
        repeat=args.repeat,
        filter_dataset=args.dataset,
        filter_llm=args.llm,
    )


if __name__ == "__main__":
    main()

