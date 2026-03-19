"""
main.py – Entry-point for the Agentic RML Pipeline.

Usage:
    # Normal run (no evaluation)
    python main.py

    # Run + Level 1 evaluation only
    python main.py --eval 1

    # Run + all 3 evaluation levels
    python main.py --eval 1 2 3

    # Run + Level 1 and 3 (skip gold comparison)
    python main.py --eval 1 3

    # Specify a custom gold KG path for Level 2
    python main.py --eval 1 2 3 --gold data/gold/bikeshare_gold.nt
"""

import argparse
import os
import sys
import time
import uuid
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Agentic RML Pipeline with optional multi-level evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval", nargs="*", type=int, default=None, dest="eval_levels",
        help="Evaluation levels to run after the pipeline. "
             "Any combination of 1, 2, 3.  Omit to skip evaluation.",
    )
    parser.add_argument(
        "--gold", type=str, default=None,
        help="Path to the gold-standard KG (N-Triples) for Level 2 evaluation.",
    )
    parser.add_argument(
        "--dashboard", "-d", nargs="?", type=int, const=5, default=None,
        metavar="N",
        help="Launch the metrics dashboard after the run. "
             "Optionally pass the number of latest runs to compare (default: 5). "
             "Example: --dashboard 3",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from graph.workflow import build_rml_graph

    app = build_rml_graph()

    # Create a unique timestamped directory for this specific run
    current_run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = f"data/output/run_{current_run_timestamp}"
    os.makedirs(run_directory, exist_ok=True)

    initial_state = {
        "csv_path": os.getenv("INPUT_CSV_PATH"),
        "ontology_path": os.getenv("INPUT_ONTOLOGY_PATH"),
        "base_uri": os.getenv("BASE_URI", "http://example.org/"),
        "schema_info": {},
        "ontology_info": {},
        "mapping_plan": {},
        "yarrrml_output": "",
        "rdf_output": "",
        "feedback": "",
        "retry_count": 0,
        "messages": [],
        "run_dir": run_directory,
        "predicate_conflict_cols": [],
    }

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("\n" + "=" * 50)
    print(" STARTING AGENTIC RML PIPELINE")
    print("=" * 50 + "\n")

    # ── Execute pipeline with timing ─────────────────────────────
    t0 = time.time()

    for event in app.stream(initial_state, config):
        for node_name, output in event.items():
            print(f" [STAGE]: {node_name}")

            if "messages" in output:
                print(f"    {output['messages'][-1]}")

            if node_name == "validate_yarrrml":
                status = "VALID" if "PASSED" in output.get("feedback", "") else "INVALID"
                print(f"    Syntax Status: {status}")

            if node_name == "refine_logic":
                print(f"    Refiner Feedback: {output.get('feedback', 'No feedback')}")
                status = " APPROVED" if output.get("feedback") == "APPROVED" else " NEEDS FIX"
                print(f"    Logic Review: {status}")

    elapsed = time.time() - t0

    print("\n" + "=" * 50)
    print(" PIPELINE COMPLETE")
    print("=" * 50)

    result = app.get_state(config).values

    if result.get("rdf_output"):
        print(f" Success! Knowledge Graph: {result['rdf_output']}")
    else:
        retries = result.get('retry_count', 0)
        feedback = result.get('feedback', '')
        if retries >= 6 and "LOGIC_ERROR" in feedback:
            print(f"[FAIL] Failure: Logic retry limit reached ({retries} attempts). Last feedback:")
            # Print just the first few lines of feedback for clarity
            for line in feedback.split('\n')[:8]:
                print(f"    {line}")
        elif retries >= 10 and "SYNTAX_ERROR" in feedback:
            print(f"[FAIL] Failure: Syntax retry limit reached ({retries} attempts).")
        else:
            print("[FAIL] Failure: Knowledge Graph was not generated.")

    # Save YARRRML
    yarrrml_content = result.get("yarrrml_output", "")
    if yarrrml_content:
        mapping_filename = os.path.join(run_directory, "final_mapping.yaml")
        with open(mapping_filename, "w") as f:
            f.write(yarrrml_content)
        print(f" Mapping saved to: {mapping_filename}")

    # Check RDF output
    rdf_path = result.get("rdf_output", "")
    if rdf_path and os.path.exists(rdf_path):
        print(f" Knowledge Graph generated at: {rdf_path}")
    else:
        print("Knowledge Graph was not generated. Check agent feedback.")

    print("-" * 30)
    print(f"Total loop attempts: {result.get('retry_count', 0)}")
    print(f"Total time: {elapsed:.1f}s")
    print("-" * 30)

    # ── Evaluation ───────────────────────────────────────────────
    eval_levels = args.eval_levels
    if eval_levels is not None:
        # --eval with no numbers means "all levels"
        if not eval_levels:
            eval_levels = [1, 2, 3]

        # Validate
        eval_levels = sorted(set(eval_levels))
        invalid = [l for l in eval_levels if l not in (1, 2, 3)]
        if invalid:
            print(f"[WARN] Ignoring unknown eval levels: {invalid}")
            eval_levels = [l for l in eval_levels if l in (1, 2, 3)]

        if eval_levels:
            from evaluation.metrics import evaluate, print_metrics
            from config.settings import get_llm_metadata
            import json

            gold_path = args.gold
            # Fallback: try standard location
            if gold_path is None and 2 in eval_levels:
                default_gold = "data/gold/bikeshare_gold.nt"
                if os.path.isfile(default_gold):
                    gold_path = default_gold
                else:
                    print("[WARN] No --gold path provided and data/gold/bikeshare_gold.nt not found.")
                    print("   Level 2 evaluation will be skipped.")

            # Ensure the pipeline result has csv_path for Level 3
            if "csv_path" not in result:
                result["csv_path"] = os.getenv("INPUT_CSV_PATH")

            metrics = evaluate(
                levels=eval_levels,
                pipeline_result=result,
                gold_kg_path=gold_path,
                elapsed_time=elapsed,
            )

            # Stamp model / temperature configuration into metrics
            metrics.update(get_llm_metadata())

            print_metrics(metrics, eval_levels)

            # Save metrics JSON alongside the run
            metrics_file = os.path.join(run_directory, "eval_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            print(f"\nEvaluation metrics saved to: {metrics_file}")

    # ── Dashboard ────────────────────────────────────────────────
    if args.dashboard is not None:
        from tools.dashboard import main as dashboard_main
        print(f"\nLaunching metrics dashboard (last {args.dashboard} runs)...")
        sys.argv = [
            "dashboard",
            "--runs", str(args.dashboard),
            "--output-dir", "data/output",
        ]
        dashboard_main()


if __name__ == "__main__":
    main()

