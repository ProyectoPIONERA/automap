"""
Side-by-side quality comparison:
  - Research pipeline  →  experiments/llm4rml/diamonds-json/diamonds-reference.nt
  - Your pipeline      →  /home/naveen/Documents/LLM-Agents_RML/data/output/.../knowledge_graph.nt

Both are evaluated against the same gold standard (diamonds-reference.nt IS the
research pipeline's gold/reference, so its score is the theoretical ceiling).

Run from the kg-pipeline project root:
    python compare_my_pipeline.py
"""

import json, os
from rdflib import Graph
from kg_core.metrics.metrics import RML_Evaluation, precision_score, recall_score, f1_score

# ── Paths ─────────────────────────────────────────────────────────────────────
GOLD_STANDARD   = "experiments/llm4rml/diamonds-json/diamonds-reference.nt"
YOUR_KG         = "/home/naveen/Documents/LLM-Agents_RML/data/output/run_20260527_123946/knowledge_graph.nt"
RESEARCH_KG     = "experiments/llm4rml/diamonds-json/gpt-4-0125-preview_101.ttl.nt"   # research pipeline output = gold standard


def _fuzzy_subject_score(test_g: Graph, ref_g: Graph) -> dict:
    """Match subjects by their nm/tt ID token, ignoring different base IRIs."""
    def subject_ids(g):
        return set(str(s).rstrip('/').split('/')[-1] for s, p, o in g)

    test_ids = subject_ids(test_g)
    ref_ids  = subject_ids(ref_g)
    tp = len(test_ids & ref_ids)
    fp = len(test_ids) - tp
    fn = len(ref_ids)  - tp
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision_score(tp, fp),
        'recall':    recall_score(tp, fn),
        'f1':        f1_score(tp, fp, fn),
    }


def evaluate(test_path: str, gold_path: str, base_iri: str, label: str) -> dict:
    print(f"\nEvaluating [{label}] …")
    gold_g = Graph()
    gold_g.parse(gold_path, format="nt")

    test_g = Graph()
    fmt = "nt" if test_path.endswith(".nt") else "turtle"
    test_g.parse(test_path, format=fmt)

    print(f"  Gold triples : {len(gold_g)}")
    print(f"  Test triples : {len(test_g)}")

    ev = RML_Evaluation(test_g, gold_g, base_iri=base_iri)

    return {
        'label':               label,
        'total_triples_test':  len(test_g),
        'total_triples_gold':  len(gold_g),
        'triples':             ev.tripleScore(),
        'subjects_exact':      ev.subjects(),
        'subjects_fuzzy':      _fuzzy_subject_score(test_g, gold_g),
        'classes':             ev.classes(),
        'classes_unique':      ev.classes_unique(),
        'predicates':          ev.properties(),
        'predicates_unique':   ev.properties_unique(),
        'objects_literals':    ev.objects_literals(),
        'objects_uris':        ev.objects_uris(),
        'additional':          ev.additionalStats(),
    }


# ── Run both evaluations ───────────────────────────────────────────────────────
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")   # suppress "does not look like a valid URI" noise

    research = evaluate(
        RESEARCH_KG,
        GOLD_STANDARD,
        base_iri="http://mykg.org/resource/",
        label="Research pipeline (LLM4RML / gold standard)"
    )

    yours = evaluate(
        YOUR_KG,
        GOLD_STANDARD,
        base_iri="http://mykg.org/resource/",
        label="Your pipeline (Qwen 2.5 14B)"
    )

# ── Side-by-side comparison table ─────────────────────────────────────────────
METRICS = [
    ("triples",           "Exact triple match"),
    ("subjects_exact",    "Subject URIs (exact)"),
    ("subjects_fuzzy",    "Subject IDs (fuzzy, IRI-agnostic)"),
    ("classes",           "Class assignments (rdf:type)"),
    ("classes_unique",    "Unique classes used"),
    ("predicates",        "Predicate usage"),
    ("predicates_unique", "Unique predicates used"),
    ("objects_literals",  "Literal values"),
    ("objects_uris",      "URI object values"),
]

COL_W = 28

def f1_of(d):
    return d.get('f1', 0) if isinstance(d, dict) else 0

def fmt_row(name, r_val, y_val, label_w=36):
    r_f1 = f"{r_val:.3f}" if isinstance(r_val, float) else str(r_val)
    y_f1 = f"{y_val:.3f}" if isinstance(y_val, float) else str(y_val)
    delta = y_val - r_val if isinstance(y_val, float) and isinstance(r_val, float) else ""
    delta_s = f"  ({'+' if delta >= 0 else ''}{delta:.3f})" if delta != "" else ""
    return f"  {name:<{label_w}} {r_f1:>8}   {y_f1:>8}{delta_s}"

print("\n")
print("=" * 75)
print("  SIDE-BY-SIDE COMPARISON  (F1 scores)")
print("=" * 75)
print(f"  {'Metric':<36} {'Research':>8}   {'Yours':>8}   Delta")
print("-" * 75)

for key, label in METRICS:
    r = f1_of(research.get(key, {}))
    y = f1_of(yours.get(key, {}))
    print(fmt_row(label, r, y))

print("-" * 75)

# Overall average F1
r_avg = sum(f1_of(research.get(k, {})) for k, _ in METRICS) / len(METRICS)
y_avg = sum(f1_of(yours.get(k, {})) for k, _ in METRICS) / len(METRICS)
print(fmt_row("AVERAGE F1", r_avg, y_avg))
print("=" * 75)

# Additional stats comparison
print("\n  ADDITIONAL STATS")
print("-" * 75)
add_keys = [
    ("allPersonHaveId",         "All 10 persons present (1=yes)"),
    ("allAHaveId",              "All 4 actors present  (1=yes)"),
    ("countPersonIds",          "Person entity count"),
    ("countActorIds",           "Actor entity count"),
    ("countPersonIdsPersonType","Persons typed as dbo:Person"),
    ("countActorIdsActorType",  "Actors typed as dbo:Actor"),
    ("allTargetPredicatesMapped","All ref predicates mapped (1=yes)"),
    ("onlyTargetPredicatesMapped","Only ref predicates used (1=yes)"),
]
for key, label in add_keys:
    r_v = research['additional'].get(key, '?')
    y_v = yours['additional'].get(key, '?')
    print(f"  {label:<36} {str(r_v):>8}   {str(y_v):>8}")

print("=" * 75)
print(f"\n  Research pipeline total triples : {research['total_triples_test']}")
print(f"  Your pipeline total triples     : {yours['total_triples_test']}")
print(f"  Gold standard triples           : {research['total_triples_gold']}")

# ── Save full JSON results ─────────────────────────────────────────────────────
os.makedirs("target", exist_ok=True)
OUTPUT = "target/comparison_results.json"
with open(OUTPUT, "w") as f:
    json.dump({'research': research, 'yours': yours}, f, indent=2, default=str)
print(f"\n✓ Full results saved to {OUTPUT}\n")
