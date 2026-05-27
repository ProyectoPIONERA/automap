"""
evaluation/metrics.py
=====================
Three levels of evaluation metrics for the Agentic RML Pipeline.

Level 1 – Pipeline Success Metrics   (automated, zero-cost)
Level 2 – Gold-Standard KG Comparison (triple-level Precision / Recall / F1)
Level 3 – Column Coverage & Structural Completeness

Usage:
    from evaluation.metrics import evaluate

    # Run specific levels
    results = evaluate(
        levels=[1, 2, 3],           # any subset of {1, 2, 3}
        pipeline_result=result,     # dict returned by the pipeline
        gold_kg_path="data/gold/bikeshare_gold.nt",  # needed for level 2
    )
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

import pandas as pd
from rdflib import Graph, RDF, URIRef


# ────────────────────────────────────────────────────────────────────
# URI / normalisation helpers
# ────────────────────────────────────────────────────────────────────

def _uri_tail(uri_str: str) -> str:
    """Extract the trailing local name from a URI.

    Used for predicate and class names (not row IDs — see
    :func:`_extract_row_id` for that).

    Examples
    --------
    >>> _uri_tail("http://example.com/rental/42")
    '42'
    >>> _uri_tail("http://example.org/bikeshare#hasInstant")
    'hasInstant'
    >>> _uri_tail("http://example.com/temporal_context/42")
    '42'
    """
    s = str(uri_str)
    if '#' in s:
        s = s.rsplit('#', 1)[-1]
    return s.rsplit('/', 1)[-1] if '/' in s else s


def _extract_row_id(uri_str: str) -> str:
    """Extract the row-level identifier from an entity URI.

    Handles both ``/path/ID`` and ``/ClassName_ID`` patterns that LLMs
    commonly produce.  Falls back to the full trailing segment when no
    numeric suffix is found (e.g. class URIs in rdf:type objects).

    Examples
    --------
    >>> _extract_row_id("http://example.com/rental/42")
    '42'
    >>> _extract_row_id("http://example.com/BikeRentalRecord_42")
    '42'
    >>> _extract_row_id("http://example.com/TemporalContext_42")
    '42'
    >>> _extract_row_id("http://example.org/bikeshare#BikeRentalRecord")
    'BikeRentalRecord'
    """
    s = str(uri_str)
    if '#' in s:
        s = s.rsplit('#', 1)[-1]
    tail = s.rsplit('/', 1)[-1] if '/' in s else s
    # If the tail contains '_' and ends with a numeric-looking ID,
    # extract just the ID portion.  This tolerates LLM-generated
    # patterns like "BikeRentalRecord_42" or "temporal_context_42".
    if '_' in tail:
        last_seg = tail.rsplit('_', 1)[-1]
        # Accept digits, possibly with hyphens/dots (e.g. UUIDs, timestamps)
        if last_seg and (last_seg[0].isdigit() or last_seg == last_seg.lower()):
            # Only use the short segment when it looks like an ID,
            # not when it's a class-name component (starts with uppercase
            # and contains no digits).
            if last_seg[0].isdigit():
                return last_seg
    return tail


def _normalize_triple(s, p, o):
    """Create a normalised triple for URI-tolerant comparison.

    Normalisation rules
    -------------------
    - **Subject**:  reduced to the row-level identifier via
      :func:`_extract_row_id`.  This tolerates different URI path
      conventions (``/rental/42`` vs ``/BikeRentalRecord_42`` both
      become ``'42'``).
    - **Predicate**: kept as the full URI (predicate naming matters).
    - **Object URI**: tagged ``'iri'`` and reduced to its row-level
      identifier (for object-property links) or local name (for class
      URIs).
    - **Object Literal**: tagged ``'lit'`` with lexical value only
      (datatype stripped so that ``"1"^^xsd:integer`` and
      ``"1"^^xsd:float`` both become ``'1'``).

    The type tag ensures a URI reference and a literal with the same
    trailing value are never conflated (e.g.
    ``<.../location-type/1>`` vs ``"1"^^xsd:string``).

    This allows fair comparison of KGs that map the same CSV rows with
    the same predicates but use different URI path templates.
    """
    row_id = _extract_row_id(str(s))
    pred = str(p)
    if isinstance(o, URIRef):
        obj = ("iri", _extract_row_id(str(o)))
    else:
        # Literal – keep just the lexical value (str() strips datatype)
        obj = ("lit", str(o))
    return (row_id, pred, obj)


def _extract_yarrrml_columns(yarrrml_str: str) -> set[str]:
    r"""Extract all ``$(column_name)`` references from a YARRRML string.

    This is a ground-truth method: it tells us exactly which CSV columns
    were referenced in the mapping, regardless of how predicates are named.

    >>> sorted(_extract_yarrrml_columns('s: "ex:r/$(instant)"\\npo: ["p","$(temp)"]'))
    ['instant', 'temp']
    """
    if not yarrrml_str:
        return set()
    return set(re.findall(r'\$\(([^)]+)\)', yarrrml_str))


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return (precision, recall, f1) from TP / FP / FN counts."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# ────────────────────────────────────────────────────────────────────
# Public entry-point
# ────────────────────────────────────────────────────────────────────

def evaluate(
    levels: list[int],
    pipeline_result: dict[str, Any],
    gold_kg_path: str | None = None,
    elapsed_time: float | None = None,
) -> dict[str, Any]:
    """Run the requested evaluation levels and return a merged metrics dict.

    Parameters
    ----------
    levels : list[int]
        Which evaluation levels to run.  Any combination of [1, 2, 3].
    pipeline_result : dict
        The state dict returned by the LangGraph pipeline. Expected keys:
        ``yarrrml_output``, ``rdf_output``, ``retry_count``, ``csv_path``,
        ``feedback``, ``messages``.
    gold_kg_path : str | None
        Path to the gold-standard N-Triples KG file.
        Required when level 2 is selected.
    elapsed_time : float | None
        Wall-clock seconds for the pipeline run (level 1).
        If None the field is omitted.

    Returns
    -------
    dict  –  flat dictionary of all computed metrics.
    """
    metrics: dict[str, Any] = {}

    if 1 in levels:
        metrics.update(level1_pipeline_metrics(pipeline_result, elapsed_time))
    if 2 in levels:
        metrics.update(level2_gold_comparison(pipeline_result, gold_kg_path))
    if 3 in levels:
        metrics.update(level3_column_coverage(pipeline_result))

    # Always compute CQ coverage when sparql_validation_results are present
    metrics.update(level4_cq_coverage(pipeline_result))

    return metrics


# ────────────────────────────────────────────────────────────────────
# Level 1 – Pipeline Success Metrics
# ────────────────────────────────────────────────────────────────────

def level1_pipeline_metrics(
    result: dict[str, Any],
    elapsed_time: float | None = None,
) -> dict[str, Any]:
    """Compute cheap, always-available pipeline health metrics.

    Metrics produced
    ----------------
    - yarrrml_produced        : bool
    - yarrrml_syntactic_valid : bool   (parseable YAML)
    - yarrrml_translatable    : bool   (yatter succeeded at least once)
    - rml_materializable      : bool   (KG file exists and has content)
    - pipeline_success        : bool
    - retry_count             : int
    - total_triples           : int    (0 when KG was not generated)
    - total_latency_sec       : float  (if provided)
    """
    yarrrml = result.get("yarrrml_output", "")
    rdf_path = result.get("rdf_output", "")

    # -- YARRRML produced? --
    yarrrml_produced = bool(yarrrml and yarrrml.strip())

    # -- Syntactically valid YAML? --
    yarrrml_syntactic_valid = False
    if yarrrml_produced:
        try:
            from ruamel.yaml import YAML
            yaml = YAML(typ="safe", pure=True)
            yaml.load(yarrrml)
            yarrrml_syntactic_valid = True
        except Exception:
            pass

    # -- Translatable to RML? --
    yarrrml_translatable = False
    if yarrrml_syntactic_valid:
        try:
            import yatter
            from ruamel.yaml import YAML
            yaml = YAML(typ="safe", pure=True)
            data = yaml.load(yarrrml)
            rml = yatter.translate(data)
            yarrrml_translatable = rml is not None and len(rml) > 0
        except Exception:
            pass

    # -- KG materialised? --
    rml_materializable = False
    total_triples = 0
    if rdf_path and os.path.isfile(rdf_path) and os.path.getsize(rdf_path) > 0:
        rml_materializable = True
        try:
            g = Graph()
            g.parse(rdf_path, format="ntriples")
            total_triples = len(g)
        except Exception:
            total_triples = -1  # file exists but unparseable

    pipeline_success = rml_materializable and total_triples > 0

    metrics = {
        "L1_yarrrml_produced": yarrrml_produced,
        "L1_yarrrml_syntactic_valid": yarrrml_syntactic_valid,
        "L1_yarrrml_translatable": yarrrml_translatable,
        "L1_rml_materializable": rml_materializable,
        "L1_pipeline_success": pipeline_success,
        "L1_retry_count": result.get("retry_count", 0),
        "L1_total_triples": total_triples,
    }
    if elapsed_time is not None:
        metrics["L1_total_latency_sec"] = round(elapsed_time, 2)

    return metrics


# ────────────────────────────────────────────────────────────────────
# Level 2 – Gold-Standard KG Comparison
# ────────────────────────────────────────────────────────────────────

def level2_gold_comparison(
    result: dict[str, Any],
    gold_kg_path: str | None,
) -> dict[str, Any]:
    """Compare the generated KG against a gold-standard reference KG.

    Metrics produced
    ----------------
    **Normalised triple-level** (subject URIs reduced to row IDs,
    object URIs to local names, literal datatypes stripped):
        L2_norm_triple_precision, L2_norm_triple_recall, L2_norm_triple_f1

    **Predicate-level** (schema coverage):
        L2_predicate_precision, L2_predicate_recall, L2_predicate_f1

    **Class-level** (rdf:type object coverage):
        L2_class_precision, L2_class_recall, L2_class_f1
    """
    rdf_path = result.get("rdf_output", "")

    # Guard: cannot run without both paths
    if not gold_kg_path or not os.path.isfile(gold_kg_path or ""):
        return {"L2_skipped": True, "L2_skip_reason": "gold_kg_path missing or not found"}
    if not rdf_path or not os.path.isfile(rdf_path or ""):
        return {"L2_skipped": True, "L2_skip_reason": "generated KG missing"}

    g_gen = Graph()
    g_gold = Graph()
    try:
        g_gen.parse(rdf_path, format="ntriples")
        g_gold.parse(gold_kg_path, format="ntriples")
    except Exception as exc:
        return {"L2_skipped": True, "L2_skip_reason": f"parse error: {exc}"}

    # ── 1. Normalised triple comparison (tolerant) ─────────────
    #   Reduces subject URIs to their trailing row-ID segment and
    #   object URIs to local names so that KGs using different
    #   path templates (e.g. /rental/42 vs /bikerentalrecord/42)
    #   can still be compared meaningfully.
    #   Object type (iri vs lit) is preserved to avoid conflating
    #   URI references with literal values.
    gen_norm = {_normalize_triple(s, p, o) for s, p, o in g_gen}
    gold_norm = {_normalize_triple(s, p, o) for s, p, o in g_gold}

    tp_n = len(gen_norm & gold_norm)
    fp_n = len(gen_norm - gold_norm)
    fn_n = len(gold_norm - gen_norm)
    prec_n, rec_n, f1_n = _prf(tp_n, fp_n, fn_n)

    # ── 1b. Object-type mismatch diagnostic ────────────────────
    #   Identifies predicates where the generated and gold KGs agree
    #   on subject-id + predicate + object-value BUT disagree on the
    #   object type (IRI vs Literal).  This is a common issue when the
    #   YARRRML maps a column as a string literal instead of an IRI
    #   reference (e.g. locationType, wheelchairAccessible).
    gen_pred_obj_types: dict[str, set[str]] = defaultdict(set)
    gold_pred_obj_types: dict[str, set[str]] = defaultdict(set)
    for _, p_uri, (otype, _) in gen_norm:
        gen_pred_obj_types[p_uri].add(otype)
    for _, p_uri, (otype, _) in gold_norm:
        gold_pred_obj_types[p_uri].add(otype)

    type_mismatch_preds: list[str] = []
    for pred_uri in gen_pred_obj_types.keys() & gold_pred_obj_types.keys():
        gen_types = gen_pred_obj_types[pred_uri]
        gold_types = gold_pred_obj_types[pred_uri]
        if gen_types != gold_types:
            tail = _uri_tail(str(pred_uri))
            g_str = "/".join(sorted(gen_types))
            go_str = "/".join(sorted(gold_types))
            type_mismatch_preds.append(
                f"{tail}(gen={g_str}, gold={go_str})"
            )

    # ── 2. Predicate-level (schema coverage) ──────────────────
    gen_preds = {p for _, p, _ in g_gen}
    gold_preds = {p for _, p, _ in g_gold}
    pred_tp = len(gen_preds & gold_preds)
    pred_fp = len(gen_preds - gold_preds)
    pred_fn = len(gold_preds - gen_preds)
    pred_p, pred_r, pred_f1 = _prf(pred_tp, pred_fp, pred_fn)

    # ── 3. Class-level (objects of rdf:type) ──────────────────
    gen_classes = {o for _, p, o in g_gen if p == RDF.type}
    gold_classes = {o for _, p, o in g_gold if p == RDF.type}
    cls_tp = len(gen_classes & gold_classes)
    cls_fp = len(gen_classes - gold_classes)
    cls_fn = len(gold_classes - gen_classes)
    cls_p, cls_r, cls_f1 = _prf(cls_tp, cls_fp, cls_fn)

    # ── Diagnostic: mismatched predicates ─────────────────────
    missing_preds = sorted(_uri_tail(str(p)) for p in (gold_preds - gen_preds))
    extra_preds = sorted(_uri_tail(str(p)) for p in (gen_preds - gold_preds))

    return {
        "L2_skipped": False,
        # Normalised triple-level
        "L2_norm_triple_precision": round(prec_n, 4),
        "L2_norm_triple_recall": round(rec_n, 4),
        "L2_norm_triple_f1": round(f1_n, 4),
        "L2_norm_true_positives": tp_n,
        "L2_norm_false_positives": fp_n,
        "L2_norm_false_negatives": fn_n,
        "L2_total_generated": len(g_gen),
        "L2_total_gold": len(g_gold),
        # Predicate-level
        "L2_predicate_precision": round(pred_p, 4),
        "L2_predicate_recall": round(pred_r, 4),
        "L2_predicate_f1": round(pred_f1, 4),
        "L2_unique_predicates_gen": len(gen_preds),
        "L2_unique_predicates_gold": len(gold_preds),
        "L2_predicates_missing": missing_preds,
        "L2_predicates_extra": extra_preds,
        # Class-level
        "L2_class_precision": round(cls_p, 4),
        "L2_class_recall": round(cls_r, 4),
        "L2_class_f1": round(cls_f1, 4),
        "L2_classes_gen": len(gen_classes),
        "L2_classes_gold": len(gold_classes),
        # Object-type mismatch diagnostic
        "L2_object_type_mismatches": len(type_mismatch_preds),
        "L2_object_type_mismatch_details": sorted(type_mismatch_preds),
    }


# ────────────────────────────────────────────────────────────────────
# Level 3 – Column Coverage & Structural Completeness
# ────────────────────────────────────────────────────────────────────

def level3_column_coverage(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Check how many CSV columns are represented in the generated output.

    Two complementary methods are used:

    1. **YARRRML template match** (primary, ground truth) -- parses all
       ``$(column_name)`` references directly from the YARRRML output.
       This is 100% accurate regardless of predicate naming.
    2. **Literal-value match** -- for the first rows of the CSV, checks
       if a column's value appears as an RDF literal object.

    Metrics produced
    ----------------
    L3_column_coverage_by_yarrrml    : float   (0.0 - 1.0)  [PRIMARY]
    L3_column_coverage_by_value      : float   (0.0 - 1.0)
    L3_columns_total                 : int
    """
    rdf_path = result.get("rdf_output", "")
    csv_path = result.get("csv_path", "")
    yarrrml = result.get("yarrrml_output", "")

    if not csv_path or not os.path.isfile(csv_path):
        return {"L3_skipped": True, "L3_skip_reason": "csv_path missing"}

    try:
        df = pd.read_csv(csv_path, nrows=5)
    except Exception as exc:
        return {"L3_skipped": True, "L3_skip_reason": f"csv read error: {exc}"}

    csv_columns = list(df.columns)
    total = len(csv_columns)

    # ── Method 1: YARRRML template references (ground truth) ──
    yarrrml_refs = _extract_yarrrml_columns(yarrrml)
    mapped_yarrrml = {col for col in csv_columns if col in yarrrml_refs}

    # ── Method 2 requires the KG ─────────────────────────────
    g = None
    if rdf_path and os.path.isfile(rdf_path):
        try:
            g = Graph()
            g.parse(rdf_path, format="ntriples")
        except Exception:
            g = None

    # ── Method 2: Literal-value match (first row) ────────────
    mapped_val = set()
    if g is not None:
        first_row = df.iloc[0].astype(str).to_dict()
        all_objects_str = {str(o) for _, _, o in g}
        for col, val in first_row.items():
            if val and val != "nan" and val in all_objects_str:
                mapped_val.add(col)

    missing_yarrrml = sorted(set(csv_columns) - mapped_yarrrml)
    missing_val = sorted(set(csv_columns) - mapped_val)

    return {
        "L3_skipped": False,
        "L3_columns_total": total,
        # YARRRML-based (primary, ground truth)
        "L3_columns_mapped_yarrrml": len(mapped_yarrrml),
        "L3_column_coverage_by_yarrrml": round(len(mapped_yarrrml) / total, 4) if total else 0.0,
        "L3_columns_missing_yarrrml": missing_yarrrml,
        # Value-based
        "L3_columns_mapped_value": len(mapped_val),
        "L3_column_coverage_by_value": round(len(mapped_val) / total, 4) if total else 0.0,
        "L3_columns_missing_value": missing_val,
    }


# ────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────
# Level 4 – CQ / SPARQL Validation Coverage
# ────────────────────────────────────────────────────────────────────

def level4_cq_coverage(result: dict[str, Any]) -> dict[str, Any]:
    """Compute CQ validation coverage from sparql_validation_results.

    Treats null results (SPARQL generation errors) as failures so the
    reported coverage is conservative and honest.

    Metrics produced
    ----------------
    L4_cq_total       : int   — total CQs attempted
    L4_cq_passed      : int   — ASK returned true
    L4_cq_failed      : int   — ASK returned false
    L4_cq_error       : int   — SPARQL could not be generated / executed
    L4_cq_coverage    : float — passed / total (0.0 if no CQs)
    L4_cq_skipped     : bool
    """
    sparql_results = result.get("sparql_validation_results", [])
    if not sparql_results:
        return {"L4_cq_skipped": True}

    n_passed = sum(1 for r in sparql_results if r.get("passed") is True)
    n_failed = sum(1 for r in sparql_results if r.get("passed") is False)
    n_error  = sum(1 for r in sparql_results if r.get("passed") is None)
    n_total  = len(sparql_results)

    # null counts as failure for coverage calculation (conservative)
    effective_pass = n_passed
    effective_total = n_total

    return {
        "L4_cq_skipped": False,
        "L4_cq_total": n_total,
        "L4_cq_passed": n_passed,
        "L4_cq_failed": n_failed,
        "L4_cq_error": n_error,
        "L4_cq_coverage": round(effective_pass / effective_total, 4) if effective_total else 0.0,
    }


# Pretty-print helper (for terminal / main.py)
# ────────────────────────────────────────────────────────────────────

def print_metrics(metrics: dict[str, Any], levels: list[int]) -> None:
    """Human-readable summary printed to stdout."""
    print("\n" + "=" * 60)
    print(" EVALUATION RESULTS")
    print("=" * 60)

    if 1 in levels:
        print("\n── Level 1: Pipeline Success Metrics ──")
        _print_kv("YARRRML produced", metrics.get("L1_yarrrml_produced"))
        _print_kv("YARRRML syntactic valid", metrics.get("L1_yarrrml_syntactic_valid"))
        _print_kv("YARRRML translatable (yatter)", metrics.get("L1_yarrrml_translatable"))
        _print_kv("RML materializable (KG)", metrics.get("L1_rml_materializable"))
        _print_kv("Pipeline success", metrics.get("L1_pipeline_success"))
        _print_kv("Retry count", metrics.get("L1_retry_count"))
        _print_kv("Total triples", metrics.get("L1_total_triples"))
        if "L1_total_latency_sec" in metrics:
            _print_kv("Total latency (sec)", metrics["L1_total_latency_sec"])

    if 2 in levels:
        print("\n── Level 2: Gold-Standard KG Comparison ──")
        if metrics.get("L2_skipped"):
            print(f"  [WARN] Skipped -- {metrics.get('L2_skip_reason', 'unknown')}")
        else:
            print("\n  ▸ Normalised Triple Match (subject URIs -> row IDs,"
                  " object URIs -> local names)")
            _print_kv("Precision", metrics.get("L2_norm_triple_precision"))
            _print_kv("Recall", metrics.get("L2_norm_triple_recall"))
            _print_kv("F1", metrics.get("L2_norm_triple_f1"))
            _print_kv("True positives", metrics.get("L2_norm_true_positives"))
            _print_kv("False positives", metrics.get("L2_norm_false_positives"))
            _print_kv("False negatives", metrics.get("L2_norm_false_negatives"))

            print("\n  ▸ Schema-Level")
            _print_kv("Predicate precision", metrics.get("L2_predicate_precision"))
            _print_kv("Predicate recall", metrics.get("L2_predicate_recall"))
            _print_kv("Predicate F1", metrics.get("L2_predicate_f1"))
            missing_p = metrics.get("L2_predicates_missing", [])
            if missing_p:
                _print_kv("Predicates missing", ", ".join(missing_p))
            extra_p = metrics.get("L2_predicates_extra", [])
            if extra_p:
                _print_kv("Predicates extra (hallucinated)", ", ".join(extra_p))
            _print_kv("Class precision", metrics.get("L2_class_precision"))
            _print_kv("Class recall", metrics.get("L2_class_recall"))
            _print_kv("Class F1", metrics.get("L2_class_f1"))

            # Object-type mismatch diagnostic
            type_mm = metrics.get("L2_object_type_mismatches", 0)
            if type_mm > 0:
                print(f"\n  ▸ Object-Type Mismatches (IRI vs Literal)")
                _print_kv("Predicates with type mismatch", type_mm)
                details = metrics.get("L2_object_type_mismatch_details", [])
                for d in details:
                    print(f"      [WARN] {d}")

    if 3 in levels:
        print("\n── Level 3: Column Coverage ──")
        if metrics.get("L3_skipped"):
            print(f"  [WARN] Skipped -- {metrics.get('L3_skip_reason', 'unknown')}")
        else:
            _print_kv("Columns total", metrics.get("L3_columns_total"))

            print("\n  ▸ YARRRML Template References (ground truth)")
            _print_kv("Mapped", metrics.get("L3_columns_mapped_yarrrml"))
            _print_kv("Coverage", metrics.get("L3_column_coverage_by_yarrrml"))
            missing_y = metrics.get("L3_columns_missing_yarrrml", [])
            if missing_y:
                _print_kv("Missing", ", ".join(missing_y))


            print("\n  ▸ Literal Value Match (first CSV row)")
            _print_kv("Mapped", metrics.get("L3_columns_mapped_value"))
            _print_kv("Coverage", metrics.get("L3_column_coverage_by_value"))
            missing_v = metrics.get("L3_columns_missing_value", [])
            if missing_v:
                _print_kv("Missing", ", ".join(missing_v))

    # ── LLM Configuration (if stamped into metrics) ─────────
    if metrics.get("llm_provider"):
        print("\n── LLM Configuration ──")
        _print_kv("Provider", metrics.get("llm_provider"))
        _print_kv("Default model", metrics.get("llm_default_model"))
        for role in ["schema_agent", "mapper_agent", "yarrrml_architect", "refiner"]:
            model = metrics.get(f"{role}_model")
            temp = metrics.get(f"{role}_temperature")
            if model:
                print(f"  [MODEL] {role}: {model}  (temp={temp})")

    # CQ coverage (always shown if data present)
    if not metrics.get("L4_cq_skipped"):
        print("\n── Level 4: CQ / SPARQL Validation Coverage ──")
        _print_kv("Total CQs", metrics.get("L4_cq_total"))
        _print_kv("Passed", metrics.get("L4_cq_passed"))
        _print_kv("Failed", metrics.get("L4_cq_failed"))
        if metrics.get("L4_cq_error", 0):
            _print_kv("Errors (SPARQL gen failed)", metrics.get("L4_cq_error"))
        _print_kv("Coverage", metrics.get("L4_cq_coverage"))

    print()


def _print_kv(key: str, value: Any) -> None:
    if value is True:
        print(f"  [PASS] {key}: Yes")
    elif value is False:
        print(f"  [FAIL] {key}: No")
    else:
        print(f"  {key}: {value}")

