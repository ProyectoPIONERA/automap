"""
agents/ontology_entity_planner.py
===================================
Deterministic schema alignment — builds a Functional Entity Plan directly
from ontology owl:Class / owl:ObjectProperty / owl:DatatypeProperty
declarations + CSV column name-matching.

This replaces the expensive LLM call (often 200-300 s) for datasets
where the ontology fully describes the target schema.  The LLM is only
invoked as a fallback on retries or when the ontology is sparse.

Runtime: < 0.5 s vs 200-300 s for the LLM call.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher


# ─────────────────────────────────────────────────────────────────────
# 1.  Parse ontology Turtle into structured data
# ─────────────────────────────────────────────────────────────────────

def _build_prefix_map(ttl: str) -> dict[str, str]:
    """Extract ``@prefix`` declarations → {short_name: uri}."""
    result: dict[str, str] = {}
    for m in re.finditer(r'@prefix\s+(\w*):\s*<([^>]+)>', ttl):
        result[m.group(1)] = m.group(2)
    return result


def _resolve_curie(token: str, prefix_map: dict[str, str]) -> str:
    """Convert a full URI in angle brackets or a CURIE to a CURIE string."""
    token = token.strip()
    if token.startswith('<') and token.endswith('>'):
        uri = token[1:-1]
        for pfx, base in sorted(prefix_map.items(), key=lambda x: -len(x[1])):
            if uri.startswith(base):
                return f"{pfx}:{uri[len(base):]}"
        return uri
    return token


def parse_ontology(ttl: str) -> dict:
    """Parse Turtle ontology into structured dicts.

    Returns dict with keys: prefix_map, classes, obj_props, data_props.
    """
    prefix_map = _build_prefix_map(ttl)

    # ── owl:Class ────────────────────────────────────────────────
    classes: list[str] = []
    seen_classes: set[str] = set()
    for m in re.finditer(r'([\w:]+|<[^>]+>)\s+a\s+owl:Class\b', ttl):
        curie = _resolve_curie(m.group(1), prefix_map)
        if curie not in seen_classes:
            classes.append(curie)
            seen_classes.add(curie)

    # ── owl:ObjectProperty ───────────────────────────────────────
    obj_props: list[dict] = []
    for m in re.finditer(
        r'([\w:]+|<[^>]+>)\s+a\s+owl:ObjectProperty\s*;([^.]+)\.', ttl, re.DOTALL
    ):
        name = _resolve_curie(m.group(1), prefix_map)
        body = m.group(2)
        dm = re.search(r'rdfs:domain\s+([\w:]+|<[^>]+>)', body)
        rm = re.search(r'rdfs:range\s+([\w:]+|<[^>]+>)', body)
        lm = re.search(r'rdfs:label\s+"([^"]+)"', body)
        domain = _resolve_curie(dm.group(1), prefix_map) if dm else None
        range_ = _resolve_curie(rm.group(1), prefix_map) if rm else None
        label = lm.group(1) if lm else None
        obj_props.append({"name": name, "domain": domain, "range": range_, "label": label})

    # ── owl:DatatypeProperty ─────────────────────────────────────
    data_props: list[dict] = []
    for m in re.finditer(
        r'([\w:]+|<[^>]+>)\s+a\s+owl:DatatypeProperty\s*;([^.]+)\.', ttl, re.DOTALL
    ):
        name = _resolve_curie(m.group(1), prefix_map)
        body = m.group(2)
        dm = re.search(r'rdfs:domain\s+([\w:]+|<[^>]+>)', body)
        rm = re.search(r'rdfs:range\s+([\w:]+|<[^>]+>)', body)
        lm = re.search(r'rdfs:label\s+"([^"]+)"', body)
        domain = _resolve_curie(dm.group(1), prefix_map) if dm else None
        range_ = _resolve_curie(rm.group(1), prefix_map) if rm else None
        label = lm.group(1) if lm else None
        data_props.append({"name": name, "domain": domain, "range": range_, "label": label})

    return {
        "prefix_map": prefix_map,
        "classes": classes,
        "obj_props": obj_props,
        "data_props": data_props,
    }


# ─────────────────────────────────────────────────────────────────────
# 2.  Match CSV columns to ontology properties
# ─────────────────────────────────────────────────────────────────────

MATCH_THRESHOLD = 0.60


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _score_col_prop(col: str, prop_name: str, prop_label: str | None) -> float:
    """Return similarity score between a CSV column and an ontology property."""
    local = prop_name.split(":")[-1] if ":" in prop_name else prop_name
    score = _similarity(col, local)
    if prop_label:
        score = max(score, _similarity(col, prop_label))

    # Exact case-insensitive match
    if col.lower() == local.lower():
        return 1.0

    # Prefix / suffix overlap
    c_low, l_low = col.lower(), local.lower()
    if l_low.startswith(c_low) or c_low.startswith(l_low):
        score = max(score, 0.85)
    if l_low.endswith(c_low) or c_low.endswith(l_low):
        score = max(score, 0.80)

    return score


def match_columns_to_properties(
    columns: list[str],
    parsed: dict,
) -> dict[str, dict]:
    """Map each CSV column to its best-scoring ontology property."""
    all_props = (
        [(p, "object") for p in parsed["obj_props"]]
        + [(p, "data") for p in parsed["data_props"]]
    )
    result: dict[str, dict] = {}
    for col in columns:
        best_score = 0.0
        best_prop = None
        best_type = None
        for prop, ptype in all_props:
            s = _score_col_prop(col, prop["name"], prop.get("label"))
            if s > best_score:
                best_score = s
                best_prop = prop
                best_type = ptype
        if best_score >= MATCH_THRESHOLD and best_prop:
            result[col] = {
                "prop": best_prop["name"],
                "domain": best_prop.get("domain"),
                "range": best_prop.get("range"),
                "prop_type": best_type,
                "score": best_score,
                "xsd_range": best_prop.get("range", "xsd:string"),
            }
    return result


# ─────────────────────────────────────────────────────────────────────
# 3.  FK column detection
# ─────────────────────────────────────────────────────────────────────

def _find_fk_col_for_class(
    cls: str,
    obj_props: list[dict],
    columns: list[str],
) -> str | None:
    """Find the CSV column most likely to be the FK for a given class.

    Uses only generic ontology-driven matching (no hardcoded domain hints):
    1. Column name contains a fragment of the class local name
    2. Column name matches an ObjectProperty predicate name (via similarity)
    """
    cls_local = cls.split(":")[-1] if ":" in cls else cls

    # 1. Column name contains a meaningful fragment of the class local name
    if len(cls_local) >= 4:
        frag = cls_local[:5].lower()
        for col in columns:
            if frag in col.lower():
                return col

    # 2. Column name matches the ObjectProperty predicate name
    op_for_cls = [op for op in obj_props if op.get("range") == cls]
    for col in columns:
        for op in op_for_cls:
            op_local = op["name"].split(":")[-1] if ":" in op["name"] else op["name"]
            if _similarity(col, op_local) >= 0.70:
                return col

    return None


# ─────────────────────────────────────────────────────────────────────
# 4.  Build the entity plan text
# ─────────────────────────────────────────────────────────────────────

_INFRA_PREFIXES = {
    'rdf', 'rdfs', 'xsd', 'owl', 'rr', 'rml', 'ql',
}


def _primary_domain_prefix(prefix_map: dict[str, str]) -> str:
    """Pick the most likely 'domain' (non-infrastructure) prefix."""
    skip = _INFRA_PREFIXES | {'schema', 'gr', 'foaf', 'dc', 'dcterms', 'skos',
                               'dbo', 'dbr', 'geo', 'vcard', 'prov'}
    for pfx in prefix_map:
        if pfx not in skip:
            return pfx
    for pfx in prefix_map:
        if pfx not in _INFRA_PREFIXES:
            return pfx
    return 'ex'


def build_deterministic_entity_plan(
    ontology_ttl: str,
    csv_columns: list[str],
    base_uri: str = "http://example.org/",
) -> tuple[str, bool]:
    """Build a Functional Entity Plan deterministically from ontology + CSV.

    Returns (entity_plan_text, multi_node_bool).

    Raises ValueError when the ontology has no owl:Class declarations
    (caller should fall back to LLM).
    """
    parsed = parse_ontology(ontology_ttl)

    if not parsed["classes"]:
        raise ValueError("No owl:Class declarations found — cannot build deterministic plan")

    col_map = match_columns_to_properties(csv_columns, parsed)
    obj_props = parsed["obj_props"]
    prefix_map = parsed["prefix_map"]
    domain_pfx = _primary_domain_prefix(prefix_map)

    # Classes that are targets of ObjectProperties → need their own mappings
    range_classes: set[str] = {
        op["range"] for op in obj_props if op.get("range")
    }

    # All classes to create mappings for (declared + range classes)
    all_classes = list(dict.fromkeys(parsed["classes"] + list(range_classes)))
    multi_node = len(all_classes) > 1

    # Detect primary key column generically: first column whose name ends with
    # 'id', 'no', 'num', 'code', 'key', or equals the class local name
    def _looks_like_pk(col: str) -> bool:
        c = col.lower().replace("_", "").replace("-", "")
        return c.endswith(("id", "no", "num", "code", "key"))

    pk_col = next((c for c in csv_columns if _looks_like_pk(c)), None)

    lines: list[str] = [
        f"MULTI_NODE: {'true' if multi_node else 'false'}",
        f"PRIMARY_PREFIX: {domain_pfx}",
    ]
    if all_classes:
        lines.append(f"PRIMARY_CLASS: {all_classes[0]}")
    lines.append("")

    assigned_cols: set[str] = set()

    for cls in all_classes:
        cls_local = cls.split(":")[-1] if ":" in cls else cls
        mapping_name = f"{cls_local}Mapping"

        # ── Identify identifier column ─────────────────────────────
        if cls in range_classes:
            id_col = (
                _find_fk_col_for_class(cls, obj_props, csv_columns)
                or pk_col
                or csv_columns[0]
            )
        else:
            id_col = pk_col or csv_columns[0]

        # Subject template
        subj_template = f"ex:{cls_local}/$({id_col})"

        lines.append(f"ENTITY: {mapping_name}")
        lines.append(f"  CLASS: {cls}")
        lines.append(f"  IDENTIFIER: {id_col}")
        lines.append(f"  SUBJECT_TEMPLATE: {subj_template}")

        # ── Data properties for this class ─────────────────────────
        class_data = [
            (col, info) for col, info in col_map.items()
            if info["prop_type"] == "data" and info.get("domain") == cls
        ]
        if class_data:
            lines.append("  DATA_PROPERTIES:")
            for col, info in class_data:
                xsd = info.get("xsd_range", "xsd:string")
                lines.append(f"    - {col} -> {info['prop']} ({xsd})")
                assigned_cols.add(col)

        # ── Object-property links FROM this class ──────────────────
        outgoing = [op for op in obj_props if op.get("domain") == cls]
        if outgoing:
            lines.append("  URI_TEMPLATE_LINKS:")
            for op in outgoing:
                range_cls = op.get("range") or ""
                range_local = range_cls.split(":")[-1] if ":" in range_cls else range_cls
                fk_col = _find_fk_col_for_class(range_cls, obj_props, csv_columns)
                if fk_col:
                    lines.append(
                        f"    - {op['name']} -> ex:{range_local}/$({fk_col})~iri"
                    )

        lines.append("")

    # ── Report unmatched columns ───────────────────────────────────
    unmatched = [
        c for c in csv_columns
        if c not in col_map and c not in assigned_cols
    ]
    if unmatched:
        lines.append(
            "# UNMATCHED COLUMNS (no exact property match — entity agent will assign):"
        )
        lines.append(f"# {', '.join(unmatched)}")
        lines.append("")

    return "\n".join(lines), multi_node

