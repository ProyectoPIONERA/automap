"""
agents/cq_to_sparql_agent.py
============================
LLM agent that translates a single Competency Question (CQ) into an
ASK SPARQL query grounded in the provided ontology context.

Design principles (from Tufek et al.):
- Feed ontology prefix map + classes/properties into the prompt so URIs
  are grounded to the actual ontology, not hallucinated.
- Use ASK queries (true/false) — avoids "empty SELECT vs. wrong pattern"
  ambiguity.
- Wrap execution in try/except; if SPARQL is syntactically invalid,
  re-prompt with the parse error as feedback (self-correction loop).
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_llm

# Prefixes always injected into generated SPARQL to avoid "undefined prefix" errors
_ALWAYS_DECLARE = {
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
}

# ── Static system prompt ──────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a SPARQL query generator for RDF Knowledge Graphs.

Your task: translate a natural-language Competency Question (CQ) into a
SPARQL 1.1 ASK query that can be executed against the generated KG.

### RULES
1. Always produce an ASK query (never SELECT or CONSTRUCT).
2. Use ONLY the prefixes provided in the MAPPING PREFIXES section below.
   Do NOT invent new namespace URIs — use the exact URIs from the mapping.
3. CRITICAL: Copy the PREFIX declarations VERBATIM from MAPPING PREFIXES.
   Do NOT change http:// to https:// or vice versa.  Do NOT change any URI.
   The KG was built with these exact URIs — any change causes queries to fail.
4. Use PREFIX declarations at the top of every query.
5. Use ?var names that match the domain concepts in the CQ.
6. Keep the triple pattern minimal — capture the essential structure.
7. If the CQ asks about a relationship between two entities, express it
   as a path: ?a ex:relatesTo ?b .
8. For ASK queries — use simple existence patterns, NOT FILTER on specific
   values. For boolean-like fields, check triple EXISTENCE only: ?s ex:prop ?val
   Do NOT filter on specific boolean values like true/false/"Yes"/"No".
9. Output ONLY the SPARQL query — no markdown fences, no explanation.
10. CRITICAL: Write numeric literals as BARE values, NEVER quoted typed literals.
    CORRECT:   FILTER(?val = 0.98)    FILTER(?count = 5)
    WRONG:     FILTER(?val = "0.98"^^xsd:decimal)   ← INVALID in strict SPARQL parsers
    This applies to ALL numeric types: decimal, integer, float, double, long.
11. CRITICAL: NEVER use a prefix that is not listed in MAPPING PREFIXES.
    If you want to use schema:, foaf:, dbo: etc. — check they are in MAPPING PREFIXES.
    If a concept maps to an undeclared prefix (e.g. schema:Patient), use ex:Patient instead.
    Using an undeclared prefix causes a parse error that fails the entire CQ.
12. CRITICAL: Use ONLY the entity types and predicates from MAPPING CONTEXT below.
    Do NOT invent class names. If the mapping has schema:OrderItem, do NOT use schema:Order.
    Use the EXACT class and predicate names shown in MAPPING CONTEXT.

### OUTPUT FORMAT
PREFIX ex: <http://example.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX ...

ASK {
  ?subject a ex:Class ;
           ex:property ?object .
}
"""


def _fix_typed_numeric_literals(sparql: str) -> str:
    """Convert quoted typed numeric literals to bare literals.

    SPARQL strict parsers (pyoxigraph) reject:
      "3.14"^^xsd:decimal   "5"^^xsd:integer
    and require:
      3.14                  5

    This post-processor normalises all such patterns deterministically.
    """
    # Decimal / double / float: "3.14"^^xsd:decimal → 3.14
    sparql = re.sub(
        r'"(-?\d+\.\d+)"\^\^xsd:(?:decimal|double|float)',
        r'\1',
        sparql,
    )
    # Integer / int / long: "5"^^xsd:integer → 5
    sparql = re.sub(
        r'"(-?\d+)"\^\^xsd:(?:integer|int|long)',
        r'\1',
        sparql,
    )
    # Boolean: "true"^^xsd:boolean → true
    sparql = re.sub(
        r'"(true|false)"\^\^xsd:boolean',
        r'\1',
        sparql,
        flags=re.IGNORECASE,
    )
    return sparql


def fix_undeclared_prefix_refs(sparql: str, prefix_map: dict[str, str]) -> str:
    """Replace any prefix:Term usage in the WHERE body where prefix is not declared.

    When the LLM writes e.g. `schema:Patient` but `schema` is not in the
    YARRRML prefix map, the SPARQL parser throws a syntax error.  This post-
    processor detects such tokens and remaps them to the `ex:` domain prefix
    (which is always declared) so the query at least parses cleanly.

    Fully agnostic — reads the actual prefix map from the generated YARRRML.
    """
    merged = dict(_ALWAYS_DECLARE)
    merged.update(prefix_map)
    declared = set(merged.keys())

    body_start = sparql.find("{")
    if body_start == -1:
        return sparql

    header = sparql[:body_start]
    body = sparql[body_start:]

    # Also collect prefixes already declared in the header
    header_declared = set(re.findall(r'PREFIX\s+(\w+)\s*:', header, re.IGNORECASE))
    all_declared = declared | header_declared

    def _replace_unknown_prefix(m: re.Match) -> str:
        pfx = m.group(1)
        local = m.group(2)
        if pfx in all_declared:
            return m.group(0)
        # Remap to ex: (always declared)
        return f"ex:{local}"

    body = re.sub(
        r'\b([A-Za-z][A-Za-z0-9_-]*):([A-Za-z][A-Za-z0-9_]*)\b',
        _replace_unknown_prefix,
        body,
    )
    return header + body


def probe_kg_types_and_predicates(kg_path: str) -> dict[str, list[str]]:
    """Query the materialised KG for all distinct rdf:type values and predicates.

    Returns {"classes": [...], "predicates": [...]} with the actual URIs
    present in the KG.  This is used to ground the SPARQL generation prompt
    so the LLM uses classes/predicates that ACTUALLY EXIST in the KG.

    Falls back to empty lists on any error (non-blocking).
    """
    result = {"classes": [], "predicates": []}
    if not kg_path:
        return result
    try:
        import pyoxigraph
        store = pyoxigraph.Store()
        with open(kg_path, "rb") as _f:
            store.load(_f, format=pyoxigraph.RdfFormat.N_TRIPLES)

        classes_q = "SELECT DISTINCT ?t WHERE { ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?t } LIMIT 40"
        preds_q   = "SELECT DISTINCT ?p WHERE { ?s ?p ?o } LIMIT 60"

        classes = []
        for row in store.query(classes_q):
            val = str(row[0])
            if val not in ("http://www.w3.org/1999/02/22-rdf-syntax-ns#type",):
                classes.append(val)

        preds = []
        for row in store.query(preds_q):
            val = str(row[0])
            if val != "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                preds.append(val)

        result["classes"] = classes
        result["predicates"] = preds
    except Exception:
        pass
    return result


def build_kg_grounding_block(kg_probe: dict[str, list[str]]) -> str:
    """Build a prompt block from the KG probe result.

    This block is injected into the SPARQL generation prompt so the LLM
    uses ONLY types and predicates that actually exist in the materialised KG.
    """
    if not kg_probe or (not kg_probe.get("classes") and not kg_probe.get("predicates")):
        return ""
    lines = [
        "### KG_CLASSES — ONLY use these rdf:type values in your SPARQL (they are the ACTUAL types in the KG):",
    ]
    for cls in kg_probe.get("classes", [])[:30]:
        lines.append(f"  <{cls}>")
    lines.append("### KG_PREDICATES — ONLY use these predicates in your SPARQL:")
    for pred in kg_probe.get("predicates", [])[:50]:
        lines.append(f"  <{pred}>")
    lines.append(
        "CRITICAL: Do NOT write ?s a schema:Order if schema:Order is NOT in KG_CLASSES above. "
        "Use the EXACT URIs listed in KG_CLASSES and KG_PREDICATES."
    )
    return "\n".join(lines)


def extract_yarrrml_mapping_context(yarrrml_str: str) -> str:
    """Extract the entity types and predicates actually used in the YARRRML mapping.

    This is injected into the CQ→SPARQL prompt so the LLM uses the CORRECT
    class names from the KG rather than hallucinating wrong ones.

    Example output:
      ### MAPPING CONTEXT (use ONLY these types and predicates in SPARQL):
      Entity types declared in mapping:
        - schema:OrderItem  (in OrderItemMapping)
        - schema:Person     (in CustomerMapping)
      Predicates declared in mapping:
        - schema:quantity, schema:customer, schema:orderedItem, ...

    Fully agnostic — reads directly from the generated YARRRML.
    """
    if not yarrrml_str:
        return ""

    try:
        import yaml as _yaml
        data = _yaml.safe_load(yarrrml_str)
    except Exception:
        return ""

    if not isinstance(data, dict) or "mappings" not in data:
        return ""

    mappings = data.get("mappings", {})
    entity_types: list[str] = []
    predicates: list[str] = []
    seen_preds: set[str] = set()

    for mname, mdef in (mappings or {}).items():
        if not isinstance(mdef, dict):
            continue
        for entry in (mdef.get("po") or []):
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            pred = str(entry[0])
            if pred in ("a", "rdf:type"):
                # Entity type declaration
                type_val = str(entry[1])
                entity_types.append(f"  - {type_val}  (in {mname})")
            elif pred not in seen_preds:
                predicates.append(pred)
                seen_preds.add(pred)

    if not entity_types and not predicates:
        return ""

    lines = ["### MAPPING CONTEXT (use ONLY these types and predicates in SPARQL):"]
    if entity_types:
        lines.append("Entity types declared in mapping:")
        lines.extend(entity_types[:20])
    if predicates:
        lines.append("Predicates declared in mapping:")
        lines.append("  " + ", ".join(predicates[:40]))
    return "\n".join(lines)


def extract_yarrrml_prefixes(yarrrml_str: str) -> dict[str, str]:
    """Extract the prefix → URI map declared in a YARRRML mapping string.

    Handles both quoted and unquoted URI values:
      ex: "http://example.org/fraud#"
      ex: http://example.org/fraud#
      ex: 'http://example.org/fraud#'

    This is the KEY fix for Bug 2 — ensures SPARQL queries use the same
    namespaces as the materialised KG, not hallucinated URIs.

    Auto-generated mapping-name prefixes (e.g. MetadataMapping, OrderItemMapping)
    are filtered out — they are YARRRML coordinator artefacts and have no place
    in SPARQL queries.
    """
    prefixes: dict[str, str] = {}
    # Look for lines inside the prefixes: block
    in_prefixes = False
    for line in yarrrml_str.splitlines():
        stripped = line.strip()
        if stripped.startswith("prefixes:"):
            in_prefixes = True
            continue
        if in_prefixes:
            if stripped and not stripped.startswith("#"):
                # Check indent — prefixes block entries are indented 2 spaces
                if not line.startswith("  ") and not line.startswith("\t"):
                    break  # left the prefixes block
                # Parse:  name: "URI"  or  name: URI
                m = re.match(
                    r"""^\s{2}([A-Za-z][A-Za-z0-9_]*):\s*['"<]?(https?://[^'"\s>]+)['">\s]?""",
                    line,
                )
                if m:
                    name = m.group(1)
                    uri = m.group(2).rstrip("/#") + \
                          ("/" if m.group(2).endswith("/") else
                           "#" if m.group(2).endswith("#") else "")
                    # ── Filter out mapping-name artefact prefixes ──────────
                    # These are auto-injected by the coordinator with URIs like
                    # http://example.com/MetadataMapping# and pollute SPARQL.
                    # Heuristic: skip if the URI path ends with the prefix name
                    # (case-insensitive) and the domain is example.com.
                    uri_lower = uri.lower()
                    name_lower = name.lower()
                    if (
                        "example.com" in uri_lower
                        and name_lower.replace("_", "") in uri_lower.replace("_", "")
                    ):
                        continue  # skip artefact prefix
                    prefixes[name] = uri
    # Normalise well-known prefix URIs to their canonical form.
    # The LLM may write https://schema.org/ but morph-kgc materialises
    # http://schema.org/, causing SPARQL ASK queries to return false.
    _well_known = {
        "schema": "http://schema.org/",
        "schema1": "http://schema.org/",
        "foaf": "http://xmlns.com/foaf/0.1/",
    }
    for pfx, canonical in _well_known.items():
        if pfx in prefixes:
            prefixes[pfx] = canonical
    return prefixes


def _build_prefix_block(prefix_map: dict[str, str]) -> str:
    """Build a PREFIX block string for injection into SPARQL prompts and queries."""
    merged = dict(_ALWAYS_DECLARE)
    merged.update(prefix_map)   # actual mapping prefixes override defaults
    lines = []
    for name, uri in sorted(merged.items()):
        uri_str = uri if uri.endswith(("#", "/")) else uri + "#"
        lines.append(f"PREFIX {name}: <{uri_str}>")
    return "\n".join(lines)


def _inject_prefix_declarations(sparql: str, prefix_map: dict[str, str]) -> str:
    """Ensure the generated SPARQL has all required PREFIX declarations AND
    that existing declarations use the correct URIs from the actual mapping.

    Two operations:
    1. REPLACE any prefix whose URI differs from the mapping's URI.
       (Fixes: LLM writes https://schema.org/ but KG uses http://schema.org/)
    2. ADD prefixes that the LLM omitted entirely.

    This is the core fix for the https vs http schema.org mismatch that
    causes all ASK queries to return false despite a valid KG.
    """
    merged = dict(_ALWAYS_DECLARE)
    merged.update(prefix_map)   # mapping prefixes are authoritative

    # ── Step 1: Replace wrong URI declarations ──────────────────
    def _replace_prefix_uri(m: re.Match) -> str:
        name = m.group(1)
        declared_uri = m.group(2).rstrip("/#")
        if name in merged:
            correct_uri = merged[name].rstrip("/#")
            # Normalise trailing separator for comparison
            if declared_uri != correct_uri:
                new_uri = merged[name] if merged[name].endswith(("#", "/")) else merged[name] + "#"
                return f"PREFIX {name}: <{new_uri}>"
        return m.group(0)  # no change

    sparql = re.sub(
        r'PREFIX\s+(\w+)\s*:\s*<([^>]+)>',
        _replace_prefix_uri,
        sparql,
        flags=re.IGNORECASE,
    )

    # ── Step 2: Add missing prefix declarations ──────────────────
    existing = set(re.findall(r'PREFIX\s+(\w+)\s*:', sparql, re.IGNORECASE))
    additions = []
    for name, uri in sorted(merged.items()):
        if name not in existing:
            uri_str = uri if uri.endswith(("#", "/")) else uri + "#"
            additions.append(f"PREFIX {name}: <{uri_str}>")

    if additions:
        return "\n".join(additions) + "\n" + sparql
    return sparql


def _extract_ontology_context(ontology_info: dict) -> str:
    """Build a compact ontology context string for the prompt."""
    raw = ontology_info.get("raw", {})

    if not isinstance(raw, dict):
        # raw is a plain Turtle/text string — return it directly
        return str(raw)[:3000]

    # Collect classes
    classes = raw.get("classes", [])
    # Collect properties
    props = raw.get("object_properties", []) + raw.get("data_properties", [])
    # Collect prefixes
    prefixes = raw.get("prefixes", {})

    lines = []
    if prefixes:
        lines.append("### Declared Prefixes")
        for k, v in prefixes.items():
            lines.append(f"  {k}: <{v}>")

    if classes:
        lines.append("\n### Ontology Classes")
        for c in classes[:30]:
            lines.append(f"  - {c}")

    if props:
        lines.append("\n### Ontology Properties")
        for p in props[:40]:
            lines.append(f"  - {p}")

    if not lines:
        return str(raw)[:3000]

    return "\n".join(lines)


def _extract_sparql_from_response(text: str) -> str:
    """Extract the SPARQL query from LLM output, stripping markdown fences."""
    text = re.sub(r"```(?:sparql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text.strip()


def cq_to_sparql(
    cq: str,
    ontology_info: dict,
    base_uri: str = "http://example.org/",
    previous_error: str | None = None,
    previous_sparql: str | None = None,
    llm=None,
    yarrrml_prefix_map: dict[str, str] | None = None,
    mapping_context: str | None = None,
    kg_grounding_block: str | None = None,
) -> str:
    """Translate a single CQ to an ASK SPARQL query.

    Parameters
    ----------
    cq :
        Natural-language Competency Question.
    ontology_info :
        Ontology info dict from AgentState.
    base_uri :
        Base URI used in the pipeline.
    previous_error :
        Parse error from a previous attempt (triggers self-correction).
    previous_sparql :
        The previously generated (invalid) SPARQL query.
    llm :
        LLM instance; uses default cq_validator role if None.
    yarrrml_prefix_map :
        Prefix → URI map extracted directly from the generated YARRRML.
        When supplied, these take precedence over ontology-derived prefixes
        because they reflect the actual namespaces in the materialised KG.
    mapping_context :
        Pre-built string listing entity types and predicates from the YARRRML.
        Injected into the prompt so the LLM uses correct class names.
    kg_grounding_block :
        Pre-built string with ACTUAL classes and predicates from the live KG.
        Strongest signal — overrides ontology and mapping_context hints.
    """
    if llm is None:
        llm = get_llm(role="cq_validator")

    ontology_ctx = _extract_ontology_context(ontology_info)

    # Build the authoritative prefix block from the actual YARRRML mapping
    mapping_prefix_block = ""
    if yarrrml_prefix_map:
        mapping_prefix_block = f"""
### MAPPING PREFIXES (use ONLY these — they match the actual KG namespaces):
{_build_prefix_block(yarrrml_prefix_map)}
"""

    # Inject entity types and predicates from the actual YARRRML so the LLM
    # uses the correct class names (e.g. schema:OrderItem, not schema:Order)
    mapping_ctx_block = ""
    if mapping_context:
        mapping_ctx_block = f"\n{mapping_context}\n"

    # Live KG grounding — strongest signal, injected right after prefix block
    kg_block = ""
    if kg_grounding_block:
        kg_block = f"\n{kg_grounding_block}\n"

    correction_block = ""
    if previous_error and previous_sparql:
        correction_block = f"""
### PREVIOUS ATTEMPT (INVALID — fix it)
{previous_sparql}

### ERROR FROM PREVIOUS ATTEMPT
{previous_error}

Correct the query above so it is valid SPARQL 1.1.
"""

    human_prompt = f"""### COMPETENCY QUESTION
{cq}
{mapping_prefix_block}{kg_block}{mapping_ctx_block}
### ONTOLOGY CONTEXT (for term guidance only — use KG_CLASSES/KG_PREDICATES above for URIs)
{ontology_ctx}

### BASE URI
{base_uri}
{correction_block}
Generate the ASK SPARQL query now."""

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ])

    raw = response.content if hasattr(response, "content") else str(response)
    sparql = _extract_sparql_from_response(raw)
    sparql = _fix_typed_numeric_literals(sparql)
    sparql = _inject_prefix_declarations(sparql, yarrrml_prefix_map or {})
    sparql = fix_undeclared_prefix_refs(sparql, yarrrml_prefix_map or {})
    return sparql


def batch_cq_to_sparql(
    cqs: list[str],
    ontology_info: dict,
    base_uri: str = "http://example.org/",
    max_correction_attempts: int = 2,
    yarrrml_prefix_map: dict[str, str] | None = None,
    mapping_context: str | None = None,
    kg_grounding_block: str | None = None,
) -> list[dict]:
    """Convert a list of CQs to SPARQL ASK queries with self-correction.

    Parameters
    ----------
    cqs :
        List of natural-language CQs.
    ontology_info :
        Ontology info dict from AgentState.
    base_uri :
        Base URI.
    max_correction_attempts :
        How many times to retry a CQ if SPARQL is syntactically invalid.
    yarrrml_prefix_map :
        Prefix → URI map extracted from the generated YARRRML (Bug 2 fix).
        Ensures SPARQL queries use the same namespaces as the materialised KG.
    mapping_context :
        Entity types and predicates from YARRRML — keeps the LLM grounded
        to actual class names in the KG.
    kg_grounding_block :
        Actual classes and predicates present in the live KG.
        Strongest grounding signal — prevents LLM from writing nonexistent types.

    Returns
    -------
    list of dicts:
        [{"cq": str, "sparql": str, "valid": bool, "error": str | None}]
    """
    import pyoxigraph

    llm = get_llm(role="cq_validator")
    results = []

    for cq in cqs:
        sparql = None
        error = None

        for attempt in range(max_correction_attempts + 1):
            sparql = cq_to_sparql(
                cq,
                ontology_info,
                base_uri=base_uri,
                previous_error=error if attempt > 0 else None,
                previous_sparql=sparql if attempt > 0 else None,
                llm=llm,
                yarrrml_prefix_map=yarrrml_prefix_map,
                mapping_context=mapping_context,
                kg_grounding_block=kg_grounding_block,
            )

            try:
                _store = pyoxigraph.Store()
                _store.query(sparql)
                error = None
                break
            except Exception as e:
                error = str(e)
                if attempt < max_correction_attempts:
                    print(
                        f"    [CQ->SPARQL] Syntax error for CQ '{cq[:50]}' "
                        f"(attempt {attempt + 1}): {error[:80]}"
                    )

        results.append({
            "cq": cq,
            "sparql": sparql,
            "valid": error is None,
            "error": error,
        })

    return results

