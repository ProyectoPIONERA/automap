"""
Entity Builder Agent — generates YARRRML mapping blocks with
sources, subjects, ``rdf:type``, and data properties.

Does NOT generate relationship links (joins) between mappings —
that is handled by the Relationship Agent.

Uses the one-shot example from ``config/yarrrml_examples.py`` to teach
local LLMs the correct YARRRML entity mapping syntax.

Prompt is split into a STATIC system message (cached by llama.cpp after
the first call) and a DYNAMIC human message (changes per call/retry).
"""

import os

from config.settings import get_llm
from config.yarrrml_examples import EXAMPLE_FOR_ENTITY_BUILDER
from config.structured_output import (
    structured_output_enabled, MappingsOutput, mappings_to_yaml,
)
from langchain_core.messages import SystemMessage, HumanMessage

# ── Static system prompt (identical every call → KV-cached by llama.cpp) ──
_SYSTEM_PROMPT = f"""{EXAMPLE_FOR_ENTITY_BUILDER}

You generate ONLY the `mappings:` block for a YARRRML mapping file.
Do NOT include the `prefixes:` section — that is handled separately.

### CRITICAL — SUBJECT URI FORMAT:
The subject `s:` MUST use the ontology's primary prefix and class name.
Example: `s: podio:ApprovedPolicy/$(id)`
Do NOT use `http://example.com/...` or `http://example.org/...`
Find the primary prefix in the ontology and use it.

### CRITICAL — IRI COLUMNS VS DATA COLUMNS:
- If a CSV column contains URLs/IRIs (publisher, language, source, etc.
  that look like http:// links or WikiData URIs), map as:
  `[predicate, $(column)~iri]`   (2-item entry, direct IRI reference)
  Do NOT create a separate mapping for these.
- If a column contains literal values (text, numbers, dates), map as:
  `[predicate, $(column), xsd:type]`   (3-item entry)

### CRITICAL — ENTITY ID COLUMNS (columns ending in ID/Id/_id that are NOT the PK):
If a CSV column is a *foreign-key ID* referencing a distinct real-world entity
(e.g. CustomerID, DriverID, StoreID, ProductID — NOT the main row key), you MUST:
1. Create a SEPARATE mapping for that entity type.
2. In the referencing mapping, link to it via URI template (NOT a literal):
     CORRECT: [schema:customer, ex:customer/$(CustomerID)~iri]
     WRONG:   [schema:customer, $(CustomerID), xsd:string]  ← literal, NOT linked
This makes the KG traversable. A literal CustomerID is useless for graph queries.

### CRITICAL — SAME-CSV FOREIGN KEYS:
If a column is a foreign key referencing another row in the SAME CSV
(like `parent_id`), link via a URI template:
  `[predicate, prefix:ClassName/$(fk_column)~iri]`
Do NOT use `joins: [child: ..., parent: ...]` — that is ONLY for
cross-CSV files.  URI templates are simpler and more performant.

### CRITICAL — METADATA NODE SPLITTING:
If the ontology defines a property like `lkg:metadata` or `onto:hasMetadata`
whose range is the same class, create a secondary mapping with a `/Metadata`
suffix in the subject:
  Primary:   `s: prefix:Class/$(id)`
  Secondary: `s: prefix:Class/$(id)/Metadata`
The PRIMARY links TO metadata:  `[lkg:metadata, prefix:Class/$(id)/Metadata~iri]`
The Metadata mapping does NOT link to itself.

### CRITICAL — PARENT ENTITY:
If the ontology defines `is_part_of` / `has_part` and the CSV has a
`parent_id` column, create a SEPARATE parent mapping:
  `s: prefix:Class/$(parent_id)` — subject uses the PARENT FK column
  `[eli:has_part, prefix:Class/$(id)~iri]` — INVERSE link to child
Map `parent_*` columns (like `parent_source`) in the parent mapping.

### CRITICAL — PROPERTY DISTRIBUTION (do NOT duplicate everything):
When creating multiple mappings, DISTRIBUTE properties:
  - **Primary entity**: content/payload properties (content, wordCount,
    description) + all direct IRI links + link TO metadata + link TO parent.
  - **Metadata**: administrative properties (title, subject, jurisdiction,
    localId, versionDate, hasPDF, summary).  NOT content properties.
  - **Parent**: parent_* columns + inverse link + shared IRI columns.
A column CAN appear in multiple mappings with DIFFERENT predicates
(e.g. $(description) → terms:description in Primary, lkg:summary in Metadata).
Within a SINGLE mapping, each column must have a unique predicate.

### ARCHITECTURE:
1. Do NOT create separate mappings for IRI value columns.
   Map publisher, language, audience, creator, source with $(column)~iri.
2. Create secondary mappings ONLY for:
   - Metadata sub-resources (ontology defines a metadata property)
   - Parent entities (ontology defines is_part_of / has_part)
3. The PRIMARY mapping includes ALL object-property links (to metadata,
   to parent, to IRI columns).  These are NOT "relationship joins" —
   they are URI template PO entries that belong in the entity block.

### SYNTAX FORMAT:
1. Use exactly 2 spaces for indentation. No tabs.
2. Subject: `s: prefix:ClassName/$(id_column)` — SINGLE string.
3. rdf:type: `[a, prefix:ClassName]`
4. Every column MUST have a UNIQUE predicate name.
5. ALL po: entries MUST use INLINE/FLOW-STYLE YAML lists on a single line:
   CORRECT: `- [schema:name, $(name), xsd:string]`
   WRONG:   block-style with separate lines per element

### CRITICAL — NON-IRI COLUMN VALUES:
If a column contains usernames (like @user9), handles, codes, plain text,
categories, event types, locales, or any value that is NOT a valid URL/IRI,
it MUST be mapped as a LITERAL with xsd type:
  `[predicate, $(column), xsd:string]`
NEVER use `$(column)~iri` for these — it produces INVALID RDF triples.
Only use ~iri when the column actually contains http:// or https:// URLs.

### OUTPUT RULES:
- Output ONLY the `mappings:` block (valid YAML).
- Start with `mappings:` on the first line.
- No `prefixes:` section.
- No markdown code blocks, preamble, or explanations.

### GOLDEN RULES (violations cause pipeline failure):
RULE 10: METADATA NODE TYPE — The Metadata mapping MUST use a DIFFERENT
  rdf:type than the primary entity. Append 'Metadata' to the primary class:
  CORRECT: [a, ex:BuildingMetadata]  or  [a, ex:EncounterMetadata]
  WRONG:   [a, ex:Building]  ← same as primary = INVALID

RULE 12: NEVER use a full URL as a prefix name in the prefixes block.
  Prefix names must be short alphanumeric identifiers (ex:, schema:, xsd:).
  NEVER write: 'http://example.com/': 'http://example.com/'
  Instead declare: ex: 'http://example.com/'

RULE 13: Multi-value columns (diag_1, diag_2, diag_3) that each hold a
  separate entity value MUST each get their own mapping with a DISTINCT subject:
  DiagnosisMapping_1: s: ex:Diagnosis/$(diag_1)  po: [ex:icdCode, $(diag_1)]
  DiagnosisMapping_2: s: ex:Diagnosis/$(diag_2)  po: [ex:icdCode, $(diag_2)]

RULE 14: When a dataset has many columns representing the same entity type
  (e.g. 23 drug columns), create one mapping per drug with a composite subject:
  s: ex:MedicationRecord/$(encounter_id)/metformin
  po:
    - [a, ex:MedicationRecord]
    - [ex:drugName, 'metformin', xsd:string]
    - [ex:dosageStatus, $(metformin), xsd:string]

RULE 15: Do NOT create a secondary mapping (Metadata, Info, etc.) UNLESS
  the ontology explicitly declares a named object property (e.g. lkg:metadata,
  onto:hasMetadata) whose range is a distinct class.
  For FLAT datasets with no such property, ALL columns belong in the PRIMARY
  mapping — do NOT invent Metadata or auxiliary mappings.
  Ask yourself: "Is there an ontology property that links the primary entity
  to this sub-entity?" If the answer is NO, do NOT create the sub-mapping.
"""


def _strip_markdown(content: str, marker: str = "mappings:") -> str:
    """Remove markdown code fences from LLM output."""
    if "```" not in content:
        return content.strip()
    parts = content.split("```")
    for part in parts:
        if marker in part:
            content = part
            break
    lines = content.splitlines()
    if lines and lines[0].strip().lower() in ("yaml", "yml"):
        content = "\n".join(lines[1:])
    return content.strip()


def call_entity_agent(state: dict) -> str:
    """Generate the ``mappings:`` block with entity definitions and data
    properties (no relationship joins).

    Parameters
    ----------
    state : dict
        Pipeline state containing CSV path, ontology, mapping plan, etc.

    Returns
    -------
    str
        The ``mappings:`` YAML block (plain text, no fences).
    """
    llm = get_llm(role="entity_builder")

    csv_path = state["csv_path"]
    csv_name = os.path.basename(csv_path)
    csv_source = f"{csv_path}~csv"
    base_uri = state.get("base_uri", "http://example.org/")
    mapping = state["mapping_plan"].get("analysis", "")
    ontology = state["ontology_info"].get("raw", "")

    # Truncate large inputs to prevent prompt from exceeding context/timeout
    MAX_ONTOLOGY = 4000
    MAX_MAPPING = 2000
    if len(ontology) > MAX_ONTOLOGY:
        ontology = ontology[:MAX_ONTOLOGY] + "\n... (truncated)"
    if len(mapping) > MAX_MAPPING:
        mapping = mapping[:MAX_MAPPING] + "\n... (truncated)"

    # Schema alignment plan (Functional Entity Plan)
    alignment = state.get("schema_alignment", {})
    entity_plan = alignment.get("entity_plan", "")
    is_multi_node = alignment.get("multi_node", False)

    # Competency Questions
    cqs = state.get("competency_questions", [])
    cq_section = ""
    if cqs:
        cq_list = "\n".join(f"  - {q}" for q in cqs)
        cq_section = f"""
### COMPETENCY QUESTIONS (the mapping MUST produce a KG that can answer ALL of these):
{cq_list}
"""

    # Explicit CSV column list — prevents LLM from hallucinating columns
    csv_columns = state.get("schema_info", {}).get("raw", {}).get("columns", [])
    col_list = "\n".join(f"  - {c}" for c in csv_columns)
    csv_column_section = f"""
### STRICT RULE: Only use columns from this exact list — do NOT invent column names:
AVAILABLE CSV COLUMNS:
{col_list}
Any column reference like $(xyz) MUST match one of the above exactly.
"""

    # ── Primary key / subject identifier hint (Fix 2) ────────────────────
    # If the dataset has no natural unique ID column, the LLM often picks a
    # low-cardinality categorical column (e.g. Orientation with 4 values) as
    # the subject ID — producing only 4 URIs for thousands of rows and making
    # subject-level triple matching impossible.
    pk_hint_section = ""
    if csv_columns:
        # Agnostic PK detection: only generic suffix/exact patterns, no dataset-specific names
        _pk_generics = {'id', 'idx', 'index', 'row_id', 'rowid'}
        _pk_suffixes = ('_id', '_no', '_num', '_code', '_key', '_number', '_uuid', '_hash')
        pk_candidates = [
            c for c in csv_columns
            if c.lower() in _pk_generics or c.lower().endswith(_pk_suffixes)
        ]
        if pk_candidates:
            pk_hint_section = f"""
### PRIMARY KEY COLUMN: Use `$({pk_candidates[0]})` as the subject identifier.
This column has unique values per row — ideal for URI templates.
"""
        else:
            # No natural PK — try to detect high-cardinality numeric columns
            # that could form a composite key.  Warn away from low-cardinality ones.
            try:
                import pandas as pd
                _df_sample = pd.read_csv(state["csv_path"], nrows=500)
                _cardinality = {c: _df_sample[c].nunique() for c in _df_sample.columns}
                _n_rows = len(_df_sample)
                # Low-cardinality: fewer than 10% unique values
                low_card = [c for c, u in _cardinality.items() if u < max(5, _n_rows * 0.1)]
                high_card = sorted(
                    [c for c, u in _cardinality.items() if u >= _n_rows * 0.5],
                    key=lambda c: -_cardinality[c]
                )
                low_card_str = ", ".join(f"'{c}' ({_cardinality[c]} values)" for c in low_card[:6])
                composite_str = "/".join(f"$({c})" for c in high_card[:3]) if high_card else "$(col1)/$(col2)"
                pk_hint_section = f"""
### NO UNIQUE ID COLUMN DETECTED — CRITICAL SUBJECT URI RULES:
This dataset has NO dedicated primary key column.
Low-cardinality columns (DO NOT use as sole subject identifier): {low_card_str or 'none detected'}
Using a low-cardinality column as the only subject part produces very few URIs
(e.g. only 4 URIs for 10,000 rows) and makes the KG nearly useless.

Instead, build a COMPOSITE subject from 2-3 high-cardinality columns:
  GOOD:  s: ex:Building/{composite_str}
  BAD:   s: ex:Building/$(Orientation)  ← only 4 distinct values = only 4 URIs

Pick columns with the MOST unique values (closest to one-per-row).
"""
            except Exception:
                pk_hint_section = """
### NO UNIQUE ID COLUMN DETECTED:
Build a COMPOSITE subject URI from 2-3 columns that together are unique per row.
Do NOT use a single categorical/enum column as the sole subject identifier.
Example: s: ex:Entity/$(col1)/$(col2)/$(col3)
"""

    # ── Coordinate-group disambiguation hints ─────────────────────
    # Detect groups of columns that share a naming prefix ending in
    # a geo-coordinate suffix (lat/long/lon/latitude/longitude).
    # When multiple groups exist (e.g. lat/long AND merch_lat/merch_long)
    # inject an explicit placement rule so the LLM doesn't conflate them.
    import re as _re

    def _find_coord_groups(cols):
        """Return dict: group_prefix → {'lat': col, 'long': col}"""
        lat_re = _re.compile(r'^(.*)_(lat|latitude)$', _re.IGNORECASE)
        lon_re = _re.compile(r'^(.*)_(lon|long|longitude)$', _re.IGNORECASE)
        groups: dict[str, dict[str, str]] = {}
        for c in cols:
            m = lat_re.match(c)
            if m:
                groups.setdefault(m.group(1), {})['lat'] = c
                continue
            m = lon_re.match(c)
            if m:
                groups.setdefault(m.group(1), {})['long'] = c
        # Also handle bare "lat" / "long" / "lon" (no prefix)
        bare: dict[str, str] = {}
        for c in cols:
            if c.lower() in ('lat', 'latitude'):
                bare['lat'] = c
            elif c.lower() in ('lon', 'long', 'longitude'):
                bare['long'] = c
        if bare:
            groups[''] = bare
        return groups

    coord_groups = _find_coord_groups(csv_columns)
    mapper_hint_section = ""
    if len(coord_groups) > 1:
        rules = []
        for prefix, pair in sorted(coord_groups.items()):
            lat_col = pair.get('lat', '?')
            lon_col = pair.get('long', '?')
            group_label = prefix if prefix else "(bare)"
            rules.append(
                f"  - Group '{group_label}': $({lat_col}) → geo:lat, "
                f"$({lon_col}) → geo:long  "
                f"[place these in the mapping whose name contains '{prefix or 'primary'}']"
            )
        mapper_hint_section = f"""
### COORDINATE COLUMN GROUPS — each group belongs to a DIFFERENT mapping:
{chr(10).join(rules)}
Do NOT mix columns from different groups into the same mapping.
$({list(coord_groups.values())[0].get('lat', '')}) and $({list(coord_groups.values())[-1].get('lat', '')}) are DIFFERENT columns for DIFFERENT entities.
"""

    # Build alignment section
    alignment_section = ""
    if entity_plan:
        alignment_section = f"""
### SCHEMA ALIGNMENT PLAN (follow this structure precisely):
{entity_plan}

{"This dataset requires MULTI-NODE mappings — create separate mappings per the plan above." if is_multi_node else "This dataset uses a FLAT mapping — a single mapping is sufficient per the plan above."}
You MUST follow the entity structure, identifier columns, and data property
assignments from the plan above.  Do NOT deviate from it.
"""

    # On retries, include targeted feedback
    feedback = state.get("feedback", "")
    persistent_fails = state.get("persistent_cq_failures", [])
    feedback_section = ""
    if feedback and "PASSED" not in feedback and "APPROVED" not in feedback:
        if "CQ_ERROR" in feedback:
            cq_fail_lines = [l.strip() for l in feedback.split("\n")
                             if "FAIL" in l.upper() and "CQ" in l.upper()]
            cq_targets = "\n".join(cq_fail_lines) if cq_fail_lines else feedback

            # Persistent failures — these have failed across multiple attempts
            persistent_section = ""
            if persistent_fails:
                pf_list = "\n".join(f"  !! {q}" for q in persistent_fails)
                persistent_section = f"""
### PERSISTENT FAILURES (failed on 2+ consecutive attempts — HIGHEST PRIORITY):
{pf_list}
These MUST be resolved first before addressing other issues.
"""

            feedback_section = f"""
### TARGETED CQ FIXES REQUIRED — each failing CQ needs a specific structural change:
{cq_targets}
{persistent_section}
For EACH failing CQ above:
1. Identify the missing entity class or object property link.
2. Add it as a new mapping or new po: entry.
3. Do NOT remove existing working mappings — only ADD what is missing.
4. Ensure every entity mentioned in a CQ has its own mapping with proper links.
"""
        elif "SYNTAX_ERROR" in feedback:
            feedback_section = f"""
### FIX REQUIRED — YARRRML translation FAILED:
{feedback}

Check that all po: entries are valid YAML lists.
Do NOT use joins for same-CSV references — use URI templates instead.
"""
        elif any(kw in feedback.lower() for kw in [
            "column", "coverage", "missing", "duplicate", "predicate",
            "redundant", "data property",
        ]):
            # ── Fix 1: Extract specific missing columns and inject as mandatory constraints ──
            try:
                from agents.refiner_agent import (
                    _extract_missing_columns_from_feedback,
                    _build_mandatory_column_injection,
                )
                missing_cols = _extract_missing_columns_from_feedback(feedback)
                if missing_cols:
                    mandatory_block = _build_mandatory_column_injection(missing_cols)
                    feedback_section = f"""
### FIX REQUIRED — Previous output had entity/column issues:
{mandatory_block}

{feedback}

Make sure ALL CSV columns are mapped and each column appears in exactly ONE mapping.
"""
                else:
                    feedback_section = f"""
### FIX REQUIRED — Previous output had entity/column issues:
{feedback}

Make sure ALL CSV columns are mapped and each column appears in exactly ONE mapping.
"""
            except ImportError:
                feedback_section = f"""
### FIX REQUIRED — Previous output had entity/column issues:
{feedback}

Make sure ALL CSV columns are mapped and each column appears in exactly ONE mapping.
"""

    # Fix 3 — Inject persisted column constraints from previous refiner auto-inject
    injected_constraints = state.get("injected_column_constraints", {})
    injected_constraints_section = ""
    if injected_constraints:
        constraint_lines = "\n".join(
            f"  - $({{col}}) → {assignment}"
            for col, assignment in sorted(injected_constraints.items())
        )
        injected_constraints_section = f"""
### PERSISTED COLUMN CONSTRAINTS (from previous auto-fix — HONOUR THESE EXACTLY):
{constraint_lines}
These columns were auto-injected in a previous attempt. Place them exactly as specified.
"""

    # ── Base URI instruction ───────────────────────────────────────────────
    # When the user has supplied a custom base URI, instruct the LLM to use
    # it directly.  This reduces post-processing rewrites and makes the
    # subject URIs semantically correct from the start.
    _DEFAULT_BASE = "http://example.org/"
    # Normalise: add http:// scheme if user passed bare domain like "mykg.org"
    if base_uri and not base_uri.startswith(("http://", "https://", "urn:")):
        base_uri = "http://" + base_uri
    base_uri_section = ""
    if base_uri and base_uri.rstrip("/") != _DEFAULT_BASE.rstrip("/"):
        if not base_uri.endswith(("/", "#")):
            base_uri = base_uri + "/"
        try:
            from urllib.parse import urlparse as _up
            _host = (_up(base_uri).hostname or "base").replace("www.", "")
            _parts = [p for p in _host.split(".") if p and p not in ("com", "org", "net", "io", "eu")]
            _pfx = (_parts[0] if _parts else "base").replace("-", "").replace("_", "")
            if not _pfx.isidentifier():
                _pfx = "base"
        except Exception:
            _pfx = "base"
        base_uri_section = f"""
### CRITICAL — BASE URI FOR ALL ENTITY SUBJECTS:
The user's knowledge graph lives at: {base_uri}
Declare this prefix:  {_pfx}: "{base_uri}"
Use it for ALL subject URI templates (s: fields):
  CORRECT: s: {_pfx}:Film/$(movie_id)
  WRONG:   s: dbo:Film/$(movie_id)   ← do NOT use ontology namespaces for subjects
  WRONG:   s: http://example.org/Film/$(movie_id)  ← no bare URIs in s:
Predicates (po: first items) may still use ontology prefixes like dbo:, schema:, etc.
Only the SUBJECT templates must use the {_pfx}: prefix.
"""

    # ── Dynamic human message (changes per call) ──
    # At attempt 3+, aggressively truncate to prevent context overflow (400 error).
    retry_count = state.get("retry_count", 0)
    if retry_count >= 2:
        # Keep only essentials: columns, PK hint, reduced alignment plan, top-3 failing CQs
        alignment_section_trimmed = alignment_section[:1200] + "\n... (truncated)" if len(alignment_section) > 1200 else alignment_section
        feedback_trimmed = feedback_section[:800] + "\n... (truncated)" if len(feedback_section) > 800 else feedback_section
        ontology_trimmed = ontology[:1500] + "\n... (truncated)" if len(ontology) > 1500 else ontology
        human_prompt = f"""{base_uri_section}{csv_column_section}
{pk_hint_section}
{alignment_section_trimmed}
{feedback_trimmed}
Target CSV: {csv_name}
Source format (use exactly): [{csv_source}]
Ontology Context: {ontology_trimmed}

Generate the mappings: block now.
"""
    else:
        human_prompt = f"""{base_uri_section}{csv_column_section}
{pk_hint_section}
{mapper_hint_section}
{alignment_section}
{cq_section}
{feedback_section}
{injected_constraints_section}
Target CSV: {csv_name}
Source format (use exactly): [{csv_source}]
Ontology Context: {ontology}
Mapping Plan: {mapping}

Generate the mappings: block now.
"""

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]

    # ── Structured output path (eliminates syntax retry loops) ────
    if structured_output_enabled():
        try:
            structured_llm = llm.with_structured_output(MappingsOutput)
            result: MappingsOutput = structured_llm.invoke(messages)
            return mappings_to_yaml(result)
        except Exception as e:
            print(f"    [EntityAgent] Structured output failed ({e}), falling back to free-text")

    # ── Free-text fallback (streaming to keep connection alive) ───
    result = ""
    for chunk in llm.stream(messages):
        result += chunk.content
    return _strip_markdown(result.strip())

