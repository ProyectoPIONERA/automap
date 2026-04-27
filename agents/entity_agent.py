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

    # ── Coordinate-group disambiguation hints (Fix 2) ─────────────
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

    # ── Dynamic human message (changes per call) ──
    human_prompt = f"""{csv_column_section}
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

