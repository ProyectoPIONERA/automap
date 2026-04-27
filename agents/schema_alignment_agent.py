"""
Schema Alignment Agent — analyses the ontology and CSV structure to
produce a **Functional Entity Plan** before any YARRRML is generated.

The agent detects hierarchical and multi-node patterns in the ontology
(e.g. ``eli:is_part_of``, ``lkg:metadata``, self-referencing properties,
parent-child class relationships) and outputs a structured plan that
tells downstream agents (Entity Builder, Relationship Linker) exactly
which entity classes to create and how they relate.

This step prevents the common LLM failure mode where a flat CSV is
naively mapped 1:1 to a single entity when the ontology actually
requires multiple interconnected nodes.
"""

from config.settings import get_llm, get_llm_with_retry
from config.yarrrml_examples import GOLDEN_RULES
from langchain_core.messages import SystemMessage, HumanMessage


def call_schema_alignment_agent(state: dict) -> dict:
    """Analyse ontology + CSV + mapping plan to produce a Functional
    Entity Plan.

    Parameters
    ----------
    state : dict
        Pipeline state with ``ontology_info``, ``schema_info``,
        ``mapping_plan``, and optionally ``competency_questions``.

    Returns
    -------
    dict
        ``entity_plan``  – structured text describing each entity class,
        its identifying column, data properties, and relationships.
        ``multi_node``   – bool indicating whether multi-node mapping is
        required.
    """
    llm = get_llm_with_retry(role="schema_alignment")

    ontology = state.get("ontology_info", {}).get("raw", "")
    schema = state.get("schema_info", {})
    columns = schema.get("raw", {}).get("columns", [])
    unique_cols = schema.get("raw", {}).get("unique_columns", [])
    sample = schema.get("raw", {}).get("sample", "")
    analysis = schema.get("analysis", "")
    mapping_plan = state.get("mapping_plan", {}).get("analysis", "")
    base_uri = state.get("base_uri", "http://example.org/")

    # Include competency questions if provided
    cq_section = ""
    cqs = state.get("competency_questions", [])
    if cqs:
        cq_list = "\n".join(f"  - {q}" for q in cqs)
        cq_section = f"""
### COMPETENCY QUESTIONS (the final KG must be able to answer these):
{cq_list}

When building the entity plan, ensure each CQ can be answered by the
proposed entity structure.  If a CQ requires distinguishing a parent
entity from a child entity, you MUST split them into separate mappings.
"""

    # Parse CQs to extract mandatory entity classes
    cq_entities: list[str] = []
    for cq in cqs:
        cq_lower = cq.lower()
        if any(kw in cq_lower for kw in ["diagnosis", "icd", "diag"]):
            cq_entities.append("DiagnosisMapping (required by CQ about diagnoses)")
        if any(kw in cq_lower for kw in ["drug", "medication", "insulin", "dosage", "prescri"]):
            cq_entities.append("MedicationRecordMapping (required by CQ about drugs/medication)")
        if any(kw in cq_lower for kw in ["age", "race", "gender", "demographic"]):
            cq_entities.append("PatientDemographics or MetadataMapping (required by CQ about demographics)")
        if any(kw in cq_lower for kw in ["patient", "person", "cardholder"]):
            cq_entities.append("PatientMapping or PersonMapping (required by CQ about patients/persons)")
        if any(kw in cq_lower for kw in ["merchant", "vendor", "seller"]):
            cq_entities.append("MerchantMapping (required by CQ about merchants)")
        if any(kw in cq_lower for kw in ["transaction", "payment", "order"]):
            cq_entities.append("TransactionMapping (required by CQ about transactions)")
        if any(kw in cq_lower for kw in ["address", "location", "city", "street"]):
            cq_entities.append("AddressMapping (required by CQ about addresses/locations)")

    if cq_entities:
        unique_entities = sorted(set(cq_entities))
        entity_list = "\n".join(f"  - {e}" for e in unique_entities)
        cq_section += f"""
### MANDATORY ENTITY CLASSES — these MUST appear as separate mappings:
{entity_list}
Do NOT merge these into a flat single mapping. Each must be a distinct
mapping block with its own subject template and rdf:type.
"""

    # On retries after CQ failure, inject the failure feedback so
    # the alignment agent knows what structural problems to fix.
    # Only inject the REQUIRED FIXES section to avoid prompt bloat.
    cq_feedback = state.get("feedback", "")
    persistent_fails = state.get("persistent_cq_failures", [])
    if "CQ_ERROR" in cq_feedback:
        if "REQUIRED FIXES" in cq_feedback:
            fixes_only = cq_feedback.split("REQUIRED FIXES")[-1].strip()
        else:
            fixes_only = "\n".join(
                l.strip() for l in cq_feedback.split("\n")
                if "FAIL" in l.upper() and "CQ" in l.upper()
            )
        fixes_only = fixes_only[:1500]  # hard cap to prevent prompt explosion

        # Build a structured diff showing what failed and what exists
        current_entity_plan = state.get("schema_alignment", {}).get("entity_plan", "")
        failed_cq_lines = [
            l.strip() for l in cq_feedback.split("\n")
            if "FAIL" in l.upper() and "CQ" in l.upper()
        ]
        failed_cqs_fmt = "\n".join(f"  !! {l}" for l in failed_cq_lines[:10])

        # Extract missing concepts from CQ failure text
        import re as _re
        concept_keywords = _re.findall(
            r"'([A-Za-z][A-Za-z0-9_]+)'|\"([A-Za-z][A-Za-z0-9_]+)\"",
            fixes_only
        )
        missing_concepts = sorted(set(
            w for pair in concept_keywords for w in pair if w
        ))[:8]

        persistent_section = ""
        if persistent_fails:
            pf_list = "\n".join(f"  !! {q}" for q in persistent_fails)
            persistent_section = f"""
### PERSISTENT CQ FAILURES (failed across multiple attempts — MUST fix in new entity plan):
{pf_list}
"""
        cq_section += f"""
### STRUCTURED REPLAN DIFF
FAILED CQs:
{failed_cqs_fmt}

MISSING CONCEPTS (likely need new entity or predicate):
{', '.join(missing_concepts) if missing_concepts else '(see REQUIRED FIXES below)'}

CURRENT ENTITY PLAN (what you must CHANGE, not copy):
{current_entity_plan[:800]}
... (truncated — do NOT repeat this, produce a REVISED plan)

REQUIRED FIXES:
{fixes_only}
{persistent_section}
Rebuild the entity plan to address the above. Do NOT copy the previous plan.
"""

    # ── Static system prompt (identical every call → KV-cached) ──
    system_prompt = f"""You are a Schema Alignment Expert.  Your job is to analyse an ontology
and a CSV dataset to produce a **Functional Entity Plan** — a structured
blueprint that downstream agents will follow when generating YARRRML.

{GOLDEN_RULES}

### YOUR TASK

1. **Identify the PRIMARY ONTOLOGY PREFIX** from the ontology.
   This is the prefix used in subject URIs (e.g. ``podio:``, ``gtfs:``).
   All subject templates MUST use this prefix: ``prefix:ClassName/$(id)``

2. **Classify each CSV column** into one of:
   - **IRI column** — contains full URLs/IRIs (http://, https://, etc.)
     → map as ``$(column)~iri`` directly. Do NOT create separate mappings.
   - **Foreign key column** — contains an ID referencing another entity
     in the SAME CSV (like ``parent_id``)
     → link via URI template: ``prefix:Class/$(fk_col)~iri``
     NEVER via joins (joins are only for different CSV files).
   - **Data property column** — contains literal values
     → map as ``[predicate, $(column), xsd:type]``

3. **Check for Semantic Node Splitting triggers:**
   Scan the ontology for object properties like ``metadata``, ``hasPart``,
   ``is_part_of`` whose domain and range are the same class.
   If found → create a secondary mapping with a ``/Suffix`` URI pattern.

4. **Check for Parent Entity triggers:**
   If the ontology has ``is_part_of`` / ``has_part`` and the CSV has a
   column like ``parent_id``, create a SEPARATE parent mapping:
   - Subject uses the parent FK: ``prefix:Class/$(parent_id)``
   - Inverse link: ``[eli:has_part, prefix:Class/$(id)~iri]``
   - Map ``parent_*`` columns (like ``parent_source``) in this mapping.

5. **DISTRIBUTE properties across mappings (CRITICAL):**
   Do NOT copy all columns into every mapping.  Split semantically:
   - **Primary entity**: content/payload properties (content, wordCount)
     + main IRI links + link TO metadata + link TO parent.
   - **Metadata sub-resource**: admin/bibliographic properties
     (title, subject, jurisdiction, localId, versionDate, hasPDF).
   - **Parent entity**: parent_* columns + inverse link + shared IRIs.
   A column CAN appear in multiple mappings with DIFFERENT predicates
   (e.g. ``date`` → ``terms:created`` in Primary, ``eli:version_date``
   in Metadata).

6. **Link direction:** The PRIMARY links TO Metadata.  Metadata does NOT
   link to itself.

7. **Determine: flat vs multi-node**
   - FLAT: one class, no metadata/hierarchy properties
   - MULTI-NODE: has metadata, parent-child, or multiple classes

### OUTPUT FORMAT (strict — follow exactly)

```
MULTI_NODE: true/false
PRIMARY_PREFIX: <prefix_name>
PRIMARY_CLASS: <prefix:ClassName>

ENTITY: <PrimaryMappingName>
  CLASS: <ontology class URI>
  IDENTIFIER: <column_name>
  SUBJECT_TEMPLATE: <prefix:ClassName/$(column_name)>
  CONTENT_PROPERTIES (belong HERE, not in metadata):
    - <column> -> <ontology_predicate> (xsd:<type>)
  IRI_COLUMNS:
    - <column> -> <ontology_predicate> (direct ~iri)
  URI_TEMPLATE_LINKS:
    - <ontology_predicate> -> <prefix:TargetClass/$(fk_col)~iri>
    - lkg:metadata -> <prefix:Class/$(id)/Metadata~iri>

ENTITY: <ParentMappingName>  (if is_part_of / parent_id exists)
  CLASS: <same class URI>
  IDENTIFIER: <parent_id_column>
  SUBJECT_TEMPLATE: <prefix:ClassName/$(parent_id)>
  PARENT_SPECIFIC_COLUMNS:
    - <parent_source> -> <predicate> (~iri)
  INVERSE_LINK:
    - eli:has_part -> <prefix:Class/$(id)~iri>
  SHARED_IRI_COLUMNS:
    - <column> -> <predicate> (~iri)

ENTITY: <MetadataMappingName>  (if metadata property exists)
  CLASS: <same class URI>
  IDENTIFIER: <id_column>
  SUBJECT_TEMPLATE: <prefix:ClassName/$(id)/Metadata>
  ADMIN_PROPERTIES (belong HERE, not in primary):
    - <column> -> <ontology_predicate> (xsd:<type>)
  NOTE: Does NOT contain a link to itself.
```

### RULES
1. DISTRIBUTE properties — do NOT put everything in every mapping.
2. Use ONLY class URIs and property URIs from the ontology.
3. Subject templates MUST use the ontology prefix, NOT base_uri.
4. NEVER plan joins for same-CSV references — use URI templates.
5. If a column holds URLs/IRIs (values starting with http:// or https://),
   classify it as IRI_COLUMN.  If a column holds usernames, handles
   (e.g. @user9), plain text, codes, categories, or locales, it is a
   DATA PROPERTY — NEVER mark it as IRI_COLUMN.  Check the sample data.
6. A column CAN appear in multiple entities with DIFFERENT predicates.
7. Primary links TO Metadata.  Metadata does NOT self-link.
8. Output ONLY the entity plan — no YARRRML, no code blocks.
"""

    # ── Dynamic human message (changes per call) ──
    human_prompt = f"""### INPUT

**Ontology:**
{ontology}

**CSV Columns:** {columns}
**Columns with ALL unique values (safe as URI keys):** {unique_cols if unique_cols else 'NONE — consider using a composite key or row index'}
**CSV Sample Data:** {sample}

**Schema Analysis (from previous agent):**
{analysis}

**Mapping Plan (from previous agent — follow these column-to-class assignments):**
{mapping_plan}

IMPORTANT: The mapping plan above represents validated column-to-class assignments
from the semantic analysis agent. You MUST follow these assignments — do NOT
override or ignore them. Place each column in the mapping that matches the class
it was assigned to.

**Base URI:** {base_uri}
{cq_section}

Produce the Functional Entity Plan now.
"""

    # Use streaming to keep connection alive during long generations
    plan_text = ""
    for chunk in llm.stream([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]):
        plan_text += chunk.content
    plan_text = plan_text.strip()

    # Determine if multi-node was detected (case-insensitive)
    import re
    multi_node = bool(re.search(r'multi[_\s-]*node\s*:\s*true', plan_text, re.IGNORECASE))

    return {
        "entity_plan": plan_text,
        "multi_node": multi_node,
    }
