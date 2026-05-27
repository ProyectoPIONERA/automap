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

from config.settings import get_llm_with_retry
from config.yarrrml_examples import GOLDEN_RULES
from langchain_core.messages import SystemMessage, HumanMessage
import re as _re


def _parse_object_properties(ontology_raw: str) -> list[dict]:
    """Parse owl:ObjectProperty declarations from Turtle ontology text.

    Returns list of dicts: {name, domain, range} using CURIE forms.
    Used to inject deterministic hints into the schema alignment prompt
    so the LLM knows which predicates MUST link to IRI entities.
    """
    if not ontology_raw:
        return []

    # Build prefix map from @prefix declarations
    prefix_map: dict[str, str] = {}
    for pm in _re.finditer(r'@prefix\s+(\w*):\s*<([^>]+)>', ontology_raw):
        prefix_map[pm.group(1)] = pm.group(2)

    def _uri_to_curie(uri: str) -> str:
        for pfx, base in sorted(prefix_map.items(), key=lambda x: -len(x[1])):
            if uri.startswith(base):
                return f"{pfx}:{uri[len(base):]}"
        return uri

    obj_props: list[dict] = []
    # Find blocks like: <uri> a owl:ObjectProperty ; rdfs:domain <x> ; rdfs:range <y>
    # Also handles short-form: propName a owl:ObjectProperty ;
    block_re = _re.compile(
        r'([\w:]+|<[^>]+>)\s+a\s+owl:ObjectProperty\s*;([^.]+)\.',
        _re.DOTALL,
    )
    for m in block_re.finditer(ontology_raw):
        prop_raw = m.group(1).strip('<>').strip()
        body = m.group(2)

        domain_m = _re.search(r'rdfs:domain\s+([\w:]+|<[^>]+>)', body)
        range_m = _re.search(r'rdfs:range\s+([\w:]+|<[^>]+>)', body)

        prop_curie = _uri_to_curie(prop_raw) if prop_raw.startswith('http') else prop_raw
        domain_curie = None
        range_curie = None

        if domain_m:
            d = domain_m.group(1).strip('<>')
            domain_curie = _uri_to_curie(d) if d.startswith('http') else d
        if range_m:
            r = range_m.group(1).strip('<>')
            range_curie = _uri_to_curie(r) if r.startswith('http') else r

        if range_curie:
            obj_props.append({"name": prop_curie, "domain": domain_curie, "range": range_curie})

    return obj_props


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
    ontology = state.get("ontology_info", {}).get("raw", "")
    schema = state.get("schema_info", {})
    columns = schema.get("raw", {}).get("columns", [])
    base_uri = state.get("base_uri", "http://example.org/")
    feedback = state.get("feedback", "")

    # ── Fast path: deterministic plan from ontology (< 0.5 s) ────────
    # Skip on retries after CQ/logic failures so the LLM can fix
    # structural problems the heuristic can't catch on its own.
    is_retry = bool(feedback) and not any(
        kw in feedback for kw in ("APPROVED", "PASSED")
    )

    if not is_retry:
        try:
            from agents.ontology_entity_planner import build_deterministic_entity_plan
            plan_text, multi_node = build_deterministic_entity_plan(
                ontology, columns, base_uri
            )
            if "ENTITY:" in plan_text:
                print("  [Schema Alignment] Deterministic plan built (<1s) — skipping LLM")
                return {"entity_plan": plan_text, "multi_node": multi_node}
        except Exception as _det_err:
            print(f"  [Schema Alignment] Deterministic plan failed ({_det_err}), falling back to LLM")

    # ── Slow path: LLM (retries or sparse ontology) ───────────────────
    print("  [Schema Alignment] Using LLM (retry or sparse ontology)...")
    llm = get_llm_with_retry(role="schema_alignment")

    unique_cols = schema.get("raw", {}).get("unique_columns", [])
    sample = schema.get("raw", {}).get("sample", "")
    analysis = schema.get("analysis", "")
    mapping_plan = state.get("mapping_plan", {}).get("analysis", "")

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
   - **Entity ID column** — contains IDs that identify a DISTINCT ENTITY TYPE
     (e.g. ``CustomerID``, ``ProductID``, ``DriverID``, ``StoreID``).
     Multiple rows share the same ID value, meaning each unique value
     represents a real-world entity (a customer, a product, etc.).
     → Create a SEPARATE mapping for that entity:
       ```
       CustomerMapping:
         s: ex:customer/$(CustomerID)
         po:
           - [a, schema:Person]
           - [schema:identifier, $(CustomerID), xsd:string]
       ```
     → In the REFERENCING mapping (e.g. InvoiceMapping), link via IRI:
       ```[schema:customer, ex:customer/$(CustomerID)~iri]```
     KEY HEURISTIC: A column is an entity ID if (a) its name ends in ``ID``
     or ``Id`` or ``_id``, AND (b) it is NOT the primary/composite key of
     the main entity (e.g. ``InvoiceNo`` is the PK, ``CustomerID`` is FK).
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
9. **CRITICAL — ONTOLOGY OBJECT PROPERTIES:**
   Scan the ontology for ALL ``owl:ObjectProperty`` declarations.
   For each ObjectProperty with a declared ``rdfs:range`` class:
   - The range class MUST become a separate entity mapping.
   - The CSV column that contains IDs for that range class MUST be
     identified as an ENTITY ID COLUMN (see rule 2 above).
   - The referencing mapping MUST use a URI_TEMPLATE_LINK:
       ``schema:customer -> ex:customer/$(CustomerID)~iri``
   Example: if ontology has:
     ``schema:customer a owl:ObjectProperty ; rdfs:domain schema:Order ; rdfs:range schema:Person``
   Then you MUST create:
     - A ``CustomerMapping`` with ``s: ex:customer/$(CustomerID)`` and class ``schema:Person``
     - In ``InvoiceMapping``: ``URI_TEMPLATE_LINKS: schema:customer -> ex:customer/$(CustomerID)~iri``
   Do NOT map ObjectProperty range values as literals (xsd:string).
"""

    # ── Parse ObjectProperties from ontology for deterministic hints ──
    obj_props = _parse_object_properties(ontology)
    obj_prop_section = ""
    if obj_props:
        lines = ["### ONTOLOGY OBJECT PROPERTIES — each range class MUST become a separate mapping:"]
        for op in obj_props:
            domain = op.get("domain") or "?"
            range_ = op["range"]
            name = op["name"]
            lines.append(
                f"  - {name}: domain={domain}, range={range_}  "
                f"→ create a separate mapping for {range_} entities "
                f"and link via URI template"
            )
        obj_prop_section = "\n".join(lines) + "\n"

    # ── Dynamic human message (changes per call) ──
    # At attempt 3+ aggressively truncate to prevent context overflow (400 Bad Request).
    retry_count = state.get("retry_count", 0)
    if retry_count >= 2:
        # Slim-mode: only columns, top-3 failing CQs, truncated ontology
        ontology_trimmed = ontology[:2000] + "\n...(truncated)" if len(ontology) > 2000 else ontology
        # Keep only the first failing CQs from cq_section
        cq_section_trimmed = cq_section[:600] + "\n...(truncated)" if len(cq_section) > 600 else cq_section
        human_prompt = f"""### INPUT (slim — attempt {retry_count + 1})

**Ontology (truncated):**
{ontology_trimmed}

**CSV Columns:** {columns}

**Base URI:** {base_uri}
{obj_prop_section}{cq_section_trimmed}

Produce the Functional Entity Plan now.
"""
    else:
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
{obj_prop_section}{cq_section}

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
