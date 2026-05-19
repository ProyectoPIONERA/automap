"""
agents/cq_generator_agent.py
=============================
LLM agent that auto-generates Competency Questions (CQs) from the CSV schema
and ontology when the user has not provided any CQs.

These CQs serve TWO purposes in the pipeline:
  1. Navigation signal — guide the YARRRML generator toward the intended
     semantic structure (what entities and relationships matter).
  2. SPARQL validation target — after KG materialisation, each CQ is
     translated to an ASK query and executed to confirm coverage.

As discussed in the design rationale, auto-generated CQs are an internal
consistency check, not an independent ground-truth validation.  The gold KG
comparison remains the definitive accuracy measure.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_llm

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an ontology and knowledge-graph engineering expert.

Your task: given a CSV dataset description and an ontology, generate a set of
Competency Questions (CQs) that the resulting Knowledge Graph should be able
to answer.

### RULES FOR GOOD CQs
1. Each CQ must be a natural-language question (start with Who/What/Which/How).
2. Each CQ must be answerable by querying the RDF graph produced from this CSV.
3. CQs must cover:
   - At least one "entity existence" question (e.g. "Which X exist in the dataset?")
   - At least one "attribute" question (e.g. "What is the Y of a given X?")
   - At least one "relationship" question if the ontology has object properties
     (e.g. "Which X are linked to Y via Z?")
   - Domain-specific questions based on the dataset's subject matter.
4. Do NOT ask about data that is clearly not in the CSV columns.
5. Keep questions concise — one concept per question.
6. Generate between 5 and 10 questions.

### OUTPUT FORMAT
Return ONLY a numbered list, one question per line:
1. Which ...
2. What ...
3. ...
"""


def generate_cqs(
    schema_info: dict,
    ontology_info: dict,
    base_uri: str = "http://example.org/",
    entity_plan: str | None = None,
    llm=None,
) -> list[str]:
    """Auto-generate Competency Questions from CSV schema + ontology.

    Parameters
    ----------
    schema_info :
        AgentState ``schema_info`` dict (contains ``raw.columns``, ``analysis``).
    ontology_info :
        AgentState ``ontology_info`` dict (contains ``raw`` subgraph).
    base_uri :
        Pipeline base URI (used for context only).
    entity_plan :
        Functional Entity Plan from the schema alignment agent. When provided,
        CQs are grounded to the ACTUAL entity types and predicates that will
        be in the KG (e.g. ``schema:OrderItem``) rather than abstract ontology
        classes (e.g. ``schema:Order``). This prevents SPARQL validation from
        failing on concepts the KG never materialises.
    llm :
        LLM instance; uses default role if None.

    Returns
    -------
    list[str]
        List of generated CQ strings (without the leading number).
    """
    if llm is None:
        llm = get_llm(role="cq_validator")

    # ── Build context ──────────────────────────────────────────────────────
    raw_schema = schema_info.get("raw", {})
    columns = raw_schema.get("columns", []) if isinstance(raw_schema, dict) else []
    sample  = raw_schema.get("sample", {})  if isinstance(raw_schema, dict) else {}
    schema_analysis = schema_info.get("analysis", "")

    ontology_raw = ontology_info.get("raw", {})
    if isinstance(ontology_raw, dict):
        classes    = ontology_raw.get("classes", [])
        obj_props  = ontology_raw.get("object_properties", [])
        data_props = ontology_raw.get("data_properties", [])
        ontology_ctx = (
            f"Classes: {classes[:20]}\n"
            f"Object properties: {obj_props[:20]}\n"
            f"Data properties: {data_props[:20]}"
        )
    else:
        # raw is a plain Turtle/text string — use it directly
        ontology_ctx = str(ontology_raw)[:3000]

    sample_rows = ""
    if isinstance(sample, dict):
        for col, vals in list(sample.items())[:5]:
            sample_rows += f"  {col}: {vals[:3]}\n"
    elif isinstance(sample, list):
        for row in sample[:2]:
            sample_rows += f"  {row}\n"

    # ── Entity plan grounding block ────────────────────────────────────────
    # Extract the actual entity CLASS lines and predicate lines from the plan
    # so CQs reference real KG types, not hallucinated ontology concepts.
    entity_plan_block = ""
    if entity_plan:
        # Extract CLASS: and SUBJECT_TEMPLATE: lines as ground truth for the LLM
        plan_lines = [
            l.strip() for l in entity_plan.splitlines()
            if any(l.strip().startswith(kw) for kw in (
                "ENTITY:", "CLASS:", "SUBJECT_TEMPLATE:", "URI_TEMPLATE_LINKS:",
                "DATA_PROPERTIES:", "  - ",
            ))
        ]
        if plan_lines:
            entity_plan_block = f"""
### ENTITY PLAN — CRITICAL: base ALL CQs on THESE entity types and predicates ONLY.
Do NOT reference any class or predicate that is NOT listed here.
{chr(10).join(plan_lines[:40])}
"""

    human_prompt = f"""### CSV SCHEMA
Columns: {columns}
Sample values:
{sample_rows}
Schema analysis: {schema_analysis[:500]}
{entity_plan_block}
### ONTOLOGY CONTEXT (for background only — use entity plan above for type names)
{ontology_ctx}

### BASE URI
{base_uri}

Generate {5}-{10} Competency Questions for this dataset now."""

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ])

    raw = response.content if hasattr(response, "content") else str(response)
    return _parse_cq_list(raw)


def _parse_cq_list(text: str) -> list[str]:
    """Parse a numbered list of CQs from LLM output."""
    cqs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading number + dot/paren: "1. ...", "1) ...", "1 ..."
        cleaned = re.sub(r"^\d+[.)]\s*", "", line).strip()
        if cleaned and cleaned[0].isupper() and len(cleaned) > 10:
            cqs.append(cleaned)
    return cqs


