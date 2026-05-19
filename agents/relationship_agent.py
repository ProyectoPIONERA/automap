"""
Relationship Linker Agent — produces ONLY the new object-property PO
entries needed to connect mappings in a YARRRML file.

The agent outputs a lightweight ADD_TO instruction set (mapping name →
list of new po entries).  The coordinator merges these into the existing
YARRRML programmatically — eliminating the need to regenerate ~12 000
tokens of unchanged YARRRML.

Prompt is split into a STATIC system message (cached by llama.cpp after
the first call) and a DYNAMIC human message (changes per call/retry).
"""

import re
import yaml as pyyaml

from config.settings import get_llm, get_llm_with_retry
from config.yarrrml_examples import EXAMPLE_FOR_RELATIONSHIP_LINKER
from langchain_core.messages import SystemMessage, HumanMessage

# ── Static system prompt (identical every call → KV-cached by llama.cpp) ──
_SYSTEM_PROMPT = f"""{EXAMPLE_FOR_RELATIONSHIP_LINKER}

You are given a SUMMARY of existing YARRRML mappings (subjects, types,
existing predicates and links).  Your job is to identify MISSING
object-property links and output ONLY the new entries to add.

Do NOT output the full YARRRML.  Do NOT repeat existing entries.

### HOW TO ADD LINKS (prioritised):

**1. Same-CSV URI template (ALWAYS prefer this):**
  `[predicate, prefix:ClassName/$(fk_column)~iri]`
Examples:
  - Parent link:    `[eli:is_part_of, podio:ApprovedPolicy/$(parent_id)~iri]`
  - Metadata link:  `[lkg:metadata, podio:ApprovedPolicy/$(id)/Metadata~iri]`
  - IRI column:     `[terms:publisher, $(publisher)~iri]`

**2. Cross-CSV join (ONLY when data is in a DIFFERENT CSV file):**
  `[predicate, TargetMapping~iri, joins: [child: local_col, parent: remote_col]]`
RARE in single-CSV datasets.  Do NOT use joins for same-CSV data.

### CRITICAL RULES:
1. ONLY output NEW object-property PO entries that are MISSING.
2. Check "existing_links" — do NOT duplicate links that already exist.
3. NEVER use `joins:` for same-CSV references.  Use URI templates.
4. If ALL links already exist, output exactly: NONE
5. The ~iri suffix is REQUIRED for all object-property links.
6. ALL entries MUST use flow-style: `- [predicate, object~iri]`
7. NEVER use $(column)~iri for plain text, usernames, codes, etc.

### OUTPUT FORMAT (strict):

ADD_TO: MappingName1
  - [predicate, prefix:Class/$(col)~iri]

ADD_TO: MappingName2
  - [eli:has_part, prefix:Class/$(id)~iri]

If NO links needed, output exactly: NONE

No full YARRRML.  No markdown.  No explanations.
"""


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _strip_markdown(content: str) -> str:
    """Remove markdown code fences from LLM output."""
    if "```" not in content:
        return content.strip()
    parts = content.split("```")
    for part in parts:
        if "ADD_TO" in part or "NONE" in part:
            content = part
            break
    lines = content.splitlines()
    if lines and lines[0].strip().lower() in ("yaml", "yml"):
        content = "\n".join(lines[1:])
    return content.strip()


def _parse_link_instructions(raw_output: str) -> dict[str, list[str]]:
    """Parse ADD_TO output into {mapping_name: [po_entry_strings]}."""
    raw = raw_output.strip()
    if not raw or raw.upper() == "NONE":
        return {}

    result: dict[str, list[str]] = {}
    current_mapping = None

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        m = re.match(r'^ADD_TO:\s*(\S+)', stripped)
        if m:
            current_mapping = m.group(1)
            result.setdefault(current_mapping, [])
            continue

        if current_mapping and stripped.startswith("- ["):
            result[current_mapping].append(stripped)

    return result


def _merge_links_into_yarrrml(yarrrml: str, links: dict[str, list[str]]) -> str:
    """Insert new po entries into the YARRRML text at the right positions.

    For each target mapping, finds the last ``- [`` line inside its
    ``po:`` block and appends the new entries immediately after it.
    """
    if not links:
        return yarrrml

    result_lines = yarrrml.split("\n")

    for target_name, new_entries in links.items():
        if not new_entries:
            continue

        # Find the mapping block and its last po entry
        in_target = False
        in_po = False
        last_po_idx = -1
        po_indent = "      "

        for idx, line in enumerate(result_lines):
            stripped = line.lstrip()

            # Detect mapping name (2-space indent under mappings:)
            m = re.match(r'^(\s{2})(\w[\w_-]*):\s*$', line)
            if m:
                if in_target and last_po_idx >= 0:
                    break  # left the target block — stop searching
                name = m.group(2)
                if name == target_name or name.lower() == target_name.lower():
                    in_target = True
                    in_po = False
                    last_po_idx = -1
                else:
                    in_target = False
                continue

            if in_target:
                if stripped == "po:":
                    in_po = True
                elif in_po and stripped.startswith("- ["):
                    last_po_idx = idx
                    po_indent = re.match(r'^(\s*)', line).group(1)

        # Insert new entries after last po line
        if last_po_idx >= 0:
            insert_lines = []
            for entry in new_entries:
                e = entry.strip()
                if e.startswith("- "):
                    insert_lines.append(f"{po_indent}{e}")
                else:
                    insert_lines.append(f"{po_indent}- {e}")

            for j, new_line in enumerate(insert_lines):
                result_lines.insert(last_po_idx + 1 + j, new_line)

    return "\n".join(result_lines)


def _list_all_existing_predicates(yarrrml: str) -> dict[str, list[str]]:
    """Return {mapping_name: [predicate, ...]} for every PO entry in the YARRRML.

    Used to tell the relationship agent which predicates already exist so it
    does not generate duplicates.
    """
    try:
        data = pyyaml.safe_load(yarrrml)
    except Exception:
        return {}
    if not isinstance(data, dict) or "mappings" not in data:
        return {}
    result: dict[str, list[str]] = {}
    for mname, mdef in (data.get("mappings") or {}).items():
        if not isinstance(mdef, dict):
            continue
        preds: list[str] = []
        for entry in (mdef.get("po") or []):
            if isinstance(entry, list) and len(entry) >= 1:
                preds.append(str(entry[0]))
        if preds:
            result[mname] = preds
    return result


def _build_mapping_summary(yarrrml: str) -> str:
    """Build a compact summary: subject + type + predicates + existing links.

    Replaces sending the full YARRRML (~8k tokens input) with a
    ~300-500 token summary the LLM can reason about efficiently.
    """
    try:
        data = pyyaml.safe_load(yarrrml)
    except Exception:
        return yarrrml[:2000] + "\n... (truncated)"

    if not isinstance(data, dict) or "mappings" not in data:
        return yarrrml[:2000]

    summaries: list[str] = []
    mappings = data.get("mappings", {})
    if not isinstance(mappings, dict):
        return yarrrml[:2000]

    for mname, mdef in mappings.items():
        if not isinstance(mdef, dict):
            continue
        subj = mdef.get("s", "?")
        parts = [f"\n{mname}:", f"  subject: {subj}"]

        po = mdef.get("po", [])
        data_preds: list[str] = []
        obj_links: list[str] = []
        for entry in po:
            if isinstance(entry, list) and len(entry) >= 2:
                pred = str(entry[0])
                obj = str(entry[1])
                if pred in ("a", "rdf:type"):
                    parts.append(f"  type: {obj}")
                elif "~iri" in obj or ("/" in obj and "$(" in obj):
                    obj_links.append(f"    {pred} -> {obj}")
                else:
                    data_preds.append(pred)

        if data_preds:
            parts.append(f"  data_predicates: [{', '.join(data_preds)}]")
        if obj_links:
            parts.append("  existing_links:")
            parts.extend(obj_links)
        else:
            parts.append("  existing_links: NONE")

        summaries.extend(parts)

    return "\n".join(summaries)


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────

def call_relationship_agent(state: dict, current_yarrrml: str) -> str:
    """Produce new linking PO entries and merge them into the YARRRML.

    The LLM outputs ONLY new entries (~200-500 tokens) instead of
    regenerating the full YARRRML (~12 000 tokens).  The merge is
    done programmatically.

    Parameters
    ----------
    state : dict
        Pipeline state with mapping_plan, ontology_info, etc.
    current_yarrrml : str
        The assembled YARRRML (prefixes + entity mappings) without links.

    Returns
    -------
    str
        The complete YARRRML with relationship links merged in.
    """
    llm = get_llm_with_retry(role="relationship_linker")

    mapping = state["mapping_plan"].get("analysis", "")
    ontology = state["ontology_info"].get("raw", "")

    MAX_ONTOLOGY = 3000
    MAX_MAPPING = 2000
    if len(ontology) > MAX_ONTOLOGY:
        ontology = ontology[:MAX_ONTOLOGY] + "\n... (truncated)"
    if len(mapping) > MAX_MAPPING:
        mapping = mapping[:MAX_MAPPING] + "\n... (truncated)"

    alignment = state.get("schema_alignment", {})
    entity_plan = alignment.get("entity_plan", "")

    alignment_section = ""
    if entity_plan:
        alignment_section = f"""
### SCHEMA ALIGNMENT PLAN (defines which relationships must exist):
{entity_plan}

Follow the URI_TEMPLATE_LINKS from the plan exactly.
"""

    cqs = state.get("competency_questions", [])
    cq_section = ""
    if cqs:
        cq_list = "\n".join(f"  - {q}" for q in cqs)
        cq_section = f"""
### COMPETENCY QUESTIONS (the links MUST enable answering ALL of these):
{cq_list}
"""

    feedback = state.get("feedback", "")
    feedback_section = ""
    if feedback and "PASSED" not in feedback and "APPROVED" not in feedback:
        if "CQ_ERROR" in feedback:
            feedback_section = f"""
### FIX REQUIRED — CQ validation FAILED:
{feedback}
"""
        elif "SYNTAX_ERROR" in feedback:
            feedback_section = f"""
### FIX REQUIRED — YARRRML translation FAILED:
{feedback}

IMPORTANT: Do NOT use `joins:` for same-CSV references.
Use URI templates instead: `[predicate, prefix:Class/$(fk_col)~iri]`
"""
        elif any(kw in feedback.lower() for kw in [
            "join", "link", "disconnect", "island", "object property",
            "iri", "referenced", "outgoing",
        ]):
            feedback_section = f"""
### FIX REQUIRED — Previous output had relationship issues:
{feedback}
"""

    # Build compact summary instead of sending full YARRRML
    mapping_summary = _build_mapping_summary(current_yarrrml)

    # Collect all existing predicates so the agent does NOT duplicate them
    existing_predicates = _list_all_existing_predicates(current_yarrrml)
    existing_pred_section = ""
    if existing_predicates:
        existing_pred_section = (
            "\n### EXISTING PO ENTRIES — DO NOT DUPLICATE ANY OF THESE PREDICATES:\n"
            + "\n".join(f"  {m}: [{', '.join(preds)}]" for m, preds in sorted(existing_predicates.items()))
            + "\nOnly add NEW predicates that are NOT already listed above.\n"
        )

    human_prompt = f"""{alignment_section}
{cq_section}
{feedback_section}
{existing_pred_section}
### EXISTING MAPPINGS (subjects, types, predicates, and current links):
{mapping_summary}

Mapping Plan: {mapping}
Ontology Context: {ontology}

Output ONLY the new ADD_TO link entries, or NONE if all links already exist.
"""

    raw_output = ""
    for chunk in llm.stream([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]):
        raw_output += chunk.content
    raw_output = _strip_markdown(raw_output.strip())

    # Parse and merge
    link_instructions = _parse_link_instructions(raw_output)

    if not link_instructions:
        return current_yarrrml

    merged = _merge_links_into_yarrrml(current_yarrrml, link_instructions)
    return merged
