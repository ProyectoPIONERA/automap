from tools.rml_tools import get_csv_schema, get_ontology_subgraph
from agents.schema_agent import call_schema_llm
from agents.mapper_agent import call_mapper_llm
from agents.yarrrml_agent import call_yarrrml_architect_llm
from data.checkpoints import AgentState
from agents.refiner_agent import call_refiner_llm
from datetime import datetime

import yatter
from ruamel.yaml import YAML
import os
import re

import morph_kgc


def schema_agent_node(state):
    # 1. Physical extraction (The Tool)
    raw_schema = get_csv_schema(state["csv_path"])

    # 2. Semantic understanding (The Agent)
    analysis = call_schema_llm(raw_schema)

    # 3. Update State
    return {
        "schema_info": {
            "raw": raw_schema,
            "analysis": analysis
        },
        "messages": [f"Schema Agent: Identified data as {analysis[:50]}..."]
    }


def ontology_scout_node(state):
    keywords = state["schema_info"]["raw"]["columns"]
    ontology_info = get_ontology_subgraph(state["ontology_path"], keywords)
    return {
        "ontology_info": {"raw": ontology_info},
        "messages": ["Ontology Scout: Extracted relevant ontology subgraph"]
    }


def mapper_agent_node(state):
    mapping = call_mapper_llm(
        state["schema_info"],
        state["ontology_info"]
    )
    return {
        "mapping_plan": {"analysis": mapping},
        "messages": [f"Mapper Agent: {mapping[:50]}..."]
    }


def _fix_subject_lists(yarrrml: str) -> str:
    """Fix ``s:`` keys that the LLM emitted as lists instead of strings.

    Handles two patterns the LLM produces and converts them into the
    single-string format Yatter requires, **without** re-serializing
    the full YAML (which would destroy flow-style ``po:``/``sources:``
    entries that Yatter needs).

    Pattern A (inline):
        s: ["ex:stop/", "$(stop_id)"]
        →  s: "ex:stop/$(stop_id)"

    Pattern B (multi-line):
        s:
          - ["ex:stop/", "$(stop_id)"]
        →  s: "ex:stop/$(stop_id)"
    """
    import ast

    lines = yarrrml.split('\n')
    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # ── Pattern A: s: ["prefix:path/", "$(col)"] ─────────
        m_inline = re.match(r'^(\s*s:\s*)(\[.+\])\s*$', line)
        if m_inline:
            indent_and_key = m_inline.group(1)   # e.g. "    s: "
            list_str = m_inline.group(2)
            joined = _try_join_list(list_str)
            if joined is not None:
                result.append(f'{indent_and_key}"{joined}"')
                i += 1
                continue

        # ── Pattern B: s:  (bare, value on next line) ─────────
        if re.match(r'^(\s*)s:\s*$', line) and i + 1 < len(lines):
            next_stripped = lines[i + 1].strip()
            m_next = re.match(r'^-\s*(\[.+\])\s*$', next_stripped)
            if m_next:
                indent = re.match(r'^(\s*)', line).group(1)
                list_str = m_next.group(1)
                joined = _try_join_list(list_str)
                if joined is not None:
                    result.append(f'{indent}s: "{joined}"')
                    i += 2          # skip both lines
                    continue

        result.append(line)
        i += 1

    return '\n'.join(result)


def _try_join_list(list_str: str) -> str | None:
    """Try to parse a string like '["a", "b"]' and join into "ab".

    Returns the joined string, or None if parsing fails or the value
    is not a list of strings.
    """
    import ast
    try:
        parts = ast.literal_eval(list_str)
        if not isinstance(parts, list):
            return None
        # Flatten one level: [["a", "b"]] → ["a", "b"]
        while len(parts) == 1 and isinstance(parts[0], list):
            parts = parts[0]
        if all(isinstance(p, str) for p in parts):
            return "".join(parts)
    except (ValueError, SyntaxError):
        pass
    return None


def _strip_prefix_angle_brackets(yarrrml: str) -> str:
    """Remove angle brackets from YARRRML prefix URIs.

    The LLM often copies Turtle-style ``<URI>`` into the YARRRML
    ``prefixes:`` block.  YARRRML wants bare strings or quoted strings,
    not angle-bracket-wrapped URIs.  When Yatter translates to Turtle
    it adds its own ``<…>``, creating invalid ``<<…>>`` doubles.

    This function converts:
        prefix: <http://example.org/>
    to:
        prefix: "http://example.org/"
    """
    lines = yarrrml.split('\n')
    result: list[str] = []
    in_prefixes = False

    for line in lines:
        stripped = line.strip()

        # Detect when we enter/leave the prefixes block
        if stripped.startswith('prefixes:'):
            in_prefixes = True
            result.append(line)
            continue
        elif in_prefixes and stripped and not stripped.startswith('#'):
            # Still in prefixes if the line is indented (continuation)
            if line[0] in (' ', '\t'):
                # Strip angle brackets: `key: <URI>` → `key: "URI"`
                m = re.match(r'^(\s+\S+:\s*)<([^>]+)>\s*$', line)
                if m:
                    result.append(f'{m.group(1)}"{m.group(2)}"')
                    continue
            else:
                in_prefixes = False

        result.append(line)

    return '\n'.join(result)


def yarrrml_architect_node(state):
    current_retries = state.get("retry_count", 0)
    yarrrml = call_yarrrml_architect_llm(state)

    # ── Strip <> from prefix URIs (Turtle-style → plain YARRRML) ─
    yarrrml = _strip_prefix_angle_brackets(yarrrml)

    # ── Fix s: when LLM outputs it as a list instead of a string ─
    # Must run BEFORE ~iri handling so we don't accidentally inject
    # ~iri into subject URI templates.
    yarrrml = _fix_subject_lists(yarrrml)

    # ── Smart ~iri handling ──────────────────────────────────
    # LLMs sometimes put ~iri in wrong places (data properties,
    # subjects). We strip all ~iri first, then re-add it ONLY to
    # 2-item PO entries that are object-property links — i.e.
    # entries whose object is a URI template (starts with a prefix
    # or http://) and contains $(col).
    yarrrml = yarrrml.replace("~iri", "")
    yarrrml = re.sub(
        r'("(?:https?://|[a-zA-Z][a-zA-Z0-9]*:)[^"]*\$\([^)]+\)[^"]*)("\s*\])',
        r'\1~iri\2',
        yarrrml
    )


    # Save for debug
    os.makedirs("data/output/debug", exist_ok=True)
    with open("data/output/debug/last_attempt.yaml", "w") as f:
        f.write(yarrrml)

    return {
        "yarrrml_output": yarrrml,
        "retry_count": current_retries + 1,
        "feedback": "",
        "messages": [f"Architect: Generated attempt #{current_retries + 1}"]
    }



def validation_node(state: AgentState):
    yarrrml_content = state["yarrrml_output"]
    yaml = YAML(typ='safe', pure=True)

    try:
        # Load the string as YAML and attempt Yatter translation
        yarrrml_data = yaml.load(yarrrml_content)
        rml_content = yatter.translate(yarrrml_data)

        # If successful, move to logic check
        return {
            "messages": ["Validator: Syntax is valid."],
            "feedback": "PASSED_SYNTAX"
        }
    except Exception as e:
        # Capture the specific error (e.g., ScannerError, YatterException)
        error_log = str(e)
        return {
            "messages": [f"Validator: Syntax Error found: {error_log[:100]}..."],
            "feedback": f"SYNTAX_ERROR: {error_log}",
            "retry_needed": True
        }


def refiner_agent_node(state):
    # The refiner now performs:
    #   Phase 1a: structural checks (islands, duplicates, redundancy)
    #             + auto-fix for duplicate predicates
    #   Phase 1b: column-coverage (tolerant of predicate conflicts)
    #   Phase 2:  LLM-based semantic / URI-logic review
    refiner_result = call_refiner_llm(state)
    logic_feedback = refiner_result["feedback"]
    conflict_cols = refiner_result.get("predicate_conflict_cols", [])
    fixed_yarrrml = refiner_result.get("fixed_yarrrml")

    # Base result dict — all branches share these keys
    result: dict = {
        "predicate_conflict_cols": conflict_cols,
    }

    # Propagate auto-fixed YARRRML into the state so the next stage
    # (KG generation or next architect retry) uses the corrected version.
    if fixed_yarrrml:
        result["yarrrml_output"] = fixed_yarrrml
        # Also save for debug
        os.makedirs("data/output/debug", exist_ok=True)
        with open("data/output/debug/auto_fixed.yaml", "w") as f:
            f.write(fixed_yarrrml)

    if "APPROVED" in logic_feedback:
        result["feedback"] = "APPROVED"
        result["messages"] = ["Refiner: [PASS] All checks passed (columns, structure, URIs)."]
        return result
    elif "STRUCTURAL PROBLEMS" in logic_feedback:
        n_errors = logic_feedback.count("\n  ")
        result["feedback"] = logic_feedback  # already prefixed with LOGIC_ERROR
        result["messages"] = [f"Refiner: [FAIL] {n_errors} structural problem(s) detected -- sending back to architect."]
        return result
    elif "LOGIC_ERROR" in logic_feedback:
        result["feedback"] = logic_feedback
        result["messages"] = ["Refiner: [FAIL] Deterministic check failed -- sending back to architect."]
        return result
    else:
        result["feedback"] = f"LOGIC_ERROR: {logic_feedback}"
        result["messages"] = ["Refiner: Found logic/URI issues — sending back to architect."]
        return result



def _internal_yarrrml_to_rml(yarrrml_content, csv_path):
    """
    Helper to convert YARRRML string to RML string and patch CSV paths.
    Handles both full-relative-path and basename-only sources.
    """
    # Use pure=True to match validation_node — the C-extension parser
    # can be stricter about unquoted colons in flow sequences.
    yaml = YAML(typ='safe', pure=True)
    yarrrml_data = yaml.load(yarrrml_content)

    # Translate YARRRML to RML (Turtle syntax)
    rml_content = yatter.translate(yarrrml_data)

    # Patch the rml:source to use the absolute path of the CSV.
    # The source in the RML may be the full relative path (e.g. data/input/file.csv)
    # or just the basename (file.csv) depending on what yatter received.
    csv_filename = os.path.basename(csv_path)
    abs_csv_path = os.path.abspath(csv_path)

    patched_rml = rml_content
    # Try full relative path first, then basename
    for candidate in [csv_path, csv_filename]:
        needle = f'rml:source "{candidate}"'
        if needle in patched_rml:
            patched_rml = patched_rml.replace(needle, f'rml:source "{abs_csv_path}"')
            break

    return patched_rml


def kg_generation_node(state):
    """
    Converts the approved YARRRML to RML and runs Morph-KGC.
    Outputs to a dedicated run directory.
    """
    yarrrml_content = state.get("yarrrml_output")
    csv_path = state.get("csv_path")

    # Use the run_dir defined in the state, or fallback to a timestamped one
    run_dir = state.get("run_dir", f"data/output/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    output_path = os.path.abspath(os.path.join(run_dir, "knowledge_graph.nt"))

    try:
        print(f"[System] Converting YARRRML to RML in {run_dir}...")
        rml_content = _internal_yarrrml_to_rml(yarrrml_content, csv_path)

        # Save RML temporarily inside the run directory
        rml_tmp_path = os.path.join(run_dir, "temp_mapping.ttl")
        with open(rml_tmp_path, "w") as f:
            f.write(rml_content)

        config_str = f"""
[CONFIGURATION]
output_file: {output_path}
output_format: N-TRIPLES

[DataSource1]
mappings: {rml_tmp_path}
        """

        print("[System] Materializing Knowledge Graph...")
        g_rdf = morph_kgc.materialize(config_str)

        if not os.path.exists(output_path):
            from rdflib import Graph, ConjunctiveGraph
            if isinstance(g_rdf, (Graph, ConjunctiveGraph)):
                g_rdf.serialize(destination=output_path, format="ntriples")

        return {
            "rdf_output": output_path,
            "messages": [f"KG Generation: Success! Created in {run_dir}"]
        }

    except Exception as e:
        error_msg = f"KG Generation Error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {"messages": [error_msg]}