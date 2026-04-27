from tools.rml_tools import get_csv_schema, get_ontology_subgraph
from agents.schema_agent import call_schema_llm
from agents.mapper_agent import call_mapper_llm
from agents.schema_alignment_agent import call_schema_alignment_agent
from agents.cq_validator_agent import call_cq_validator_agent
from agents.yarrrml_coordinator import coordinate_yarrrml_generation
from data.checkpoints import AgentState
from agents.refiner_agent import call_refiner_llm
from datetime import datetime

import yaml as pyyaml   # PyYAML for normalisation
import yatter
from ruamel.yaml import YAML
import os
import re

import morph_kgc


# ────────────────────────────────────────────────────────────────────
# YARRRML structure normaliser
# ────────────────────────────────────────────────────────────────────

def _normalize_yarrrml_structure(yarrrml_str: str) -> str:
    """Parse YARRRML, fix malformed po/sources entries, and re-serialise
    in a format that Yatter can consume.

    Fixes:
      - po entries that are dicts (``- a: Class``) → lists (``[a, Class]``)
      - Nested single-element wrappers
      - Ensures sources are ``[[path~csv]]``
      - Re-serialises with flow-style po entries
      - Sanitises non-IRI values used with ~iri suffix
    """
    try:
        data = pyyaml.safe_load(yarrrml_str)
    except Exception:
        return yarrrml_str  # can't parse — return as-is

    if not isinstance(data, dict) or "mappings" not in data:
        return yarrrml_str

    mappings = data.get("mappings", {})
    if not isinstance(mappings, dict):
        return yarrrml_str

    for mname, mblock in mappings.items():
        if not isinstance(mblock, dict):
            continue

        # ── Fix sources ──────────────────────────────────────────
        sources = mblock.get("sources")
        if isinstance(sources, list):
            fixed_sources = []
            for s in sources:
                if isinstance(s, str):
                    fixed_sources.append([s])
                elif isinstance(s, list) and len(s) == 1 and isinstance(s[0], list):
                    fixed_sources.append(s[0])  # unwrap [[x]] → [x]
                else:
                    fixed_sources.append(s)
            mblock["sources"] = fixed_sources

        # ── Fix po entries ───────────────────────────────────────
        po = mblock.get("po")
        if not isinstance(po, list):
            continue

        fixed_po = []
        for entry in po:
            if isinstance(entry, dict):
                # Convert {pred: obj} → [pred, obj]
                for k, v in entry.items():
                    if isinstance(v, dict):
                        # e.g. {pred: {value: x, datatype: y}} → [pred, x, y]
                        val = v.get("value", v.get("~iri", ""))
                        dt = v.get("datatype", v.get("type", ""))
                        if dt:
                            fixed_po.append([str(k), str(val), str(dt)])
                        else:
                            fixed_po.append([str(k), str(val)])
                    elif isinstance(v, list):
                        fixed_po.append([str(k)] + [str(x) for x in v])
                    else:
                        fixed_po.append([str(k), str(v)])
            elif isinstance(entry, list):
                # Already a list — flatten single-wrapped: [[a,b]] → [a,b]
                if len(entry) == 1 and isinstance(entry[0], list):
                    entry = entry[0]
                # Ensure all elements are strings
                fixed_po.append([str(x) if not isinstance(x, str) else x for x in entry])
            else:
                fixed_po.append(entry)
        mblock["po"] = fixed_po

    # ── Re-serialise ─────────────────────────────────────────────
    # Build YAML manually for precise control over flow-style
    lines = []

    # Prefixes
    prefixes = data.get("prefixes", {})
    if prefixes:
        lines.append("prefixes:")
        for k, v in prefixes.items():
            # Ensure quoted URI
            v_str = str(v)
            if not v_str.startswith('"'):
                v_str = f'"{v_str}"' if ' ' in v_str or ':' in v_str else v_str
            lines.append(f"  {k}: {v_str}")
    lines.append("")

    # Mappings
    lines.append("mappings:")
    for mname, mblock in mappings.items():
        if not isinstance(mblock, dict):
            continue
        lines.append(f"  {mname}:")

        # Sources
        sources = mblock.get("sources", [])
        lines.append("    sources:")
        for s in sources:
            if isinstance(s, list):
                inner = ", ".join(str(x) for x in s)
                lines.append(f"      - [{inner}]")
            else:
                lines.append(f"      - [{s}]")

        # Subject
        subj = mblock.get("s", "")
        lines.append(f"    s: {subj}")

        # PO entries — flow-style lists
        po = mblock.get("po", [])
        if po:
            lines.append("    po:")
            for entry in po:
                if isinstance(entry, list):
                    parts = []
                    for x in entry:
                        x_str = str(x)
                        # Quote strings that contain commas or special chars
                        if ',' in x_str or ('\n' in x_str):
                            x_str = f'"{x_str}"'
                        parts.append(x_str)
                    lines.append(f"      - [{', '.join(parts)}]")
                else:
                    lines.append(f"      - {entry}")

        lines.append("")

    return "\n".join(lines)


def _sanitize_iri_values(yarrrml_str: str) -> str:
    """Remove ~iri suffix from bare column references like $(user_handle)~iri
    that are NOT valid URI templates.

    A valid IRI reference has a prefix path before the column reference:
      prefix:Class/$(col)~iri   ← KEEP (URI template)
      $(col)~iri                ← STRIP if it's in a 2-item po entry
                                   (bare column values like @user9 aren't IRIs)

    We only strip when:
      - The value is JUST $(column)~iri with no prefix/path
      - It appears in a po entry (inside [...])
    """
    # Match po entries like: [predicate, $(col)~iri]  (2-item, bare column ~iri)
    # But NOT: [predicate, prefix:Class/$(col)~iri]  (has prefix path)
    yarrrml_str = re.sub(
        r'(\[\s*[^,\]]+,\s*)\$\(([^)]+)\)~iri(\s*\])',
        r'\1$(\2), xsd:string\3',
        yarrrml_str,
    )
    return yarrrml_str


def schema_agent_node(state):
    # 1. Physical extraction (The Tool)
    raw_schema = get_csv_schema(state["csv_path"])

    # 2. Detect columns with all unique values — safe as URI keys
    try:
        import pandas as pd
        df = pd.read_csv(state["csv_path"])
        unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
    except Exception:
        unique_cols = []
    raw_schema["unique_columns"] = unique_cols

    # 3. Semantic understanding (The Agent)
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


def schema_alignment_node(state):
    """Analyse ontology + CSV to produce a Functional Entity Plan.

    Detects hierarchical patterns (is_part_of, parent-child, etc.)
    and determines whether multi-node mappings are needed.
    """
    print("  [Schema Alignment] Analysing ontology for entity structure...")
    result = call_schema_alignment_agent(state)
    multi = result.get("multi_node", False)
    tag = "MULTI-NODE" if multi else "FLAT"
    return {
        "schema_alignment": result,
        "alignment_changed": True,
        "messages": [f"Schema Alignment: [{tag}] Entity plan created."]
    }


def cq_validator_node(state):
    """Validate the current YARRRML against user-provided Competency Questions.

    If no CQs are provided, passes through without blocking.
    """
    cqs = state.get("competency_questions", [])
    if not cqs:
        return {
            "cq_validation": {"feedback": "CQ_SKIPPED", "cq_results": []},
            "feedback": "CQ_SKIPPED",
            "messages": ["CQ Validator: No competency questions provided — skipping."]
        }

    print("  [CQ Validator] Checking YARRRML against Competency Questions...")
    result = call_cq_validator_agent(state)
    feedback = result["feedback"]
    cq_results = result["cq_results"]

    n_passed = sum(1 for r in cq_results if r["passed"])
    n_total = len(cq_results)
    n_layer_a = sum(1 for r in cq_results if r.get("layer") == "A")
    n_layer_b = sum(1 for r in cq_results if r.get("layer") == "B")

    current_cq_retries = state.get("cq_retry_count", 0)

    layer_note = f" (LayerA:{n_layer_a} LayerB:{n_layer_b})"

    if "CQ_PASSED" in feedback:
        return {
            "cq_validation": result,
            "feedback": "CQ_PASSED",
            "persistent_cq_failures": result.get("persistent_cq_failures", []),
            "messages": [f"CQ Validator: [PASS] {n_passed}/{n_total} CQ(s) satisfied{layer_note}."]
        }
    else:
        return {
            "cq_validation": result,
            "feedback": feedback,
            "cq_retry_count": current_cq_retries + 1,
            "persistent_cq_failures": result.get("persistent_cq_failures", []),
            "messages": [f"CQ Validator: [FAIL] {n_total - n_passed}/{n_total} CQ(s) failed{layer_note} — routing back for fix."]
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
                # Skip invalid prefix lines where a URL is used as prefix name
                # e.g. "http://example.com/": "http://example.com/"
                if re.match(r'^\s+"?https?://', line):
                    continue  # drop this invalid line entirely

                # Strip angle brackets: `key: <URI>` → `key: "URI"`
                m = re.match(r'^(\s+\S+:\s*)<([^>]+)>\s*$', line)
                if m:
                    result.append(f'{m.group(1)}"{m.group(2)}"')
                    continue
            else:
                in_prefixes = False

        result.append(line)

    return '\n'.join(result)


def _dedup_po_entries(yarrrml_str: str) -> tuple[str, list[str]]:
    """Remove duplicate PO entries within each mapping.

    Keeps the LAST occurrence of each predicate so that the relationship
    agent's improved version overwrites the entity agent's initial entry.
    rdf:type / ``a`` entries are always kept as-is (duplicates allowed).

    Returns
    -------
    (fixed_yarrrml, list_of_fix_descriptions)
    """
    try:
        from ruamel.yaml import YAML as _YAML
        from io import StringIO
        _yaml = _YAML()
        _yaml.preserve_quotes = True
        _yaml.indent(mapping=2, sequence=4, offset=2)
        data = _yaml.load(yarrrml_str)
    except Exception:
        return yarrrml_str, []

    if not data or not isinstance(data.get("mappings"), dict):
        return yarrrml_str, []

    fixes: list[str] = []
    for mname, mdef in data["mappings"].items():
        if not isinstance(mdef, dict):
            continue
        po = mdef.get("po") or []

        # Count occurrences per predicate (skip rdf:type / a)
        pred_count: dict[str, int] = {}
        for entry in po:
            if isinstance(entry, list) and len(entry) >= 2:
                pred = str(entry[0])
                if pred not in ("a", "rdf:type"):
                    pred_count[pred] = pred_count.get(pred, 0) + 1

        dup_preds = {p for p, c in pred_count.items() if c > 1}
        if not dup_preds:
            continue

        # Reverse pass: keep only the LAST occurrence of each dup predicate
        new_po: list = []
        kept: set[str] = set()
        for entry in reversed(po):
            if isinstance(entry, list) and len(entry) >= 2:
                pred = str(entry[0])
                if pred in dup_preds:
                    if pred not in kept:
                        kept.add(pred)
                        new_po.insert(0, entry)
                    else:
                        fixes.append(f"Removed duplicate '{pred}' in {mname}")
                    continue
            new_po.insert(0, entry)

        mdef["po"] = new_po

    if not fixes:
        return yarrrml_str, []

    try:
        from io import StringIO
        buf = StringIO()
        _yaml.dump(data, buf)
        return buf.getvalue().strip(), fixes
    except Exception:
        return yarrrml_str, []


def yarrrml_coordinator_node(state):
    """Orchestrate YARRRML generation via three specialised sub-agents
    (PrefixAgent, EntityAgent, RelationshipAgent) and apply
    post-processing fixes.
    """
    current_retries = state.get("retry_count", 0)
    cq_retries = state.get("cq_retry_count", 0)
    max_syntax = 10
    max_logic = 6
    max_cq = 5
    attempt_num = current_retries + 1
    print(f"    [Progress] Attempt {attempt_num} | "
          f"syntax retries left: {max_syntax - current_retries} | "
          f"CQ retries left: {max_cq - cq_retries} | "
          f"logic retries left: {max_logic - current_retries}")

    # ── Delegate to the coordinator ───────────────────────────────
    coord_result = coordinate_yarrrml_generation(state)
    yarrrml = coord_result["yarrrml_output"]

    # ── Strip <> from prefix URIs (Turtle-style → plain YARRRML) ─
    yarrrml = _strip_prefix_angle_brackets(yarrrml)

    # ── Fix s: when LLM outputs it as a list instead of a string ─
    yarrrml = _fix_subject_lists(yarrrml)

    # ── Fix bare {col} brace templates → $(col) in subject and IRI values ─
    yarrrml, brace_fixed = _fix_bare_brace_subjects(yarrrml)
    if brace_fixed:
        print("    [Coordinator] Fixed bare {col} brace templates → $(col)")

    # ── Replace http://example.com/ subjects with real ontology prefix ──
    yarrrml, excom_fixed = _fix_example_com_subjects(yarrrml)
    if excom_fixed:
        print("    [Coordinator] Fixed http://example.com/ subjects → ontology prefix")

    # ── Targeted ~iri handling ──────────────────────────────
    # Only strip ~iri from LITERAL po entries (3-item with xsd: type).
    # Do NOT strip from join targets (MappingName~iri) or standalone
    # IRI refs ($(col)~iri in 2-item po entries) — Yatter needs those.
    yarrrml = re.sub(
        r'(\$\([^)]+\))~iri(\s*,\s*xsd:)',
        r'\1\2',
        yarrrml,
    )

    # Ensure URI-template values in po entries have ~iri suffix.
    # Matches: "prefix:path/$(col)" or "http://…/$(col)" at end of a
    # po list item, where ~iri is missing.
    yarrrml = re.sub(
        r'("(?:https?://|[a-zA-Z][a-zA-Z0-9]*:)[^"]*\$\([^)]+\)[^"]*")'
        r'(?!~iri)(\s*\])',
        r'\1~iri\2',
        yarrrml,
    )

    # ── Deduplicate PO entries (same predicate from entity + relationship agents) ─
    yarrrml, dedup_fixes = _dedup_po_entries(yarrrml)
    if dedup_fixes:
        for fix in dedup_fixes:
            print(f"    [Coordinator] Dedup: {fix}")

    # ── Normalise YARRRML structure ─────────────────────────────
    # Converts block-style po entries (dicts, nested lists) to flow-style
    # lists that Yatter can process.  This is the critical fix for the
    # "can only concatenate str (not 'dict') to str" error.
    yarrrml = _normalize_yarrrml_structure(yarrrml)

    # ── Sanitise bare $(col)~iri references ──────────────────────
    # Strips ~iri from bare column refs that aren't URI templates
    # (e.g. $(user_handle)~iri → $(user_handle), xsd:string)
    yarrrml = _sanitize_iri_values(yarrrml)

    # ── Save per-attempt debug files ─────────────────────────
    run_dir = state.get("run_dir", "data/output/debug")
    debug_dir = os.path.join(run_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    attempt_num = current_retries + 1
    with open(os.path.join(debug_dir, f"attempt_{attempt_num}.yaml"), "w") as f:
        f.write(yarrrml)

    # Also save as last_attempt for quick access
    global_debug = "data/output/debug"
    os.makedirs(global_debug, exist_ok=True)
    with open(os.path.join(global_debug, "last_attempt.yaml"), "w") as f:
        f.write(yarrrml)

    # ── Write diff log between this and previous attempt ──────────
    if attempt_num > 1:
        prev_path = os.path.join(debug_dir, f"attempt_{attempt_num - 1}.yaml")
        if os.path.exists(prev_path):
            try:
                from difflib import unified_diff
                with open(prev_path) as pf:
                    prev_lines = pf.read().splitlines()
                curr_lines = yarrrml.splitlines()
                diff = list(unified_diff(
                    prev_lines, curr_lines,
                    fromfile=f"attempt_{attempt_num - 1}.yaml",
                    tofile=f"attempt_{attempt_num}.yaml",
                    lineterm=""
                ))
                diff_path = os.path.join(debug_dir, f"diff_{attempt_num - 1}_to_{attempt_num}.txt")
                with open(diff_path, "w") as df:
                    df.write("\n".join(diff))
            except Exception:
                pass  # diff is best-effort, never block the pipeline

    return {
        "yarrrml_output": yarrrml,
        "prefixes_output": coord_result.get("prefixes_output", ""),
        "entity_yarrrml": coord_result.get("entity_yarrrml", ""),
        "alignment_changed": False,
        "retry_count": current_retries + 1,
        "feedback": "",
        "messages": [f"Coordinator: Generated attempt #{attempt_num} (prefix → entity → relationship)"]
    }


def _fix_example_com_subjects(yarrrml_str: str) -> tuple[str, bool]:
    """Replace ``http://example.com/...`` subject templates with the primary
    ontology prefix declared in the ``prefixes:`` block.

    When the LLM ignores the ontology prefix and falls back to
    ``http://example.com/buildings/$(col)`` we look for a non-standard
    prefix declared in the prefixes block (not rdf/rdfs/owl/xsd/schema/foaf
    etc.) and use that instead.  This is a last-resort normalization step
    applied after all other fixes.

    Only replaces subjects — not po-entry object values.
    """
    # Only act when the mapping contains example.com subjects
    if "http://example.com/" not in yarrrml_str:
        return yarrrml_str, False

    # Extract declared prefixes
    skip_prefixes = {
        "rdf", "rdfs", "owl", "xsd", "schema", "schema1", "foaf",
        "dc", "dcterms", "skos", "prov", "void", "geo", "sosa", "ssn",
    }
    declared: dict[str, str] = {}
    for m in re.finditer(r'^\s{2}(\w+):\s+"?(https?://[^"\s]+)"?', yarrrml_str, re.MULTILINE):
        name, uri = m.group(1), m.group(2)
        if name.lower() not in skip_prefixes:
            declared[name] = uri.rstrip("#/")

    if not declared:
        return yarrrml_str, False

    # Pick the domain-specific prefix (shortest URI that is not example.com)
    best = None
    for name, uri in declared.items():
        if "example.com" not in uri and "example.org" not in uri:
            if best is None or len(uri) < len(declared[best]):
                best = name

    if not best:
        return yarrrml_str, False

    ontology_base = declared[best]
    prefix_short = f"{best}:"

    # Replace s: http://example.com/ClassName/... → s: best:ClassName/...
    # Pattern: captures the class-path after example.com/
    lines = yarrrml_str.split("\n")
    result = []
    changed = False
    for line in lines:
        stripped = line.lstrip()
        if re.match(r's:\s+http://example\.com/', stripped):
            # Extract path after example.com/
            path = re.sub(r's:\s+http://example\.com/', '', stripped)
            indent = line[: len(line) - len(line.lstrip())]
            new_line = f"{indent}s: {prefix_short}{path}"
            result.append(new_line)
            changed = True
            continue
        result.append(line)

    return "\n".join(result), changed


def _fix_bare_brace_subjects(yarrrml_str: str) -> tuple[str, bool]:
    """Convert bare ``{col}`` templates to YARRRML ``$(col)`` in subject lines
    and IRI object values.

    The LLM sometimes emits:
      s: http://example.com/buildings/{Surface_Area}_{Overall_Height}
    instead of:
      s: http://example.com/buildings/$(Surface_Area)_$(Overall_Height)

    Applies ONLY to ``s:`` lines and IRI-looking values in po entries (those
    starting with ``http://`` or a ``prefix:``).  Literal string values with
    ``{col}`` are left unchanged because they are concatenation templates that
    the refiner handles separately.
    """
    lines = yarrrml_str.split("\n")
    changed = False

    def _replace_braces(s: str) -> str:
        """Replace {word} with $(word) — only simple single-word identifiers."""
        return re.sub(r'\{([A-Za-z][A-Za-z0-9_]*)\}', r'$(\1)', s)

    result = []
    for line in lines:
        stripped = line.lstrip()
        # Subject lines
        if re.match(r's:\s+', stripped):
            fixed = _replace_braces(line)
            if fixed != line:
                changed = True
                line = fixed
        # IRI-valued po entries: - [pred, http://...{col}...~iri]
        elif stripped.startswith("- [") and "{" in line:
            # Only fix if the value looks like a URI (contains ://)
            if "://" in line or re.search(r'\[\s*\w+:\w', line):
                fixed = _replace_braces(line)
                if fixed != line:
                    changed = True
                    line = fixed
        result.append(line)

    return "\n".join(result), changed


def _auto_repair_syntax(yarrrml_str: str, error_msg: str) -> tuple[str, bool]:
    """Attempt deterministic YAML repair before triggering full regeneration.

    Tries (in order):
      1. Fix ``{$(col)}`` double-wrap → ``$(col)``
      2. Remove obviously broken lines (bare URIs used as prefix names)
      3. Deduplicate predicate entries in the same mapping block

    Returns (fixed_yarrrml, was_repaired).
    """
    original = yarrrml_str
    changed = False

    # Fix 1: {$(col)} double-wrap
    fixed = re.sub(r'\{\$\(([^)]+)\)\}', r'$(\1)', yarrrml_str)
    if fixed != yarrrml_str:
        yarrrml_str = fixed
        changed = True

    # Fix 2: bare http:// lines used as YAML keys (invalid prefix names)
    lines = yarrrml_str.split("\n")
    clean_lines = []
    for line in lines:
        if re.match(r'^\s+"?https?://', line):
            changed = True
            continue  # drop the line
        clean_lines.append(line)
    yarrrml_str = "\n".join(clean_lines)

    # Fix 3: if a specific line number is given, try to fix that line
    m = re.search(r'line (\d+)', error_msg)
    if m:
        line_no = int(m.group(1)) - 1
        lines = yarrrml_str.split("\n")
        if 0 <= line_no < len(lines):
            bad = lines[line_no]
            # Fix unbalanced brackets in flow sequences
            open_b = bad.count("[")
            close_b = bad.count("]")
            if open_b > close_b:
                lines[line_no] = bad + "]" * (open_b - close_b)
                changed = True
            elif close_b > open_b:
                lines[line_no] = "[" * (close_b - open_b) + bad
                changed = True
            yarrrml_str = "\n".join(lines)

    return yarrrml_str, changed


def validation_node(state: AgentState):
    yarrrml_content = state["yarrrml_output"]
    # Normalisation already done in yarrrml_coordinator_node — skip here
    yaml = YAML(typ='safe', pure=True)

    try:
        import logging
        import io

        # Capture Yatter's log output to detect silent failures.
        # Yatter may use the root logger or a module-level logger,
        # so we attach to the root logger with an ERROR-level filter.
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.ERROR)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        try:
            # Load the string as YAML and attempt Yatter translation
            yarrrml_data = yaml.load(yarrrml_content)
            rml_content = yatter.translate(yarrrml_data)
        finally:
            root_logger.removeHandler(handler)

        # Check for silent failures: Yatter may log errors without raising
        captured_errors = log_capture.getvalue().strip()

        # Also verify translation produced valid RML (must contain @prefix)
        rml_str = str(rml_content) if rml_content else ""
        has_rml_output = rml_str.strip() and "@prefix" in rml_str

        if not has_rml_output or captured_errors:
            error_detail = captured_errors or "Yatter produced empty/invalid RML output"
            # ── Try deterministic repair before full regeneration ──
            repaired, was_repaired = _auto_repair_syntax(yarrrml_content, error_detail)
            if was_repaired and repaired != yarrrml_content:
                # Re-attempt translation with the repaired YARRRML
                try:
                    repaired_data = yaml.load(repaired)
                    rml2 = yatter.translate(repaired_data)
                    rml2_str = str(rml2) if rml2 else ""
                    if rml2_str.strip() and "@prefix" in rml2_str:
                        print("    [Validator] Auto-repaired YARRRML — translation succeeded.")
                        return {
                            "yarrrml_output": repaired,
                            "messages": ["Validator: Syntax is valid (after auto-repair)."],
                            "feedback": "PASSED_SYNTAX",
                        }
                except Exception:
                    pass  # repair didn't help — fall through to normal error
            return {
                "messages": [f"Validator: Translation failed: {error_detail[:150]}..."],
                "feedback": f"SYNTAX_ERROR: YARRRML-to-RML translation failed: {error_detail}. "
                            f"Check join syntax — use 'TargetMapping~iri' (with ~iri suffix) "
                            f"for join targets. Ensure all po: entries are valid YAML lists.",
                "retry_needed": True,
            }

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
    injected_constraints = refiner_result.get("injected_column_constraints", {})

    # Base result dict — all branches share these keys
    result: dict = {
        "predicate_conflict_cols": conflict_cols,
        "injected_column_constraints": injected_constraints,
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
    # Normalisation already done in yarrrml_coordinator_node — skip here
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










