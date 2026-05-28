from tools.rml_tools import get_csv_schema, get_ontology_subgraph
from agents.schema_agent import call_schema_llm
from agents.mapper_agent import call_mapper_llm
from agents.schema_alignment_agent import call_schema_alignment_agent
from agents.cq_to_sparql_agent import batch_cq_to_sparql, extract_yarrrml_prefixes, extract_yarrrml_mapping_context, probe_kg_types_and_predicates, build_kg_grounding_block
from agents.cq_generator_agent import generate_cqs
from agents.yarrrml_coordinator import coordinate_yarrrml_generation
from data.checkpoints import AgentState
from agents.refiner_agent import call_refiner_llm, build_column_alias_map, restore_column_names
from config.prefixes import WELL_KNOWN_PREFIXES
from datetime import datetime

import yaml as pyyaml   # PyYAML for normalisation
import yatter
from ruamel.yaml import YAML
import os
import re

import morph_kgc


# ────────────────────────────────────────────────────────────────────
# RML prefix safety-net
# ────────────────────────────────────────────────────────────────────

# URI schemes that must not be treated as prefixes when scanning RML
_RML_URI_SCHEMES = {"http", "https", "ftp", "urn", "mailto", "file"}

# Tokens used as keywords in Turtle/RML — never real prefixes
_TURTLE_KEYWORDS = {
    "a", "true", "false", "BASE", "PREFIX",
    "rr", "rml", "ql", "fnml", "fno",   # always emitted by yatter itself
}

# Well-known prefix → URI for RML-level auto-resolution.
# Imported from config/prefixes.py (single source of truth).
# NOTE: Do NOT add dataset-specific prefixes here (e.g. lkg, podio).
#       Those are declared in the ontology and auto-detected from YARRRML.
_RML_WELL_KNOWN_PREFIXES: dict[str, str] = dict(WELL_KNOWN_PREFIXES)

_RML_PREFIX_USAGE_RE = re.compile(r'\b([a-zA-Z][a-zA-Z0-9_]*):[a-zA-Z_]')


# ────────────────────────────────────────────────────────────────────
# Post-materialisation KG cleanup helpers
# ────────────────────────────────────────────────────────────────────

def _clean_invalid_rdf_type_literals(nt_path: str) -> int:
    """Remove triples where rdf:type has a literal object from an N-Triples file.

    By definition, rdf:type must always have an IRI as its object (RDF spec).
    When morph-kgc produces ``<s> rdf:type "N"^^xsd:integer`` triples they
    are structurally invalid and are silently removed here.

    Returns the number of lines removed.
    """
    RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
    if not nt_path or not os.path.exists(nt_path):
        return 0
    try:
        with open(nt_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        kept: list[str] = []
        removed = 0
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                kept.append(line)
                continue
            parts = stripped.split(None, 2)
            if len(parts) >= 3 and parts[1] == RDF_TYPE:
                obj = parts[2].strip().rstrip(" .")
                if obj.startswith('"') or obj.startswith("'"):
                    removed += 1
                    continue
            kept.append(line)
        if removed:
            with open(nt_path, "w", encoding="utf-8") as fh:
                fh.writelines(kept)
            print(f"    [KG Cleanup] Removed {removed} invalid rdf:type literal triple(s)")
        return removed
    except Exception as exc:
        print(f"    [KG Cleanup] Warning: {exc}")
        return 0


def _fix_iri_template_for_objectproperty(
    yarrrml_str: str,
    violations: list[str],
    obj_props: set[str],
    base_prefix: str,
    base_uri: str,
) -> tuple[str, list[str]]:
    """Deterministic SHACL-triggered fix for ObjectProperty violations.

    Two-pass strategy (both are dataset-agnostic):

    **Pass A — prefix rewrite**: When a po entry already uses ``~iri`` but with
    the wrong prefix (e.g. ``dbo:Profession/$(col)~iri``), rewrite it to use the
    subject base prefix (e.g. ``example:Profession/$(col)~iri``).  Morph-KGC
    reliably materialises IRI templates under the base namespace; foreign
    ontology namespace templates can silently produce Literals.

    **Pass B — literal→IRI conversion**: When a po entry maps an
    owl:ObjectProperty to a *literal* (e.g. ``[dbo:profession, $(category),
    xsd:string]``), replace it with an IRI template using the column value as a
    local name (e.g. ``[dbo:profession, example:Profession/$(category)~iri]``).
    The class name is derived from the property's local name (capitalised).

    Returns (fixed_yarrrml, list_of_fix_descriptions).
    """
    if not yarrrml_str or not violations:
        return yarrrml_str, []

    # Collect predicates flagged as ObjectProperty violations
    failing_preds: set[str] = set()
    for v in violations:
        m = re.search(r'path=(<[^>]+>|[\w:]+)', v)
        if not m:
            continue
        pred_raw = m.group(1).strip("<>")
        if not obj_props or pred_raw in obj_props:
            failing_preds.add(pred_raw)

    if not failing_preds:
        return yarrrml_str, []

    try:
        import yaml as _yaml
        data = _yaml.safe_load(yarrrml_str) or {}
    except Exception:
        data = {}
    declared_prefixes: dict[str, str] = data.get("prefixes", {}) or {}
    uri_to_prefix: dict[str, str] = {v: k for k, v in declared_prefixes.items()}
    needs_prefix_injection = base_prefix not in declared_prefixes and bool(base_uri)

    fixes: list[str] = []
    result = yarrrml_str

    for pred_uri in failing_preds:
        pred_forms: list[str] = [pred_uri]
        for pfx_uri, pfx_alias in uri_to_prefix.items():
            if pred_uri.startswith(pfx_uri):
                pred_forms.append(f"{pfx_alias}:{pred_uri[len(pfx_uri):]}")

        # ── Pass A: rewrite wrong-prefix ~iri templates ───────────────────
        for pred_form in pred_forms:
            pat = re.compile(
                r'(\[\s*' + re.escape(pred_form) + r'\s*,\s*)'
                r'([a-zA-Z][a-zA-Z0-9_]*)'
                r'(:(?:[^,\]]*)\$\([^)]+\)~iri)'
                r'(\s*\])',
            )
            _fixes_local: list[str] = []

            def _rewrite(m: re.Match, bp: str = base_prefix,
                         fl: list = _fixes_local) -> str:
                if m.group(2) == bp:
                    return m.group(0)
                fl.append(
                    f"IRI template prefix '{m.group(2)}:' → '{bp}:' "
                    f"for predicate '{pred_form}'"
                )
                return f"{m.group(1)}{bp}{m.group(3)}{m.group(4)}"

            new_result = pat.sub(_rewrite, result)
            if _fixes_local:
                result = new_result
                fixes.extend(_fixes_local)
                break

        # ── Pass B: convert literal mapping → IRI template ───────────────
        # Matches: [pred_form, $(col), xsd:anyType]  (3-item literal entry)
        # Rewrites to: [pred_form, base_prefix:LocalClass/$(col)~iri]
        for pred_form in pred_forms:
            # Derive class name from property local name (e.g. profession → Profession)
            local_name = pred_uri.split("/")[-1].split("#")[-1]
            class_name = local_name[0].upper() + local_name[1:] if local_name else "Resource"

            lit_pat = re.compile(
                r'(\[\s*' + re.escape(pred_form) + r'\s*,\s*)'
                r'\$\(([^)]+)\)'          # group 2: column name
                r'\s*,\s*xsd:[a-zA-Z]+'  # xsd datatype — confirm it's a literal
                r'(\s*\])',
            )
            _lit_fixes: list[str] = []

            def _lit_rewrite(m: re.Match, bp: str = base_prefix,
                              cn: str = class_name, pf: str = pred_form,
                              fl: list = _lit_fixes) -> str:
                col = m.group(2)
                fl.append(
                    f"Converted literal → IRI template for ObjectProperty '{pf}': "
                    f"$({{col}}) xsd:string → {bp}:{cn}/$({{col}})~iri"
                    .format(col=col)
                )
                return f"[{pf}, {bp}:{cn}/$({col})~iri]"

            new_result = lit_pat.sub(_lit_rewrite, result)
            if _lit_fixes:
                result = new_result
                fixes.extend(_lit_fixes)
                needs_prefix_injection = (
                    needs_prefix_injection
                    or base_prefix not in declared_prefixes
                )
                break  # only apply first matching form

    if fixes and needs_prefix_injection and base_prefix not in declared_prefixes:
        result = result.replace(
            "prefixes:",
            f"prefixes:\n  {base_prefix}: \"{base_uri}\"",
            1,
        )
        fixes.append(f"Injected missing prefix '{base_prefix}: {base_uri}'")

    return result, fixes


def _inject_missing_rml_prefixes(yarrrml_data: dict, rml_content: str,
                                  yarrrml_text: str = "") -> str:
    """Guarantee that every prefix used in the RML/Turtle output has a
    corresponding ``@prefix`` declaration before morph_kgc parses it.

    Two-pass strategy:
      1. Collect prefixes *declared* in the original YARRRML ``prefixes:`` block
         (yatter may have silently dropped some).  Also scan the raw YARRRML
         text for declared prefixes (catches any that were text-injected and
         not yet reflected in the parsed ``yarrrml_data`` dict).
      2. Scan the *RML Turtle text* for ``prefix:localname`` patterns that have
         no ``@prefix`` declaration.

    Resolution order for unknown prefixes:
      a. YARRRML ``prefixes:`` block  →  use declared URI
      b. ``_RML_WELL_KNOWN_PREFIXES`` table  →  use canonical URI
      c. Fallback  →  synthesize ``http://example.org/{prefix}/``

    This makes the fix fully dataset-agnostic and permanent.
    """
    if not rml_content:
        return rml_content

    # --- Pass 1a: declared in YARRRML data dict ---
    yarrrml_prefixes: dict[str, str] = {}
    if yarrrml_data:
        yarrrml_prefixes = {k: v for k, v in (yarrrml_data.get("prefixes") or {}).items()}

    # --- Pass 1b: declared in YARRRML text (more robust — catches text-injected entries) ---
    if yarrrml_text:
        for m in re.finditer(r'^\s{2}(\w+):\s+"?(https?://[^"\s]+)"?', yarrrml_text, re.MULTILINE):
            name, uri = m.group(1), m.group(2)
            if name not in yarrrml_prefixes:
                yarrrml_prefixes[name] = uri

    # --- Pass 2: used in RML Turtle but never declared at all ---
    existing_declared = set(re.findall(r'@prefix\s+(\w+)\s*:', rml_content))

    used_in_rml: set[str] = set()
    for m in _RML_PREFIX_USAGE_RE.finditer(rml_content):
        p = m.group(1)
        if p.lower() not in _RML_URI_SCHEMES and p not in _TURTLE_KEYWORDS:
            used_in_rml.add(p)

    # Combine: anything declared in YARRRML but absent from RML, plus
    # anything used in RML but missing a @prefix declaration
    need_injection = (set(yarrrml_prefixes.keys()) | used_in_rml) - existing_declared

    if not need_injection:
        return rml_content

    # Build a resolution map: YARRRML declarations > well-known > synthetic
    resolution: dict[str, str] = dict(_RML_WELL_KNOWN_PREFIXES)
    resolution.update(yarrrml_prefixes)   # YARRRML declarations win

    missing_lines: list[str] = []
    for prefix in sorted(need_injection):
        uri = resolution.get(prefix)
        if not uri:
            uri = f"http://example.org/{prefix}/"
            print(f"    [RML-FIX] Unknown prefix '{prefix}' — synthesized URI <{uri}>")
        else:
            if not uri.endswith(("/", "#")):
                uri = uri + "/"
        missing_lines.append(f"@prefix {prefix}: <{uri}> .")
        print(f"    [RML-FIX] Injected @prefix {prefix}: <{uri}>")

    injection = "\n".join(missing_lines) + "\n"
    first_prefix = rml_content.find("@prefix")
    if first_prefix >= 0:
        return rml_content[:first_prefix] + injection + rml_content[first_prefix:]
    return injection + rml_content


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


def generate_cqs_node(state):
    """Auto-generate Competency Questions when the user has not provided any.

    If the user already provided CQs (via --cqs) this node is a no-op —
    their CQs are used as-is.  Auto-generated CQs are stored in
    ``generated_cqs`` (separate from ``competency_questions``) so the
    pipeline always knows which came from the user.

    Either way, the active CQ list is saved to <run_dir>/cqs.txt so it
    is always clear which questions were used in this run.
    """
    run_dir = state.get("run_dir", "data/output/debug")
    user_cqs = state.get("competency_questions", [])

    if user_cqs:
        _log_section("CQ Generator")
        print(f"  Using {len(user_cqs)} user-provided CQ(s) — skipping auto-generation.")
        for i, cq in enumerate(user_cqs, 1):
            print(f"    [{i}] {cq}")
        _save_cqs_to_file(user_cqs, run_dir, source="user-provided")
        return {
            "generated_cqs": [],
            "messages": [f"CQ Generator: Using {len(user_cqs)} user-provided CQ(s)."],
        }

    _log_section("CQ Generator")
    print("  No CQs provided — auto-generating from schema + ontology ...")

    # Pass the entity plan (built by align_schema before this node) so
    # the CQ generator is grounded to actual entity types in the KG.
    entity_plan = state.get("schema_alignment", {}).get("entity_plan", "")

    cqs = generate_cqs(
        schema_info=state.get("schema_info", {}),
        ontology_info=state.get("ontology_info", {}),
        base_uri=state.get("base_uri", "http://example.org/"),
        entity_plan=entity_plan if entity_plan else None,
    )

    if cqs:
        print(f"  [OK] Generated {len(cqs)} CQ(s):")
        for i, cq in enumerate(cqs, 1):
            print(f"    [{i}] {cq}")
    else:
        print("  [WARNING] No CQs could be generated -- SPARQL validation will be skipped.")

    _save_cqs_to_file(cqs, run_dir, source="auto-generated")

    return {
        "generated_cqs": cqs,
        "messages": [f"CQ Generator: Auto-generated {len(cqs)} CQ(s) from schema + ontology."],
    }


def _save_cqs_to_file(cqs: list, run_dir: str, source: str = "generated") -> None:
    """Write CQs to <run_dir>/cqs.txt for audit/debug purposes."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "cqs.txt")
    with open(path, "w") as f:
        f.write(f"# Competency Questions ({source})\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total: {len(cqs)}\n\n")
        for i, cq in enumerate(cqs, 1):
            f.write(f"{i}. {cq}\n")
    print(f"  [saved] CQs --> {path}")


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
        # EXCEPT: when choosing between a 2-item (IRI) and 3-item (literal)
        # entry for the same predicate, ALWAYS keep the 2-item IRI entry.
        # This fixes Bug 3 (atMerchant/usesCard kept as literal not IRI).
        def _is_iri_entry(e: list) -> bool:
            """Return True if this is a 2-item IRI link (not a literal)."""
            if len(e) == 2:
                val = str(e[1])
                return '~iri' in val or val.startswith('http')
            return False

        # Group entries by predicate to pick the best one
        pred_entries: dict[str, list[list]] = {}
        for entry in po:
            if isinstance(entry, list) and len(entry) >= 2:
                pred = str(entry[0])
                if pred in dup_preds:
                    pred_entries.setdefault(pred, []).append(entry)

        # Choose winner for each dup predicate
        pred_winner: dict[str, list] = {}
        for pred, entries in pred_entries.items():
            iri_entries = [e for e in entries if _is_iri_entry(e)]
            pred_winner[pred] = iri_entries[0] if iri_entries else entries[-1]

        new_po: list = []
        kept: set[str] = set()
        for entry in po:
            if isinstance(entry, list) and len(entry) >= 2:
                pred = str(entry[0])
                if pred in dup_preds:
                    if pred not in kept:
                        new_po.append(pred_winner[pred])
                        kept.add(pred)
                    else:
                        fixes.append(f"Removed duplicate '{pred}' in {mname}")
                    continue
            new_po.append(entry)

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

    # ── Final prefix reconciliation (safety net) ─────────────────
    # _normalize_yarrrml_structure and _dedup_po_entries both perform YAML
    # load+dump cycles that can silently drop auto-injected prefixes
    # (e.g. lkg:, eli:).  Re-running reconciliation here guarantees every
    # prefix:localName in the final YARRRML is declared, regardless of what
    # any earlier transform did to the prefix block.
    from agents.yarrrml_coordinator import _reconcile_prefixes as _final_reconcile, _apply_base_uri_to_subjects as _apply_base_uri
    _ont_raw = state.get("ontology_info", {}).get("raw", "")
    yarrrml = _final_reconcile(yarrrml, _ont_raw)

    # ── Re-apply base URI as the VERY LAST step ───────────────────
    # _normalize_yarrrml_structure and _final_reconcile above both do YAML
    # round-trips that can overwrite the prefix URI set by the coordinator's
    # step 9 (e.g. putting back mykg: http://example.org/mykg# instead of
    # http://mykg.org/resource/).  Re-applying here ensures the user's
    # BASE_URI is always respected in the final output.
    _user_base_uri = state.get("base_uri", "http://example.org/")
    yarrrml, _base_uri_node_changes = _apply_base_uri(yarrrml, _user_base_uri)
    for _chg in _base_uri_node_changes:
        print(f"    [Coordinator/node] {_chg}")

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

            # ── Pre-flight: auto-fix missing prefixes in YARRRML before
            #    passing to yatter.  This prevents yatter from silently
            #    producing Turtle with undeclared prefixes.
            from agents.refiner_agent import _auto_fix_missing_prefixes as _fix_pfx
            _fixed_content, _pfx_fixes = _fix_pfx(yarrrml_content, yarrrml_data)
            if _pfx_fixes:
                for _f in _pfx_fixes:
                    print(f"    [Validator-FIX] {_f}")
                yarrrml_content = _fixed_content
                yarrrml_data = yaml.load(yarrrml_content)

            rml_content = yatter.translate(yarrrml_data)

            # ── Post-flight: inject any prefixes yatter still dropped ──
            rml_content = _inject_missing_rml_prefixes(yarrrml_data, rml_content,
                                                        yarrrml_text=yarrrml_content)
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

        # If successful, move to logic check.
        # Always propagate the (possibly prefix-fixed) YARRRML so all
        # downstream nodes use the corrected version.
        return {
            "yarrrml_output": yarrrml_content,
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
    yaml = YAML(typ='safe', pure=True)
    yarrrml_data = yaml.load(yarrrml_content)

    # ── Pre-flight: ensure all prefixes used in the YARRRML are declared
    #    before handing to yatter.  Island-wiring and column auto-injection
    #    in the refiner can introduce new ex: usages AFTER the refiner's
    #    early prefix check.  Yatter silently returns None when it encounters
    #    an undeclared prefix, which then crashes the CSV path-patching below.
    #
    #    IMPORTANT: update BOTH yarrrml_content AND yarrrml_data so yatter
    #    receives the corrected version.
    from agents.refiner_agent import _auto_fix_missing_prefixes as _preflight_fix
    _pf_fixed, _pf_fixes = _preflight_fix(yarrrml_content, yarrrml_data)
    if _pf_fixes:
        for _f in _pf_fixes:
            print(f"    [RML-PREFLIGHT] {_f}")
        yarrrml_content = _pf_fixed          # ← update text too
        yarrrml_data = yaml.load(_pf_fixed)

    # Translate YARRRML to RML (Turtle syntax)
    rml_content = yatter.translate(yarrrml_data)

    # Guard: yatter returns None when it encounters an undeclared prefix.
    if rml_content is None:
        raise ValueError(
            "Yatter returned None — the YARRRML still contains an undeclared prefix. "
            "Check that every prefix:localname in the mappings has a matching entry "
            "in the prefixes: block."
        )
    rml_content = str(rml_content)

    # Safety net: inject any @prefix declarations that yatter still dropped.
    # Pass the YARRRML text as well so text-injected prefixes are caught.
    rml_content = _inject_missing_rml_prefixes(yarrrml_data, rml_content,
                                                yarrrml_text=yarrrml_content)

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

    # ── Post-processing: fix any predicate map with rr:termType rr:Literal ──
    # RDF predicates can ONLY be IRIs — rr:Literal as a predicate termType is
    # invalid and causes morph-kgc to abort with "Found an invalid predicate
    # termtype".  This happens when a YARRRML po entry has a datatype annotation
    # on the predicate position (e.g. a malformed 3-element entry).
    patched_rml = _fix_predicate_termtype(patched_rml)

    return patched_rml


def _fix_predicate_termtype(rml_content: str) -> str:
    """Remove rr:termType rr:Literal (or any non-IRI termType) from predicate maps.

    Predicates in RML must always be IRIs.  Yatter can occasionally emit
    ``rr:termType rr:Literal`` inside a ``rr:predicateMap [ ... ]`` block when
    a po entry is malformed (e.g. the predicate slot has a datatype annotation).

    Returns the cleaned RML string.
    """
    if "rr:termType" not in rml_content:
        return rml_content

    lines = rml_content.split("\n")
    result: list[str] = []
    in_predicate_map = False
    bracket_depth = 0

    for line in lines:
        # Detect start of rr:predicateMap [ ...
        if re.search(r'rr:predicateMap\s*\[', line):
            in_predicate_map = True
            bracket_depth = line.count("[") - line.count("]")
            if bracket_depth <= 0:
                in_predicate_map = False  # single-line block
            result.append(line)
            continue

        if in_predicate_map:
            bracket_depth += line.count("[") - line.count("]")
            if bracket_depth <= 0:
                in_predicate_map = False
            # Inside a predicateMap block — drop any non-IRI termType line
            if re.search(r'rr:termType\s+(?!rr:IRI)', line):
                print(f"    [RML-FIX] Removed invalid predicate termType: {line.strip()}")
                continue  # skip this line

        result.append(line)

    return "\n".join(result)


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

        # ── Post-materialisation cleanup ──────────────────────────
        # Remove structurally-invalid rdf:type triples whose object is a
        # literal (e.g. integers from an ordering column that morph-kgc
        # incorrectly materialises as rdf:type values).
        _clean_invalid_rdf_type_literals(output_path)

        return {
            "rdf_output": output_path,
            "messages": [f"KG Generation: Success! Created in {run_dir}"]
        }

    except Exception as e:
        error_msg = f"KG Generation Error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {"messages": [error_msg]}


# ────────────────────────────────────────────────────────────────────
# SPARQL-based CQ Validator Node (post-KG generation)
# ────────────────────────────────────────────────────────────────────

def _extract_triple_patterns(sparql: str) -> str:
    """Extract the triple patterns inside the ASK { ... } block for the diagnosis."""
    upper = sparql.upper()
    ask_pos = upper.find("ASK")
    if ask_pos == -1:
        return sparql.strip()
    start = sparql.find("{", ask_pos)
    if start == -1:
        return sparql.strip()
    depth = 0
    end = start
    for i, ch in enumerate(sparql[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    return sparql[start + 1:end].strip()


def _select_probe_diagnosis(store, ask_sparql: str) -> str:
    """When an ASK query returns false, run targeted SELECT probes to diagnose why."""
    import re as _re
    try:
        type_pat = _re.search(
            r'\?(\w+)\s+a\s+(<[^>]+>|[\w]+:[\w]+)',
            ask_sparql,
        )
        if not type_pat:
            return ""
        type_val = type_pat.group(2)
        prefix_lines = "\n".join(
            l for l in ask_sparql.splitlines()
            if l.strip().upper().startswith("PREFIX")
        )
        probe = f"{prefix_lines}\nSELECT (COUNT(?s) AS ?cnt) WHERE {{ ?s a {type_val} }} LIMIT 1"
        rows = list(store.query(probe))
        count = int(str(rows[0][0])) if rows else 0
        if count == 0:
            return (
                f"Type {type_val} does not exist in KG — the mapping may not create "
                f"resources of this type. Check entity agent output."
            )
        else:
            return (
                f"Type {type_val} exists ({count} instance(s)) but the full triple "
                f"pattern did not match — a predicate or value may differ from the KG."
            )
    except Exception:
        return ""


def _log_section(title: str, width: int = 56) -> None:
    """Print a clearly visible section header for a pipeline stage."""
    bar = "-" * width
    print(f"\n  +{bar}+")
    print(f"  |  {title:<{width - 2}}|")
    print(f"  +{bar}+")


def _ask_user_continue(n_passed: int, n_total: int, timeout: int = 5) -> bool:
    """Prompt the user whether to continue optimizing when pass rate hits threshold.

    Waits ``timeout`` seconds for input on stdin.  Returns:
      - True  → user typed 'y' / 'yes'  → pipeline keeps retrying
      - False → any other input, or no input within timeout
                → pipeline accepts the current KG as-is

    Uses ``select`` for non-blocking stdin (Linux / macOS).
    Falls back silently to False on Windows or redirected stdin.
    """
    import sys
    import select as _select

    pct = int(n_passed / n_total * 100) if n_total else 0
    bar = "─" * 57

    print(f"\n  +{bar}+")
    print(f"  │  ⚡ {n_passed}/{n_total} ({pct}%) competency questions answered        │")
    print(f"  │                                                         │")
    print(f"  │  The KG partially satisfies your questions.             │")
    print(f"  │  Continue optimizing to reach 100%?                     │")
    print(f"  │  [y = yes  /  Enter or anything else = accept & stop]   │")
    print(f"  │  Auto-accepts in {timeout}s if no input.                      │")
    print(f"  +{bar}+")
    print(f"  > ", end="", flush=True)

    try:
        rlist, _, _ = _select.select([sys.stdin], [], [], timeout)
        if rlist:
            answer = sys.stdin.readline().strip().lower()
            if answer in ("y", "yes"):
                print(f"  → Continuing optimization "
                      f"({n_total - n_passed} question(s) still failing)...")
                return True
            else:
                print(f"  → Accepted. Keeping current KG ({n_passed}/{n_total} passed).")
                return False
        else:
            print(f"\n  → No response in {timeout}s — "
                  f"accepting current KG ({n_passed}/{n_total} passed).")
            return False
    except Exception:
        # Windows or non-interactive stdin — accept silently
        print(f"\n  → Non-interactive mode — "
              f"accepting current KG ({n_passed}/{n_total} passed).")
        return False


def _save_sparql_report(results: list, run_dir: str, retry: int = 0) -> None:
    """Write the full SPARQL validation report (CQs + queries + results) to file."""
    import json as _json
    os.makedirs(run_dir, exist_ok=True)
    suffix = f"_retry{retry}" if retry > 0 else ""
    txt_path = os.path.join(run_dir, f"sparql_validation{suffix}.txt")
    json_path = os.path.join(run_dir, f"sparql_validation{suffix}.json")

    with open(txt_path, "w") as f:
        f.write(f"# SPARQL CQ Validation Report\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Retry: {retry}\n")
        f.write(f"# Total checks: {len(results)}\n\n")
        for i, r in enumerate(results, 1):
            status = "PASS" if r["passed"] is True else ("FAIL" if r["passed"] is False else "SKIP")
            src = r.get("source", "unknown")
            f.write(f"[{i}] {status}  [{src}]\n")
            f.write(f"  CQ      : {r['cq']}\n")
            f.write(f"  SPARQL  :\n")
            for line in r["sparql"].strip().splitlines():
                f.write(f"    {line}\n")
            if r.get("diagnosis"):
                f.write(f"  Diagnosis: {r['diagnosis']}\n")
            f.write("\n")

    with open(json_path, "w") as f:
        _json.dump(results, f, indent=2)

    print(f"  [saved] SPARQL validation report --> {txt_path}")


def sparql_cq_validator_node(state: AgentState):
    """Validate Competency Questions via SPARQL execution on the materialized KG.

    Handles three input modes (all optional, combinable):
      1. User-provided CQs (``--cqs``)    → translated to ASK SPARQL by LLM
      2. Auto-generated CQs               → translated to ASK SPARQL by LLM
      3. User-provided SPARQL queries     → executed directly (no LLM translation)

    Steps
    -----
    1. Collect all CQs (user + generated) and direct SPARQL queries.
    2. Skip if none of the above are present.
    3. Load the generated KG into an in-memory pyoxigraph store.
    4. For direct SPARQL: execute as-is.
    5. For CQs: use CQ→SPARQL LLM agent with self-correction, then execute.
    6. Build structured feedback with exact failing triple patterns for refiner.
    """
    import pyoxigraph

    _log_section("SPARQL CQ Validator")

    user_cqs = state.get("competency_questions", [])
    generated_cqs = state.get("generated_cqs", [])
    user_sparql = state.get("user_sparql_queries", [])  # direct SPARQL from user

    # Combine all CQs (user takes priority, then generated)
    all_cqs = list(user_cqs) + [cq for cq in generated_cqs if cq not in user_cqs]

    if not all_cqs and not user_sparql:
        return {
            "sparql_validation_results": [],
            "feedback": "CQ_SPARQL_PASSED",
            "messages": ["SPARQL CQ Validator: No CQs or SPARQL queries provided — skipping."],
        }

    kg_path = state.get("rdf_output", "")
    if not kg_path or not os.path.exists(kg_path):
        return {
            "sparql_validation_results": [],
            "feedback": "CQ_SPARQL_PASSED",
            "messages": ["SPARQL CQ Validator: KG not available — skipping CQ validation."],
        }

    cq_sparql_retry_count = state.get("cq_sparql_retry_count", 0)
    max_cq_sparql_retries = int(os.environ.get("CQ_SPARQL_MAX_RETRIES", "3"))

    source_summary = []
    if user_cqs:
        source_summary.append(f"{len(user_cqs)} user CQ(s)")
    if generated_cqs:
        source_summary.append(f"{len(generated_cqs)} auto-generated CQ(s)")
    if user_sparql:
        source_summary.append(f"{len(user_sparql)} direct SPARQL query(ies)")
    print(f"  [SPARQL CQ Validator] Loading KG from {kg_path} "
          f"[{', '.join(source_summary)}] ...")

    # ── Load KG into in-memory pyoxigraph store ──────────────────────────
    store = pyoxigraph.Store()
    try:
        with open(kg_path, "rb") as f:
            store.load(f, format=pyoxigraph.RdfFormat.N_TRIPLES)
    except Exception as e:
        return {
            "sparql_validation_results": [],
            "feedback": "CQ_SPARQL_PASSED",
            "messages": [f"SPARQL CQ Validator: Could not load KG ({e}) — skipping."],
        }

    results = []

    # ── Mode 1: Execute user-provided SPARQL directly (no LLM) ──────────
    for sparql in user_sparql:
        label = sparql.strip()[:60]
        try:
            ask_result = store.query(sparql)
            passed = bool(ask_result)
        except Exception as e:
            print(f"    [SPARQL CQ Validator] Execution error on user SPARQL: {e}")
            results.append({
                "cq": f"[Direct SPARQL] {label}",
                "sparql": sparql,
                "passed": None,
                "diagnosis": f"Execution error: {e}",
                "source": "user_sparql",
            })
            continue

        status = "PASS" if passed else "FAIL"
        print(f"    [{status}]  [direct SPARQL] {label}")
        results.append({
            "cq": f"[Direct SPARQL] {label}",
            "sparql": sparql,
            "passed": passed,
            "diagnosis": "" if passed else f"ASK returned false. Pattern: {_extract_triple_patterns(sparql)}",
            "source": "user_sparql",
        })

    # ── Mode 2 & 3: Translate CQs → SPARQL then execute ─────────────────
    if all_cqs:
        ontology_info = state.get("ontology_info", {})
        base_uri = state.get("base_uri", "http://example.org/")

        # Extract actual prefixes from the generated YARRRML so SPARQL queries
        # use the same namespaces as the materialised KG (Bug 2 fix)
        yarrrml_str = state.get("yarrrml_output", "")
        yarrrml_prefix_map = extract_yarrrml_prefixes(yarrrml_str) if yarrrml_str else {}
        if yarrrml_prefix_map:
            print(f"  [SPARQL CQ Validator] Using {len(yarrrml_prefix_map)} prefix(es) "
                  f"from YARRRML: {list(yarrrml_prefix_map.keys())}")

        # Extract entity types and predicates from YARRRML to ground SPARQL generation
        mapping_context = extract_yarrrml_mapping_context(yarrrml_str) if yarrrml_str else None

        # Probe the live KG for actual classes and predicates — strongest grounding signal.
        # This prevents the LLM from writing ?s a schema:Order when only schema:OrderItem exists.
        kg_path = state.get("rdf_output", "")
        kg_probe = probe_kg_types_and_predicates(kg_path) if kg_path else {}
        kg_grounding = build_kg_grounding_block(kg_probe)
        if kg_grounding:
            print(f"  [SPARQL CQ Validator] KG probe: {len(kg_probe.get('classes', []))} class(es), "
                  f"{len(kg_probe.get('predicates', []))} predicate(s) found")

        print(f"  [SPARQL CQ Validator] Translating {len(all_cqs)} CQ(s) to SPARQL ...")
        cq_sparql_list = batch_cq_to_sparql(
            all_cqs, ontology_info, base_uri=base_uri,
            yarrrml_prefix_map=yarrrml_prefix_map,
            mapping_context=mapping_context,
            kg_grounding_block=kg_grounding or None,
        )

        for item in cq_sparql_list:
            cq = item["cq"]
            sparql = item["sparql"]
            source = "user_cq" if cq in user_cqs else "generated_cq"

            if not item["valid"]:
                print(f"    [WARNING] Invalid SPARQL for: {cq[:60]}")
                results.append({
                    "cq": cq,
                    "sparql": sparql,
                    "passed": None,
                    "diagnosis": f"SPARQL generation failed: {item['error']}",
                    "source": source,
                })
                continue

            # Execute ASK query
            try:
                ask_result = store.query(sparql)
                passed = bool(ask_result)
            except Exception as e:
                # Fallback: re-prompt with error
                from agents.cq_to_sparql_agent import cq_to_sparql
                corrected = cq_to_sparql(
                    cq, ontology_info, base_uri=base_uri,
                    previous_error=str(e), previous_sparql=sparql,
                    yarrrml_prefix_map=yarrrml_prefix_map,
                    mapping_context=mapping_context,
                    kg_grounding_block=kg_grounding or None,
                )
                try:
                    ask_result = store.query(corrected)
                    passed = bool(ask_result)
                    sparql = corrected
                except Exception as e2:
                    results.append({
                        "cq": cq, "sparql": corrected,
                        "passed": None,
                        "diagnosis": f"Execution error after correction: {e2}",
                        "source": source,
                    })
                    continue

            status = "PASS" if passed else "FAIL"
            src_label = "user CQ" if source == "user_cq" else "auto CQ"
            print(f"    [{status}]  [{src_label}] {cq[:70]}")

            diagnosis = ""
            if not passed:
                # Run a SELECT probe to give a better diagnosis:
                # try to find which part of the triple pattern actually exists.
                triple_pat = _extract_triple_patterns(sparql)
                probe_diagnosis = _select_probe_diagnosis(store, sparql)
                diagnosis = probe_diagnosis if probe_diagnosis else f"ASK returned false. Missing: {triple_pat}"

            results.append({
                "cq": cq,
                "sparql": sparql,
                "passed": passed,
                "diagnosis": diagnosis,
                "source": source,
            })

    # ── Evaluate overall result ──────────────────────────────────────────
    definite_failures = [r for r in results if r["passed"] is False]
    syntax_failures   = [r for r in results if r["passed"] is None]   # SPARQL gen failed
    n_total  = len(results)
    n_passed = sum(1 for r in results if r["passed"] is True)
    n_unknown = len(syntax_failures)

    # ── Save SPARQL validation report to run_dir ─────────────────────────
    _save_sparql_report(results, state.get("run_dir", "data/output/debug"),
                        retry=cq_sparql_retry_count)

    if not definite_failures:
        print(f"  [PASS] All {n_passed}/{n_total} check(s) covered by the KG")
        if n_unknown:
            print(f"  [WARNING] {n_unknown} check(s) had SPARQL generation errors — treated as skipped")
        return {
            "sparql_validation_results": results,
            "feedback": "CQ_SPARQL_PASSED",
            "persistent_cq_failures": [],
            "messages": [f"SPARQL CQ Validator: [PASS] {n_passed}/{n_total} check(s) covered by KG."],
        }

    # ── Fix 3: Distinguish syntax failures from KG coverage failures ─────
    # A "syntax failure" is a result where passed=None (SPARQL could not even
    # be parsed/executed).  These indicate the SPARQL generator produced invalid
    # queries — they do NOT indicate anything wrong with the YARRRML/KG.
    # Re-generating YARRRML for syntax failures wastes 3×55s and changes nothing.
    #
    # A "KG coverage failure" is a result where passed=False (valid SPARQL but
    # ASK returned false) — these indicate the KG is missing triples, so
    # YARRRML re-generation is the right response.
    all_failures_are_syntax = (
        len(definite_failures) == 0
        and len(syntax_failures) > 0
    )
    mixed_but_mostly_syntax = (
        len(syntax_failures) > 0
        and len(definite_failures) <= 1
        and len(syntax_failures) >= len(definite_failures) * 2
    )

    if all_failures_are_syntax or mixed_but_mostly_syntax:
        # The SPARQL generator is producing invalid queries.
        # Mark as "SPARQL_SYNTAX_ONLY" so the workflow knows NOT to
        # re-generate YARRRML — it should only re-run SPARQL translation.
        print(
            f"  [SPARQL CQ Validator] {len(syntax_failures)} syntax-only SPARQL failure(s) "
            f"— YARRRML is fine, only SPARQL queries need fixing."
        )
        return {
            "sparql_validation_results": results,
            "feedback": "CQ_SPARQL_PASSED",   # treat as pass — not a KG problem
            "persistent_cq_failures": [],
            "messages": [
                f"SPARQL CQ Validator: {len(syntax_failures)} SPARQL syntax error(s) "
                f"(not a KG issue) — accepting current KG."
            ],
        }

    # ── Partial-pass threshold: ask user whether to keep retrying ────────
    # If pass_rate >= CQ_CONTINUE_THRESHOLD (default 70%) but not 100%,
    # prompt the user interactively.  Auto-accepts after CQ_CONTINUE_TIMEOUT
    # seconds (default 5s) if there is no response.
    _threshold = float(os.environ.get("CQ_CONTINUE_THRESHOLD", "0.70"))
    _pass_rate  = n_passed / n_total if n_total > 0 else 0.0
    _timeout    = int(os.environ.get("CQ_CONTINUE_TIMEOUT", "5"))

    if _pass_rate >= _threshold and definite_failures:
        _user_continues = _ask_user_continue(n_passed, n_total, timeout=_timeout)
        if not _user_continues:
            # User accepted partial results (or timed out) — treat as passed
            return {
                "sparql_validation_results": results,
                "feedback": "CQ_SPARQL_PASSED",
                "persistent_cq_failures": [],
                "messages": [
                    f"SPARQL CQ Validator: [ACCEPTED] {n_passed}/{n_total} "
                    f"({int(_pass_rate * 100)}%) — user accepted partial results."
                ],
            }
        # User said 'yes' → fall through to build error feedback and retry

    # ── Build structured feedback for refiner ────────────────────────────
    failure_lines = []
    for f in definite_failures:
        src_tag = {"user_cq": "user CQ", "generated_cq": "auto CQ", "user_sparql": "direct SPARQL"}.get(f["source"], f["source"])
        failure_lines.append(
            f"  [{src_tag}] CQ: \"{f['cq']}\"\n"
            f"  SPARQL tried: {f['sparql'].strip()}\n"
            f"  Diagnosis: {f['diagnosis']}"
        )

    feedback = (
        f"CQ_SPARQL_ERROR: {len(definite_failures)}/{n_total} check(s) "
        f"are NOT covered by the generated KG (verified by SPARQL execution).\n\n"
        f"FAILED CHECKS:\n" + "\n\n".join(failure_lines) + "\n\n"
        f"INSTRUCTIONS: Fix the YARRRML mapping to produce the missing triple patterns "
        f"shown above. Add or correct the predicateObjectMap for the relevant CSV columns. "
        f"Do NOT change mappings for passing checks."
    )

    prev_persistent = state.get("persistent_cq_failures", [])
    prev_failing_cqs = {p["cq"] if isinstance(p, dict) else p for p in prev_persistent}
    now_failing_cqs = {f["cq"] for f in definite_failures}

    return {
        "sparql_validation_results": results,
        "feedback": feedback,
        "cq_sparql_retry_count": cq_sparql_retry_count + 1,
        "persistent_cq_failures": definite_failures,
        "messages": [
            f"SPARQL CQ Validator: [FAIL] {len(definite_failures)}/{n_total} check(s) "
            f"not covered — routing to refiner "
            f"(attempt {cq_sparql_retry_count + 1}/{max_cq_sparql_retries})."
        ],
    }


# ────────────────────────────────────────────────────────────────────────────
# SHACL validation helpers  (dataset-agnostic, no hard-coded ontology knowledge)
# ────────────────────────────────────────────────────────────────────────────

def _parse_shacl_violations(results_text: str) -> list[str]:
    """Extract human-readable, deduplicated violation summaries from pyshacl output.

    Parses the real pyshacl text format:
      Constraint Violation in NodeKindConstraintComponent (...):
          Severity: sh:Violation
          Source Shape: ...
          Focus Node: ...
          Value Node: ...
          Result Path: ...   ← present for sh:property constraints
          Message: ...

    Groups by (component, path-or-message) to collapse per-node repeats into
    a single summarised entry with a count and example focus nodes.
    """
    import collections as _coll
    import re as _re

    raw: list[dict] = []
    cur: dict = {}

    for line in results_text.splitlines():
        s = line.strip()
        # Header: "Constraint Violation in NodeKindConstraintComponent (...):"
        _m = _re.match(r'Constraint Violation in\s+(\w+)\s*\(', s)
        if _m:
            if cur:
                raw.append(cur)
            cur = {"component": _m.group(1)}
            continue
        if s.startswith("Result Path:"):
            cur["path"] = s.split(":", 1)[-1].strip()
        elif s.startswith("Message:"):
            cur["message"] = s.split(":", 1)[-1].strip()
        elif s.startswith("Focus Node:"):
            cur.setdefault("focus_nodes", []).append(s.split(":", 1)[-1].strip())
        elif s.startswith("Source Shape:"):
            cur["shape"] = s.split(":", 1)[-1].strip()

    if cur:
        raw.append(cur)

    if not raw:
        # Plain-text fallback
        return [l.strip() for l in results_text.splitlines()
                if "Violation" in l or "Message:" in l][:20]

    groups: dict[str, list[dict]] = _coll.OrderedDict()
    for v in raw:
        key = str((v.get("component", "?"), v.get("path", v.get("message", "?"))))
        groups.setdefault(key, []).append(v)

    summaries: list[str] = []
    for instances in groups.values():
        rep = instances[0]
        component = rep.get("component", "unknown")
        path = rep.get("path", "")
        message = rep.get("message", "")
        count = len(instances)
        parts: list[str] = []
        if path:
            parts.append(f"path={path}")
        if message:
            parts.append(message)
        if count > 1:
            parts.append(f"({count} nodes affected)")
        sample: list[str] = []
        for inst in instances[:2]:
            sample.extend(inst.get("focus_nodes", [])[:1])
        if sample:
            parts.append(f"e.g. {', '.join(sample[:2])}")
        summaries.append(f"[{component}] {' | '.join(parts)}")

    return summaries


def _shacl_violation_fingerprint(violations: list[str]) -> str:
    """Stable 12-char MD5 digest of the sorted violation set.

    Used to detect when the identical violations recur across consecutive
    SHACL retries — a signal that the generator cannot fix them (the root
    cause is the data itself, e.g. literal role strings mapped to an
    owl:ObjectProperty that expects IRIs).
    """
    import hashlib as _hl
    return _hl.md5("|".join(sorted(violations)).encode()).hexdigest()[:12]


def _build_shacl_actionable_feedback(
    violations: list[str],
    ontology_path: str,
    shapes_source: str,
) -> str:
    """Convert raw violation summaries into generator-actionable fix instructions.

    Looks up each offending property in the ontology to determine whether it is
    an owl:ObjectProperty or owl:DatatypeProperty, then adds a concrete remedy:

    • ObjectProperty + literal values  →
        "construct an IRI from the column: prefix:Class/$(col)~iri"
    • DatatypeProperty + IRI values    →
        "remove the ~iri suffix"
    • NodeKind IRI on a subject (no path) →
        "the s: line is producing a literal — use a URI template"
    """
    import re as _re

    obj_props: set[str] = set()
    data_props: set[str] = set()
    if ontology_path and os.path.exists(ontology_path):
        try:
            from rdflib import Graph as _G, OWL as _OWL, RDF as _RDF
            _g = _G()
            _g.parse(ontology_path)
            obj_props  = {str(p) for p in _g.subjects(_RDF.type, _OWL.ObjectProperty)}
            data_props = {str(p) for p in _g.subjects(_RDF.type, _OWL.DatatypeProperty)}
        except Exception:
            pass

    lines = [
        "SHACL_ERROR: SHACL VIOLATIONS DETECTED",
        f"  {len(violations)} unique constraint violation type(s) found in the generated KG.",
        f"  Shapes derived from: {shapes_source}",
        "",
        "  ⚠️  CRITICAL RULE — READ BEFORE EDITING:",
        "  Do NOT add new intermediate mappings (e.g. CastMapping, RoleMapping).",
        "  Do NOT create cast nodes, join nodes, or reification mappings.",
        "  Do NOT change the overall mapping architecture.",
        "  ONLY fix the specific predicate(s) listed below — nothing else.",
        "  The previous mapping structure was correct; just fix the listed predicates.",
        "",
        "  Each violation is listed with a SPECIFIC FIX INSTRUCTION.",
        "  Apply ALL fixes before regenerating the YARRRML.",
        "",
    ]

    for v in violations[:8]:
        lines.append(f"  Violation: {v[:140]}")
        prop_m   = _re.search(r'path=(<[^>]+>|[\w:]+)', v)
        prop_raw = prop_m.group(1) if prop_m else ""
        prop_uri = prop_raw.strip("<>")
        prop_local = prop_uri.split("/")[-1].split("#")[-1] if prop_uri else ""
        class_name = prop_local[0].upper() + prop_local[1:] if prop_local else "Resource"

        if prop_uri and prop_uri in obj_props:
            lines.append(
                f"  FIX: <{prop_uri}> is owl:ObjectProperty — values MUST be IRIs.\n"
                "       OPTION A (preferred): build an IRI from the column value:\n"
                f"         Change:  [<{prop_uri}>, $(column), xsd:string]\n"
                f"         To:      [<{prop_uri}>, example:{class_name}/$(column)~iri]\n"
                "       OPTION B: replace with a DatatypeProperty that accepts strings:\n"
                f"         Change:  [<{prop_uri}>, $(column), xsd:string]\n"
                f"         To:      [schema:roleName, $(column), xsd:string]  (or similar)\n"
                "       Do NOT use the ontology prefix (e.g. dbo:) in the IRI template —\n"
                "       use the subject base namespace prefix (ex: or example:) instead.\n"
                "       Do NOT create any new intermediate mapping to resolve this."
            )
        elif prop_uri and prop_uri in data_props:
            lines.append(
                f"  FIX: <{prop_uri}> is owl:DatatypeProperty — values MUST be literals.\n"
                "       Remove the '~iri' suffix from this property's PO entry.\n"
                f"         Change:  [<{prop_uri}>, $(column)~iri]\n"
                f"         To:      [<{prop_uri}>, $(column), xsd:string]"
            )
        elif not prop_uri and "IRI" in v:
            lines.append(
                "  FIX: A subject URI template is producing a literal.\n"
                "       The 's:' line must use a URI template: 'prefix:Class/$(id_column)'.\n"
                "       Do NOT restructure the mapping."
            )
        else:
            lines.append(
                f"  FIX: Ensure the value for '{prop_uri or 'this property'}' matches the "
                "node kind (IRI vs. Literal) required by the ontology.\n"
                "  Do NOT restructure the mapping — only fix this property."
            )
        lines.append("")

    return "\n".join(lines)


# ── Shape generation helpers ─────────────────────────────────────────────────

def _sanitize_shacl_shapes(shapes_ttl: str) -> str:
    """Remove invalid shapes from Astrea output before passing to pyshacl.

    pyshacl strictly requires every sh:PropertyShape to have exactly one
    sh:path predicate.  Astrea occasionally emits PropertyShapes that have
    no sh:path (e.g. the 'profession' and 'minute' shapes in some DBpedia
    ontology responses).  These cause pyshacl to raise:
      "A shape defined as a PropertyShape must include one `sh:path` property."
    and abort the entire validation run.

    This function:
    1. Parses the Turtle with rdflib.
    2. Identifies every sh:PropertyShape subject.
    3. Removes all triples whose subject is a PropertyShape without sh:path.
    4. Re-serialises as Turtle.

    Returns the sanitised Turtle string (or the original if parsing fails).
    """
    try:
        from rdflib import Graph as _G, URIRef as _U, RDF as _RDF
        _SH_PS   = _U("http://www.w3.org/ns/shacl#PropertyShape")
        _SH_PATH = _U("http://www.w3.org/ns/shacl#path")
        _SH_PROP = _U("http://www.w3.org/ns/shacl#property")

        g = _G()
        g.parse(data=shapes_ttl, format="turtle")

        # Find PropertyShapes that have NO sh:path
        bad_shapes: set = set()
        for s in g.subjects(_RDF.type, _SH_PS):
            if not any(True for _ in g.objects(s, _SH_PATH)):
                bad_shapes.add(s)

        if not bad_shapes:
            return shapes_ttl   # nothing to remove

        print(
            f"    [SHACL] Sanitising Astrea shapes: removing "
            f"{len(bad_shapes)} PropertyShape(s) without sh:path"
        )

        # Remove all triples whose subject is a bad shape
        for s in bad_shapes:
            for p, o in list(g.predicate_objects(s)):
                g.remove((s, p, o))
            # Also remove sh:property references pointing to bad shapes
            for subj, pred in list(g.subject_predicates(s)):
                if pred == _SH_PROP:
                    g.remove((subj, pred, s))

        return g.serialize(format="turtle")
    except Exception as _e:
        print(f"    [SHACL] Shape sanitisation failed ({_e}) — using original shapes")
        return shapes_ttl


def _astrea_generate_shapes(ontology_path: str) -> str | None:
    """Try to fetch SHACL shapes from the Astrea REST API.

    Uses the correct Astrea endpoint as documented in the Swagger UI:
      POST /api/shacl/document
      Content-Type: application/json
      Body: {"ontology": "<ontology text>", "serialisation": "<FORMAT>"}
      Response: text/rdf+turtle

    Supported serialisation values: TURTLE, RDF_XML, N_TRIPLES, JSON_LD, etc.

    Returns Turtle string on success, None on any failure.
    """
    import requests as _req
    import json as _json

    if not ontology_path or not os.path.exists(ontology_path):
        return None

    # ── Step 1: Read ontology as raw text ────────────────────────────────
    try:
        with open(ontology_path, "r", encoding="utf-8") as _fh:
            _onto_text = _fh.read()
    except Exception as _e:
        print(f"    [SHACL] Astrea: cannot read ontology file: {_e}")
        return None

    # ── Step 2: Detect serialisation format from file extension ──────────
    _ext = os.path.splitext(ontology_path)[1].lower()
    _FORMAT_MAP = {
        ".ttl":    "TURTLE",
        ".turtle": "TURTLE",
        ".rdf":    "RDF_XML",
        ".xml":    "RDF_XML",
        ".nt":     "N_TRIPLES",
        ".jsonld": "JSON_LD",
        ".json":   "JSON_LD",
        ".trig":   "TRIG",
        ".nq":     "N_QUADS",
        ".trix":   "TRIX",
    }
    _serialisation = _FORMAT_MAP.get(_ext, "TURTLE")

    # ── Step 3: POST to /api/shacl/document ──────────────────────────────
    _ENDPOINT = "https://astrea.linkeddata.es/api/shacl/document"
    _payload = _json.dumps({
        "ontology":      _onto_text,
        "serialisation": _serialisation,
    })

    try:
        _r = _req.post(
            _ENDPOINT,
            data=_payload,
            headers={
                "Content-Type": "application/json",
                "Accept":       "text/turtle, text/rdf+turtle, */*",
            },
            timeout=30,
        )
        if _r.status_code == 200 and _r.text.strip():
            _body = _r.text.strip()
            if "sh:NodeShape" in _body or "@prefix" in _body or "sh:" in _body:
                print(
                    f"    [SHACL] Astrea ✓ "
                    f"(format={_serialisation}, {len(_body)} chars)"
                )
                return _body
            else:
                print("    [SHACL] Astrea 200 but response doesn't look like SHACL — falling back")
                return None
        else:
            print(
                f"    [SHACL] Astrea HTTP {_r.status_code} "
                f"(format={_serialisation}): {_r.text[:200]}"
            )
            return None
    except _req.exceptions.ConnectionError:
        print("    [SHACL] Astrea unreachable — falling back to local rdflib shape generation")
        return None
    except _req.exceptions.Timeout:
        print("    [SHACL] Astrea timed out — falling back to local rdflib shape generation")
        return None
    except Exception as _exc:
        print(f"    [SHACL] Astrea error: {_exc}")
        return None


def _rdflib_generate_shapes(ontology_path: str) -> str | None:
    """Generate SHACL shapes locally from an OWL ontology using rdflib.

    Produces NodeShape entries for every class and PropertyShape entries for
    every ObjectProperty (sh:nodeKind sh:IRI) and DatatypeProperty
    (sh:nodeKind sh:Literal) declared in the ontology.

    Returns Turtle string on success, None if the ontology cannot be parsed.
    """
    if not ontology_path or not os.path.exists(ontology_path):
        return None
    try:
        from rdflib import Graph as _G, OWL as _OWL, RDF as _RDF, RDFS as _RDFS
        _g = _G()
        _g.parse(ontology_path)

        _SH = "http://www.w3.org/ns/shacl#"
        _lines: list[str] = [
            "@prefix sh: <http://www.w3.org/ns/shacl#> .",
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "",
        ]

        _shape_idx = 0

        # NodeShapes for every owl:Class
        for _cls in _g.subjects(_RDF.type, _OWL.Class):
            _cls_s = str(_cls)
            if _cls_s.startswith("http") and "owl#" not in _cls_s:
                _lines.append(f"<{_cls_s}Shape> a sh:NodeShape ;")
                _lines.append(f"    sh:targetClass <{_cls_s}> ;")
                _lines.append("    sh:nodeKind sh:IRI .")
                _lines.append("")
                _shape_idx += 1

        # PropertyShapes for ObjectProperties → nodeKind IRI
        for _prop in _g.subjects(_RDF.type, _OWL.ObjectProperty):
            _prop_s = str(_prop)
            if _prop_s.startswith("http"):
                _lines.append(f"[] a sh:PropertyShape ;")
                _lines.append(f"    sh:path <{_prop_s}> ;")
                _lines.append("    sh:nodeKind sh:IRI .")
                _lines.append("")
                _shape_idx += 1

        # PropertyShapes for DatatypeProperties → nodeKind Literal
        for _prop in _g.subjects(_RDF.type, _OWL.DatatypeProperty):
            _prop_s = str(_prop)
            if _prop_s.startswith("http"):
                _lines.append(f"[] a sh:PropertyShape ;")
                _lines.append(f"    sh:path <{_prop_s}> ;")
                _lines.append("    sh:nodeKind sh:Literal .")
                _lines.append("")
                _shape_idx += 1

        if _shape_idx == 0:
            return None

        print(f"    [SHACL] rdflib-local: generated {_shape_idx} shapes from ontology")
        return "\n".join(_lines)
    except Exception as _e:
        print(f"    [SHACL] rdflib-local shape generation failed: {_e}")
        return None


# ── SHACL Validation Node ────────────────────────────────────────────────────

def shacl_validation_node(state: AgentState) -> dict:
    """Run SHACL validation on the materialised KG.

    Passthrough when ``shacl_enabled`` is False.
    On violations: builds actionable feedback and sets SHACL_ERROR in feedback.
    On conformance: sets SHACL_PASSED.

    Persistent-violation guard: if the same (or superset) violations recur
    across two consecutive retries, the node treats them as unresolvable and
    passes through to avoid infinite loops.
    """
    if not state.get("shacl_enabled", False):
        return {"feedback": "SHACL_SKIP"}

    import tempfile as _tmp

    kg_path     = state.get("rdf_output", "")
    onto_path   = state.get("ontology_path", "")
    run_dir     = state.get("run_dir", "")
    base_uri    = state.get("base_uri", "http://example.org/mykg#")
    shacl_retry = state.get("shacl_retry_count", 0)
    prev_viols  = state.get("_prev_shacl_violations") or []
    prev_fp     = state.get("shacl_violation_fingerprint", "")

    print(f"\n[SHACL Validator] Running SHACL validation (retry #{shacl_retry}) …")

    if not kg_path or not os.path.exists(kg_path):
        print("  [SHACL] No KG file found — skipping.")
        return {"feedback": "SHACL_SKIP"}

    # ── 1. Obtain SHACL shapes ──────────────────────────────────────────────
    shapes_ttl    = _astrea_generate_shapes(onto_path)
    shapes_source = "Astrea" if shapes_ttl else None

    if not shapes_ttl:
        print("  [SHACL] Astrea unavailable — trying local rdflib shape generation …")
        shapes_ttl    = _rdflib_generate_shapes(onto_path)
        shapes_source = "rdflib-local" if shapes_ttl else None

    if not shapes_ttl:
        print("  [SHACL] Using minimal structural fallback shapes.")
        shapes_ttl = (
            "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
            "[] a sh:NodeShape ; sh:targetSubjectsOf "
            "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ; sh:nodeKind sh:IRI .\n"
        )
        shapes_source = "structural-fallback"

    # ── 1b. Sanitise shapes — remove PropertyShapes without sh:path ────────
    # Astrea occasionally emits path-less PropertyShapes that cause pyshacl
    # to abort with "A shape defined as a PropertyShape must include one
    # `sh:path` property."  Strip them before validation.
    if shapes_source in ("Astrea",):
        shapes_ttl = _sanitize_shacl_shapes(shapes_ttl)

    # ── 2. Write shapes to file ─────────────────────────────────────────────
    _shapes_file: str
    if run_dir:
        _shapes_file = os.path.join(run_dir, "shacl_shapes.ttl")
        with open(_shapes_file, "w") as _f:
            _f.write(shapes_ttl)
    else:
        _fd, _shapes_file = _tmp.mkstemp(suffix=".ttl")
        with os.fdopen(_fd, "w") as _f:
            _f.write(shapes_ttl)

    # ── 3. Run pyshacl ─────────────────────────────────────────────────────
    try:
        import pyshacl as _pyshacl
        _conforms, _results_g, _results_text = _pyshacl.validate(
            kg_path,
            shacl_graph=_shapes_file,
            inference="none",
            serialize_report_graph=True,
        )
    except Exception as _e:
        print(f"  [SHACL] pyshacl error: {_e}")
        return {"feedback": "SHACL_SKIP"}

    # ── 4. Print / save report ─────────────────────────────────────────────
    print(f"\n  [SHACL] Shapes source: {shapes_source}")
    print(f"  [SHACL] {_results_text}")

    if run_dir:
        try:
            with open(os.path.join(run_dir, "shacl_report.txt"), "w") as _f:
                _f.write(f"Shapes source: {shapes_source}\n\n")
                _f.write(_results_text)
        except Exception:
            pass

    if _conforms:
        print("  [SHACL Validator] [PASS] KG conforms to all SHACL shapes.")
        return {
            "feedback": "SHACL_PASSED",
            "shacl_retry_count": shacl_retry,
            "_prev_shacl_violations": [],
            "shacl_violation_fingerprint": "",
        }

    # ── 5. Parse violations ────────────────────────────────────────────────
    _viols = _parse_shacl_violations(_results_text)
    cur_fp = _shacl_violation_fingerprint(_viols)
    print(f"  [SHACL Validator] ✗ {len(_viols)} unique violation type(s) found (shapes: {shapes_source})")
    for _v in _viols[:5]:
        print(f"    - {_v[:120]}")

    # ── 6. Persistent-violation guard ──────────────────────────────────────
    if prev_fp and prev_viols:
        prev_paths = {
            re.search(r'path=(\S+)', v).group(1)
            for v in prev_viols if re.search(r'path=(\S+)', v)
        }
        cur_paths = {
            re.search(r'path=(\S+)', v).group(1)
            for v in _viols if re.search(r'path=(\S+)', v)
        }
        _is_persistent = bool(prev_paths and prev_paths.issubset(cur_paths)) or (cur_fp == prev_fp)
        if _is_persistent:
            print(
                "  [SHACL Validator] Persistent violations — "
                "passing through to avoid infinite loop."
            )
            return {
                "feedback": "SHACL_PASSED",
                "shacl_retry_count": shacl_retry,
                "_prev_shacl_violations": _viols,
                "shacl_violation_fingerprint": cur_fp,
            }

    # ── 7. Deterministic Pass-B fix (literal → IRI for ObjectProperty) ────
    yarrrml = state.get("yarrrml_output", "")
    if yarrrml and onto_path:
        # Collect obj_props from ontology
        _obj_props: set[str] = set()
        try:
            from rdflib import Graph as _G2, OWL as _OWL2, RDF as _RDF2
            _g2 = _G2()
            _g2.parse(onto_path)
            _obj_props = {str(p) for p in _g2.subjects(_RDF2.type, _OWL2.ObjectProperty)}
        except Exception:
            pass

        # Derive base prefix from base_uri
        _base_pfx = "ex"
        try:
            import yaml as _yaml2
            _yd = _yaml2.safe_load(yarrrml) or {}
            _pfxs = _yd.get("prefixes", {}) or {}
            for _k, _v in _pfxs.items():
                if isinstance(_v, str) and base_uri and _v.rstrip("/#") == base_uri.rstrip("/#"):
                    _base_pfx = _k
                    break
        except Exception:
            pass

        _fixed, _fix_msgs = _fix_iri_template_for_objectproperty(
            yarrrml, _viols, _obj_props, _base_pfx, base_uri
        )
        if _fix_msgs:
            for _msg in _fix_msgs:
                print(f"  [SHACL] Deterministic fix: {_msg}")
            print(f"  [SHACL] Deterministic fix applied ({len(_fix_msgs)} patch(es)) — re-materialising without LLM.")
            return {
                "yarrrml_output": _fixed,
                "feedback": "SHACL_ERROR",
                "shacl_retry_count": shacl_retry + 1,
                "_prev_shacl_violations": _viols,
                "shacl_violation_fingerprint": cur_fp,
            }

    # ── 8. Build LLM feedback ──────────────────────────────────────────────
    _fb = _build_shacl_actionable_feedback(_viols, onto_path, shapes_source)
    print("  [SHACL Validator] Sending violation feedback to YARRRML generator …")
    return {
        "feedback": _fb,
        "shacl_retry_count": shacl_retry + 1,
        "_prev_shacl_violations": _viols,
        "shacl_violation_fingerprint": cur_fp,
    }

