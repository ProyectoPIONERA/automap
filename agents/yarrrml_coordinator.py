"""
YARRRML Coordinator — orchestrates three specialised sub-agents to
generate a complete YARRRML mapping file.

Sub-agents
----------
1. **PrefixAgent**        → generates the ``prefixes:`` block
2. **EntityAgent**        → generates ``mappings:`` with data properties
3. **RelationshipAgent**  → adds joins / links between mappings

The coordinator assembles intermediate outputs, normalises CSV source
paths, and returns the final YARRRML together with intermediate
artefacts (for debugging / state inspection).
"""

import os
import re
import yaml as pyyaml

# ── Well-known prefixes for auto-resolution ──────────────────────
_WELL_KNOWN_PREFIXES: dict[str, str] = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "schema": "http://schema.org/",
    "schema1": "http://schema.org/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "terms": "http://purl.org/dc/terms/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "dbo": "http://dbpedia.org/ontology/",
    "dbr": "http://dbpedia.org/resource/",
    "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "vcard": "http://www.w3.org/2006/vcard/ns#",
    "prov": "http://www.w3.org/ns/prov#",
    "time": "http://www.w3.org/2006/time#",
    "gr": "http://purl.org/goodrelations/v1#",
    "org": "http://www.w3.org/ns/org#",
    "sosa": "http://www.w3.org/ns/sosa/",
    "ssn": "http://www.w3.org/ns/ssn/",
    "eli": "http://data.europa.eu/eli/ontology#",
    "gtfs": "http://vocab.gtfs.org/terms#",
    "ql": "http://semweb.mmlab.be/ns/ql#",
    "rr": "http://www.w3.org/ns/r2rml#",
    "rml": "http://semweb.mmlab.be/ns/rml#",
}


# ────────────────────────────────────────────────────────────────────
# Prefix reconciliation
# ────────────────────────────────────────────────────────────────────

def _reconcile_prefixes(yarrrml_str: str, ontology_raw: str = "") -> str:
    """Detect prefixes used in mappings but missing from the prefixes
    block and inject them automatically.

    Resolution order for unknown prefixes:
      1. Well-known prefix table (``_WELL_KNOWN_PREFIXES``)
      2. Prefixes declared in the ontology Turtle file
      3. Base-URI-derived placeholder (``http://example.com/{prefix}``)

    This is model-agnostic — it post-processes any YARRRML string.
    """
    try:
        data = pyyaml.safe_load(yarrrml_str)
    except Exception:
        return yarrrml_str  # unparseable — return as-is

    if not isinstance(data, dict) or "mappings" not in data:
        return yarrrml_str

    declared: dict[str, str] = data.get("prefixes", {}) or {}
    declared_lower = {k.lower() for k in declared}

    # ── Collect every prefix:localName used in mappings ──────────
    mappings_str = pyyaml.dump(data.get("mappings", {}), default_flow_style=False)
    # Match prefix:localName but not http: / https: / urn: and not
    # inside full URIs.  Also skip YAML keys like "sources:", "po:", "s:"
    used_prefixes: set[str] = set()
    for m in re.finditer(r'(?<![/\w])([a-zA-Z][\w]*):', mappings_str):
        candidate = m.group(1)
        # Skip YAML structural keys and URI scheme prefixes
        if candidate.lower() in (
            "sources", "po", "s", "a", "mappings", "graph",
            "http", "https", "urn", "ftp", "mailto",
            "condition", "targets", "graphs", "joins",
            "child", "parent",
        ):
            continue
        used_prefixes.add(candidate)

    missing = {p for p in used_prefixes if p.lower() not in declared_lower}
    if not missing:
        return yarrrml_str

    # ── Build ontology prefix lookup ─────────────────────────────
    onto_prefixes: dict[str, str] = {}
    if ontology_raw:
        for om in re.finditer(
            r'@prefix\s+(\w+):\s*<([^>]+)>', ontology_raw
        ):
            onto_prefixes[om.group(1)] = om.group(2)

    # ── Resolve each missing prefix ──────────────────────────────
    new_lines: list[str] = []
    for prefix in sorted(missing):
        uri = (
            _WELL_KNOWN_PREFIXES.get(prefix)
            or _WELL_KNOWN_PREFIXES.get(prefix.lower())
            or onto_prefixes.get(prefix)
            or onto_prefixes.get(prefix.lower())
        )
        if not uri:
            # Derive from base URI as last resort
            uri = f"http://example.com/{prefix}#"
        new_lines.append(f'  {prefix}: "{uri}"')
        print(f"    [Coordinator] Auto-injected missing prefix: {prefix}: <{uri}>")

    if not new_lines:
        return yarrrml_str

    # ── Inject into the YARRRML string ───────────────────────────
    # Insert right after the "prefixes:" line
    lines = yarrrml_str.split("\n")
    result: list[str] = []
    injected = False
    for line in lines:
        result.append(line)
        if not injected and line.strip() == "prefixes:":
            result.extend(new_lines)
            injected = True

    # If no standalone "prefixes:" line found, prepend
    if not injected:
        result = ["prefixes:"] + new_lines + [""] + lines

    return "\n".join(result)


# ────────────────────────────────────────────────────────────────────
# Assembly helper
# ────────────────────────────────────────────────────────────────────

def _assemble_yarrrml(prefixes: str, entities: str) -> str:
    """Merge the ``prefixes:`` and ``mappings:`` blocks into one
    YARRRML document.
    """
    prefixes = prefixes.strip()
    entities = entities.strip()

    # Ensure the prefixes block starts with the key
    if not prefixes.startswith("prefixes:"):
        prefixes = "prefixes:\n" + prefixes

    # Ensure the mappings block starts with the key
    if not entities.startswith("mappings:"):
        entities = "mappings:\n" + entities

    return f"{prefixes}\n\n{entities}\n"


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────

def _prefixes_need_regeneration(old_plan: str, new_plan: str) -> bool:
    """Return True only if the new entity plan introduces new namespace requirements.

    Compares the set of short prefix names (e.g. 'ex', 'schema', 'gtfs')
    used in both plans.  If the new plan uses prefixes not seen in the old
    plan, the PrefixAgent must re-run to declare them.
    """
    _prefix_re = re.compile(r'\b([a-z][a-z0-9]*):', re.IGNORECASE)
    _skip = {"http", "https", "ftp", "urn", "mailto"}
    old_pfx = {m.group(1).lower() for m in _prefix_re.finditer(old_plan)
               if m.group(1).lower() not in _skip}
    new_pfx = {m.group(1).lower() for m in _prefix_re.finditer(new_plan)
               if m.group(1).lower() not in _skip}
    return bool(new_pfx - old_pfx)


def coordinate_yarrrml_generation(state: dict) -> dict:
    """Orchestrate YARRRML generation via three specialised sub-agents.

    Steps
    -----
    1. **PrefixAgent**       – produce the ``prefixes:`` block.
    2. **EntityAgent**       – produce ``mappings:`` with data properties
       (no joins between mappings).
    3. **Assemble**          – merge prefixes + entities into intermediate
       YARRRML.
    4. **RelationshipAgent** – add join / object-property PO entries to
       connect the mappings.
    5. **Normalise**         – fix CSV source paths in the final output.

    Parameters
    ----------
    state : dict
        The full pipeline ``AgentState``.

    Returns
    -------
    dict
        ``prefixes_output``  – raw PrefixAgent output
        ``entity_yarrrml``   – raw EntityAgent output
        ``yarrrml_output``   – final assembled YARRRML (with relationships)
    """
    # Lazy imports to avoid circular dependencies
    from agents.prefix_agent import call_prefix_agent
    from agents.entity_agent import call_entity_agent
    from agents.relationship_agent import call_relationship_agent

    csv_path = state["csv_path"]
    csv_source = f"{csv_path}~csv"
    csv_name = os.path.basename(csv_path)

    # Attempt number for per-attempt debug files
    attempt_num = state.get("retry_count", 0) + 1
    run_dir = state.get("run_dir", "data/output/debug")
    debug_dir = os.path.join(run_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    DEBUG_MODE = os.getenv("DEBUG_PIPELINE", "true").lower() == "true"

    # ── Step 1: Generate prefixes (cached across retries) ─────────
    existing_prefixes = state.get("prefixes_output", "")
    alignment_changed = state.get("alignment_changed", False)
    current_entity_plan = state.get("schema_alignment", {}).get("entity_plan", "")
    prev_entity_plan = state.get("_prev_entity_plan", "")

    # Only regenerate prefixes if the new entity plan needs new namespaces
    needs_regen = (
        not existing_prefixes
        or (alignment_changed and _prefixes_need_regeneration(prev_entity_plan, current_entity_plan))
    )

    if needs_regen:
        print("    [Coordinator] Step 1/3: Generating prefixes...")
        prefixes_output = call_prefix_agent(state)
    else:
        print("    [Coordinator] Step 1/3: Reusing cached prefixes...")
        prefixes_output = existing_prefixes

    if DEBUG_MODE:
        with open(os.path.join(debug_dir, f"attempt_{attempt_num}_prefixes.yaml"), "w") as f:
            f.write(prefixes_output)

    # ── Step 2: Generate entity mappings (data properties only) ───
    print("    [Coordinator] Step 2/3: Building entity mappings...")
    entity_output = call_entity_agent(state)

    with open(os.path.join(debug_dir, f"attempt_{attempt_num}_entities.yaml"), "w") as f:
        f.write(entity_output)

    # ── Step 3: Assemble intermediate YARRRML ─────────────────────
    intermediate = _assemble_yarrrml(prefixes_output, entity_output)

    if DEBUG_MODE:
        with open(os.path.join(debug_dir, f"attempt_{attempt_num}_intermediate.yaml"), "w") as f:
            f.write(intermediate)

    # ── Step 4: Add relationships / joins ─────────────────────────
    is_multi_node = state.get("schema_alignment", {}).get("multi_node", False)
    if is_multi_node:
        print("    [Coordinator] Step 3/3: Adding relationships and joins...")
        try:
            final_yarrrml = call_relationship_agent(state, intermediate)
        except Exception as e:
            print(f"    [Coordinator] WARN: Relationship agent failed ({type(e).__name__}), using intermediate YARRRML")
            final_yarrrml = intermediate
    else:
        print("    [Coordinator] Step 3/3: FLAT dataset — skipping relationship agent.")
        final_yarrrml = intermediate

    if DEBUG_MODE:
        with open(os.path.join(debug_dir, f"attempt_{attempt_num}_after_links.yaml"), "w") as f:
            f.write(final_yarrrml)

    # ── Step 5: Normalise CSV source paths ────────────────────────
    final_yarrrml = re.sub(
        r'(-\s*)\[?[^\n]*?' + re.escape(csv_name) + r'~csv[^\n]*',
        r'\g<1>[' + csv_source + ']',
        final_yarrrml,
    )

    # ── Step 6: Reconcile prefixes ────────────────────────────────
    # Auto-detect prefixes used in mappings but not declared, and
    # inject them.  This eliminates "Prefix X not bound" errors that
    # otherwise cause retry loops.
    ontology_raw = state.get("ontology_info", {}).get("raw", "")
    final_yarrrml = _reconcile_prefixes(final_yarrrml, ontology_raw)

    return {
        "prefixes_output": prefixes_output,
        "entity_yarrrml": entity_output,
        "yarrrml_output": final_yarrrml.strip(),
        "_prev_entity_plan": current_entity_plan,
    }

