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

from config.prefixes import WELL_KNOWN_PREFIXES as _WELL_KNOWN_PREFIXES

# ── Well-known prefixes imported from config/prefixes.py ─────────────────────
# (Previously duplicated here; now a single shared source of truth.)


# ────────────────────────────────────────────────────────────────────
# Prefix reconciliation
# ────────────────────────────────────────────────────────────────────

def _normalize_well_known_prefix_uris(yarrrml_str: str) -> str:
    """Correct LLM-generated prefix URIs that differ from well-known canonical URIs.

    The most common case is ``schema: "https://schema.org/"`` being emitted by
    the LLM while morph-kgc normalises the materialised KG to use
    ``http://schema.org/``.  This causes every SPARQL ASK query that references
    ``schema:`` to return false.

    We iterate over the declared prefixes and replace any whose URI doesn't
    match the well-known canonical URI.
    """
    try:
        data = pyyaml.safe_load(yarrrml_str)
    except Exception:
        return yarrrml_str

    if not isinstance(data, dict):
        return yarrrml_str

    declared: dict[str, str] = (data.get("prefixes") or {})
    replacements: list[tuple[str, str, str]] = []  # (prefix, old_uri, new_uri)

    for prefix, declared_uri in declared.items():
        canonical = _WELL_KNOWN_PREFIXES.get(prefix) or _WELL_KNOWN_PREFIXES.get(prefix.lower())
        if not canonical:
            continue
        # Normalise trailing separator for comparison
        declared_norm = declared_uri.rstrip("/#")
        canonical_norm = canonical.rstrip("/#")
        if declared_norm != canonical_norm:
            replacements.append((prefix, declared_uri, canonical))

    if not replacements:
        return yarrrml_str

    for prefix, old_uri, new_uri in replacements:
        # Replace in the prefixes block (quoted and unquoted forms)
        yarrrml_str = re.sub(
            r'(^\s{2}' + re.escape(prefix) + r':\s*)["\']?' + re.escape(old_uri) + r'["\']?',
            r'\g<1>"' + new_uri + '"',
            yarrrml_str,
            flags=re.MULTILINE,
        )
        print(f"    [Coordinator] Normalised prefix URI: {prefix}: <{old_uri}> → <{new_uri}>")

    return yarrrml_str


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

    # ── Collect mapping names to skip (avoid mapping names becoming prefixes) ─
    mapping_names_lower = {
        k.lower() for k in (data.get("mappings") or {}).keys()
    }

    missing = {p for p in used_prefixes
               if p.lower() not in declared_lower
               and p.lower() not in mapping_names_lower}
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
            # Agnostic fallback: synthesize a URI for completely unknown prefixes.
            # Using example.org (not example.com) to stay consistent with
            # the ex: well-known prefix and avoid accidentally colliding with
            # domain-specific namespaces.
            uri = f"http://example.org/{prefix}#"
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


def _apply_base_uri_to_subjects(yarrrml_str: str, base_uri: str) -> tuple[str, list[str]]:
    """Rewrite all subject (``s:``) URI templates to use the user's base URI.

    Only subjects are rewritten — predicates, object properties, and
    ontology-based class URIs in rdf:type entries are left unchanged.
    This makes the generated KG's entity URIs reflect the user's own
    domain instead of the ontology's namespace.

    Example
    -------
    base_uri = "http://mykg.org/"
    s: dbo:Film/$(movie_id)  →  s: mykg:Film/$(movie_id)
    + adds  mykg: "http://mykg.org/"  to the prefixes block.

    The prefix name is derived from the hostname:
      "http://mykg.org/"      → mykg
      "https://data.acme.com/kg/" → acme
      "http://example.org/"   → (unchanged — default, skip rewrite)

    Returns
    -------
    (rewritten_yarrrml, list_of_changes)
    """
    _DEFAULT_BASE = "http://example.org/"
    if not base_uri:
        return yarrrml_str, []

    # Normalise: add http:// scheme if missing (e.g. "mykg.org" → "http://mykg.org/")
    if not base_uri.startswith(("http://", "https://", "urn:")):
        base_uri = "http://" + base_uri
        print(f"    [Coordinator] Normalised base URI: added http:// scheme → {base_uri}")

    # Normalise: must end with / or #
    if not base_uri.endswith(("/", "#")):
        base_uri = base_uri + "/"

    if base_uri.rstrip("/") == _DEFAULT_BASE.rstrip("/"):
        return yarrrml_str, []

    # Derive a short prefix name from the hostname
    try:
        from urllib.parse import urlparse
        _parsed = urlparse(base_uri)
        _host = (_parsed.hostname or "base").replace("www.", "")
        # Take the first meaningful part: "mykg.org" → "mykg"
        _parts = [p for p in _host.split(".") if p and p not in ("com", "org", "net", "io", "eu")]
        prefix_name = (_parts[0] if _parts else "base").replace("-", "").replace("_", "")
        if not prefix_name.isidentifier() or not prefix_name:
            prefix_name = "base"
    except Exception:
        prefix_name = "base"

    try:
        import yaml as _yaml
        data = _yaml.safe_load(yarrrml_str)
    except Exception:
        return yarrrml_str, []

    if not data or not isinstance(data.get("mappings"), dict):
        return yarrrml_str, []

    changes: list[str] = []

    # Register the new prefix
    if "prefixes" not in data or not isinstance(data["prefixes"], dict):
        data["prefixes"] = {}
    data["prefixes"][prefix_name] = base_uri

    # Regex: match  somePrefix:ClassName/rest  or  somePrefix:ClassName_rest
    # where ClassName starts with a capital letter (class convention).
    _SUBJ_RE = re.compile(r'^([a-zA-Z][a-zA-Z0-9_]*):(([A-Z][A-Za-z0-9_]*)(/.+)?)$')

    for mname, mdef in data["mappings"].items():
        if not isinstance(mdef, dict):
            continue
        subj = str(mdef.get("s", ""))
        m = _SUBJ_RE.match(subj)
        if not m:
            continue
        old_prefix = m.group(1)
        rest = m.group(2)       # ClassName/$(id) or ClassName_$(id)

        # Don't rewrite if already using the target prefix
        if old_prefix == prefix_name:
            pass
        else:
            new_subj = f"{prefix_name}:{rest}"
            mdef["s"] = new_subj
            changes.append(
                f"[BaseURI] {mname}: s: {subj!r} → {new_subj!r}"
            )

    # ── Pass B: rewrite IRI template objects in po: entries ──────────
    # e.g. [dbo:starring, dbo:Person/$(person_id)~iri]
    #   →  [dbo:starring, mykg:Person/$(person_id)~iri]
    # Only rewrite templates whose ClassName already exists as the class
    # of a mapping subject (i.e. it's a self-namespace entity, not an
    # external ontology resource).  Fully dataset-agnostic.
    known_classes: set[str] = set()
    for _mdef in data["mappings"].values():
        if not isinstance(_mdef, dict):
            continue
        _s = str(_mdef.get("s", ""))
        _sm = _SUBJ_RE.match(_s)
        if _sm:
            # class name is everything up to the first /$(
            _cls = _sm.group(2).split("/$(")[0].split("_$(")[0]
            known_classes.add(_cls)

    def _rewrite_po_iri_templates(po_list, mapping_name: str) -> None:
        """Recursively rewrite IRI template objects inside po entries."""
        if not isinstance(po_list, list):
            return
        for idx, entry in enumerate(po_list):
            if isinstance(entry, list) and len(entry) >= 2:
                obj = entry[1]
                if isinstance(obj, str):
                    # Match: somePrefix:ClassName/$(col)~iri
                    _om = re.match(
                        r'^([a-zA-Z][a-zA-Z0-9_]*):(([A-Z][A-Za-z0-9_]*)(/.+~iri))$',
                        obj.strip(),
                    )
                    if _om:
                        pfx = _om.group(1)
                        cls_path = _om.group(2)     # e.g. Person/$(person_id)~iri
                        cls_name = _om.group(3)     # e.g. Person
                        if pfx != prefix_name and cls_name in known_classes:
                            new_obj = f"{prefix_name}:{cls_path}"
                            entry[1] = new_obj
                            changes.append(
                                f"[BaseURI-po] {mapping_name}: "
                                f"IRI template '{obj}' → '{new_obj}'"
                            )
            elif isinstance(entry, list):
                _rewrite_po_iri_templates(entry, mapping_name)

    for mname, mdef in data["mappings"].items():
        if not isinstance(mdef, dict):
            continue
        po = mdef.get("po", [])
        _rewrite_po_iri_templates(po, mname)

    # Always re-serialise when the prefix URI was updated (even if no subjects
    # needed renaming), so the YAML reflects the corrected URI.
    old_prefix_uri = None
    try:
        import yaml as _yaml_check
        _old_data = _yaml_check.safe_load(yarrrml_str)
        old_prefix_uri = (_old_data or {}).get("prefixes", {}).get(prefix_name)
    except Exception:
        pass

    prefix_uri_changed = old_prefix_uri != base_uri  # also True when key was absent

    if not changes and not prefix_uri_changed:
        return yarrrml_str, []

    if not changes and prefix_uri_changed:
        changes.append(f"[BaseURI] Updated prefix {prefix_name}: <{old_prefix_uri}> → <{base_uri}>")

    # Re-serialise — use ruamel to preserve flow-style po entries
    try:
        from io import StringIO
        from ruamel.yaml import YAML as _RuamelYAML
        _ry = _RuamelYAML()
        _ry.default_flow_style = False
        _ry.preserve_quotes = True
        _ry.indent(mapping=2, sequence=4, offset=2)
        _buf = StringIO()
        _ry.dump(data, _buf)
        return _buf.getvalue().strip(), changes
    except Exception:
        return yarrrml_str, []


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

    # ── Fix 3: Cache invalidation on repeated logic failures ──────
    # If the same COLUMN COVERAGE FAILURE repeats across 2+ retries it means
    # the LLM is reproducing identical output from stale cached state.
    # Force prefix regeneration so the full prompt is rebuilt from scratch.
    retry_count = state.get("retry_count", 0)
    prev_feedback = state.get("feedback", "")
    logic_fail_streak = state.get("_logic_fail_streak", 0)

    is_logic_failure = (
        "COLUMN COVERAGE FAILURE" in prev_feedback
        or "LOGIC_ERROR" in prev_feedback
        or "STRUCTURAL PROBLEMS" in prev_feedback
    )
    if is_logic_failure:
        logic_fail_streak += 1
        state["_logic_fail_streak"] = logic_fail_streak
    else:
        state["_logic_fail_streak"] = 0
        logic_fail_streak = 0

    # After 2 identical logic failures, invalidate prefix cache so the
    # next generation starts with a fresh full prompt (no KV-cache hit).
    cache_invalidated = logic_fail_streak >= 2
    if cache_invalidated:
        print(f"    [Coordinator] Logic failure streak={logic_fail_streak} — invalidating prefix cache to force full rebuild.")
        existing_prefixes = ""
        state["_logic_fail_streak"] = 0  # reset after invalidation

    # Only regenerate prefixes if the new entity plan needs new namespaces
    needs_regen = (
        not existing_prefixes
        or cache_invalidated
        or (alignment_changed and _prefixes_need_regeneration(prev_entity_plan, current_entity_plan))
    )

    if needs_regen:
        # ── Steps 1 + 2: Run prefix and entity agents IN PARALLEL ────
        # Both agents only read state — no write conflicts.
        # This saves ~50% coordinator wall-time when both take similar time.
        import concurrent.futures as _cf
        print("    [Coordinator] Steps 1+2: Generating prefixes + entities in parallel...")
        with _cf.ThreadPoolExecutor(max_workers=2) as _pool:
            _fut_pfx = _pool.submit(call_prefix_agent, state)
            _fut_ent = _pool.submit(call_entity_agent, state)
            prefixes_output = _fut_pfx.result()
            entity_output = _fut_ent.result()
    else:
        print("    [Coordinator] Step 1/3: Reusing cached prefixes...")
        prefixes_output = existing_prefixes
        # ── Step 2: Generate entity mappings ──────────────────────
        print("    [Coordinator] Step 2/3: Building entity mappings...")
        entity_output = call_entity_agent(state)

    if DEBUG_MODE:
        with open(os.path.join(debug_dir, f"attempt_{attempt_num}_prefixes.yaml"), "w") as f:
            f.write(prefixes_output)

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
    # 6a. Normalise well-known prefix URIs (e.g. https://schema.org/ → http://schema.org/)
    #     before reconciliation so the KG namespaces match SPARQL queries.
    final_yarrrml = _normalize_well_known_prefix_uris(final_yarrrml)
    # 6b. Auto-detect prefixes used in mappings but not declared, and
    #     inject them.  This eliminates "Prefix X not bound" errors that
    #     otherwise cause retry loops.
    ontology_raw = state.get("ontology_info", {}).get("raw", "")
    final_yarrrml = _reconcile_prefixes(final_yarrrml, ontology_raw)

    # ── Step 7: Restore hyphenated column names ───────────────────
    # LLMs silently convert hyphens in column names to underscores
    # (e.g. metformin-pioglitazone → metformin_pioglitazone).
    # This causes Morph-KGC to fail with "column not found".
    # Restore them deterministically using the actual CSV headers.
    from agents.refiner_agent import build_column_alias_map, restore_column_names, _fix_predicate_separator_typo, _fix_yaml_breaking_predicates, _fix_unprefixed_predicates
    try:
        import pandas as _pd
        _df_headers = _pd.read_csv(csv_path, nrows=0)
        _csv_cols = set(_df_headers.columns)
    except Exception:
        _csv_cols = set()

    _alias_map = build_column_alias_map(_csv_cols)
    if _alias_map:
        final_yarrrml, _restored = restore_column_names(final_yarrrml, _alias_map)
        for _fix in _restored:
            print(f"    [Coordinator] {_fix}")

    # ── Step 8: Fix predicate separator typos: ex#local → ex:local ──
    # Also sanitise YAML-breaking predicates (spaces/extra colons) before
    # parsing — e.g. "ex:Unnamed: 0" becomes "ex:Unnamed_0"
    final_yarrrml, _pred_fixes = _fix_yaml_breaking_predicates(final_yarrrml)
    for _fix in _pred_fixes:
        print(f"    [Coordinator] {_fix}")
    try:
        import yaml as _yaml2
        _coord_data = _yaml2.safe_load(final_yarrrml)
        _coord_pfx = set((_coord_data or {}).get("prefixes", {}).keys())
    except Exception:
        _coord_pfx = set()
    _coord_pfx.update({"rdf", "rdfs", "xsd", "owl"})
    final_yarrrml, _sep_fixes = _fix_predicate_separator_typo(final_yarrrml, _coord_pfx)
    for _fix in _sep_fixes:
        print(f"    [Coordinator] {_fix}")

    # ── Fix unprefixed predicates: isAdult → ex:isAdult ─────────
    try:
        import yaml as _yaml3
        _coord_data2 = _yaml3.safe_load(final_yarrrml)
    except Exception:
        _coord_data2 = {}
    final_yarrrml, _unpfx_fixes = _fix_unprefixed_predicates(final_yarrrml, _coord_data2 or {})
    for _fix in _unpfx_fixes:
        print(f"    [Coordinator] {_fix}")

    # ── Step 9: Enforce user's base URI on all subject templates ─────────
    # When the user supplies --base-uri (or BASE_URI env var) that differs
    # from the default http://example.org/, rewrite every s: template so
    # entity URIs live under the user's domain, not the ontology's namespace.
    user_base_uri = state.get("base_uri", "http://example.org/")
    final_yarrrml, base_uri_changes = _apply_base_uri_to_subjects(final_yarrrml, user_base_uri)
    for _chg in base_uri_changes:
        print(f"    [Coordinator] {_chg}")

    return {
        "prefixes_output": prefixes_output,
        "entity_yarrrml": entity_output,
        "yarrrml_output": final_yarrrml.strip(),
        "_prev_entity_plan": current_entity_plan,
    }
