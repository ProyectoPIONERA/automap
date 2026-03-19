import re
from config.settings import get_llm
from data.checkpoints import AgentState


def _extract_yarrrml_columns(yarrrml_str: str) -> set[str]:
    """Return all $(column) references found in the YARRRML string."""
    if not yarrrml_str:
        return set()
    return set(re.findall(r'\$\(([^)]+)\)', yarrrml_str))


def _parse_yarrrml(yarrrml_str: str):
    """Parse YARRRML string into a dict.  Returns None on failure."""
    try:
        from ruamel.yaml import YAML
        yaml = YAML(typ="safe", pure=True)
        return yaml.load(yarrrml_str)
    except Exception:
        return None


def _detect_empty_columns(csv_path: str) -> set[str]:
    """Return the set of CSV columns that are entirely empty/NaN.

    Columns with no data cannot be meaningfully mapped to RDF and
    should not cause column-coverage failures.
    """
    if not csv_path:
        return set()
    try:
        import pandas as pd
        import os
        if not os.path.isfile(csv_path):
            return set()
        df = pd.read_csv(csv_path)
        # A column is "empty" if every value is NaN or empty string
        empty = set()
        for col in df.columns:
            if df[col].dropna().astype(str).str.strip().replace("", None).dropna().empty:
                empty.add(col)
        return empty
    except Exception:
        return set()


# ────────────────────────────────────────────────────────────────────
# Well-known prefix URIs for deterministic auto-fix
# ────────────────────────────────────────────────────────────────────

_WELL_KNOWN_PREFIXES: dict[str, str] = {
    "rdf":     "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs":    "http://www.w3.org/2000/01/rdf-schema#",
    "owl":     "http://www.w3.org/2002/07/owl#",
    "xsd":     "http://www.w3.org/2001/XMLSchema#",
    "schema":  "http://schema.org/",
    "foaf":    "http://xmlns.com/foaf/0.1/",
    "dc":      "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "dct":     "http://purl.org/dc/terms/",
    "skos":    "http://www.w3.org/2004/02/skos/core#",
    "geo":     "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "dcat":    "http://www.w3.org/ns/dcat#",
    "prov":    "http://www.w3.org/ns/prov#",
    "gtfs":    "http://vocab.gtfs.org/terms#",
    "vcard":   "http://www.w3.org/2006/vcard/ns#",
    "void":    "http://rdfs.org/ns/void#",
    "time":    "http://www.w3.org/2006/time#",
    "sosa":    "http://www.w3.org/ns/sosa/",
    "ssn":     "http://www.w3.org/ns/ssn/",
    "wgs":     "http://www.w3.org/2003/01/geo/wgs84_pos#",
}

# Prefixes that YARRRML / Yatter treats as implicitly available
_IMPLICIT_PREFIXES = {"xsd", "rdf", "rdfs"}

# URI schemes to exclude when scanning for prefix usage
_URI_SCHEMES = {"http", "https", "ftp", "urn", "mailto", "file"}

_PREFIX_USAGE_RE = re.compile(r'\b([a-zA-Z][a-zA-Z0-9_]*):[a-zA-Z]')


def _extract_used_prefixes(data: dict) -> set[str]:
    """Extract all prefix names used in YARRRML mapping values.

    Scans subject templates, predicate-object entries, and type
    declarations for ``prefix:localName`` patterns.  Ignores URI
    schemes (http:, https:, etc.).
    """
    used: set[str] = set()

    def _scan(val):
        if isinstance(val, str):
            for m in _PREFIX_USAGE_RE.finditer(val):
                prefix = m.group(1)
                if prefix.lower() not in _URI_SCHEMES:
                    used.add(prefix)
        elif isinstance(val, list):
            for item in val:
                _scan(item)
        elif isinstance(val, dict):
            for v in val.values():
                _scan(v)

    mappings = data.get("mappings", {})
    if isinstance(mappings, dict):
        for mdef in mappings.values():
            if isinstance(mdef, dict):
                _scan(mdef)

    return used


def _extract_ontology_prefixes(ontology_info: str) -> dict[str, str]:
    """Extract prefix -> URI mappings from the ontology info string.

    Handles the ``@prefix name: <URI> .`` format produced by
    ``rml_tools.get_ontology_subgraph``.
    """
    prefixes: dict[str, str] = {}
    for m in re.finditer(r'@prefix\s+(\w+):\s*<([^>]+)>', ontology_info):
        prefixes[m.group(1)] = m.group(2)
    return prefixes


def _check_prefix_completeness(data: dict) -> tuple[list[str], set[str]]:
    """Check that all prefixes used in mappings are declared.

    Returns
    -------
    (errors, missing_prefix_names)
    """
    declared = set(data.get("prefixes", {}).keys()) if data.get("prefixes") else set()
    used = _extract_used_prefixes(data)
    missing = used - declared - _IMPLICIT_PREFIXES

    errors: list[str] = []
    if missing:
        errors.append(
            f"PREFIX COMPLETENESS -- these prefixes are used but NOT declared "
            f"in the `prefixes:` section: [{', '.join(sorted(missing))}]. "
            f"Add them to the `prefixes:` block with their full URI."
        )
    return errors, missing


def _auto_fix_missing_prefixes(
    yarrrml_str: str,
    data: dict,
    ontology_info: str = "",
) -> tuple[str, list[str]]:
    """Auto-fix missing prefix declarations in the YARRRML text.

    Resolution order for each missing prefix:
      1. Ontology prefix declarations (pipeline's own ontology)
      2. Well-known prefix URI table

    Returns
    -------
    (fixed_yarrrml, list_of_fix_descriptions)
    """
    _, missing = _check_prefix_completeness(data)
    if not missing:
        return yarrrml_str, []

    # Ontology prefixes take precedence over well-known defaults
    available: dict[str, str] = dict(_WELL_KNOWN_PREFIXES)
    available.update(_extract_ontology_prefixes(ontology_info))

    fixes: list[str] = []
    additions: list[str] = []
    for prefix in sorted(missing):
        uri = available.get(prefix)
        if uri:
            additions.append(f'  {prefix}: "{uri}"')
            fixes.append(f"Added missing prefix '{prefix}: {uri}'")

    if not additions:
        return yarrrml_str, []

    # Insert new prefix lines right after the `prefixes:` line
    lines = yarrrml_str.split('\n')
    result: list[str] = []
    inserted = False

    for line in lines:
        result.append(line)
        if not inserted and line.strip().startswith('prefixes:'):
            result.extend(additions)
            inserted = True

    if not inserted:
        # No prefixes block found — create one at the top
        result = ['prefixes:'] + additions + [''] + lines
        return '\n'.join(result), fixes

    return '\n'.join(result), fixes


def _strip_unused_prefixes(yarrrml_str: str, data: dict) -> tuple[str, list[str]]:
    """Remove prefix declarations that are never used in the mappings.

    LLMs often dump all ontology prefixes into the YARRRML prefixes
    block.  Unused prefixes confuse the refiner LLM and bloat output.
    This function strips them deterministically.

    Returns
    -------
    (cleaned_yarrrml, list_of_removed_prefix_names)
    """
    declared = data.get("prefixes", {})
    if not declared:
        return yarrrml_str, []

    used = _extract_used_prefixes(data)
    # Always keep implicitly-needed prefixes even if not directly referenced
    keep = used | _IMPLICIT_PREFIXES | {"rml", "ql"}

    to_remove = set(declared.keys()) - keep
    if not to_remove:
        return yarrrml_str, []

    # Remove the lines from the text
    lines = yarrrml_str.split('\n')
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Check if this line declares a prefix we want to remove
        # Format: "  prefixName: "URI""  or  "  prefixName: URI"
        is_removed = False
        for prefix in to_remove:
            if stripped.startswith(f'{prefix}:') or stripped.startswith(f'"{prefix}":'):
                is_removed = True
                break
        if not is_removed:
            result.append(line)

    return '\n'.join(result), sorted(to_remove)


# ────────────────────────────────────────────────────────────────────
# Deterministic structural checks
# ────────────────────────────────────────────────────────────────────

def _check_islands(mappings: dict) -> list[str]:
    """Detect mappings whose subjects are never referenced as objects
    anywhere else AND that have no outgoing object-property links.

    Skips the check when there is only one mapping — a single-class
    dataset is a valid topology and cannot be "disconnected".
    Self-referential links (e.g. a parent reference within the same
    class) count as valid connectivity.
    """
    errors: list[str] = []

    # A single-mapping YARRRML is a valid flat dataset — nothing to check.
    valid_mappings = {n: m for n, m in mappings.items() if isinstance(m, dict)}
    if len(valid_mappings) <= 1:
        return errors

    # Build subject-base for every mapping  (strip $(col) and trailing _ /)
    subject_bases: dict[str, str] = {}
    for name, mdef in valid_mappings.items():
        subj = str(mdef.get("s", "") or "")
        base = re.sub(r'\$\([^)]+\)', '', subj).rstrip('_/').replace('~iri', '')
        if base:
            subject_bases[name] = base

    # Collect every object string **per mapping**.
    po_objects_by_mapping: dict[str, list[str]] = {}
    for mname, mdef in valid_mappings.items():
        objs: list[str] = []
        for entry in (mdef.get("po") or []):
            if isinstance(entry, list) and len(entry) >= 2:
                objs.append(str(entry[1]).replace('~iri', ''))
        po_objects_by_mapping[mname] = objs

    for name, base in subject_bases.items():
        # Is this mapping's subject referenced from ANY OTHER mapping?
        is_referenced = False
        for other_name, objs in po_objects_by_mapping.items():
            if other_name == name:
                continue          # skip self-references
            if any(base in obj for obj in objs):
                is_referenced = True
                break

        # Does this mapping have outgoing links?
        # We check for links to OTHER mappings first, then also accept
        # self-referential links (e.g. a parent FK pointing back to
        # the same class).
        has_outgoing = False
        for entry in (valid_mappings[name].get("po") or []):
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            obj = str(entry[1]).replace('~iri', '')
            pred = str(entry[0])
            if pred in ("rdf:type", "a"):
                continue
            # Check links to OTHER mappings
            for other_name, other_base in subject_bases.items():
                if other_name != name and other_base and other_base in obj:
                    has_outgoing = True
                    break
            # Also accept self-referential object-property links
            # (e.g. an entity linking to its parent within the same class)
            if not has_outgoing and base and base in obj and '$(' in obj:
                has_outgoing = True
            if has_outgoing:
                break

        if not is_referenced and not has_outgoing:
            errors.append(
                f"Mapping '{name}' is DISCONNECTED — its subject is never "
                f"referenced from another mapping and it has no outgoing "
                f"object-property links.  The primary mapping MUST include "
                f"an object property (2-item PO entry) linking to this "
                f"mapping's subject URI template."
            )

    return errors


def _to_camel_case(snake_str: str) -> str:
    """Convert a snake_case column name to a camelCase predicate name."""
    parts = snake_str.split('_')
    return parts[0] + ''.join(p.capitalize() for p in parts[1:])


def _check_duplicate_predicates(mappings: dict) -> tuple[list[str], set[str]]:
    """Detect the same predicate used for different $(columns).

    Returns
    -------
    (errors, conflicting_columns) where *conflicting_columns* is the
    set of column names involved in duplicate-predicate conflicts.
    """
    errors: list[str] = []
    conflicting_cols: set[str] = set()
    for name, mdef in mappings.items():
        if not isinstance(mdef, dict):
            continue
        pred_to_cols: dict[str, set[str]] = {}
        for entry in (mdef.get("po") or []):
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            pred = str(entry[0])
            if pred in ("rdf:type", "a"):
                continue
            cols = set(re.findall(r'\$\(([^)]+)\)', str(entry[1])))
            if cols:
                pred_to_cols.setdefault(pred, set()).update(cols)

        for pred, cols in pred_to_cols.items():
            if len(cols) > 1:
                conflicting_cols.update(cols)
                sorted_cols = sorted(cols)
                # Build concrete alternative-predicate suggestions
                prefix = pred.split(":")[0] if ":" in pred else "ex"
                suggestions = []
                for col in sorted_cols[1:]:
                    camel = _to_camel_case(col)
                    suggestions.append(f"'{prefix}:{camel}' for $({col})")
                suggestion_text = ", ".join(suggestions)
                errors.append(
                    f"In mapping '{name}', predicate '{pred}' is reused "
                    f"for columns {sorted_cols}.  Every column MUST have "
                    f"a UNIQUE predicate name.  Suggested fix: keep "
                    f"'{pred}' for $({sorted_cols[0]}) and use "
                    f"{suggestion_text}.  Alternatively, you may OMIT the "
                    f"secondary column if no suitable predicate exists."
                )
    return errors, conflicting_cols


def _auto_fix_duplicate_predicates(
    yarrrml_str: str,
    mappings: dict,
) -> tuple[str, list[str]]:
    """Deterministically fix duplicate-predicate violations in the YARRRML text.

    For each mapping that reuses the same predicate for multiple columns,
    this keeps the predicate for the first column (alphabetically) and
    renames subsequent columns' predicates to a ``prefix:camelCase`` form
    derived from the column name.

    The fix is a targeted line-level text replacement — only lines
    containing both the offending predicate AND the ``$(column)``
    reference are modified.

    Returns
    -------
    (fixed_yarrrml, list_of_fix_descriptions)
        If no fixes are needed or all attempts fail, returns
        ``(original_yarrrml, [])``.
    """
    fixes: list[str] = []
    result = yarrrml_str

    for mname, mdef in mappings.items():
        if not isinstance(mdef, dict):
            continue

        # Build predicate → list of columns (preserving PO order)
        pred_to_cols: dict[str, list[str]] = {}
        for entry in (mdef.get("po") or []):
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            pred = str(entry[0])
            if pred in ("rdf:type", "a"):
                continue
            for col in re.findall(r'\$\(([^)]+)\)', str(entry[1])):
                pred_to_cols.setdefault(pred, []).append(col)

        # Collect all predicates already in use (to avoid naming collisions)
        all_preds_in_use = set(pred_to_cols.keys())

        for pred, cols in pred_to_cols.items():
            if len(cols) <= 1:
                continue

            # Keep the predicate for the first column alphabetically;
            # rename the rest.
            sorted_cols = sorted(cols)
            prefix = pred.split(":")[0] if ":" in pred else "ex"

            for col in sorted_cols[1:]:
                camel = _to_camel_case(col)
                new_pred = f"{prefix}:{camel}"

                # Avoid collisions with existing predicates
                if new_pred in all_preds_in_use or new_pred == pred:
                    new_pred = f"{prefix}:has{camel[0].upper()}{camel[1:]}"
                if new_pred in all_preds_in_use:
                    continue  # can't safely rename — skip

                # Targeted line replacement: match lines with BOTH the
                # predicate AND $(col).  Use a regex word boundary to
                # avoid partial matches (e.g. gtfs:id inside gtfs:identifier).
                pat = (
                    r'(?<![a-zA-Z0-9_])'
                    + re.escape(pred)
                    + r'(?![a-zA-Z0-9_])'
                )
                lines = result.split('\n')
                applied = False
                for i, line in enumerate(lines):
                    if f'$({col})' in line and re.search(pat, line):
                        new_line = re.sub(pat, new_pred, line, count=1)
                        if new_line != line:
                            lines[i] = new_line
                            applied = True
                            break

                if applied:
                    result = '\n'.join(lines)
                    all_preds_in_use.add(new_pred)
                    fixes.append(
                        f"'{pred}' → '{new_pred}' for $({col}) in '{mname}'"
                    )

    return result, fixes


def _check_redundancy(mappings: dict) -> list[str]:
    """Detect data columns that appear in PO lists of multiple mappings."""
    col_to_mappings: dict[str, set[str]] = {}
    for name, mdef in mappings.items():
        if not isinstance(mdef, dict):
            continue
        for entry in (mdef.get("po") or []):
            if not isinstance(entry, list):
                continue
            # Only flag 3-item entries (data properties).
            # 2-item entries are object-property links — the $(col) in
            # those is used as an identifier, not as duplicated data.
            if len(entry) >= 3:
                for col in re.findall(r'\$\(([^)]+)\)', str(entry[1])):
                    col_to_mappings.setdefault(col, set()).add(name)

    redundant = {c: m for c, m in col_to_mappings.items() if len(m) > 1}
    if not redundant:
        return []

    items = [f"$({c}) in [{', '.join(sorted(m))}]"
             for c, m in sorted(redundant.items())]
    return [
        f"REDUNDANT COLUMNS — these data columns appear in MULTIPLE "
        f"mappings: {'; '.join(items)}.  Each data column must appear "
        f"in EXACTLY ONE mapping.  The primary class should link to "
        f"secondary classes via object properties, NOT duplicate their data."
    ]


def _auto_fix_redundancy(
    yarrrml_str: str,
    data: dict,
    mappings: dict,
) -> tuple[str, list[str]]:
    """Deterministically remove duplicated data columns across mappings.

    Strategy
    --------
    - Compute duplicated columns from 3-item PO entries.
    - Keep each duplicated column in a single "primary" mapping.
      Primary mapping = one with most 3-item data properties.
    - Remove duplicate data-property lines from other mappings.

    This avoids architect retry loops where the model keeps creating
    overlapping secondary mappings (e.g. stop + station duplicates).
    """
    if not mappings:
        return yarrrml_str, []

    # Build column -> mappings map from 3-item PO entries only.
    col_to_mappings: dict[str, set[str]] = {}
    data_prop_counts: dict[str, int] = {}
    for mname, mdef in mappings.items():
        if not isinstance(mdef, dict):
            continue
        count = 0
        for entry in (mdef.get("po") or []):
            if not isinstance(entry, list) or len(entry) < 3:
                continue
            count += 1
            for col in re.findall(r'\$\(([^)]+)\)', str(entry[1])):
                col_to_mappings.setdefault(col, set()).add(mname)
        data_prop_counts[mname] = count

    redundant = {c: ms for c, ms in col_to_mappings.items() if len(ms) > 1}
    if not redundant:
        return yarrrml_str, []

    # Primary mapping: most data props, then lexical tie-break.
    primary = sorted(
        data_prop_counts.items(),
        key=lambda kv: (-kv[1], kv[0])
    )[0][0] if data_prop_counts else sorted(mappings.keys())[0]

    fixes: list[str] = []
    changed = False

    for col, mnames in sorted(redundant.items()):
        keep_mapping = primary if primary in mnames else sorted(mnames)[0]
        for mname in sorted(mnames):
            if mname == keep_mapping:
                continue
            mdef = mappings.get(mname)
            if not isinstance(mdef, dict):
                continue
            po = mdef.get("po") or []
            new_po = []
            removed_here = False
            for entry in po:
                if (
                    isinstance(entry, list)
                    and len(entry) >= 3
                    and f"$({col})" in str(entry[1])
                ):
                    removed_here = True
                    changed = True
                    continue
                new_po.append(entry)
            if removed_here:
                mdef["po"] = new_po
                fixes.append(
                    f"Removed duplicated $({col}) from '{mname}' "
                    f"(kept in '{keep_mapping}')"
                )

    if not changed:
        return yarrrml_str, []

    # Dump the patched YAML back to text.
    try:
        from io import StringIO
        from ruamel.yaml import YAML

        yaml = YAML()
        yaml.default_flow_style = False
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)

        buf = StringIO()
        yaml.dump(data, buf)
        return buf.getvalue().strip(), fixes
    except Exception:
        return yarrrml_str, []


# ────────────────────────────────────────────────────────────────────
# Concrete column-assignment recipe for architect
# ────────────────────────────────────────────────────────────────────

def _build_column_assignment_hint(
    parsed_yaml: dict | None,
    csv_columns: set[str],
    mappings: dict,
) -> str:
    """Build a concrete column-assignment recipe that tells the architect
    exactly which columns currently live where, which are duplicated, and
    which are missing.  This removes ambiguity for the LLM.
    """
    if not mappings or not csv_columns:
        return ""

    lines: list[str] = ["CURRENT COLUMN DISTRIBUTION (for reference):\n"]

    # Where each column currently appears (ALL PO entries — both
    # 3-item data properties and 2-item object-property links).
    col_to_mappings: dict[str, list[str]] = {}
    for name, mdef in mappings.items():
        if not isinstance(mdef, dict):
            continue
        for entry in (mdef.get("po") or []):
            if isinstance(entry, list) and len(entry) >= 2:
                for col in re.findall(r'\$\(([^)]+)\)', str(entry[1])):
                    col_to_mappings.setdefault(col, []).append(name)

    # Show current state per mapping
    mapping_to_cols: dict[str, list[str]] = {}
    for col, mnames in col_to_mappings.items():
        for m in mnames:
            mapping_to_cols.setdefault(m, []).append(col)

    for mname in sorted(mappings.keys()):
        cols = sorted(mapping_to_cols.get(mname, []))
        lines.append(f"  {mname}: {cols if cols else '(no data columns)'}")

    # Highlight duplicates
    duplicated = {c: ms for c, ms in col_to_mappings.items() if len(ms) > 1}
    if duplicated:
        lines.append("\n  [WARN] DUPLICATED (must appear in only ONE mapping):")
        for col, ms in sorted(duplicated.items()):
            lines.append(f"    $({col}) → currently in [{', '.join(ms)}]")

    # Highlight missing
    mapped_cols = set(col_to_mappings.keys())
    missing = sorted(csv_columns - mapped_cols)
    if missing:
        lines.append(f"\n  [WARN] MISSING COLUMNS (must be added): {missing}")

    # Identifier column hint
    # Find the column used in most subject templates
    subj_cols: dict[str, int] = {}
    for mdef in mappings.values():
        if not isinstance(mdef, dict):
            continue
        for col in re.findall(r'\$\(([^)]+)\)', str(mdef.get("s", ""))):
            subj_cols[col] = subj_cols.get(col, 0) + 1
    if subj_cols:
        id_col = max(subj_cols, key=subj_cols.get)
        lines.append(
            f"\n  [INFO] Identifier column '$({id_col})' is used in subject templates — "
            f"add it as a data property in the PRIMARY mapping ONLY, not in secondary mappings."
        )

    lines.append("")
    return "\n".join(lines) + "\n"


# ────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────

def call_refiner_llm(state: AgentState) -> dict:
    """Three-phase refinement:

    Phase 1a (deterministic) — Structural checks: missing prefixes,
        islands, duplicate predicates, redundancy.  **Missing prefixes
        and duplicate predicates are auto-fixed when possible** before
        being reported as errors.
    Phase 1b (deterministic) — Column-coverage audit (tolerant of
        known predicate-conflict columns).
    Phase 2  (LLM)           — Semantic / URI-logic review.

    Returns
    -------
    dict with keys ``feedback`` (str), ``predicate_conflict_cols`` (list),
    and optionally ``fixed_yarrrml`` (str) when an auto-fix was applied.
    """

    yarrrml = state.get("yarrrml_output", "")
    original_yarrrml = yarrrml          # keep a reference for diff detection
    schema_info = state.get("schema_info", {})
    csv_columns = set(schema_info.get("raw", {}).get("columns", []))

    # Columns flagged by earlier retries as having predicate conflicts
    prev_conflict_cols = set(state.get("predicate_conflict_cols", []))

    all_errors: list[str] = []
    current_conflict_cols: set[str] = set()

    # ── Phase 1a: structural checks (run FIRST to learn conflicts) ──
    data = _parse_yarrrml(yarrrml)
    mappings = {}
    if data and isinstance(data.get("mappings"), dict):
        mappings = data["mappings"]

        # ── Auto-fix missing prefixes before other checks ─────────
        ontology_raw = state.get("ontology_info", {}).get("raw", "")
        prefix_fixed, prefix_fixes = _auto_fix_missing_prefixes(
            yarrrml, data, ontology_raw
        )
        if prefix_fixes:
            fixed_data = _parse_yarrrml(prefix_fixed)
            if fixed_data:
                for fix in prefix_fixes:
                    print(f"    [FIX] Auto-fix: {fix}")
                yarrrml = prefix_fixed
                data = fixed_data
                if isinstance(data.get("mappings"), dict):
                    mappings = data["mappings"]

        # Check for remaining undeclared prefixes (ones we couldn't fix)
        prefix_errors, _ = _check_prefix_completeness(data)
        all_errors.extend(prefix_errors)

        # ── Strip unused prefixes (reduces noise for Phase 2 LLM) ──
        cleaned, removed = _strip_unused_prefixes(yarrrml, data)
        if removed:
            cleaned_data = _parse_yarrrml(cleaned)
            if cleaned_data:
                print(f"    [CLEAN] Removed {len(removed)} unused prefix(es): {removed}")
                yarrrml = cleaned
                data = cleaned_data
                if isinstance(data.get("mappings"), dict):
                    mappings = data["mappings"]

        all_errors.extend(_check_islands(mappings))

        dup_errors, current_conflict_cols = _check_duplicate_predicates(mappings)

        # ── Auto-fix duplicate predicates before reporting them ────
        if dup_errors:
            fixed_yarrrml, fixes = _auto_fix_duplicate_predicates(yarrrml, mappings)
            if fixes:
                # Verify the fix actually resolved the duplicates
                fixed_data = _parse_yarrrml(fixed_yarrrml)
                if fixed_data and isinstance(fixed_data.get("mappings"), dict):
                    fixed_mappings = fixed_data["mappings"]
                    dup_errors2, conflict_cols2 = _check_duplicate_predicates(fixed_mappings)
                    if not dup_errors2:
                        # Auto-fix succeeded — adopt the fixed version
                        for fix in fixes:
                            print(f"    [FIX] Auto-fix: {fix}")
                        yarrrml = fixed_yarrrml
                        data = fixed_data
                        mappings = fixed_mappings
                        dup_errors = []
                        current_conflict_cols = conflict_cols2

        all_errors.extend(dup_errors)

        # ── Auto-fix redundant columns before reporting them ───────
        redundancy_errors = _check_redundancy(mappings)
        if redundancy_errors:
            fixed_yarrrml, red_fixes = _auto_fix_redundancy(yarrrml, data, mappings)
            if red_fixes:
                fixed_data = _parse_yarrrml(fixed_yarrrml)
                if fixed_data and isinstance(fixed_data.get("mappings"), dict):
                    fixed_mappings = fixed_data["mappings"]
                    redundancy_errors2 = _check_redundancy(fixed_mappings)
                    if not redundancy_errors2:
                        for fix in red_fixes:
                            print(f"    [FIX] Auto-fix: {fix}")
                        yarrrml = fixed_yarrrml
                        data = fixed_data
                        mappings = fixed_mappings
                        redundancy_errors = []

        all_errors.extend(redundancy_errors)

    # Merge current + previous conflict columns
    all_conflict_cols = current_conflict_cols | prev_conflict_cols

    # ── Phase 1b: column-coverage (tolerant of known conflicts) ──
    if csv_columns:
        mapped = _extract_yarrrml_columns(yarrrml)
        missing = sorted(csv_columns - mapped)
        # Columns that were intentionally omitted to avoid duplicate-
        # predicate deadlocks are acceptable — don't flag them.
        if all_conflict_cols:
            missing = [c for c in missing if c not in all_conflict_cols]
        # Columns that are entirely empty in the CSV cannot be
        # meaningfully mapped — tolerate their omission.
        if missing:
            csv_path = state.get("csv_path", "")
            empty_cols = _detect_empty_columns(csv_path)
            if empty_cols:
                missing = [c for c in missing if c not in empty_cols]
        if missing:
            all_errors.append(
                f"COLUMN COVERAGE FAILURE — {len(missing)} CSV column(s) "
                f"are NOT referenced: [{', '.join(missing)}]."
            )

    # ── Return early if any deterministic check failed ────────
    if all_errors:
        numbered = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(all_errors))

        # ── Build a concrete column-assignment recipe ──────────
        assignment_hint = _build_column_assignment_hint(
            data, csv_columns, mappings
        )

        feedback = (
            f"LOGIC_ERROR: STRUCTURAL PROBLEMS DETECTED\n"
            f"{numbered}\n\n"
            f"{assignment_hint}"
            f"INSTRUCTIONS FOR ARCHITECT:\n"
            f"1. Fix ALL listed problems in a single revision.\n"
            f"2. Each data column → EXACTLY ONE mapping's po: list.\n"
            f"3. The primary class must link to every secondary class "
            f"via a 2-item object-property PO entry.\n"
            f"4. Every column needs a UNIQUE predicate name.\n"
            f"5. If no unique ontology predicate exists for a column, "
            f"you may OMIT that column or use a derived predicate "
            f"(e.g. column 'my_col' → 'prefix:myCol').\n"
            f"6. Output the complete corrected YARRRML."
        )
        return {
            "feedback": feedback,
            "predicate_conflict_cols": sorted(all_conflict_cols),
            "fixed_yarrrml": yarrrml if yarrrml != original_yarrrml else None,
        }

    # ── Phase 2: LLM-based semantic / URI-logic review ────────
    retry_count = state.get("retry_count", 0)
    llm = get_llm(role="refiner", retry_count=retry_count)

    ontology = state.get("ontology_info", {}).get("raw", "No ontology provided.")
    csv_col_list = ", ".join(sorted(csv_columns)) if csv_columns else "unknown"

    prompt = f"""You are a YARRRML mapping validator.  Your ONLY job is to check
whether the mapping below will translate and materialise correctly.

ONTOLOGY CONTEXT (for reference only -- do NOT check prefix declarations here):
{ontology}

CSV COLUMNS:
{csv_col_list}

CURRENT YARRRML:
{yarrrml}

NOTE: Prefix completeness has ALREADY been verified by an automated check.
Do NOT re-check whether prefixes are declared.  Do NOT flag unused prefixes.

CHECK ONLY THESE (all must pass for APPROVED):

1. DATA TYPING
   - Numeric columns -> xsd:integer, xsd:float, or xsd:double
   - Boolean columns -> xsd:boolean
   - Date/time columns -> xsd:date or xsd:dateTime
   - String columns -> xsd:string (or untyped, which defaults to string)
   - Alternative valid type choices are NOT errors (e.g. xsd:float vs xsd:double).

2. rdf:type DECLARATION
   - Every mapping SHOULD have an rdf:type or `a` PO entry.

DO NOT FLAG ANY OF THE FOLLOWING:
- Prefix declarations or unused prefixes (already verified)
- URI template consistency warnings (handled by deterministic checks)
- Missing value / null handling (YARRRML has NO conditional logic)
- Data validation (e.g. "check if numeric columns have valid numbers")
- Empty URI concerns (e.g. "a foreign-key column may be empty")
- Subjective predicate naming preferences
- Controlled vocabulary or enum constraints
- Orphaned entity concerns from optional foreign keys
- Whether something "should be" an IRI vs a literal (both are valid mappings)

If ALL checks above pass, respond with EXACTLY: APPROVED
Otherwise, list ONLY the failing checks as a SHORT bullet-list (max 3 lines).
Do NOT rewrite the YARRRML.  Do NOT explain passing checks.
"""

    response = llm.invoke(prompt)
    llm_feedback = response.content.strip()

    # ── Post-processing: filter out non-actionable LLM feedback ──
    # Despite the prompt instructions, LLMs often flag issues that
    # cannot be fixed in YARRRML or that are already handled by
    # deterministic checks.  Filter these out to prevent infinite loops.
    if "APPROVED" not in llm_feedback.upper():
        llm_feedback = _filter_non_actionable_feedback(llm_feedback)

    return {
        "feedback": llm_feedback,
        "predicate_conflict_cols": sorted(all_conflict_cols),
        "fixed_yarrrml": yarrrml if yarrrml != original_yarrrml else None,
    }


# ────────────────────────────────────────────────────────────────────
# Non-actionable feedback filter
# ────────────────────────────────────────────────────────────────────

# Patterns that indicate a feedback line is NOT actionable in YARRRML.
# These are checked as case-insensitive substrings.
#
# IMPORTANT — only include patterns that reflect genuine YARRRML
# limitations (no conditional logic, no null handling, no data
# validation).  Do NOT add broad phrasing patterns that could
# accidentally suppress legitimate feedback for any dataset.
_NON_ACTIONABLE_PATTERNS = [
    # ── Null / empty value handling (YARRRML has NO conditional logic) ──
    "may be empty",
    "may be null",
    "null or empty",
    "empty or null",
    "null value",
    "could lead to invalid uri",       # speculative URI concern from nulls
    "could result in invalid uri",
    "leading to invalid uri",
    "might result in invalid",
    "might cause broken",
    "broken links",                     # speculative from optional FK
    "no explicit handling of missing",
    "no explicit handling of null",

    # ── Conditional logic (YARRRML doesn't support it) ──
    "conditional logic",
    "without checking if",
    "without checking for",

    # ── Data validation (YARRRML doesn't validate input values) ──
    "no validation that",
    "valid numeric value",
    "valid number",

    # ── Orphaned entities (unavoidable with optional foreign keys) ──
    "orphaned entit",
    "orphan entit",

    # ── Controlled vocabulary / enum (YARRRML can't enforce these) ──
    "controlled vocabulary",
    "enum-like",
    "enum constraint",
    "conform to expected value",

    # ── Harmless style issues (unused prefixes are not errors) ──
    "but not used",
    "declared but not used",

    # ── Subjective data typing preferences (debatable, not errors) ──
    "could be a potential issue",
    "should probably be xsd:",
    "should be a string",
    "should be an integer",
    "gtfs specification",
    "this should probably be",
    "based on typical",
    "which is acceptable",
    "which is correct",

    # ── URI consistency narration (non-actionable in LLM phase) ──
    "uri template consistency",
    "there is no mapping defining subjects",
    "with that exact pattern",

    # ── PREFIX COMPLETENESS hallucinations (handled deterministically) ──
    "prefix completeness",
    "in the ontology context",
    "used in the ontology",
    "not declared in the `prefixes",
    "not declared in the prefixes",
    "prefix is used but not",
    "prefix is not declared",

    # ── Passing / non-applicable checks the LLM narrates instead of omitting ──
    "not applicable",
    "is not applicable",
    "check is not applicable",
    "correctly assigned",
    "correctly applied",
    "correctly typed",
    "data types are correct",
    "is present in the mapping",
    "declaration is present",
    "this check passes",
    "this check might pass",
    "no issues here",
    "satisfies the",

    # ── LLM hedging / uncertainty language (not definitive errors) ──
    "however, re-evaluating",
    "upon closer inspection",
    "depending on how the system",
    "but typically such",
    "failing check",
    "failing checks",
    "the failing check is",
]


def _filter_non_actionable_feedback(feedback: str) -> str:
    """Remove non-actionable items from LLM feedback.

    Splits the feedback into individual bullet/line items and removes
    any that match known non-actionable patterns (null handling, data
    validation, etc.).

    If all items are filtered out, returns "APPROVED".
    """
    lines = feedback.strip().splitlines()
    kept: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check if this line matches any non-actionable pattern
        lower = stripped.lower()
        is_non_actionable = any(
            pattern in lower for pattern in _NON_ACTIONABLE_PATTERNS
        )

        if not is_non_actionable:
            kept.append(line)

    # If nothing actionable remains, auto-approve
    if not kept:
        return "APPROVED"

    # If the only remaining lines are decorative (headers, separators),
    # also auto-approve
    substantive = [
        l for l in kept
        if l.strip() and not l.strip().startswith("---")
        and len(l.strip()) > 5
    ]
    if not substantive:
        return "APPROVED"

    return "\n".join(kept)

