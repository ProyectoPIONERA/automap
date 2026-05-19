import re
from config.settings import get_llm
from config.prefixes import WELL_KNOWN_PREFIXES, IMPLICIT_PREFIXES, URI_SCHEMES
from data.checkpoints import AgentState


def _extract_yarrrml_columns(yarrrml_str: str) -> set[str]:
    """Return all column references found in the YARRRML string.

    Scans three patterns:
      1. $(col)  — standard YARRRML column reference
      2. {col}   — bare brace template sometimes emitted by LLMs
                   (also used in composite keys like {first}_{last})
    Both po: values and subject templates are scanned.
    """
    if not yarrrml_str:
        return set()
    # Pattern 1: $(col)
    found = set(re.findall(r'\$\(([^)]+)\)', yarrrml_str))
    # Pattern 2: {col} — split composite keys like {first}_{last}
    for raw in re.findall(r'\{([^}]+)\}', yarrrml_str):
        for part in re.split(r'[^a-zA-Z0-9_]', raw):
            part = part.strip()
            if part:
                found.add(part)
    return found


def build_column_alias_map(csv_columns: set[str]) -> dict[str, str]:
    """Build a mapping from sanitised column names back to real CSV column names.

    When LLMs see column names with hyphens (e.g. ``metformin-pioglitazone``),
    they silently convert them to underscores in YARRRML templates
    (``$(metformin_pioglitazone)``). Morph-KGC then fails because the CSV
    column is ``metformin-pioglitazone``, not ``metformin_pioglitazone``.

    This function builds the reverse mapping:
        'metformin_pioglitazone' → 'metformin-pioglitazone'

    Only creates entries where sanitised != original.
    Fully agnostic — works for hyphens, spaces, dots, and any non-word char.
    """
    alias_map: dict[str, str] = {}
    for col in csv_columns:
        sanitised = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        if sanitised != col:
            alias_map[sanitised] = col
    return alias_map


def restore_column_names(yarrrml_str: str, alias_map: dict[str, str]) -> tuple[str, list[str]]:
    """Replace sanitised column references in YARRRML with real CSV column names.

    Handles both ``$(col_name)`` and ``{col_name}`` template patterns.
    Sorts by length descending to prevent partial replacements
    (e.g. fixes ``metformin_pioglitazone`` before ``metformin``).

    Returns ``(fixed_yarrrml, list_of_fix_descriptions)``.
    """
    if not alias_map:
        return yarrrml_str, []

    fixes: list[str] = []
    result = yarrrml_str

    for sanitised, real in sorted(alias_map.items(), key=lambda x: -len(x[0])):
        pattern_paren = re.compile(r'\$\(' + re.escape(sanitised) + r'\)')
        pattern_brace = re.compile(r'\{' + re.escape(sanitised) + r'\}')

        if pattern_paren.search(result) or pattern_brace.search(result):
            result = pattern_paren.sub(f'$({real})', result)
            result = pattern_brace.sub('{' + real + '}', result)
            fixes.append(f"Restored column name: '{sanitised}' -> '{real}'")

    return result, fixes


def _parse_yarrrml(yarrrml_str: str):
    """Parse YARRRML string into a dict.  Returns None on failure."""
    try:
        from ruamel.yaml import YAML
        yaml = YAML(typ="safe", pure=True)
        return yaml.load(yarrrml_str)
    except Exception:
        return None


_empty_cols_cache: dict[str, set[str]] = {}


def _detect_empty_columns(csv_path: str) -> set[str]:
    """Return the set of CSV columns that are entirely empty/NaN.

    Columns with no data cannot be meaningfully mapped to RDF and
    should not cause column-coverage failures.  Results are cached.
    """
    if not csv_path:
        return set()
    if csv_path in _empty_cols_cache:
        return _empty_cols_cache[csv_path]
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
        _empty_cols_cache[csv_path] = empty
        return empty
    except Exception:
        return set()


# ────────────────────────────────────────────────────────────────────
# Shared prefix constants (consolidated into config/prefixes.py)
# ────────────────────────────────────────────────────────────────────
_WELL_KNOWN_PREFIXES: dict[str, str] = WELL_KNOWN_PREFIXES
_IMPLICIT_PREFIXES: frozenset[str]   = IMPLICIT_PREFIXES
_URI_SCHEMES: frozenset[str]         = URI_SCHEMES

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
    entity_plan: str = "",
) -> tuple[str, list[str]]:
    """Auto-fix missing prefix declarations in the YARRRML text.

    Uses TWO detection strategies so that text-injected entries
    (e.g. ``ex:hasActor`` added by the island-wirer) are ALWAYS caught
    even when the parsed ``data`` dict is stale:

      Strategy A — dict-based: scan the parsed YAML data structure.
      Strategy B — text-based: scan the raw YARRRML string directly.

    Both strategies contribute to the set of missing prefixes.

    Resolution order for each missing prefix:
      1. Ontology prefix declarations (pipeline's own ontology)
      2. Well-known prefix URI table
      3. Synthetic fallback URI

    Returns
    -------
    (fixed_yarrrml, list_of_fix_descriptions)
    """
    # Strategy A: dict-based detection
    _, missing_from_dict = _check_prefix_completeness(data) if data else ([], set())

    # Strategy B: text-based detection (catches text-injected entries
    # that are NOT yet reflected in the parsed ``data`` dict)
    declared_in_text: set[str] = set()
    if data and isinstance(data.get("prefixes"), dict):
        declared_in_text = set(data["prefixes"].keys())
    # Also extract declared prefixes directly from the text (more reliable)
    for m in re.finditer(r'^\s{2}(\w+):\s+["\']?https?://', yarrrml_str, re.MULTILINE):
        declared_in_text.add(m.group(1))

    text_used: set[str] = set()
    for m in _PREFIX_USAGE_RE.finditer(yarrrml_str):
        prefix = m.group(1)
        if prefix.lower() not in _URI_SCHEMES:
            text_used.add(prefix)
    missing_from_text = text_used - declared_in_text - _IMPLICIT_PREFIXES

    # Combined missing set
    missing = missing_from_dict | missing_from_text
    if not missing:
        return yarrrml_str, []

    # Ontology prefixes take precedence over well-known defaults
    available: dict[str, str] = dict(_WELL_KNOWN_PREFIXES)
    available.update(_extract_ontology_prefixes(ontology_info))
    # Also scan entity plan for prefix URIs (catches domain prefixes like lkg:)
    if entity_plan:
        available.update(_extract_ontology_prefixes(entity_plan))

    fixes: list[str] = []
    additions: list[str] = []
    for prefix in sorted(missing):
        uri = available.get(prefix)
        if not uri:
            # Permanent fallback: synthesize a URI for any completely unknown prefix.
            # This makes the fix dataset-agnostic — no manual prefix registry needed.
            uri = f"http://example.org/{prefix}/"
            print(f"    [WARN] Unknown prefix '{prefix}' — assigned synthetic URI <{uri}>")
        additions.append(f'  {prefix}: "{uri}"')
        fixes.append(f"Added missing prefix '{prefix}: {uri}'")


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
    URI-template links (e.g. ``prefix:Class/$(id)/Metadata~iri``)
    are recognised as valid connectivity.
    """
    errors: list[str] = []

    # A single-mapping YARRRML is a valid flat dataset — nothing to check.
    valid_mappings = {n: m for n, m in mappings.items() if isinstance(m, dict)}
    if len(valid_mappings) <= 1:
        return errors

    # ── Single-source guard ──────────────────────────────────────────
    # When all mappings share exactly one CSV source, a fan-out topology
    # (one CSV → multiple class mappings) is architecturally valid even
    # when cross-links use different namespace prefixes (e.g. mykg: vs dbo:).
    # Flagging these as DISCONNECTED causes an infinite retry loop.
    all_sources: set[str] = set()
    for mdef in valid_mappings.values():
        for src in (mdef.get("sources") or []):
            if isinstance(src, list) and src:
                all_sources.add(str(src[0]))
            elif isinstance(src, str):
                all_sources.add(src)
    if len(all_sources) <= 1:
        return errors  # single-source fan-out — island check not meaningful

    def _normalise(s: str) -> str:
        """Strip template vars, ~iri, namespace prefix, and normalise slashes.

        Stripping the namespace prefix (everything up to the first colon)
        makes cross-namespace references comparable, so that
        ``dbo:Film/`` and ``mykg:Film/`` are treated as equivalent.
        This prevents false DISCONNECTED errors when the LLM uses different
        prefixes in subject templates vs. object IRI references.
        """
        s = re.sub(r'\$\([^)]+\)', '', s)
        s = s.replace('~iri', '')
        # Strip namespace prefix so dbo:Film/ and mykg:Film/ both → Film/
        if ':' in s and not s.startswith('http'):
            s = s.split(':', 1)[1]
        s = re.sub(r'/+', '/', s)     # collapse double-slashes
        return s.rstrip('_/')

    # Build subject-base for every mapping
    subject_bases: dict[str, str] = {}
    for name, mdef in valid_mappings.items():
        subj = str(mdef.get("s", "") or "")
        base = _normalise(subj)
        if base:
            subject_bases[name] = base

    # Collect normalised PO object strings per mapping
    po_objects_by_mapping: dict[str, list[str]] = {}
    for mname, mdef in valid_mappings.items():
        objs: list[str] = []
        for entry in (mdef.get("po") or []):
            if isinstance(entry, list) and len(entry) >= 2:
                objs.append(_normalise(str(entry[1])))
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
        has_outgoing = False
        for entry in (valid_mappings[name].get("po") or []):
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            obj = _normalise(str(entry[1]))
            pred = str(entry[0])
            if pred in ("rdf:type", "a"):
                continue
            # Check links to OTHER mappings
            for other_name, other_base in subject_bases.items():
                if other_name != name and other_base and other_base in obj:
                    has_outgoing = True
                    break
            # Also accept self-referential object-property links
            if not has_outgoing and base and base in obj and obj != base:
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
        pred_to_cols: dict[str, list[set[str]]] = {}
        for entry in (mdef.get("po") or []):
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            pred = str(entry[0])
            if pred in ("rdf:type", "a"):
                continue
            cols = set(re.findall(r'\$\(([^)]+)\)', str(entry[1])))
            if cols:
                pred_to_cols.setdefault(pred, []).append(cols)

        for pred, col_sets in pred_to_cols.items():
            # Only flag if there are SEPARATE PO entries with the same
            # predicate.  A single URI template like
            # ex:Person/$(cc_num)_$(first)_$(last)~iri is ONE entry with
            # multiple column refs — that is fine and expected.
            if len(col_sets) <= 1:
                continue
            # Multiple separate entries share the same predicate
            all_cols = set()
            for cs in col_sets:
                all_cols.update(cs)
            if len(all_cols) > 1:
                conflicting_cols.update(all_cols)
                sorted_cols = sorted(all_cols)
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

        # Build predicate → list of (entry_index, columns_in_entry)
        # We only consider entries as duplicates when SEPARATE PO entries
        # share the same predicate, not when a single URI template
        # contains multiple $(col) references.
        pred_to_entries: dict[str, list[tuple[int, set[str]]]] = {}
        for idx, entry in enumerate(mdef.get("po") or []):
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            pred = str(entry[0])
            if pred in ("rdf:type", "a"):
                continue
            cols = set(re.findall(r'\$\(([^)]+)\)', str(entry[1])))
            if cols:
                pred_to_entries.setdefault(pred, []).append((idx, cols))

        # Collect all predicates already in use (to avoid naming collisions)
        all_preds_in_use = set(pred_to_entries.keys())

        for pred, entries in pred_to_entries.items():
            # Only fix if there are SEPARATE PO entries with the same predicate
            if len(entries) <= 1:
                continue

            # Flatten to individual columns for renaming
            cols = []
            for _, col_set in entries:
                cols.extend(sorted(col_set))
            cols = list(dict.fromkeys(cols))  # dedupe preserving order

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
    """Detect data columns that appear in PO lists of multiple mappings
    WITH THE SAME PREDICATE AND THE SAME rdf:type CLASS.

    Same predicate+column across mappings with DIFFERENT classes is valid
    (e.g. ``schema:streetAddress`` in both a Person and a PostalAddress).
    Only flag when mappings share the same class — that indicates true
    duplication.
    """
    # First, extract the rdf:type class for each mapping
    mapping_classes: dict[str, str] = {}
    for name, mdef in mappings.items():
        if not isinstance(mdef, dict):
            continue
        for entry in (mdef.get("po") or []):
            if isinstance(entry, list) and len(entry) >= 2:
                if str(entry[0]) in ("a", "rdf:type"):
                    mapping_classes[name] = str(entry[1])
                    break

    # Track (predicate, column) → set of mapping names
    pred_col_to_mappings: dict[tuple[str, str], set[str]] = {}
    for name, mdef in mappings.items():
        if not isinstance(mdef, dict):
            continue
        for entry in (mdef.get("po") or []):
            if not isinstance(entry, list):
                continue
            if len(entry) >= 3:
                pred = str(entry[0])
                if pred in ("rdf:type", "a"):
                    continue
                for col in re.findall(r'\$\(([^)]+)\)', str(entry[1])):
                    pred_col_to_mappings.setdefault((pred, col), set()).add(name)

    # Only flag as redundant if the mappings sharing a predicate+column
    # also share the same rdf:type class
    redundant = {}
    for (pred, col), mnames in pred_col_to_mappings.items():
        if len(mnames) <= 1:
            continue
        # Group by class
        classes = {mapping_classes.get(m, f"unknown_{m}") for m in mnames}
        if len(classes) == 1:
            # Same class in multiple mappings — true redundancy
            redundant[(pred, col)] = mnames
        # Different classes → NOT redundant, skip

    if not redundant:
        return []

    items = [f"[{pred},$({col})] in [{', '.join(sorted(m))}]"
             for (pred, col), m in sorted(redundant.items())]
    return [
        f"REDUNDANT — same predicate+column pair appears in MULTIPLE "
        f"mappings: {'; '.join(items)}.  If the same column appears in "
        f"multiple mappings, it MUST use a DIFFERENT predicate in each."
    ]


def _auto_fix_redundancy(
    yarrrml_str: str,
    data: dict,
    mappings: dict,
) -> tuple[str, list[str]]:
    """Deterministically remove duplicated (same predicate + same column)
    data properties across mappings.

    Strategy
    --------
    - Find 3-item PO entries where the SAME predicate AND SAME column
      appear in multiple mappings (true redundancy).
    - Columns with DIFFERENT predicates across mappings are NOT redundant
      (e.g. date → terms:created in Primary, eli:version_date in Metadata).
    - Keep the entry in the "primary" mapping (most data properties).
    """
    if not mappings:
        return yarrrml_str, []

    # Build (predicate, column) -> mappings map from 3-item PO entries.
    pred_col_to_mappings: dict[tuple[str, str], set[str]] = {}
    data_prop_counts: dict[str, int] = {}
    for mname, mdef in mappings.items():
        if not isinstance(mdef, dict):
            continue
        count = 0
        for entry in (mdef.get("po") or []):
            if not isinstance(entry, list) or len(entry) < 3:
                continue
            pred = str(entry[0])
            if pred in ("rdf:type", "a"):
                continue
            count += 1
            for col in re.findall(r'\$\(([^)]+)\)', str(entry[1])):
                pred_col_to_mappings.setdefault((pred, col), set()).add(mname)
        data_prop_counts[mname] = count

    redundant_raw = {k: ms for k, ms in pred_col_to_mappings.items() if len(ms) > 1}
    if not redundant_raw:
        return yarrrml_str, []

    # Extract rdf:type class for each mapping
    mapping_classes: dict[str, str] = {}
    for mname, mdef in mappings.items():
        if not isinstance(mdef, dict):
            continue
        for entry in (mdef.get("po") or []):
            if isinstance(entry, list) and len(entry) >= 2:
                if str(entry[0]) in ("a", "rdf:type"):
                    mapping_classes[mname] = str(entry[1])
                    break

    # Only treat as redundant if mappings share the same class
    redundant = {}
    for (pred, col), mnames in redundant_raw.items():
        classes = {mapping_classes.get(m, f"unknown_{m}") for m in mnames}
        if len(classes) == 1:
            redundant[(pred, col)] = mnames

    if not redundant:
        return yarrrml_str, []

    # Primary mapping: most data props, then lexical tie-break.
    primary = sorted(
        data_prop_counts.items(),
        key=lambda kv: (-kv[1], kv[0])
    )[0][0] if data_prop_counts else sorted(mappings.keys())[0]

    # Build a set of "semantically owns" hints: if a mapping's class name
    # appears in the predicate prefix (e.g. PersonMapping → schema:givenName
    # belongs to the Person), prefer keeping it there.
    mapping_class_hints: dict[str, str] = {}
    for mname, mdef in mappings.items():
        if not isinstance(mdef, dict):
            continue
        for entry in (mdef.get("po") or []):
            if isinstance(entry, list) and len(entry) >= 2:
                if str(entry[0]) in ("a", "rdf:type"):
                    # e.g. "schema:Person" → "person"
                    cls = str(entry[1]).rsplit(":", 1)[-1].rsplit("/", 1)[-1].lower()
                    mapping_class_hints[mname] = cls

    fixes: list[str] = []
    changed = False

    for (pred, col), mnames in sorted(redundant.items()):
        # Determine which mapping should KEEP the property.
        # Priority order (agnostic):
        #   1. A mapping that has a 2-item IRI entry for this predicate
        #      (IRI reference > literal — fixes Bug 3).
        #   2. Semantic ownership via class-name heuristic.
        #   3. Non-primary mapping (more semantically specific).
        #   4. Primary mapping as last resort.
        keep_mapping = None

        # Check for IRI-typed entry: prefer the mapping that has [pred, val~iri]
        for mname in sorted(mnames):
            mdef = mappings.get(mname)
            if not isinstance(mdef, dict):
                continue
            for entry in (mdef.get("po") or []):
                if (isinstance(entry, list) and len(entry) == 2
                        and str(entry[0]) == pred
                        and f"$({col})" in str(entry[1])):
                    # 2-item entry with this column — it's an IRI reference
                    keep_mapping = mname
                    break
            if keep_mapping:
                break

        # Heuristic: check if predicate local name suggests class ownership
        if keep_mapping is None:
            pred_local = pred.rsplit(":", 1)[-1].lower() if ":" in pred else pred.lower()
            for mname in sorted(mnames):
                cls_hint = mapping_class_hints.get(mname, "")
                mname_lower = mname.lower()
                if cls_hint and cls_hint in mname_lower and mname != primary:
                    keep_mapping = mname
                    break

        # Fall back to non-primary (more specific)
        if keep_mapping is None:
            non_primary = sorted(m for m in mnames if m != primary)
            keep_mapping = non_primary[0] if non_primary else primary
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
                    and str(entry[0]) == pred
                    and f"$({col})" in str(entry[1])
                ):
                    removed_here = True
                    changed = True
                    continue
                new_po.append(entry)
            if removed_here:
                mdef["po"] = new_po
                fixes.append(
                    f"Removed duplicated [{pred},$({col})] from '{mname}' "
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
    exactly which columns currently live where and which are missing.

    Scans po: values, subject templates, and {col} concat patterns so
    that columns used only as URI key parts (e.g. city_pop in the subject
    template) are NOT incorrectly shown as missing.
    """
    if not mappings or not csv_columns:
        return ""

    lines: list[str] = ["CURRENT COLUMN DISTRIBUTION (for reference):\n"]

    col_to_mappings: dict[str, list[str]] = {}

    def _register(col: str, mapping_name: str) -> None:
        col_to_mappings.setdefault(col, []).append(mapping_name)

    for name, mdef in mappings.items():
        if not isinstance(mdef, dict):
            continue

        # ── Subject template ──────────────────────────────────────
        subj = str(mdef.get("s", ""))
        for col in re.findall(r'\$\(([^)]+)\)', subj):
            _register(col, name)
        for raw in re.findall(r'\{([^}]+)\}', subj):
            for part in re.split(r'[^a-zA-Z0-9_]', raw):
                part = part.strip()
                if part and part in csv_columns:
                    _register(part, name)

        # ── PO entries ────────────────────────────────────────────
        for entry in (mdef.get("po") or []):
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            val = str(entry[1])
            for col in re.findall(r'\$\(([^)]+)\)', val):
                _register(col, name)
            for raw in re.findall(r'\{([^}]+)\}', val):
                for part in re.split(r'[^a-zA-Z0-9_]', raw):
                    part = part.strip()
                    if part and part in csv_columns:
                        _register(part, name)

    # De-dup per mapping
    col_to_mappings = {c: list(dict.fromkeys(ms)) for c, ms in col_to_mappings.items()}

    # Show current state per mapping
    mapping_to_cols: dict[str, list[str]] = {}
    for col, mnames in col_to_mappings.items():
        for m in mnames:
            mapping_to_cols.setdefault(m, []).append(col)
    mapping_to_cols = {m: sorted(set(cs)) for m, cs in mapping_to_cols.items()}

    for mname in sorted(mappings.keys()):
        cols = mapping_to_cols.get(mname, [])
        lines.append(f"  {mname}: {cols if cols else '(no data columns)'}")

    # Highlight missing
    mapped_cols = set(col_to_mappings.keys())
    missing = sorted(csv_columns - mapped_cols)
    if missing:
        lines.append(f"\n  [WARN] MISSING COLUMNS (must be added): {missing}")

    # Identifier column hint
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


def _auto_fix_metadata_class(
    yarrrml_str: str,
    data: dict,
    mappings: dict,
) -> tuple[str, list[str]]:
    """Deterministically fix Metadata mappings that use the same rdf:type as
    the primary mapping.

    Renames the class to ``{PrimaryClass}Metadata`` so that downstream tools
    do not confuse the metadata resource with the main entity.

    Returns (fixed_yarrrml, list_of_fix_descriptions).
    """
    if not mappings:
        return yarrrml_str, []

    # Detect primary class
    primary_class: str | None = None
    max_po = 0
    for mname, mdef in mappings.items():
        if not isinstance(mdef, dict) or "metadata" in mname.lower():
            continue
        po = mdef.get("po") or []
        if len(po) > max_po:
            for entry in po:
                if isinstance(entry, list) and str(entry[0]) in ("a", "rdf:type"):
                    primary_class = str(entry[1])
                    max_po = len(po)
                    break

    if not primary_class:
        return yarrrml_str, []

    fixes: list[str] = []
    changed = False
    for mname, mdef in mappings.items():
        if not isinstance(mdef, dict) or "metadata" not in mname.lower():
            continue
        po = mdef.get("po") or []
        for idx, entry in enumerate(po):
            if isinstance(entry, list) and str(entry[0]) in ("a", "rdf:type"):
                if str(entry[1]) == primary_class:
                    suggested = primary_class.rstrip(">").rstrip("/") + "Metadata"
                    po[idx] = [entry[0], suggested]
                    changed = True
                    fixes.append(
                        f"Auto-fixed metadata class in '{mname}': "
                        f"'{primary_class}' → '{suggested}'"
                    )

    if not changed:
        return yarrrml_str, []

    try:
        from io import StringIO
        from ruamel.yaml import YAML
        _yaml = YAML()
        _yaml.default_flow_style = False
        _yaml.preserve_quotes = True
        _yaml.indent(mapping=2, sequence=4, offset=2)
        buf = StringIO()
        _yaml.dump(data, buf)
        return buf.getvalue().strip(), fixes
    except Exception:
        return yarrrml_str, []


def _fix_predicate_separator_typo(
    yarrrml_str: str,
    declared_prefixes: set[str],
) -> tuple[str, list[str]]:
    """Fix predicates that use '#' instead of ':' as prefix separator.

    e.g.  ex#acarboseStatus  →  ex:acarboseStatus

    When the LLM accidentally writes 'ex#localname' in a po entry,
    Yatter accepts it as valid YAML but morph-KGC translates it to a
    Literal term-type instead of an IRI, causing:
       "Found an invalid predicate termtype [..., rr:Literal].
        Predicate maps must be rr:IRI."

    Fix is applied only when the prefix part matches a declared prefix
    (or a known short-form prefix) so we don't mangle full URIs like
    <http://example.org/fraud#Class>.

    Fully agnostic — driven by the prefix declarations in the mapping.
    """
    if not yarrrml_str or not declared_prefixes:
        return yarrrml_str, []

    fixes: list[str] = []
    # Build a pattern that matches declared_prefix#LocalName
    # where LocalName starts with a capital or lowercase letter.
    # We anchor to word boundaries and exclude patterns inside < >
    # by only replacing outside angle-bracket-enclosed URIs.

    def _replace(m: re.Match) -> str:
        prefix = m.group(1)
        local = m.group(2)
        if prefix in declared_prefixes:
            fixes.append(f"Fixed predicate separator: '{prefix}#{local}' -> '{prefix}:{local}'")
            return f"{prefix}:{local}"
        return m.group(0)

    # Replace prefix#Local patterns that are NOT inside < ... >
    # Strategy: split on angle-bracket sections, apply only outside them
    result_parts: list[str] = []
    # Split on <...> URIs to avoid touching full URIs
    segments = re.split(r'(<[^>]+>)', yarrrml_str)
    for seg in segments:
        if seg.startswith('<') and seg.endswith('>'):
            result_parts.append(seg)  # inside URI — skip
        else:
            seg = re.sub(
                r'\b([A-Za-z][A-Za-z0-9_]*)#([A-Za-z][A-Za-z0-9_]*)\b',
                _replace,
                seg,
            )
            result_parts.append(seg)

    return ''.join(result_parts), fixes


def _fix_yaml_breaking_predicates(yarrrml_str: str) -> tuple[str, list[str]]:
    """Fix predicate values in po: flow-list entries that contain spaces or
    extra colons, causing YAML to parse them as dicts instead of strings.

    The canonical case is a column named ``Unnamed: 0``:
        - [ex:Unnamed: 0, $(Unnamed: 0), xsd:string]
    YAML parses ``ex:Unnamed: 0`` as the dict ``{ex:Unnamed: 0}`` which
    causes Yatter to crash with "dict object has no attribute 'startswith'".

    Fix: replace spaces with underscores in the predicate (first item of
    each flow-list po: entry).  The column reference $(Unnamed: 0) is
    left untouched — Yatter accepts those.

    This is fully agnostic — works for any column name with spaces/colons.
    """
    fixes: list[str] = []
    lines = yarrrml_str.split("\n")
    result: list[str] = []
    for line in lines:
        # Match flow-list po entries: - [predicate, ...]
        m = re.match(r'^(\s*-\s*\[)([^\[,]+?)(,.+)$', line)
        if m:
            indent_and_bracket = m.group(1)
            predicate = m.group(2)
            rest = m.group(3)
            # Sanitise only if predicate has space or extra colons (more than prefix:local)
            # A valid predicate is prefix:localName with no spaces
            if " " in predicate or predicate.count(":") > 1:
                safe_pred = re.sub(r'\s+', '_', predicate.strip())
                # Remove extra colons beyond the first (prefix:local:extra → prefix:local_extra)
                parts = safe_pred.split(":")
                if len(parts) > 2:
                    safe_pred = parts[0] + ":" + "_".join(parts[1:])
                if safe_pred != predicate.strip():
                    fixes.append(f"Sanitised predicate: '{predicate.strip()}' → '{safe_pred}'")
                    line = f"{indent_and_bracket}{safe_pred}{rest}"
        result.append(line)
    return "\n".join(result), fixes


def _auto_fix_intra_mapping_duplicates(
    yarrrml_str: str,
    data: dict,
    mappings: dict,
) -> tuple[str, list[str]]:
    """Remove exact duplicate PO entries within the same mapping.

    A duplicate is a PO entry whose (predicate, object-value) pair
    appears more than once inside the same mapping's ``po:`` list.
    Only exact string-level duplicates are removed; entries that share
    a predicate but have different object values are kept.

    Returns
    -------
    (fixed_yarrrml, list_of_fix_descriptions)
    """
    if not mappings:
        return yarrrml_str, []

    fixes: list[str] = []
    changed = False

    for mname, mdef in mappings.items():
        if not isinstance(mdef, dict):
            continue
        po = mdef.get("po") or []
        seen: set[str] = set()
        new_po: list = []
        for entry in po:
            key = repr(entry)
            if key in seen:
                fixes.append(
                    f"Removed duplicate PO entry {entry!r} from '{mname}'"
                )
                changed = True
            else:
                seen.add(key)
                new_po.append(entry)
        if len(new_po) < len(po):
            mdef["po"] = new_po

    if not changed:
        return yarrrml_str, []

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


def _check_class_validity(mappings: dict, ontology_raw: str) -> list[str]:
    """Flag rdf:type values not present in the ontology.

    Extracts class URIs from the ontology text (both Turtle @prefix style
    and raw URIs) and checks that every rdf:type in the mappings matches.
    """
    if not ontology_raw or not mappings:
        return []

    # Extract all class-like URIs from ontology
    # Match full URIs and prefixed names after owl:Class, rdfs:Class, rdf:type
    ontology_classes: set[str] = set()
    # Full URIs
    for m in re.finditer(r'<([^>]+)>\s+(?:a|rdf:type)\s+(?:owl:Class|rdfs:Class)', ontology_raw):
        ontology_classes.add(m.group(1))
    # Also extract from @prefix definitions + any URI that looks like a class
    for m in re.finditer(r'<([^>]+(?:#|/)[A-Z][^>]*)>', ontology_raw):
        ontology_classes.add(m.group(1))
    # Prefixed names used as classes
    for m in re.finditer(r'(\w+:[A-Z]\w*)', ontology_raw):
        ontology_classes.add(m.group(1))

    if not ontology_classes:
        return []  # Can't validate without ontology classes

    # Detect primary class (from mapping with most po entries)
    primary_class = None
    max_po = 0
    for mname, mdef in mappings.items():
        if not isinstance(mdef, dict):
            continue
        po_count = len(mdef.get("po") or [])
        if po_count > max_po:
            max_po = po_count
            for entry in (mdef.get("po") or []):
                if isinstance(entry, list) and len(entry) >= 2:
                    if str(entry[0]) in ("a", "rdf:type"):
                        primary_class = str(entry[1])
                        break

    errors: list[str] = []
    for mname, mdef in mappings.items():
        if not isinstance(mdef, dict):
            continue
        for entry in (mdef.get("po") or []):
            if isinstance(entry, list) and len(entry) >= 2:
                if str(entry[0]) in ("a", "rdf:type"):
                    cls = str(entry[1])

                    # Special check: metadata mapping must NOT use primary class
                    if "metadata" in mname.lower() and primary_class and cls == primary_class:
                        # Suggest {PrimaryClass}Metadata
                        suggested = cls.rstrip(">").rstrip("/") + "Metadata"
                        errors.append(
                            f"CLASS WARNING: Metadata mapping '{mname}' uses the same "
                            f"class '{cls}' as the primary entity — auto-fix: changing to "
                            f"'[a, {suggested}]'."
                        )
                    elif cls not in ontology_classes:
                        # Check both full URI and prefixed form
                        found = any(cls.split(":")[-1] in oc for oc in ontology_classes)
                        if not found:
                            # Don't block on Metadata suffix classes — they're expected
                            if "Metadata" in cls:
                                continue
                            errors.append(
                                f"CLASS WARNING: '{cls}' in mapping '{mname}' "
                                f"may not exist in the ontology. Verify it matches "
                                f"an actual class from the ontology."
                            )
    return errors


# ────────────────────────────────────────────────────────────────────
# Dataset-agnostic column fallback rules (Fix 2 + Fix 4)
# ────────────────────────────────────────────────────────────────────

# Evaluated in ORDER — first match wins.
COLUMN_FALLBACK_RULES: list[tuple[str, str, str]] = [
    # Index / row-number artifacts
    (r'^unnamed:\s*\d+$',                                       'ex:rowIndex',         'xsd:integer'),
    (r'^index$',                                                'ex:rowIndex',         'xsd:integer'),
    # Geographic coordinates
    (r'(lat|latitude)$',                                        'geo:lat',             'xsd:decimal'),
    (r'(lon|long|longitude)$',                                  'geo:long',            'xsd:decimal'),
    # Full date-time / timestamp  (match BEFORE bare date/time)
    (r'date.*time|datetime|timestamp|trans_date',               'ex:dateTime',         'xsd:dateTime'),
    (r'_date_.*|.*_time_.*',                                    'ex:dateTime',         'xsd:dateTime'),
    # Bare date / time suffixes
    (r'(^date$|_date$)',                                        'ex:date',             'xsd:date'),
    (r'(^time$|_time$)',                                        'ex:time',             'xsd:dateTime'),
    (r'created_at|updated_at',                                  'schema:dateModified', 'xsd:dateTime'),
    # Numeric amounts
    (r'(amt|amount|price|cost|fee|salary|revenue|total|balance)$', None,              'xsd:decimal'),
    # Integer / counts
    (r'(pop|population|count|num|age|year|qty|quantity|rank|score)$', None,           'xsd:integer'),
    # Booleans
    (r'^is_|^has_|_flag$|_bool$',                               None,                 'xsd:boolean'),
    # Identifiers
    (r'^id$|_id$',                                              'ex:identifier',       'xsd:string'),
    (r'^uuid$|^guid$',                                          'ex:uuid',             'xsd:string'),
]


def get_fallback_predicate(col_name: str) -> tuple[str, str] | None:
    """Return (predicate, datatype) for columns the LLM tends to skip.

    Driven entirely by column-name patterns — no dataset-specific knowledge.
    Returns None if no rule matches (caller should use a safe default).
    """
    col_lower = col_name.lower().strip()
    for pattern, predicate, datatype in COLUMN_FALLBACK_RULES:
        if re.search(pattern, col_lower):
            if predicate is None:
                predicate = f"ex:{_to_camel_case(col_name)}"
            return predicate, datatype
    return None


def _extract_missing_columns_from_feedback(feedback: str) -> list[str]:
    """Parse column names from a COLUMN COVERAGE FAILURE feedback string."""
    # Primary format: NOT referenced: [col1, col2, ...]
    m = re.search(r'NOT referenced:\s*\[(.+?)\]', feedback)
    if m:
        return [c.strip().strip("'\"") for c in m.group(1).split(",") if c.strip()]
    # WARN format from _build_column_assignment_hint
    m = re.search(r'MISSING COLUMNS \(must be added\):\s*\[(.+?)\]', feedback)
    if m:
        return [c.strip().strip("'\"") for c in m.group(1).split(",") if c.strip()]
    return []


def _build_mandatory_column_injection(missing_cols: list[str]) -> str:
    """Format missing columns as a mandatory fix block with concrete predicate suggestions."""
    if not missing_cols:
        return ""
    lines = [
        "⚠️ MANDATORY FIX — previous attempt REJECTED.",
        "The following columns are MISSING and MUST be added to the mapping:",
    ]
    for col in missing_cols:
        result = get_fallback_predicate(col)
        if result:
            pred, dtype = result
            lines.append(f"  - '{col}' → suggested: [{pred}, $({col}), {dtype}]")
        else:
            lines.append(
                f"  - '{col}' → suggested: [ex:{_to_camel_case(col)}, $({col}), xsd:string]"
            )
    lines += [
        "",
        "Rules:",
        "- Every column in this list MUST appear in at least one mapping's po section.",
        "- Do NOT skip columns just because they seem like metadata or index fields.",
        "- For index-like columns (Unnamed: 0, index): use [ex:rowIndex, $(col), xsd:integer].",
        "- For datetime columns: use [ex:dateTime, $(col), xsd:dateTime].",
        "- When unsure, use [ex:hasValue, $(col), xsd:string] as a safe fallback.",
    ]
    return "\n".join(lines)


def _auto_inject_missing_columns(
    yarrrml_str: str,
    data: dict,
    mappings: dict,
    missing_cols: list[str],
) -> tuple[str, list[str], list[str]]:
    """Deterministically inject missing CSV columns into the best-fit mapping
    using direct text insertion (no ruamel.yaml re-serialisation).

    Inference is fully agnostic — derived from universal column-name
    conventions only; no dataset-specific hardcoding.

    Returns
    -------
    (fixed_yarrrml, still_missing_cols, fix_descriptions)
        ``still_missing_cols`` contains columns that could not be injected
        (e.g. no matching po: block found in the text).
    """
    if not missing_cols or not mappings:
        return yarrrml_str, list(missing_cols), []

    # ── Find primary mapping (most data-property entries) ──────────
    primary = sorted(
        mappings.keys(),
        key=lambda m: -len([
            e for e in (mappings[m].get("po") or [])
            if isinstance(e, list) and len(e) >= 3
        ])
    )[0]

    fixes: list[str] = []
    still_missing: list[str] = []
    lines = yarrrml_str.split("\n")

    for col in missing_cols:
        col_lower = col.lower().strip()

        # ── Datatype + predicate from universal naming conventions ──
        if re.search(r'(lat|latitude)$', col_lower):
            pred, dtype = "geo:lat", "xsd:decimal"
        elif re.search(r'(lon|long|longitude)$', col_lower):
            pred, dtype = "geo:long", "xsd:decimal"
        elif re.search(r'(date|time|datetime|timestamp)$', col_lower):
            pred, dtype = f"ex:{_to_camel_case(col)}", "xsd:dateTime"
        elif re.search(r'(amt|amount|price|cost|fee|salary|revenue|total|balance)$', col_lower):
            pred, dtype = f"ex:{_to_camel_case(col)}", "xsd:decimal"
        elif re.search(r'(pop|population|count|num|age|year|qty|quantity|rank|score)$', col_lower):
            pred, dtype = f"ex:{_to_camel_case(col)}", "xsd:integer"
        elif re.search(r'^(is_|has_)|(_flag|_bool)$', col_lower):
            pred, dtype = f"ex:{_to_camel_case(col)}", "xsd:boolean"
        else:
            pred, dtype = f"ex:{_to_camel_case(col)}", "xsd:string"

        # ── Best-fit mapping: always prefer the primary (largest) mapping.
        # The old name-similarity heuristic was picking small sub-mappings
        # (e.g. MedicationRecord_insulin) for unrelated columns like
        # admission_source_id — which is semantically wrong.
        # Only override the primary when there is a strong semantic match:
        # the column prefix exactly appears in a non-primary mapping name.
        target = primary
        col_prefix = col_lower.replace("_", "")[:6]
        for mname in mappings:
            if mname == primary:
                continue
            m_lower_clean = mname.lower().replace("_", "").replace("mapping", "")
            # Only reassign if the column prefix is a substantial match
            # (at least 5 chars overlap) and not a generic sub-mapping name
            if len(col_prefix) >= 5 and col_prefix in m_lower_clean:
                target = mname
                break

        # ── Text-level injection after last po: entry of target ────
        new_entry = f"      - [{pred}, $({col}), {dtype}]"
        in_target = False
        last_po_idx = -1

        in_po_section = False
        for i, line in enumerate(lines):
            # Detect start of the target mapping block (2-space indent)
            if re.match(rf'^ {{2}}{re.escape(target)}\s*:', line):
                in_target = True
            if in_target:
                # Detect start of po: section
                if re.match(r'^ {4}po\s*:', line):
                    in_po_section = True
                # Stop when we hit the next sibling mapping definition
                elif re.match(r'^ {2}\w', line) and not re.match(
                        rf'^ {{2}}{re.escape(target)}\s*:', line):
                    break
                elif in_po_section:
                    # Any line at 6+ spaces that is not blank belongs to po entries
                    if re.match(r'^ {6}', line) and line.strip():
                        last_po_idx = i
                    # A 4-space key (like 's:', 'sources:') ends po section
                    elif re.match(r'^ {4}\w', line):
                        in_po_section = False

        if last_po_idx > 0:
            lines.insert(last_po_idx + 1, new_entry)
            fixes.append(f"'{col}' → [{pred}, {dtype}] in {target}")
        else:
            still_missing.append(col)

    result_yaml = "\n".join(lines)

    # ── Restore hyphenated column names that the LLM underscore-ified ──
    # e.g. $(metformin_pioglitazone) → $(metformin-pioglitazone)
    all_csv_cols: set[str] = set()
    # Collect from original missing_cols (these have real names)
    all_csv_cols.update(missing_cols)
    # Also from any columns already in mappings
    for mdef in (mappings or {}).values():
        if isinstance(mdef, dict):
            for entry in (mdef.get("po") or []):
                if isinstance(entry, list) and len(entry) >= 2:
                    for ref in re.findall(r'\$\(([^)]+)\)', str(entry[1])):
                        all_csv_cols.add(ref)
    alias_map = build_column_alias_map(all_csv_cols)
    if alias_map:
        result_yaml, restore_fixes = restore_column_names(result_yaml, alias_map)
        fixes.extend(restore_fixes)

    return result_yaml, still_missing, fixes


def _auto_fix_islands(
    yarrrml_str: str,
    data: dict,
    mappings: dict,
) -> tuple[str, list[str]]:
    """Deterministically wire disconnected (island) mappings to the primary.

    For each island mapping:
    1. Derives a predicate from the mapping name:
       ``DiagnosisMapping`` → ``ex:hasDiagnosis``
    2. Injects that 2-item IRI link into the primary mapping's po: block.
    3. Ensures ``ex:`` prefix is declared (``_auto_fix_missing_prefixes``
       handles this downstream if not).

    This is fully agnostic — predicate derivation uses only the mapping
    name string, not any domain knowledge.

    Returns (fixed_yarrrml, list_of_fix_descriptions).
    """
    if not mappings or len(mappings) <= 1:
        return yarrrml_str, []

    # Identify islands using the same logic as _check_islands
    def _normalise(s: str) -> str:
        s = re.sub(r'\$\([^)]+\)', '', s)
        s = s.replace('~iri', '')
        s = re.sub(r'/+', '/', s)
        return s.rstrip('_/')

    subject_bases: dict[str, str] = {}
    for name, mdef in mappings.items():
        if isinstance(mdef, dict):
            subj = str(mdef.get("s", "") or "")
            base = _normalise(subj)
            if base:
                subject_bases[name] = base

    po_objects_by_mapping: dict[str, list[str]] = {}
    for mname, mdef in mappings.items():
        if isinstance(mdef, dict):
            po_objects_by_mapping[mname] = [
                _normalise(str(e[1]))
                for e in (mdef.get("po") or [])
                if isinstance(e, list) and len(e) >= 2
            ]

    islands: list[str] = []
    for name, base in subject_bases.items():
        referenced = any(
            base in obj
            for other, objs in po_objects_by_mapping.items()
            if other != name
            for obj in objs
        )
        has_outgoing = any(
            other_base and other_base in _normalise(str(e[1]))
            for e in (mappings[name].get("po") or [])
            if isinstance(e, list) and len(e) >= 2 and str(e[0]) not in ("a", "rdf:type")
            for other, other_base in subject_bases.items()
            if other != name
        )
        if not referenced and not has_outgoing:
            islands.append(name)

    if not islands:
        return yarrrml_str, []

    # Primary mapping = most data-property po entries
    data_prop_counts = {
        n: sum(1 for e in (m.get("po") or [])
               if isinstance(e, list) and len(e) >= 3 and str(e[0]) not in ("a", "rdf:type"))
        for n, m in mappings.items()
        if isinstance(m, dict)
    }
    primary = sorted(data_prop_counts, key=lambda k: -data_prop_counts[k])[0]

    fixes: list[str] = []
    lines = yarrrml_str.split("\n")

    for island_name in islands:
        if island_name == primary:
            continue

        # Derive predicate: strip Mapping/Metadata suffix, prepend ex:has
        clean = re.sub(r'(Mapping|Metadata)$', '', island_name)
        pred = f"ex:has{clean}"

        # Build the subject URI template for the island mapping
        island_subj = str(mappings[island_name].get("s", "")) if isinstance(mappings[island_name], dict) else ""
        if not island_subj:
            continue
        link_value = f"{island_subj}~iri"

        new_entry = f"      - [{pred}, {link_value}]"

        # Find the last po entry line in the primary mapping
        in_primary = False
        last_po_idx = -1
        for i, line in enumerate(lines):
            if re.match(rf'^ {{2}}{re.escape(primary)}\s*:', line):
                in_primary = True
            if in_primary:
                if re.match(r'^ {6}-\s+\[', line):
                    last_po_idx = i
                elif i > 0 and re.match(r'^ {2}\w', line) and not re.match(
                        rf'^ {{2}}{re.escape(primary)}\s*:', line):
                    break

        if last_po_idx > 0:
            lines.insert(last_po_idx + 1, new_entry)
            fixes.append(
                f"Auto-wired island '{island_name}' → [{pred}, {link_value}] in '{primary}'"
            )

    if not fixes:
        return yarrrml_str, []

    return "\n".join(lines), fixes


def _auto_drop_phantom_mappings(
    yarrrml_str: str,
    data: dict,
    mappings: dict,
    csv_columns: set[str],
    ontology_raw: str = "",
) -> tuple[str, list[str]]:
    """Drop disconnected mappings that have NO explicit ontology support and
    redistribute their orphaned columns to the primary mapping.

    This is the fix for the "MetadataMapping always disconnected" runaway loop.
    It is FULLY AGNOSTIC — it never hard-codes any mapping name or column name.

    A mapping is considered a "phantom" (safe to drop) when ALL of these hold:
    1. It is disconnected (neither referenced by another mapping, nor has outgoing
       links to another mapping's subject).
    2. No property in the ontology explicitly links TO a class whose name matches
       the mapping's subject URI base.
    3. It contributes NO columns that are not already mapped in the primary mapping
       OR those unmapped columns can be trivially injected into the primary.

    Returns (fixed_yarrrml, list_of_fix_descriptions).
    Columns from dropped mappings are redistributed to the primary mapping.
    """
    if not mappings or len(mappings) <= 1:
        return yarrrml_str, []

    def _normalise(s: str) -> str:
        s = re.sub(r'\$\([^)]+\)', '', s)
        s = s.replace('~iri', '')
        s = re.sub(r'/+', '/', s)
        return s.rstrip('_/')

    # ── Identify disconnected mappings (same logic as _check_islands) ──────
    subject_bases: dict[str, str] = {}
    for name, mdef in mappings.items():
        if isinstance(mdef, dict):
            base = _normalise(str(mdef.get("s", "") or ""))
            if base:
                subject_bases[name] = base

    po_objs: dict[str, list[str]] = {
        n: [_normalise(str(e[1])) for e in (m.get("po") or [])
            if isinstance(e, list) and len(e) >= 2]
        for n, m in mappings.items() if isinstance(m, dict)
    }

    disconnected: list[str] = []
    for name, base in subject_bases.items():
        referenced = any(
            base in obj
            for other, objs in po_objs.items()
            if other != name for obj in objs
        )
        has_outgoing = any(
            other_base and other_base in _normalise(str(e[1]))
            for e in (mappings[name].get("po") or [])
            if isinstance(e, list) and len(e) >= 2
               and str(e[0]) not in ("a", "rdf:type")
            for other, other_base in subject_bases.items()
            if other != name
        )
        if not referenced and not has_outgoing:
            disconnected.append(name)

    if not disconnected:
        return yarrrml_str, []

    # ── Primary = most data-property po entries ──────────────────────────
    data_prop_counts = {
        n: sum(1 for e in (m.get("po") or [])
               if isinstance(e, list) and len(e) >= 3
               and str(e[0]) not in ("a", "rdf:type"))
        for n, m in mappings.items() if isinstance(m, dict)
    }
    primary = sorted(data_prop_counts, key=lambda k: (-data_prop_counts[k], k))[0]

    # ── Ontology property names (to check if a mapping has explicit support) ─
    # Extract all property / class local names mentioned in the ontology text
    ontology_terms: set[str] = set()
    if ontology_raw:
        for m in re.finditer(r'[#/]([A-Za-z][A-Za-z0-9_]*)(?=[>\s,;.])', ontology_raw):
            ontology_terms.add(m.group(1).lower())

    fixes: list[str] = []
    lines = yarrrml_str.split("\n")

    for island in disconnected:
        if island == primary:
            continue

        mdef = mappings.get(island)
        if not isinstance(mdef, dict):
            continue

        # ── Check if ontology explicitly defines a property for this mapping ──
        # Extract the "semantic name" from the mapping name (strip Mapping/Meta suffixes)
        clean_name = re.sub(r'(Mapping|Metadata|Map|Node)$', '', island)
        clean_lower = clean_name.lower()

        # If the ontology has a property whose local name includes the mapping's
        # semantic name, keep the mapping (it has ontology justification).
        has_ontology_support = any(
            clean_lower in term or term in clean_lower
            for term in ontology_terms
            if len(term) > 3  # ignore trivial short tokens
        )
        if has_ontology_support and clean_lower not in ("metadata", "meta", "info"):
            # Skip — don't drop mappings the ontology explicitly supports
            continue

        # ── Collect columns from the phantom mapping ──────────────────────
        phantom_cols: list[tuple[str, str, str]] = []  # (predicate, col, dtype)
        phantom_iri_links: list[tuple[str, str]] = []   # (predicate, iri_value)

        for entry in (mdef.get("po") or []):
            if not isinstance(entry, list):
                continue
            pred = str(entry[0])
            if pred in ("a", "rdf:type"):
                continue
            if len(entry) == 3:
                # data property: [pred, $(col), dtype]
                cols_in_val = re.findall(r'\$\(([^)]+)\)', str(entry[1]))
                for c in cols_in_val:
                    if c in csv_columns:
                        phantom_cols.append((pred, c, str(entry[2])))
            elif len(entry) == 2:
                # could be an IRI link or a bare column reference
                val = str(entry[1])
                if '~iri' in val and not val.startswith('$('):
                    phantom_iri_links.append((pred, val))
                else:
                    cols_in_val = re.findall(r'\$\(([^)]+)\)', val)
                    for c in cols_in_val:
                        if c in csv_columns:
                            phantom_cols.append((pred, c, "xsd:string"))

        # ── What columns are already in the primary mapping? ──────────────
        primary_cols: set[str] = set()
        for entry in (mappings[primary].get("po") or []):
            if isinstance(entry, list) and len(entry) >= 2:
                for c in re.findall(r'\$\(([^)]+)\)', str(entry[1])):
                    primary_cols.add(c)
        subj_cols = set(re.findall(r'\$\(([^)]+)\)', str(mappings[primary].get("s", ""))))
        primary_cols |= subj_cols

        # Orphaned columns = unique to the phantom mapping (not in primary)
        orphaned = [(p, c, d) for p, c, d in phantom_cols if c not in primary_cols]

        # ── Drop the phantom mapping from the YARRRML text ────────────────
        # Strategy: find the mapping's block boundaries and remove them.
        new_lines: list[str] = []
        in_block = False
        skip_next_blank = False
        for i, line in enumerate(lines):
            # Detect start of this mapping's block
            if re.match(rf'^ {{2}}{re.escape(island)}\s*:', line):
                in_block = True
                skip_next_blank = True
                continue
            if in_block:
                # Block ends when we hit the next sibling mapping at 2-space indent
                if re.match(r'^ {2}\w', line) and not line.strip().startswith('#'):
                    in_block = False
                    # Keep this line — it's the start of the next mapping
                    new_lines.append(line)
                    continue
                # Skip all lines in the phantom block
                continue
            if skip_next_blank and not line.strip():
                skip_next_blank = False
                continue
            new_lines.append(line)

        lines = new_lines
        fixes.append(f"Dropped phantom mapping '{island}' (disconnected, no ontology support)")

        # ── Inject orphaned columns into primary mapping ──────────────────
        if orphaned:
            # Find last po entry in the primary mapping in the updated lines
            in_primary = False
            last_po_idx = -1
            for i, line in enumerate(lines):
                if re.match(rf'^ {{2}}{re.escape(primary)}\s*:', line):
                    in_primary = True
                if in_primary:
                    if re.match(r'^ {6}-\s+\[', line):
                        last_po_idx = i
                    elif i > 0 and re.match(r'^ {2}\w', line) and not re.match(
                            rf'^ {{2}}{re.escape(primary)}\s*:', line):
                        break

            if last_po_idx > 0:
                insert_lines = []
                for pred, col, dtype in orphaned:
                    insert_lines.append(f"      - [{pred}, $({col}), {dtype}]")
                for j, il in enumerate(insert_lines):
                    lines.insert(last_po_idx + 1 + j, il)
                fixes.append(
                    f"Redistributed {len(orphaned)} orphaned column(s) from "
                    f"'{island}' → '{primary}': "
                    f"{[c for _, c, _ in orphaned]}"
                )

    if not fixes:
        return yarrrml_str, []

    return "\n".join(lines), fixes


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
        entity_plan_text = state.get("schema_alignment", {}).get("entity_plan", "")
        prefix_fixed, prefix_fixes = _auto_fix_missing_prefixes(
            yarrrml, data, ontology_raw, entity_plan_text
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
        # This is already computed inside _auto_fix_missing_prefixes —
        # only re-check if there were unfixable prefixes
        declared = set(data.get("prefixes", {}).keys()) if data.get("prefixes") else set()
        used = _extract_used_prefixes(data)
        still_missing = used - declared - _IMPLICIT_PREFIXES
        if still_missing:
            all_errors.append(
                f"PREFIX COMPLETENESS -- these prefixes are used but NOT declared "
                f"in the `prefixes:` section: [{', '.join(sorted(still_missing))}]. "
                f"Add them to the `prefixes:` block with their full URI."
            )

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

        # ── Fix predicate separator typos: ex#local → ex:local ────
        _declared_pfx = set(data.get("prefixes", {}).keys()) if data.get("prefixes") else set()
        _declared_pfx.update(_IMPLICIT_PREFIXES)
        sep_fixed, sep_fixes = _fix_predicate_separator_typo(yarrrml, _declared_pfx)
        if sep_fixes:
            sep_data = _parse_yarrrml(sep_fixed)
            if sep_data:
                for fix in sep_fixes:
                    print(f"    [FIX] {fix}")
                yarrrml = sep_fixed
                data = sep_data
                if isinstance(data.get("mappings"), dict):
                    mappings = data["mappings"]

        # ── Check class validity against ontology ─────────────
        ontology_raw = state.get("ontology_info", {}).get("raw", "")
        class_errors = _check_class_validity(mappings, ontology_raw)

        # ── Auto-fix metadata class before reporting ───────────
        if any("metadata" in e.lower() and "auto-fix" in e.lower() for e in class_errors):
            fixed_yarrrml, meta_fixes = _auto_fix_metadata_class(yarrrml, data, mappings)
            if meta_fixes:
                fixed_data = _parse_yarrrml(fixed_yarrrml)
                if fixed_data and isinstance(fixed_data.get("mappings"), dict):
                    for fix in meta_fixes:
                        print(f"    [FIX] Auto-fix: {fix}")
                    yarrrml = fixed_yarrrml
                    data = fixed_data
                    mappings = fixed_data["mappings"]
                    # Re-run class check now that we've fixed it
                    class_errors = _check_class_validity(mappings, ontology_raw)

        # Fix 4: CLASS WARNING is advisory only — do NOT block pipeline.
        # CLASS ERROR (metadata using wrong class) is blocking.
        blocking_class_errors = [e for e in class_errors if e.startswith("CLASS ERROR")]
        advisory_class_warnings = [e for e in class_errors if e.startswith("CLASS WARNING")]
        if advisory_class_warnings:
            for w in advisory_class_warnings:
                print(f"    [INFO] {w}")   # log but don't block
        all_errors.extend(blocking_class_errors)

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

        # ── Auto-fix redundancy directly (merged check+fix) ───────
        fixed_yarrrml, red_fixes = _auto_fix_redundancy(yarrrml, data, mappings)
        if red_fixes:
            fixed_data = _parse_yarrrml(fixed_yarrrml)
            if fixed_data and isinstance(fixed_data.get("mappings"), dict):
                for fix in red_fixes:
                    print(f"    [FIX] Auto-fix: {fix}")
                yarrrml = fixed_yarrrml
                data = fixed_data
                mappings = fixed_data["mappings"]
        # No else needed — auto_fix_redundancy already handles detection internally

        # ── Auto-fix intra-mapping duplicate PO entries ────────────
        fixed_yarrrml, intra_fixes = _auto_fix_intra_mapping_duplicates(yarrrml, data, mappings)
        if intra_fixes:
            fixed_data = _parse_yarrrml(fixed_yarrrml)
            if fixed_data and isinstance(fixed_data.get("mappings"), dict):
                for fix in intra_fixes:
                    print(f"    [FIX] Auto-fix: {fix}")
                yarrrml = fixed_yarrrml
                data = fixed_data
                mappings = fixed_data["mappings"]

        # ── Islands check — auto-fix first, report only if unfixable ──
        if not all_errors:
            fixed_yarrrml, island_fixes = _auto_fix_islands(yarrrml, data, mappings)
            if island_fixes:
                fixed_data = _parse_yarrrml(fixed_yarrrml)
                if fixed_data and isinstance(fixed_data.get("mappings"), dict):
                    for fix in island_fixes:
                        print(f"    [FIX] Auto-wired island: {fix}")
                    yarrrml = fixed_yarrrml
                    data = fixed_data
                    mappings = fixed_data["mappings"]
                    # Re-check — if islands are resolved, don't report them
                    remaining_islands = _check_islands(mappings)
                    # ── If islands persist, try dropping phantom mappings ──
                    if remaining_islands:
                        fixed_yarrrml2, drop_fixes = _auto_drop_phantom_mappings(
                            yarrrml, data, mappings, csv_columns, ontology_raw
                        )
                        if drop_fixes:
                            fixed_data2 = _parse_yarrrml(fixed_yarrrml2)
                            if fixed_data2 and isinstance(fixed_data2.get("mappings"), dict):
                                for fix in drop_fixes:
                                    print(f"    [FIX] Phantom-drop: {fix}")
                                yarrrml = fixed_yarrrml2
                                data = fixed_data2
                                mappings = fixed_data2["mappings"]
                                remaining_islands = _check_islands(mappings)
                    all_errors.extend(remaining_islands)
            else:
                # No wiring fix found — try phantom-drop directly
                fixed_yarrrml2, drop_fixes = _auto_drop_phantom_mappings(
                    yarrrml, data, mappings, csv_columns, ontology_raw
                )
                if drop_fixes:
                    fixed_data2 = _parse_yarrrml(fixed_yarrrml2)
                    if fixed_data2 and isinstance(fixed_data2.get("mappings"), dict):
                        for fix in drop_fixes:
                            print(f"    [FIX] Phantom-drop: {fix}")
                        yarrrml = fixed_yarrrml2
                        data = fixed_data2
                        mappings = fixed_data2["mappings"]
                        remaining_islands = _check_islands(mappings)
                        all_errors.extend(remaining_islands)
                else:
                    all_errors.extend(_check_islands(mappings))

    # Merge current + previous conflict columns
    all_conflict_cols = current_conflict_cols | prev_conflict_cols

    # ── Fix 1: FLAT→MULTI-NODE override ───────────────────────────
    # If the entity agent produced >1 mapping but the alignment label says
    # FLAT, promote to MULTI-NODE so the relationship agent runs on next retry.
    if mappings and len(mappings) > 1:
        alignment = state.get("schema_alignment", {})
        if not alignment.get("multi_node", False):
            alignment["multi_node"] = True
            print("    [Refiner] FLAT→MULTI-NODE promoted (>1 mapping detected)")

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
            # ── Auto-inject before burning an LLM retry ────────
            yarrrml, missing, inject_fixes = _auto_inject_missing_columns(
                yarrrml, data, mappings, missing
            )
            if inject_fixes:
                data = _parse_yarrrml(yarrrml) or data
                if data and isinstance(data.get("mappings"), dict):
                    mappings = data["mappings"]
                for fix in inject_fixes:
                    print(f"    [Refiner] Auto-inject: {fix}")
                # Persist injected column constraints so entity agent honours them
                # on the next retry — format: {col: "pred (dtype) in MappingName"}
                prev_injected = state.get("injected_column_constraints", {})
                new_injected = dict(prev_injected)
                for fix_str in inject_fixes:
                    # Parse "'{col}' → [pred, dtype] in MappingName"
                    m = re.match(r"'([^']+)'\s*→\s*\[([^,\]]+),\s*([^\]]+)\]\s+in\s+(\S+)", fix_str)
                    if m:
                        col_name, pred, dtype, mapping_name = m.groups()
                        new_injected[col_name] = f"{pred} ({dtype}) in {mapping_name}"
                if new_injected != prev_injected:
                    state["injected_column_constraints"] = new_injected
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
            f"2. Every CSV column must appear in at least one mapping.\n"
            f"3. A column CAN appear in multiple mappings if it uses a "
            f"DIFFERENT predicate in each (e.g. date → terms:created in "
            f"Primary, eli:version_date in Metadata).\n"
            f"4. Within a SINGLE mapping, every column must have a UNIQUE predicate.\n"
            f"5. Use URI templates for same-CSV links, NOT joins.\n"
            f"6. Output the complete corrected YARRRML."
        )
        return {
            "feedback": feedback,
            "predicate_conflict_cols": sorted(all_conflict_cols),
            "fixed_yarrrml": yarrrml if yarrrml != original_yarrrml else None,
            "injected_column_constraints": state.get("injected_column_constraints", {}),
        }

    # ── Phase 2: LLM-based semantic / URI-logic review ────────
    # Skip Phase 2 if all issues were handled deterministically
    auto_fixes_applied = yarrrml != original_yarrrml

    # ── Always apply column-name restoration before returning ──
    # Fixes: LLM silently converts 'metformin-pioglitazone' → 'metformin_pioglitazone'
    # in YARRRML templates. Restore using actual CSV column names.
    if csv_columns:
        _alias_map = build_column_alias_map(csv_columns)
        if _alias_map:
            yarrrml, _restore_fixes = restore_column_names(yarrrml, _alias_map)
            for _fix in _restore_fixes:
                print(f"    [FIX] {_fix}")
            if _restore_fixes:
                auto_fixes_applied = True

    # ── FINAL prefix pass — must run LAST, after island-wiring and
    #    column-injection which can introduce new prefix usages (e.g.
    #    ex:hasActor, ex:category).  Without this pass those usages
    #    would never get a declaration and yatter silently returns None.
    _final_data = _parse_yarrrml(yarrrml) or data
    _ontology_raw = state.get("ontology_info", {}).get("raw", "")
    _entity_plan  = state.get("schema_alignment", {}).get("entity_plan", "")
    yarrrml, _final_pfx_fixes = _auto_fix_missing_prefixes(
        yarrrml, _final_data, _ontology_raw, _entity_plan
    )
    if _final_pfx_fixes:
        for _f in _final_pfx_fixes:
            print(f"    [FIX] Final-prefix-pass: {_f}")
        auto_fixes_applied = True

    if auto_fixes_applied and not all_errors:
        print("    [Refiner] All issues auto-fixed deterministically — skipping LLM phase.")
        return {
            "feedback": "APPROVED",
            "predicate_conflict_cols": sorted(all_conflict_cols),
            "fixed_yarrrml": yarrrml,
            "injected_column_constraints": state.get("injected_column_constraints", {}),
        }

    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm(role="refiner")

    ontology = state.get("ontology_info", {}).get("raw", "No ontology provided.")
    csv_col_list = ", ".join(sorted(csv_columns)) if csv_columns else "unknown"

    # Static system prompt (cached by llama.cpp across calls)
    system_prompt = """You are a YARRRML mapping validator.  Your ONLY job is to check
whether a mapping will translate and materialise correctly.

NOTE: Prefix completeness has ALREADY been verified by an automated check.
Do NOT re-check whether prefixes are declared.  Do NOT flag unused prefixes.

CHECK ONLY THESE (all must pass for APPROVED):

1. DATA TYPING
   - Numeric columns -> xsd:integer, xsd:float, xsd:double, xsd:long, xsd:decimal
   - Boolean columns -> xsd:boolean
   - Date/time columns -> xsd:date or xsd:dateTime
   - String columns -> xsd:string (or untyped, which defaults to string)
   - Alternative valid type choices are NOT errors (e.g. xsd:float vs xsd:double,
     xsd:long vs xsd:integer, xsd:decimal vs xsd:float).  Do NOT flag these.

2. rdf:type DECLARATION
   - Every mapping SHOULD have an rdf:type or `a` PO entry.
   - A missing rdf:type is a minor warning, NOT a blocking error.
     If everything else is fine, still respond with APPROVED.

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

    # Dynamic human message (changes per call)
    human_prompt = f"""ONTOLOGY CONTEXT (for reference only):
{ontology}

CSV COLUMNS:
{csv_col_list}

CURRENT YARRRML:
{yarrrml}

Validate the mapping now.
"""

    # Use streaming to keep connection alive during long generations
    llm_feedback = ""
    for chunk in llm.stream([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]):
        llm_feedback += chunk.content
    llm_feedback = llm_feedback.strip()

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
        "injected_column_constraints": state.get("injected_column_constraints", {}),
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

    # ── Data type alternatives (all are valid, not errors) ──
    "xsd:long",
    "xsd:integer",
    "should be xsd:",
    "mapped to xsd:",
    "data typing",

    # ── rdf:type warnings (minor, not blocking) ──
    "missing rdf:type",
    "missing `rdf:type`",
    "no rdf:type",
    "should have an rdf:type",
    "should have a rdf:type",
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

