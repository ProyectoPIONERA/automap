import pandas as pd
from rdflib import Graph, RDF, RDFS, OWL

def get_csv_schema(path: str):
    """Tool to get headers and a 3-row sample for semantic context."""
    df = pd.read_csv(path, nrows=3)
    return {
        "columns": df.columns.tolist(),
        "sample": df.to_dict(orient='records')
    }


def get_ontology_subgraph(path: str, keywords: list):
    """Extract a structured ontology summary that preserves classes,
    properties, prefixes, and domain/range information.

    The previous keyword-filter approach lost most of the ontology because
    CSV column names (e.g. ``stop_id``) rarely appear verbatim in ontology
    URIs (e.g. ``gtfs:id``).  This version extracts the full schema
    structure so the LLM can use the correct prefixes and predicates.
    """
    g = Graph()
    g.parse(path)

    sections: list[str] = []

    # ── 1. Extract namespace prefixes ────────────────────────────
    prefixes: list[str] = []
    for prefix, uri in g.namespaces():
        if prefix and str(uri) not in (
            "http://www.w3.org/XML/1998/namespace",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        ):
            prefixes.append(f"  @prefix {prefix}: <{uri}> .")
    if prefixes:
        sections.append("PREFIXES:\n" + "\n".join(sorted(prefixes)))

    # ── 2. Extract class definitions ─────────────────────────────
    classes: list[str] = []
    for cls in g.subjects(RDF.type, OWL.Class):
        uri = str(cls)
        label = _first_literal(g, cls, RDFS.label) or ""
        comment = _first_literal(g, cls, RDFS.comment) or ""
        parent = _first_uri_local(g, cls, RDFS.subClassOf)
        line = f"  Class: <{uri}>"
        if label:
            line += f"  label=\"{label}\""
        if parent:
            line += f"  subClassOf={parent}"
        if comment:
            line += f"  — {comment[:120]}"
        classes.append(line)
    if classes:
        sections.append("CLASSES:\n" + "\n".join(sorted(classes)))

    # ── 3. Extract property definitions ──────────────────────────
    props: list[str] = []
    for prop_type, prop_label in [
        (OWL.DatatypeProperty, "DatatypeProperty"),
        (OWL.ObjectProperty, "ObjectProperty"),
    ]:
        for prop in g.subjects(RDF.type, prop_type):
            uri = str(prop)
            label = _first_literal(g, prop, RDFS.label) or ""
            comment = _first_literal(g, prop, RDFS.comment) or ""
            domain = _first_uri_local(g, prop, RDFS.domain)
            rng = _first_uri_local(g, prop, RDFS.range)
            line = f"  {prop_label}: <{uri}>"
            if label:
                line += f"  label=\"{label}\""
            if domain:
                line += f"  domain={domain}"
            if rng:
                line += f"  range={rng}"
            if comment:
                line += f"  — {comment[:100]}"
            props.append(line)
    if props:
        sections.append("PROPERTIES:\n" + "\n".join(sorted(props)))

    # ── 4. Keyword-matched triples (bonus context) ───────────────
    keywords_lower = [k.lower() for k in keywords]
    matched: list[str] = []
    for s, p, o in g:
        triple_str = f"{s} {p} {o}".lower()
        if any(kw in triple_str for kw in keywords_lower):
            matched.append(f"  {s} {p} {o} .")
    if matched:
        # Deduplicate and limit
        unique = sorted(set(matched))[:50]
        sections.append("KEYWORD-MATCHED TRIPLES:\n" + "\n".join(unique))

    return "\n\n".join(sections) if sections else "(No ontology information extracted)"


def _first_literal(g: Graph, subject, predicate) -> str | None:
    """Return the first English or untagged literal for a subject+predicate."""
    for obj in g.objects(subject, predicate):
        s = str(obj)
        # Prefer English
        if hasattr(obj, 'language') and obj.language == 'en':
            return s
    # Fallback: any literal
    for obj in g.objects(subject, predicate):
        return str(obj)
    return None


def _first_uri_local(g: Graph, subject, predicate) -> str | None:
    """Return the local name of the first URI object, or None."""
    for obj in g.objects(subject, predicate):
        uri = str(obj)
        if '#' in uri:
            return uri.rsplit('#', 1)[-1]
        if '/' in uri:
            return uri.rsplit('/', 1)[-1]
    return None
