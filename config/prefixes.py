"""
config/prefixes.py — Single source of truth for well-known prefix → URI mappings.

All pipeline modules that need to resolve an unknown prefix should import from here
instead of maintaining their own copies.  This eliminates the three duplicate
``_WELL_KNOWN_PREFIXES`` dictionaries that previously existed in:
  - agents/refiner_agent.py
  - agents/yarrrml_coordinator.py
  - graph/nodes.py

NOTE: Only include genuinely *well-known*, publicly-standardised vocabularies.
      Do NOT add dataset-specific prefixes here (e.g. ``lkg``, ``podio``).
      Those should be declared in the ontology file and auto-detected from there.
"""

WELL_KNOWN_PREFIXES: dict[str, str] = {
    # ── Core RDF/OWL ──────────────────────────────────────────────
    "rdf":      "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs":     "http://www.w3.org/2000/01/rdf-schema#",
    "owl":      "http://www.w3.org/2002/07/owl#",
    "xsd":      "http://www.w3.org/2001/XMLSchema#",
    # ── Common vocabularies ───────────────────────────────────────
    "schema":   "http://schema.org/",
    "schema1":  "http://schema.org/",   # alias emitted by some LLMs
    "foaf":     "http://xmlns.com/foaf/0.1/",
    "dc":       "http://purl.org/dc/elements/1.1/",
    "dcterms":  "http://purl.org/dc/terms/",
    "dct":      "http://purl.org/dc/terms/",
    "terms":    "http://purl.org/dc/terms/",
    "skos":     "http://www.w3.org/2004/02/skos/core#",
    # ── Geo / sensor ─────────────────────────────────────────────
    "geo":      "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "wgs":      "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "sosa":     "http://www.w3.org/ns/sosa/",
    "ssn":      "http://www.w3.org/ns/ssn/",
    # ── Provenance / cataloguing ──────────────────────────────────
    "prov":     "http://www.w3.org/ns/prov#",
    "dcat":     "http://www.w3.org/ns/dcat#",
    "void":     "http://rdfs.org/ns/void#",
    "time":     "http://www.w3.org/2006/time#",
    # ── Organisation / contact ────────────────────────────────────
    "vcard":    "http://www.w3.org/2006/vcard/ns#",
    "org":      "http://www.w3.org/ns/org#",
    "gr":       "http://purl.org/goodrelations/v1#",
    # ── Public/transport ─────────────────────────────────────────
    "gtfs":     "http://vocab.gtfs.org/terms#",
    # ── European legislation ──────────────────────────────────────
    "eli":      "http://data.europa.eu/eli/ontology#",
    # ── DBpedia ───────────────────────────────────────────────────
    "dbo":      "http://dbpedia.org/ontology/",
    "dbr":      "http://dbpedia.org/resource/",
    "dbp":      "http://dbpedia.org/property/",
    # ── Wikidata ──────────────────────────────────────────────────
    "wikidata": "http://www.wikidata.org/entity/",
    "wd":       "http://www.wikidata.org/entity/",
    "wdt":      "http://www.wikidata.org/prop/direct/",
    # ── RML / YARRRML internals ───────────────────────────────────
    "rr":       "http://www.w3.org/ns/r2rml#",
    "rml":      "http://semweb.mmlab.be/ns/rml#",
    "ql":       "http://semweb.mmlab.be/ns/ql#",
    # ── Generic example namespace (last resort fallback) ─────────
    "ex":       "http://example.org/",
}

# Prefixes that YARRRML/Yatter treats as implicitly available —
# they do NOT need to be declared in the ``prefixes:`` block.
IMPLICIT_PREFIXES: frozenset[str] = frozenset({"xsd", "rdf", "rdfs"})

# URI schemes — never treat these tokens as prefix names when scanning text.
URI_SCHEMES: frozenset[str] = frozenset({"http", "https", "ftp", "urn", "mailto", "file"})

