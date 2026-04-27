"""
One-shot YARRRML reference examples for LLM prompts.

Models like Qwen and Mistral are not natively aware of YARRRML syntax.
Including a concrete, correct example dramatically reduces syntax errors
and hallucinated constructs.

These examples are:
  - Dataset-agnostic (use a generic legislative-document domain)
  - Modifiable (edit this file to change what all agents see)
  - Centralized (imported by every sub-agent prompt)

To customize, simply edit the strings below.  Changes propagate to all
agents on the next pipeline run — no other files need to be touched.
"""

# ────────────────────────────────────────────────────────────────────
# FULL ONE-SHOT EXAMPLE — the complete, valid YARRRML file
#
# Demonstrates:
#   • Ontology prefix in subject URIs (not example.com)
#   • Direct IRI references for columns that already contain URIs
#   • URI-template linking for same-CSV foreign keys (no join)
#   • Metadata sub-resource linked FROM the primary entity
#   • Parent entity with inverse link (has_part) and parent_* columns
#   • Property DISTRIBUTION — different mappings get different properties
#   • Same column reused with DIFFERENT predicates across mappings
# ────────────────────────────────────────────────────────────────────

YARRRML_FULL_EXAMPLE = """\
prefixes:
  onto: "http://example.org/ontology#"
  foaf: "http://xmlns.com/foaf/0.1/"
  terms: "http://purl.org/dc/terms/"
  xsd: "http://www.w3.org/2001/XMLSchema#"
  rdf: "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  rdfs: "http://www.w3.org/2000/01/rdf-schema#"

mappings:
  # ── PRIMARY entity ─────────────────────────────────────────
  # Holds: content properties, direct IRI links, link TO metadata,
  # link TO parent.  Does NOT hold admin/bibliographic properties.
  DocumentMapping:
    sources:
      - [documents.csv~csv]
    s: onto:Document/$(id)
    po:
      - [a, onto:Document]
      # Link TO metadata (primary → metadata, NOT the other way)
      - [onto:metadata, onto:Document/$(id)/Metadata~iri]
      # Same-CSV parent link via URI template
      - [onto:isPartOf, onto:Document/$(parent_id)~iri]
      # Content-focused properties (belong in primary, NOT metadata)
      - [onto:content, $(content), xsd:string]
      - [terms:description, $(summary), xsd:string]
      - [onto:wordCount, $(word_count), xsd:integer]
      # IRI columns — mapped directly, no separate mapping needed
      - [terms:publisher, $(publisher)~iri]
      - [terms:language, $(language)~iri]
      - [terms:audience, $(audience)~iri]
      - [terms:creator, $(creator)~iri]
      - [terms:source, $(source)~iri]
      - [terms:created, $(date), xsd:dateTime]
      - [terms:identifier, $(id), xsd:int]

  # ── PARENT entity ──────────────────────────────────────────
  # Subject uses the PARENT's FK column: s: onto:Document/$(parent_id)
  # Uses INVERSE property (has_part instead of is_part_of)
  # Maps parent_* columns here (parent_source → terms:source)
  ParentDocumentMapping:
    sources:
      - [documents.csv~csv]
    s: onto:Document/$(parent_id)
    po:
      - [a, onto:Document]
      # Inverse link: parent has_part child
      - [onto:hasPart, onto:Document/$(id)~iri]
      # Parent-specific columns
      - [terms:source, $(parent_source)~iri]
      - [terms:identifier, $(parent_id), xsd:int]
      # Shared IRI columns (same column, repeated here for the parent)
      - [terms:language, $(language)~iri]
      - [terms:publisher, $(publisher)~iri]
      - [terms:created, $(date), xsd:dateTime]

  # ── METADATA sub-resource ──────────────────────────────────
  # URI = primary subject + /Metadata suffix
  # Holds ADMINISTRATIVE properties (title, subject, jurisdiction)
  # Same column CAN appear with a DIFFERENT predicate:
  #   $(summary) → terms:description in Primary
  #   $(summary) → onto:summary in Metadata (different predicate!)
  MetadataMapping:
    sources:
      - [documents.csv~csv]
    s: onto:Document/$(id)/Metadata
    po:
      - [a, onto:Document]
      - [onto:versionDate, $(date), xsd:dateTime]
      - [onto:localId, $(id), xsd:string]
      - [terms:subject, $(topic), xsd:string]
      - [terms:title, $(title), xsd:string]
      - [onto:jurisdiction, $(jurisdiction), xsd:string]
      - [onto:hasPDF, $(source)~iri]
      - [onto:summary, $(summary), xsd:string]
      - [terms:source, $(parent_source)~iri]
      - [onto:hasAuthority, $(publisher)~iri]
      - [terms:language, $(language)~iri]
"""

# ────────────────────────────────────────────────────────────────────
# GOLDEN RULES — injected into every agent's context
# ────────────────────────────────────────────────────────────────────

GOLDEN_RULES = """\
### GOLDEN RULES FOR LINKED DATA MAPPINGS

**RULE 1 — Templates over Joins (CRITICAL):**
If a foreign key column (like `parent_id`, `publisher`, `language`) is
in the SAME CSV row as the entity, ALWAYS link via a URI template:
  `[predicate, prefix:ClassName/$(fk_column)~iri]`
NEVER use `joins: [child: ..., parent: ...]` for same-CSV references.
Joins are ONLY for linking across DIFFERENT CSV files.

**RULE 2 — Namespace Priority (CRITICAL):**
Subject URIs MUST use the primary ontology prefix and class name:
  `s: prefix:ClassName/$(id)`
Example: `s: podio:ApprovedPolicy/$(id)`
NEVER use `http://example.com/` or `http://example.org/` as namespace.

**RULE 3 — Semantic Node Splitting:**
If the ontology defines an object property (like `lkg:metadata`) whose
range is the same class, create a secondary mapping with a URI suffix:
  Primary:   `s: prefix:Class/$(id)`
  Secondary: `s: prefix:Class/$(id)/Metadata`

**RULE 4 — Direct IRI for URL Columns:**
If a CSV column already contains a full URL or IRI, map it with:
  `[predicate, $(column)~iri]`
Do NOT create a separate mapping for these.

**RULE 5 — Property Distribution (CRITICAL — do NOT duplicate everything):**
When creating multiple mappings (Primary + Metadata + Parent), DISTRIBUTE
properties semantically — do NOT copy all properties into every mapping:
  - **Primary entity**: content/payload properties (content, wordCount,
    description) + all IRI links + link TO metadata + link TO parent.
  - **Metadata sub-resource**: administrative/bibliographic properties
    (title, subject, jurisdiction, localId, versionDate, hasPDF, summary).
  - **Parent entity**: parent-specific columns (parent_source, parent_id)
    + inverse link (has_part) + shared IRI columns.
A column CAN appear in multiple mappings if it uses DIFFERENT predicates
(e.g. $(description) → terms:description in Primary, lkg:summary in Metadata).

**RULE 6 — Parent Entity Mapping:**
If the ontology defines `is_part_of` / `has_part` and the CSV has a
`parent_id` column, create a SEPARATE parent mapping:
  `s: prefix:Class/$(parent_id)`   ← subject uses the PARENT FK column
  `[eli:has_part, prefix:Class/$(id)~iri]`   ← INVERSE link to child
Map `parent_*` columns (like `parent_source`) in the parent mapping.

**RULE 7 — Link Direction:**
The PRIMARY entity links TO the Metadata node, NOT the other way around:
  In PRIMARY:  `[lkg:metadata, prefix:Class/$(id)/Metadata~iri]`  ✓
  In METADATA: (no self-link — metadata does NOT point to itself)  ✓

**RULE 8 — Non-IRI Column Values (CRITICAL):**
Only use `$(column)~iri` when the column contains VALID IRIs/URLs
(starting with http://, https://, or a known prefix like wd:).
If the column contains usernames (e.g. @user9), codes, plain text,
or any value that is NOT a valid IRI, map it as a LITERAL:
  `[predicate, $(column), xsd:string]`
NEVER use `~iri` for columns like user_handle, username, author_name,
category, event_type, locale, etc.  These produce invalid RDF.

**RULE 9 — YAML Flow-Style Lists (CRITICAL for syntax):**
All `po:` entries MUST use inline/flow-style YAML lists:
  CORRECT:  `- [a, schema:Person]`
  CORRECT:  `- [schema:name, $(name), xsd:string]`
  WRONG:    `- - a\\n          - schema:Person`  (block style — breaks parser)
  WRONG:    `- a: schema:Person`  (dict style — breaks parser)
Always write each po entry as a single line: `- [item1, item2, item3]`

**RULE 10 — Metadata Node Type (CRITICAL):**
The Metadata mapping MUST be typed as a DIFFERENT class than the primary.
Append "Metadata" to the primary class name to create the metadata type:
  If primary is `ex:HospitalEncounter` → metadata uses `[a, ex:HospitalEncounterMetadata]`
  If primary is `ex:CreditCardTransaction` → metadata uses `[a, ex:CreditCardTransactionMetadata]`
  NEVER copy the primary class: `[a, ex:HospitalEncounter]` in a MetadataMapping is WRONG.

**RULE 11 — Object Property Direction (domain/range):**
Always check the ontology domain/range for object properties.
If a property has domain=Transaction and range=Person, it MUST appear
on the Transaction mapping pointing TO Person, NOT the reverse.
  `schema:customer` domain=Order range=Person → put on Order mapping.
Do NOT put relationship properties on the wrong entity.

**RULE 12 — Never Use a URL as a Prefix Name (CRITICAL for syntax):**
Prefix names MUST be short alphanumeric identifiers (e.g. ex:, schema:, xsd:).
NEVER use a full URL as the prefix name:
  WRONG:   `"http://example.com/": "http://example.com/"`  ← breaks RML
  CORRECT: `ex: "http://example.com/"`

**RULE 13 — Multi-Value Columns with Same Entity Type (CRITICAL):**
When multiple columns represent separate instances of the same entity
(e.g. `diag_1`, `diag_2`, `diag_3` each holding a diagnosis code),
create a SEPARATE mapping per column with a DISTINCT subject:
  CORRECT:
    DiagnosisMapping_1: s: ex:Diagnosis/$(diag_1)
      po: [ex:icdCode, $(diag_1), xsd:string]
    DiagnosisMapping_2: s: ex:Diagnosis/$(diag_2)
      po: [ex:icdCode, $(diag_2), xsd:string]
    DiagnosisMapping_3: s: ex:Diagnosis/$(diag_3)
      po: [ex:icdCode, $(diag_3), xsd:string]
  WRONG: putting diag_1, diag_2, diag_3 in one mapping with duplicate predicates.
Link each from the primary: `[ex:hasDiagnosis, ex:Diagnosis/$(diag_N)~iri]`

**RULE 14 — Many Columns of the Same Type (e.g. 20+ drug columns):**
When a dataset has many columns representing the same entity type
(e.g. metformin, insulin, glipizide — each column's VALUE is a dosage
status like "Up"/"Down"/"Steady"), create ONE mapping per column with a
composite subject and a STATIC string literal for the entity name:
  MedicationRecord_metformin:
    s: ex:MedicationRecord/$(encounter_id)/metformin
    po:
      - [a, ex:MedicationRecord]
      - [ex:drugName, "metformin", xsd:string]   ← static literal
      - [ex:dosageStatus, $(metformin), xsd:string]  ← column value
      - [ex:forEncounter, ex:PrimaryEntity/$(id)~iri]
Do NOT create 20+ po entries with duplicate predicates like `ex:drugName`.
"""

# ────────────────────────────────────────────────────────────────────
# ROLE-SPECIFIC EXCERPTS
# ────────────────────────────────────────────────────────────────────

EXAMPLE_FOR_PREFIX_MANAGER = f"""\
### YARRRML SYNTAX REFERENCE (one-shot example)
Below is a complete, valid YARRRML file.  Focus on the `prefixes:` block format:
- Each prefix is `name: "URI"` (double-quoted, NO angle brackets).
- Standard prefixes (rdf, rdfs, xsd) are always included.
- Domain-specific prefixes come from the ontology.

```yaml
{YARRRML_FULL_EXAMPLE}
```
"""

EXAMPLE_FOR_ENTITY_BUILDER = f"""\
### YARRRML SYNTAX REFERENCE (one-shot example)
Study the example carefully — notice how properties are DISTRIBUTED:
- Primary gets content properties + IRI links + link TO metadata.
- Metadata gets administrative properties (title, subject, jurisdiction).
- Parent gets parent_* columns + inverse link.
- The SAME column CAN use DIFFERENT predicates in different mappings.
- Primary links TO metadata — metadata does NOT link to itself.

```yaml
{YARRRML_FULL_EXAMPLE}
```

{GOLDEN_RULES}
"""

EXAMPLE_FOR_RELATIONSHIP_LINKER = f"""\
### YARRRML SYNTAX REFERENCE (one-shot example)
Focus on linking patterns:

**Same-CSV linking (ALWAYS use URI templates, NEVER joins):**
- Metadata link (from PRIMARY):  `[onto:metadata, prefix:Class/$(id)/Metadata~iri]`
- Parent link (from PRIMARY):    `[onto:isPartOf, prefix:Class/$(parent_id)~iri]`
- Inverse link (from PARENT):    `[onto:hasPart, prefix:Class/$(id)~iri]`
- IRI column (full URL):         `[predicate, $(column)~iri]`

**Link direction:** PRIMARY → Metadata (never Metadata → itself)

```yaml
{YARRRML_FULL_EXAMPLE}
```

{GOLDEN_RULES}
"""

EXAMPLE_FOR_YARRRML_ARCHITECT = f"""\
### YARRRML SYNTAX REFERENCE (one-shot example)
Your output MUST follow the same format and property distribution pattern.

```yaml
{YARRRML_FULL_EXAMPLE}
```

{GOLDEN_RULES}
"""
