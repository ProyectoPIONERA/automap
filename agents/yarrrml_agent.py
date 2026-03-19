from config.settings import get_llm
import os
import re
from dotenv import load_dotenv

load_dotenv()


def call_yarrrml_architect_llm(state: dict):
    retry_count = state.get("retry_count", 0)

    # Use role-specific settings with progressive temperature on retries
    llm = get_llm(role="yarrrml_architect", retry_count=retry_count)

    # ── Derive source path dynamically from state ────────────────
    # state["csv_path"] comes from the INPUT_CSV_PATH env variable
    # e.g. "data/input/weather_cleaned.csv"
    csv_path = state["csv_path"]                    # full relative path
    csv_name = os.path.basename(csv_path)           # just "weather_cleaned.csv"
    csv_source = f"{csv_path}~csv"                  # "data/input/weather_cleaned.csv~csv"

    base_uri = state.get("base_uri", "http://example.org/")
    mapping = state["mapping_plan"].get("analysis", "")
    ontology = state["ontology_info"].get("raw", "")

    feedback = state.get("feedback", "")
    previous_yarrrml = state.get("yarrrml_output", "")
    feedback_section = ""

    if feedback and "PASSED" not in feedback and "APPROVED" not in feedback:
        feedback_section = f"""
### CRITICAL: FIX PREVIOUS ERRORS
Your last output failed validation.
ERROR LOG: {feedback}

Instructions for fix:
1. Check for missing colons or incorrect indentation.
2. Ensure all referenced $(columns) exist in the CSV schema.
3. Do not include any explanation text.
"""
        # Include the previous attempt so the architect can see what to fix
        if previous_yarrrml:
            feedback_section += f"""
### YOUR PREVIOUS ATTEMPT (fix the problems listed above):
{previous_yarrrml}
"""
        if "found unexpected ':'" in feedback or "startswith" in feedback or "not valid" in feedback:
            feedback_section += f"""
YAML QUOTING FIX: You MUST wrap ALL values in double quotes, including:
- Prefix URIs: ex: "http://example.org/"
- Subject URIs: s: "ex:Class/$(id)"
- Every item in po: lists: ["ex:pred", "$(col)", "xsd:type"]
Do NOT use angle brackets <> for prefix values.
SOURCE FORMAT FIX: Use exactly this format (no quotes around the source string):
  sources:
    - [{csv_source}]
"""

        if "concatenate str" in feedback or ("list" in feedback.lower() and "str" in feedback.lower()):
            feedback_section += f"""
SUBJECT FORMAT FIX: The `s:` key MUST be a SINGLE quoted string, NOT a list.
WRONG:
  s:
    - ["ex:entity/", "$(id)"]
WRONG:
  s: ["ex:entity/", "$(id)"]
CORRECT:
  s: "{base_uri}entity/$(id)"
The entire subject URI template must be ONE string with $(column) embedded directly.
"""

        if "flow sequence" in feedback or "~csv" in feedback:
            feedback_section += f"""
SOURCE FORMAT FIX: The source MUST be exactly this (no quotes, no doubled paths):
  sources:
    - [{csv_source}]
Do NOT split the path or add extra brackets.
"""

    prompt = f"""Generate a valid YARRRML mapping file based on the provided ontology and mapping plan.
{feedback_section}
Target CSV: {csv_name}
Base URI: {base_uri}
Ontology Context: {ontology}
Mapping Plan: {mapping}

### CRITICAL — USE THE ONTOLOGY VOCABULARY:
You MUST use the prefixes, class URIs, and property URIs defined in the
ontology above.  Do NOT invent new prefixes like "ex:" or new predicates
like "ex:hasName" when the ontology already provides vocabulary.
- Use the ontology's prefix for predicates (the prefix declared in the ontology).
- Use the ontology's class names for rdf:type declarations.
- Use the ontology's property names for predicate-object entries.
- Copy prefix declarations from the ontology into the YARRRML prefixes section.

### ARCHITECTURE:
1. Prefer the MINIMAL number of mappings needed by the CSV.
   - If one mapping can represent the dataset, keep ONE mapping.
   - Create a secondary mapping ONLY when it has at least 2 unique
     data columns that do not belong in the primary mapping.
2. Each data column MUST appear in EXACTLY ONE mapping's po: list.
   Do NOT duplicate data columns across multiple mappings.
   - The primary class holds ONLY its own unique properties
     (identifiers, counts, direct attributes) and object-property
     links to secondary classes.
   - Each secondary class holds its own domain-specific columns.
3. The primary class MUST link to every secondary class via a
   2-item object-property PO entry (see LINKING section below).
4. If a secondary entity has NO unique data columns in this CSV,
   do NOT create a dedicated mapping for it.

### SYNTAX FORMAT:
1. Use exactly 2 spaces for indentation. No tabs.
2. Subject: use the short `s:` key with the BASE URI and a SINGLE
   identifier column.  Use the SAME identifier in ALL mappings.
   - Example: s: "{base_uri}class_path/$(ID_COLUMN)"
   - Do NOT use composite keys like $(col1)_$(col2) in subjects.
3. Source: use this exact format:
   sources:
     - [{csv_source}]
4. Predicate-Object: use `po:` with 2-item or 3-item lists.
   - Data property (3-item):  ["predicate", "$(column)", "xsd:datatype"]
   - Object property (2-item): ["predicate", "{base_uri}class_path/$(ID)~iri"]
5. Every URI, template, and prefix value MUST be wrapped in double quotes.
6. All prefixes used MUST be defined in the `prefixes:` section.
7. Every column MUST have a UNIQUE predicate name — never reuse the
   same predicate for two different columns.  If the ontology defines
   one generic predicate that could apply to multiple columns, keep
   that predicate for the most relevant column and derive a new
   predicate from the column name for others (e.g. column "my_col"
   with prefix "ont:" → "ont:myCol").  If no reasonable predicate
   exists, you may OMIT that column entirely.
8. PREFIX VALUES: Do NOT use angle brackets <> around prefix URIs.
   WRONG:  ont: <http://example.org/ontology#>
   CORRECT: ont: "http://example.org/ontology#"

### DATA TYPING:
Assign appropriate XSD datatypes (xsd:integer, xsd:float, xsd:boolean, xsd:date, xsd:dateTime).

### LINKING MAPPINGS:
To link the primary class to a secondary class, add a 2-item PO entry:
  ["prefix:objectProperty", "{base_uri}secondary_path/$(ID)~iri"]
The ~iri suffix is REQUIRED — it tells the processor the object is a
URI reference.  The URI template MUST match the secondary class's `s:` value.

### PREDICATE CONFLICT HANDLING:
If the ontology defines ONE generic predicate that could apply to multiple
CSV columns, you MUST NOT reuse that predicate.  Instead:
- Keep the ontology predicate for the most relevant / primary column.
- For secondary columns, derive a specific predicate from the column name
  using the same prefix (e.g. column "other_col" → "prefix:otherCol").
- As a LAST RESORT, you may omit a column if absolutely no predicate fits.

### OUTPUT RULES:
- Output ONLY valid YARRRML (YAML).
- No markdown code blocks, preamble, explanations, or post-text.
"""

    response = llm.invoke(prompt)
    content = response.content.strip()

    if "```" in content:
        parts = content.split("```")
        for part in parts:
            if "prefixes:" in part or "mappings:" in part:
                content = part
                break
        lines = content.splitlines()
        if lines and lines[0].strip().lower() in ("yaml", "yml"):
            content = "\n".join(lines[1:])
    content = re.sub(
        r'(-\s*)\[?[^\n]*?' + re.escape(csv_name) + r'~csv[^\n]*',
        r'\g<1>[' + csv_source + ']',
        content
    )

    return content.strip()
