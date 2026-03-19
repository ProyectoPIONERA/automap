from config.settings import get_llm


def call_mapper_llm(schema, ontology):
    # Use the role-aware factory so this agent can have its own
    # model/temperature configuration.
    llm = get_llm(role="mapper_agent")

    prompt = f"""You are a semantic mapping expert.  Your job is to map CSV columns
to ontology classes and properties.

CSV SCHEMA (columns + sample data):
{schema}

ONTOLOGY (classes, properties, prefixes):
{ontology}

INSTRUCTIONS:
1. Identify which ontology CLASS(es) the CSV rows represent.
   Use the EXACT class URIs from the ontology.
   Do NOT invent new classes.

2. For EACH CSV column, find the matching ontology PROPERTY.
   Use the EXACT property URIs from the ontology.
   Do NOT invent new properties like "ex:hasName" when
   the ontology already defines a property for that concept.

3. Identify which columns are:
   - Identifier / primary key (used in subject URI templates)
   - Foreign keys / object properties (link to other classes)
   - Data properties (literal values with XSD datatypes)

4. List the PREFIX declarations needed (copy from the ontology).

OUTPUT FORMAT — for each mapping:
  Class: [ontology class URI]
  Subject template: [URI pattern with identifier column]
  Properties:
    [column_name] → [ontology property URI] (datatype or object-link)

Use ONLY vocabulary from the ontology.  Do NOT invent prefixes or predicates.
"""
    response = llm.invoke(prompt)
    return response.content