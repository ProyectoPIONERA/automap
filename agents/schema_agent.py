from config.settings import get_llm


def call_schema_llm(schema_data: dict):
    # Use the role-aware factory so this agent can have its own
    # model/temperature configuration.
    llm = get_llm(role="schema_agent")

    prompt = f"""Analyze this CSV structure:
Columns: {schema_data['columns']}
Sample Data: {schema_data['sample']}

1. Identify the main entity type this data describes.
2. For each column, provide a brief description of what it represents
   (data type, likely meaning, relationship to other columns).
3. Identify which column(s) serve as unique identifiers (primary keys).
4. Note any columns that appear to be foreign keys or references to
   other entities.
"""

    response = llm.invoke(prompt)
    return response.content
