from config.settings import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

# ── Static system prompt (cached by llama.cpp across calls) ──
_SYSTEM_PROMPT = """Analyze a CSV structure.

1. Identify the main entity type this data describes.
2. For each column, provide a brief description of what it represents
   (data type, likely meaning, relationship to other columns).
3. Identify which column(s) serve as unique identifiers (primary keys).
4. Note any columns that appear to be foreign keys or references to
   other entities.
"""


def call_schema_llm(schema_data: dict):
    llm = get_llm(role="schema_agent")

    human_prompt = f"""Columns: {schema_data['columns']}
Sample Data: {schema_data['sample']}

Analyze this CSV structure now.
"""

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ])
    return response.content
