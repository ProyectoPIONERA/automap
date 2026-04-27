"""
Prefix Agent — generates the YARRRML ``prefixes:`` block.

Uses the one-shot example from ``config/yarrrml_examples.py`` to teach
local LLMs (Qwen, Mistral, etc.) the correct YARRRML prefix syntax.

Prompt is split into a STATIC system message (cached by llama.cpp after
the first call) and a DYNAMIC human message (changes per call/retry).
"""

from config.settings import get_llm
from config.yarrrml_examples import EXAMPLE_FOR_PREFIX_MANAGER
from config.structured_output import (
    structured_output_enabled, PrefixesOutput, prefixes_to_yaml,
)
from langchain_core.messages import SystemMessage, HumanMessage

# ── Static system prompt (identical every call → KV-cached by llama.cpp) ──
_SYSTEM_PROMPT = f"""{EXAMPLE_FOR_PREFIX_MANAGER}

You generate ONLY the `prefixes:` block for a YARRRML mapping file.

INSTRUCTIONS:
1. ONLY include prefixes that are ACTUALLY USED by at least one mapping.
   Do NOT include all ontology prefixes — only the ones referenced
   in subject URIs, rdf:type declarations, or predicate names.
2. Always include standard prefixes: rdf, rdfs, xsd.
3. Use the format: `name: "URI"` (double-quoted, NO angle brackets).
4. Include a prefix for the base URI if not already in the ontology.
5. Do NOT use angle brackets <> around URIs.
   WRONG:  ex: <http://example.org/>
   CORRECT: ex: "http://example.org/"

OUTPUT RULES:
- Output ONLY the `prefixes:` block (valid YAML).
- No markdown code blocks, no explanations, no other text.
- Start with `prefixes:` on the first line.
"""


def _strip_markdown(content: str, marker: str = "prefixes:") -> str:
    """Remove markdown code fences from LLM output."""
    if "```" not in content:
        return content.strip()
    parts = content.split("```")
    for part in parts:
        if marker in part:
            content = part
            break
    lines = content.splitlines()
    if lines and lines[0].strip().lower() in ("yaml", "yml"):
        content = "\n".join(lines[1:])
    return content.strip()


def call_prefix_agent(state: dict) -> str:
    """Generate ONLY the ``prefixes:`` block for the YARRRML mapping.

    Parameters
    ----------
    state : dict
        Pipeline state containing ``ontology_info`` and ``base_uri``.

    Returns
    -------
    str
        The ``prefixes:`` YAML block (plain text, no fences).
    """
    llm = get_llm(role="prefix_agent")

    ontology = state.get("ontology_info", {}).get("raw", "")
    base_uri = state.get("base_uri", "http://example.org/")

    # Include entity plan so prefix agent can detect domain-specific prefixes
    # used in the alignment (e.g. lkg:, eli:, podio:)
    entity_plan = state.get("schema_alignment", {}).get("entity_plan", "")
    entity_plan_section = ""
    if entity_plan:
        entity_plan_section = f"""
IMPORTANT: The entity plan below uses prefixes that MUST be declared.
Scan every prefix:localName pattern and include it in the prefixes block:
{entity_plan[:2000]}
"""

    # On retries, include targeted feedback for prefix issues
    feedback = state.get("feedback", "")
    feedback_section = ""
    if feedback and "PASSED" not in feedback and "APPROVED" not in feedback:
        if any(kw in feedback.lower() for kw in [
            "prefix", "declared", "not defined", "undeclared",
        ]):
            feedback_section = f"""
### FIX REQUIRED — Previous output had prefix issues:
{feedback}

Make sure ALL prefixes used in the mappings are declared.
"""

    # ── Dynamic human message (changes per call) ──
    human_prompt = f"""{feedback_section}
{entity_plan_section}

Ontology Context:
{ontology}

Base URI: {base_uri}

Generate the prefixes: block now.
"""

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]

    # ── Structured output path (eliminates syntax retry loops) ────
    if structured_output_enabled():
        try:
            structured_llm = llm.with_structured_output(PrefixesOutput)
            result: PrefixesOutput = structured_llm.invoke(messages)
            return prefixes_to_yaml(result)
        except Exception as e:
            print(f"    [PrefixAgent] Structured output failed ({e}), falling back to free-text")

    # ── Free-text fallback ────────────────────────────────────────
    response = llm.invoke(messages)
    return _strip_markdown(response.content.strip())
