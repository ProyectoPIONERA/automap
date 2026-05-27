"""
Structured Output Models for YARRRML Generation
================================================

Pydantic models that constrain LLM output to valid YARRRML structure.
When the LLM is forced to fill these schemas (via JSON-schema mode),
syntax errors become structurally impossible — eliminating most
validate_yarrrml → retry loops.

Usage:
    llm = get_llm(role="prefix_agent")
    structured_llm = llm.with_structured_output(PrefixesOutput)
    result: PrefixesOutput = structured_llm.invoke(messages)
    yaml_str = prefixes_to_yaml(result)
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field, ConfigDict


# ────────────────────────────────────────────────────────────────────
# Environment toggle
# ────────────────────────────────────────────────────────────────────

def structured_output_enabled() -> bool:
    """Return True when structured output mode is active."""
    return os.getenv("STRUCTURED_OUTPUT", "true").lower() in ("true", "1", "yes")


# ────────────────────────────────────────────────────────────────────
# Pydantic models — Prefixes
# ────────────────────────────────────────────────────────────────────

class PrefixesOutput(BaseModel):
    """The ``prefixes:`` block of a YARRRML file."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prefixes: dict[str, str] = Field(
        description=(
            "Map of prefix name to URI. Example: "
            '{"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", '
            '"xsd": "http://www.w3.org/2001/XMLSchema#"}'
        )
    )


# ────────────────────────────────────────────────────────────────────
# Pydantic models — Mappings (entity block)
# ────────────────────────────────────────────────────────────────────

class MappingBlock(BaseModel):
    """A single YARRRML mapping entry."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    sources: list[list[str]] = Field(
        description=(
            'List of source specifications. Each source is a list with one '
            'element: the CSV path with ~csv suffix. '
            'Example: [["data/input/file.csv~csv"]]'
        )
    )
    s: str = Field(
        description=(
            'Subject URI template as a single string. '
            'Example: "http://example.com/Person/$(id)"'
        )
    )
    po: list[list[str]] = Field(
        description=(
            "List of predicate-object entries. Each entry is a list of 2 or 3 "
            "strings: [predicate, object] or [predicate, object, datatype]. "
            'Examples: ["a", "schema:Person"], '
            '["schema:name", "$(name)", "xsd:string"], '
            '["schema:knows", "http://example.com/Person/$(friend_id)~iri"]'
        )
    )


class MappingsOutput(BaseModel):
    """The ``mappings:`` block of a YARRRML file."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mappings: dict[str, MappingBlock] = Field(
        description="Map of mapping name to its definition."
    )


# ────────────────────────────────────────────────────────────────────
# Converters — Pydantic → YARRRML YAML string
# ────────────────────────────────────────────────────────────────────

def prefixes_to_yaml(output: PrefixesOutput) -> str:
    """Convert a ``PrefixesOutput`` model to a YARRRML prefixes block."""
    lines = ["prefixes:"]
    for name, uri in output.prefixes.items():
        # Ensure URI is double-quoted
        lines.append(f'  {name}: "{uri}"')
    return "\n".join(lines)


def mappings_to_yaml(output: MappingsOutput) -> str:
    """Convert a ``MappingsOutput`` model to a YARRRML mappings block.

    Uses flow-style lists for ``po:`` entries (which is what Yatter
    expects) and inline ``sources:``.
    """
    lines = ["mappings:"]
    for mname, mblock in output.mappings.items():
        lines.append(f"  {mname}:")
        # Sources
        lines.append("    sources:")
        for src in mblock.sources:
            inner = ", ".join(src)
            lines.append(f"      - [{inner}]")
        # Subject
        lines.append(f'    s: {mblock.s}')
        # Predicate-object
        lines.append("    po:")
        for entry in mblock.po:
            formatted = ", ".join(entry)
            lines.append(f"      - [{formatted}]")
    return "\n".join(lines)


