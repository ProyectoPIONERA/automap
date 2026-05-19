from langgraph.graph import StateGraph, END
from data.checkpoints import AgentState
from graph.nodes import (
    schema_agent_node,
    ontology_scout_node,
    mapper_agent_node,
    schema_alignment_node,
    yarrrml_coordinator_node,
    validation_node,
    refiner_agent_node,
    kg_generation_node,
    sparql_cq_validator_node,
    generate_cqs_node,
    shacl_validation_node,
)
from langgraph.checkpoint.memory import MemorySaver
import os


def build_rml_graph():
    workflow = StateGraph(AgentState)

    # Node Definitions
    workflow.add_node("analyze_schema", schema_agent_node)
    workflow.add_node("scout_ontology", ontology_scout_node)
    workflow.add_node("map_semantics", mapper_agent_node)
    workflow.add_node("generate_cqs", generate_cqs_node)
    workflow.add_node("align_schema", schema_alignment_node)
    workflow.add_node("generate_yarrrml", yarrrml_coordinator_node)
    workflow.add_node("validate_yarrrml", validation_node)
    workflow.add_node("refine_logic", refiner_agent_node)
    workflow.add_node("generate_kg", kg_generation_node)
    workflow.add_node("shacl_validate", shacl_validation_node)
    # Deterministic SPARQL-based CQ validation on the materialized KG
    workflow.add_node("sparql_validate_cqs", sparql_cq_validator_node)

    # --- Routing Functions ---
    def syntax_check_routing(state):
        feedback = state.get("feedback", "")
        retries = state.get("retry_count", 0)
        if "SYNTAX_ERROR" in feedback:
            if retries >= 10:
                return END
            return "generate_yarrrml"
        return "refine_logic"

    def logic_check_routing(state):
        feedback = state.get("feedback", "")
        retries = state.get("retry_count", 0)
        if "LOGIC_ERROR" in feedback:
            if retries >= 6:
                return END
            return "generate_yarrrml"
        return "generate_kg"

    def shacl_routing(state):
        """Route after SHACL validation.

        - If --shacl was not requested (shacl_enabled=False), pass through immediately.
        - On violations (SHACL_ERROR), retry YARRRML generation (capped via retry_count).
        - On SHACL_PASSED or SHACL_SKIP, continue to SPARQL CQ validation.
        """
        fb = state.get("feedback", "")
        retries = state.get("retry_count", 0)
        if "SHACL_ERROR" in fb:
            if retries >= 10:
                # Retry cap reached — accept current KG and move on
                return "sparql_validate_cqs"
            return "generate_yarrrml"
        return "sparql_validate_cqs"

    def sparql_cq_routing(state):
        """Route after SPARQL-based CQ validation on the materialized KG."""
        feedback = state.get("feedback", "")
        cq_sparql_retries = state.get("cq_sparql_retry_count", 0)
        max_retries = int(os.environ.get("CQ_SPARQL_MAX_RETRIES", "3"))

        if "CQ_SPARQL_ERROR" in feedback:
            if cq_sparql_retries >= max_retries:
                print(
                    f"    [SPARQL CQ Validator] Retry cap ({max_retries}) reached "
                    "— accepting current KG."
                )
                return END
            if cq_sparql_retries >= 2:
                # Deep retry — rebuild entity plan from scratch
                return "align_schema"
            # Standard retry — re-generate YARRRML with structured SPARQL feedback
            return "generate_yarrrml"

        # CQ_SPARQL_PASSED or no CQs → done
        return END

    # --- Connections ---
    workflow.set_entry_point("analyze_schema")
    workflow.add_edge("analyze_schema", "scout_ontology")
    workflow.add_edge("scout_ontology", "map_semantics")
    workflow.add_edge("map_semantics", "align_schema")
    workflow.add_edge("align_schema", "generate_cqs")
    workflow.add_edge("generate_cqs", "generate_yarrrml")
    workflow.add_edge("generate_yarrrml", "validate_yarrrml")

    workflow.add_conditional_edges(
        "validate_yarrrml",
        syntax_check_routing,
        {"generate_yarrrml": "generate_yarrrml", "refine_logic": "refine_logic", "__end__": END}
    )

    workflow.add_conditional_edges(
        "refine_logic",
        logic_check_routing,
        {"generate_yarrrml": "generate_yarrrml", "generate_kg": "generate_kg", "__end__": END}
    )

    # After KG generation → SHACL validation (zero-cost passthrough when --shacl not set)
    workflow.add_edge("generate_kg", "shacl_validate")

    workflow.add_conditional_edges(
        "shacl_validate",
        shacl_routing,
        {
            "generate_yarrrml": "generate_yarrrml",
            "sparql_validate_cqs": "sparql_validate_cqs",
        }
    )

    workflow.add_conditional_edges(
        "sparql_validate_cqs",
        sparql_cq_routing,
        {
            "generate_yarrrml": "generate_yarrrml",
            "align_schema": "align_schema",
            "__end__": END,
        }
    )

    return workflow.compile(checkpointer=MemorySaver())
