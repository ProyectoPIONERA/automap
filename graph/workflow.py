from langgraph.graph import StateGraph, END
from data.checkpoints import AgentState
from graph.nodes import (
    schema_agent_node,
    ontology_scout_node,
    mapper_agent_node,
    schema_alignment_node,
    yarrrml_coordinator_node,
    validation_node,
    cq_validator_node,
    refiner_agent_node,
    kg_generation_node
)
from langgraph.checkpoint.memory import MemorySaver


def build_rml_graph():
    workflow = StateGraph(AgentState)

    # Node Definitions
    workflow.add_node("analyze_schema", schema_agent_node)
    workflow.add_node("scout_ontology", ontology_scout_node)
    workflow.add_node("map_semantics", mapper_agent_node)
    workflow.add_node("align_schema", schema_alignment_node)
    workflow.add_node("generate_yarrrml", yarrrml_coordinator_node)
    workflow.add_node("validate_yarrrml", validation_node)
    workflow.add_node("validate_cqs", cq_validator_node)
    workflow.add_node("refine_logic", refiner_agent_node)
    workflow.add_node("generate_kg", kg_generation_node)

    # --- Routing Functions ---
    def syntax_check_routing(state):
        feedback = state.get("feedback", "")
        retries = state.get("retry_count", 0)
        if "SYNTAX_ERROR" in feedback:
            if retries >= 10:
                return END
            return "generate_yarrrml"
        return "validate_cqs"

    def cq_check_routing(state):
        feedback = state.get("feedback", "")
        cq_retries = state.get("cq_retry_count", 0)
        if "CQ_ERROR" in feedback:
            if cq_retries >= 5:
                # Exceeded CQ retry limit — proceed to logic check anyway
                print("    [CQ Validator] Retry cap reached, proceeding to refiner.")
                return "refine_logic"
            if cq_retries >= 2:
                # Re-run schema alignment with CQ feedback to rebuild entity plan
                return "align_schema"
            return "generate_yarrrml"
        # CQ_PASSED or CQ_SKIPPED — proceed to logic check
        return "refine_logic"

    def logic_check_routing(state):
        feedback = state.get("feedback", "")
        retries = state.get("retry_count", 0)
        if "LOGIC_ERROR" in feedback:
            if retries >= 6:
                return END
            return "generate_yarrrml"
        return "generate_kg"

    # --- Connections ---
    workflow.set_entry_point("analyze_schema")
    workflow.add_edge("analyze_schema", "scout_ontology")
    workflow.add_edge("scout_ontology", "map_semantics")
    workflow.add_edge("map_semantics", "align_schema")
    workflow.add_edge("align_schema", "generate_yarrrml")
    workflow.add_edge("generate_yarrrml", "validate_yarrrml")

    workflow.add_conditional_edges(
        "validate_yarrrml",
        syntax_check_routing,
        {"generate_yarrrml": "generate_yarrrml", "validate_cqs": "validate_cqs", "__end__": END}
    )

    workflow.add_conditional_edges(
        "validate_cqs",
        cq_check_routing,
        {"generate_yarrrml": "generate_yarrrml", "align_schema": "align_schema", "refine_logic": "refine_logic"}
    )

    workflow.add_conditional_edges(
        "refine_logic",
        logic_check_routing,
        {"generate_yarrrml": "generate_yarrrml", "generate_kg": "generate_kg", "__end__": END}
    )

    workflow.add_edge("generate_kg", END)

    return workflow.compile(checkpointer=MemorySaver())