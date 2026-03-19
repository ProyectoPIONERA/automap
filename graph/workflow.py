from langgraph.graph import StateGraph, END
from data.checkpoints import AgentState
from graph.nodes import (
    schema_agent_node,
    ontology_scout_node,
    mapper_agent_node,
    yarrrml_architect_node,
    validation_node,
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
    workflow.add_node("generate_yarrrml", yarrrml_architect_node)
    workflow.add_node("validate_yarrrml", validation_node)
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
    workflow.add_edge("map_semantics", "generate_yarrrml")
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

    workflow.add_edge("generate_kg", END)

    return workflow.compile(checkpointer=MemorySaver())