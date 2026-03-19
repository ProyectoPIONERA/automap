from typing import TypedDict, List, Annotated
import operator


class AgentState(TypedDict):
    # Inputs
    csv_path: str
    ontology_path: str
    base_uri: str

    # Processed Data
    schema_info: dict
    ontology_info: dict
    mapping_plan: dict

    # Outputs
    yarrrml_output: str
    rdf_output: str

    # Run directory for outputs
    run_dir: str

    # Loop & Metadata
    feedback: str
    retry_count: int
    messages: Annotated[List[str], operator.add]
    predicate_conflict_cols: list          # columns with unresolvable predicate clashes
