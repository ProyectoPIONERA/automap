from typing import TypedDict, List, Annotated
import operator


class AgentState(TypedDict):
    # Inputs
    csv_path: str
    ontology_path: str
    base_uri: str
    competency_questions: list             # user-provided Competency Questions

    # Processed Data
    schema_info: dict
    ontology_info: dict
    mapping_plan: dict

    # Schema Alignment (intermediate step between mapper and YARRRML gen)
    schema_alignment: dict                 # Functional Entity Plan from alignment agent
    alignment_changed: bool                # True when align_schema re-runs (resets prefix cache)

    # YARRRML sub-agent intermediate outputs
    prefixes_output: str                   # PrefixAgent output
    entity_yarrrml: str                    # EntityAgent output (no joins)

    # Outputs
    yarrrml_output: str
    rdf_output: str

    # CQ Validation
    cq_validation: dict                    # per-CQ pass/fail results

    # Run directory for outputs
    run_dir: str

    # Loop & Metadata
    feedback: str
    retry_count: int
    cq_retry_count: int                    # separate retry counter for CQ failures
    messages: Annotated[List[str], operator.add]
    predicate_conflict_cols: list          # columns with unresolvable predicate clashes
    persistent_cq_failures: list          # CQ texts that have failed on 2+ consecutive attempts
    _prev_entity_plan: str                 # entity plan from previous alignment run (for prefix cache diff)
    injected_column_constraints: dict      # {col → "pred (dtype) in MappingName"} from refiner auto-inject
