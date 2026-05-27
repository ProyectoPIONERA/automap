from typing import TypedDict, List, Annotated
import operator


class AgentState(TypedDict):
    # Inputs
    csv_path: str
    ontology_path: str
    base_uri: str
    competency_questions: list             # user-provided Competency Questions (optional)
    generated_cqs: list                    # auto-generated CQs when user provides none
    user_sparql_queries: list              # user-provided SPARQL ASK queries (optional, skips CQ→SPARQL)

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

    # SPARQL-based CQ Validation (post KG generation — deterministic)
    sparql_validation_results: list        # [{"cq", "sparql", "passed", "diagnosis"}]
    cq_sparql_retry_count: int             # retry counter for SPARQL-based CQ validation

    # Run directory for outputs
    run_dir: str

    # Loop & Metadata
    feedback: str
    retry_count: int
    messages: Annotated[List[str], operator.add]
    predicate_conflict_cols: list          # columns with unresolvable predicate clashes
    persistent_cq_failures: list          # CQ failure dicts {cq, sparql, diagnosis} from SPARQL validation
    _prev_entity_plan: str                 # entity plan from previous alignment run (for prefix cache diff)
    injected_column_constraints: dict      # {col → "pred (dtype) in MappingName"} from refiner auto-inject
    shacl_enabled: bool                    # True when --shacl flag was passed
    shacl_retry_count: int                 # dedicated SHACL retry counter (independent of global retry_count)
    shacl_violation_fingerprint: str       # MD5 of last violation set — used to detect persistent violations
    _prev_shacl_violations: list           # raw violation list from previous SHACL run (for superset check)
