def get_in_domain(eval_json: dict) -> dict:
    return {
        'entity_coverage': eval_json["entity_coverage"],
        'classes_with_hierarchy': eval_json["classes_with_hierarchy"],
        'predicates_with_hierarchy': eval_json["predicates_with_hierarchy"],
        'single_property_hierarchy_scores': eval_json["single_property_hierarchy_scores"],
        'predicates_direct': eval_json["predicates_direct"],
        'predicates_inverse': eval_json["predicates_inverse"],
        'predicate_details': eval_json["predicate_details"],
    }


def get_common(eval_json: dict) -> dict:
    return {
        # Basic metrics
        'triples': eval_json["triples"],
        'subjects_unique': eval_json["subjects_unique"],
        'subjects_fuzzy_unique': eval_json["subjects_fuzzy_unique"],
        'classes': eval_json["classes"],
        'classes_unique': eval_json["classes_unique"],

        # Property metrics
        'predicates': eval_json["predicates"],
        'predicates_unique': eval_json["predicates_unique"],
        'predicate_datatype_range': eval_json["predicate_datatype_range"],
        'predicate_datatype_range_unique': eval_json["predicate_datatype_range_unique"],

        # Object metrics
        'objects': eval_json["objects"],
        'objects_uris': eval_json["objects_uris"],
        'objects_literals': eval_json["objects_literals"],
    }


def get_for_wandb(eval_json: dict) -> dict:
    wandb_metrics = {}
    common = get_common(eval_json)

    log_metrics = ['p', 'r', 'f1']
    exclude = ["errors"]

    for key, values in common.items():
        if key not in exclude:
            for metric in log_metrics:
                if metric in values:
                    wandb_metrics[f"{key}_{metric}"] = values[metric]
    return wandb_metrics
