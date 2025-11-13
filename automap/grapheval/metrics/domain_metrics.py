"""
Domain-Specific Metrics

This module provides domain-specific evaluation metrics tailored to
particular use cases (e.g., entity identification, specific property validation).
"""

from rdflib import Literal, URIRef
from .hierarchy import HierarchyScorer
from .base import Metrics


class DomainMetrics(Metrics):
    def count_entity_ids_by_type(self, entity_type: str) -> int:
        """
        Count how many entities of a specific type have IDs in test graph.

        Args:
            entity_type: The entity type URI (e.g., 'http://dbpedia.org/ontology/Person')

        Returns:
            int: Count of entities with matching IDs
        """
        entity_ids = set(self.config.ids_by_type.get(entity_type, []))
        # Optimized: Use RDFLib's subjects() method
        subjects = set([str(s) for s in self.test_graph.subjects() if str(s).startswith(self.config.base_iri)])

        return sum(1 for subject in subjects if any(entity_id in subject for entity_id in entity_ids))

    def check_all_entity_ids_present(self, entity_type: str) -> int:
        """
        Check if all expected entity IDs of a type are present.

        Args:
            entity_type: The entity type URI

        Returns:
            int: 1 if all present, 0 otherwise
        """
        entity_ids = set(self.config.ids_by_type.get(entity_type, []))
        # Optimized: Use RDFLib's subjects() method
        subjects = set([str(s) for s in self.test_graph.subjects() if str(s).startswith(self.config.base_iri)])

        matched_ids = sum(1 for entity_id in entity_ids if any(entity_id in subject for subject in subjects))
        return 1 if matched_ids == len(entity_ids) else 0

    def count_entity_ids_with_type(self, entity_type: str) -> int:
        """
        Count entities that have both the ID and the correct rdf:type.

        Args:
            entity_type: The entity type URI

        Returns:
            int: Count of correctly typed entities
        """
        entity_ids = set(self.config.ids_by_type.get(entity_type, []))
        # Optimized: Use RDFLib's subjects() method with predicate and object filters
        rdf_type = URIRef(self.config.rdf_type_uri)
        entity_type_uri = URIRef(entity_type)
        subjects_with_type = set([str(s) for s in self.test_graph.subjects(predicate=rdf_type, object=entity_type_uri)
                                  if str(s).startswith(self.config.base_iri)])

        return sum(1 for subject in subjects_with_type if any(entity_id in subject for entity_id in entity_ids))

    def evaluate_predicate_details(self, predicate: str, hierarchy_scorer: HierarchyScorer = None) -> dict:
        """
        Detailed evaluation of a specific predicate.

        Args:
            predicate: The predicate URI to evaluate
            hierarchy_scorer: Optional HierarchyScorer for advanced metrics

        Returns:
            dict: Detailed metrics for the predicate
        """
        if not hierarchy_scorer and self.ontology_graph:
            hierarchy_scorer = HierarchyScorer(
                self.ontology_graph,
                self.reference_graph,
                self.test_graph
            )

        predicate_count = len([s for s, p, o in self.test_graph if str(p) == predicate])

        result = {
            'predicate_used': 1 if predicate_count > 0 else 0,
            'usage_count': predicate_count,
            'used_with_uris': len([s for s, p, o in self.test_graph
                                   if str(p) == predicate and not isinstance(o, Literal)]),
            'used_with_literals': len([s for s, p, o in self.test_graph
                                       if str(p) == predicate and isinstance(o, Literal)])
        }

        if hierarchy_scorer:
            direct_score = hierarchy_scorer.evaluate_property_direct(predicate)
            expected_count = direct_score['tp'] + direct_score['fn']

            result.update({
                'expected_count': expected_count,
                'correct_usage_count': direct_score['tp'],
                'outdegree_correct': 1 if predicate_count == expected_count else 0,
                'fuzzy_match_correct': 1 if direct_score['tp'] == expected_count else 0
            })

            # Add datatype validation if applicable
            if result['used_with_literals'] > 0:
                datatype_score = hierarchy_scorer.evaluate_property_direct_with_datatype(predicate)
                result['datatype_correct'] = 1 if datatype_score['tp'] == expected_count else 0

        return result

    def evaluate_all_predicates_detailed(self, hierarchy_scorer: HierarchyScorer = None) -> dict:
        """
        Detailed evaluation of all configured predicates.

        Args:
            hierarchy_scorer: Optional HierarchyScorer for advanced metrics

        Returns:
            dict: Detailed metrics for each predicate
        """
        # Include predicates from reference graph plus common extras
        predicates = set([str(p) for s, p, o in self.reference_graph])

        results = {}
        for predicate in predicates:
            results[predicate] = self.evaluate_predicate_details(predicate, hierarchy_scorer)

        return results

    def summarize_entity_coverage(self) -> dict:
        """
        Summarize coverage of expected entities across all types.

        Returns:
            dict: Summary statistics for entity coverage
        """
        summary = {}

        for entity_type in self.config.ids_by_type.keys():
            type_name = entity_type.split('/')[-1]  # Extract class name
            summary[type_name] = {
                'ids_found': self.count_entity_ids_by_type(entity_type),
                'all_ids_present': self.check_all_entity_ids_present(entity_type),
                'ids_with_correct_type': self.count_entity_ids_with_type(entity_type),
                'expected_count': len(self.config.ids_by_type[entity_type])
            }

        return summary
