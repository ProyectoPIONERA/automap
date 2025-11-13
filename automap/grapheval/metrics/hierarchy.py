"""
Hierarchy-Based Scoring Module

This module provides functionality for calculating similarity scores based on
ontology hierarchies (class and property hierarchies).
"""

from rdflib import Graph
from rdflib.term import Literal
from collections import defaultdict
from .base import Metrics
from automap.utils import (
    Config,
    overlapping_lists,
    average,
    precision_score,
    recall_score,
    f1_score
)


class HierarchyScorer(Metrics):
    """
    Calculate evaluation scores based on ontology hierarchies.

    This class is useful when the reference graph uses the most specific
    classes/properties and you want to give partial credit for using
    parent classes/properties in the hierarchy.
    """

    def __init__(self, test_graph: Graph, reference_graph: Graph, ontology_graph: Graph = None, config: Config = None):
        """
        Initialize the hierarchy scorer.

        Args:
            ontology_graph: RDF graph containing the ontology with hierarchy information
            reference_graph: Ground truth RDF graph
            test_graph: Test/predicted RDF graph to evaluate
            config: Optional configuration object
        """
        super().__init__(test_graph, reference_graph, ontology_graph, config)

        self.class_paths = self._build_transitive_closure(self._extract_class_relations(ontology_graph))
        self.property_paths = self._build_transitive_closure(self._extract_property_relations(ontology_graph))
        self.subject_alignment = self._align_subjects(test_graph, reference_graph, self.config.base_iri)

    def _extract_class_relations(self, ontology_graph: Graph) -> dict:
        """Extract class-subclass relationships from ontology."""
        class_relations = {}
        query_results = ontology_graph.query(self.config.subclass_query)

        for result in query_results:
            super_class = str(result[0])
            sub_class = str(result[1])
            class_relations[sub_class] = super_class

        return class_relations

    def _extract_property_relations(self, ontology_graph: Graph) -> dict:
        """Extract property-subproperty relationships from ontology."""
        property_relations = {}
        query_results = ontology_graph.query(self.config.subproperty_query)

        for result in query_results:
            super_property = str(result[0])
            sub_property = str(result[1])
            property_relations[sub_property] = super_property

        return property_relations

    @staticmethod
    def _build_transitive_closure(hierarchy_dict: dict) -> dict:
        """
        Build transitive closure of hierarchy relationships.

        For each element, creates a list containing itself and all its ancestors
        in order from most specific to most general.
        """
        all_elements = set(hierarchy_dict.keys()) | set(hierarchy_dict.values())
        closure = defaultdict(list)

        for element in all_elements:
            current = element
            closure[element].append(current)

            while hierarchy_dict.get(current):
                current = hierarchy_dict[current]
                closure[element].append(current)

        return closure

    @staticmethod
    def _calculate_hierarchy_similarity(reference_resource: str, test_resource: str,
                                        hierarchy_paths: dict) -> float:
        """
        Calculate similarity score based on hierarchy distance.

        Returns 1.0 for exact match, 0.5^n for n steps up in hierarchy, 0.0 otherwise.
        """
        if reference_resource == test_resource:
            return 1.0
        elif test_resource in hierarchy_paths.get(reference_resource, []):
            path = hierarchy_paths[reference_resource]
            distance = path.index(test_resource)
            return 0.5 ** distance
        elif reference_resource in hierarchy_paths.get(test_resource, []):
            # TODO: Handle case where test is more specific than reference
            return 0.0
        else:
            return 0.0

    @staticmethod
    def _align_subjects(test_graph: Graph, reference_graph: Graph, prefix: str) -> list:
        """
        Align subjects between test and reference graphs based on IRI structure.

        Returns list of tuples: (reference_subject_iri, test_subject_iri)
        """
        test_subjects = set([s for s, p, o in test_graph])
        reference_subjects = set([s for s, p, o in reference_graph])
        reference_ids = [str(s).split("/")[-1] for s in reference_subjects]

        alignments = []
        for test_subject in test_subjects:
            test_subject_str = str(test_subject)
            if test_subject_str.startswith(prefix):
                for ref_id in reference_ids:
                    if ref_id in test_subject_str:
                        alignments.append((prefix + ref_id, test_subject_str))
                        break

        return alignments

    def _align_properties(self, subject_alignments: list) -> list:
        """
        Align properties between graphs based on subject-object pairs.

        Returns list of tuples: (reference_property, test_property)
        """
        subject_map = {test: ref for ref, test in subject_alignments}

        reference_so_p = [
            (str(s) + str(o), str(p))
            for s, p, o in self.reference_graph
        ]
        test_so_p = [
            (subject_map.get(str(s), str(s)) + subject_map.get(str(o), str(o)), str(p))
            for s, p, o in self.test_graph
        ]

        property_pairs = set()
        for ref_so, ref_p in reference_so_p:
            for test_so, test_p in test_so_p:
                if ref_so == test_so:
                    property_pairs.add((ref_p, test_p))

        return list(property_pairs)

    def _align_subject_properties(self, subject_alignments: list) -> list:
        """Similar to _align_properties but returns all matches, not unique pairs."""
        subject_map = {test: ref for ref, test in subject_alignments}

        reference_so_p = [
            (str(s) + str(o), str(p))
            for s, p, o in self.reference_graph
        ]
        test_so_p = [
            (subject_map.get(str(s), str(s)) + subject_map.get(str(o), str(o)), str(p))
            for s, p, o in self.test_graph
        ]

        property_pairs = []
        for ref_so, ref_p in reference_so_p:
            for test_so, test_p in test_so_p:
                if ref_so == test_so:
                    property_pairs.append((ref_p, test_p))

        return property_pairs

    def _get_subject_class(self, subject: str, graph: Graph) -> str:
        """Get the class of a subject from the graph."""
        query = self.config.subject_class_query.replace("?s", f"<{subject}>")
        result_set = graph.query(query)
        for result in result_set:
            return str(result[0])
        return None

    def calculate_class_similarity(self, reference_resource: str, test_resource: str) -> float:
        """Calculate similarity score for classes based on hierarchy."""
        return self._calculate_hierarchy_similarity(
            reference_resource, test_resource, self.class_paths
        )

    def calculate_property_similarity(self, reference_resource: str, test_resource: str) -> float:
        """Calculate similarity score for properties based on hierarchy."""
        return self._calculate_hierarchy_similarity(
            reference_resource, test_resource, self.property_paths
        )

    def evaluate_class_hierarchies(self) -> dict:
        """
        Evaluate class usage considering hierarchy.

        Returns dict with precision, recall, F1, and detailed scores.
        """
        scores = []

        s_ref = list(set([str(s) for s, p, o in self.reference_graph]))
        s_test = list(set([str(s) for s, p, o in self.test_graph]))

        for ref_subj, test_subj in self.subject_alignment:
            ref_class = self._get_subject_class(ref_subj, self.reference_graph)
            test_class = self._get_subject_class(test_subj, self.test_graph)
            if ref_class and test_class:
                similarity = self.calculate_class_similarity(ref_class, test_class)
                scores.append((ref_class, test_class, similarity))

        precision = average([score[2] for score in scores])
        # TODO: Review recall calculation
        recall = len([s for s in scores if s[2] > 0]) / len(s_ref) if s_ref else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'detailed_scores': scores,
            'reference_subjects': s_ref,
            'test_subjects': s_test
        }

    def evaluate_property_hierarchies(self) -> dict:
        """
        Evaluate property usage considering hierarchy.

        Returns dict with precision, recall, F1, and detailed scores.
        """
        scores = []
        s_alignments = self.subject_alignment
        p_alignments = self._align_properties(s_alignments)

        p_ref = list(set([str(p) for s, p, o in self.reference_graph]))
        p_test = list(set([str(p) for s, p, o in self.test_graph]))

        for ref_prop, test_prop in p_alignments:
            similarity = self.calculate_property_similarity(ref_prop, test_prop)
            scores.append((ref_prop, test_prop, similarity))

        precision = average([score[2] for score in scores])
        recall = len([s for s in scores if s[2] > 0]) / len(p_ref) if p_ref else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'subject_alignments': s_alignments,
            'property_alignments': p_alignments,
            'detailed_scores': scores,
            'reference_properties': p_ref,
            'test_properties': p_test
        }

    def evaluate_property_direct(self, target_property: str) -> dict:
        """
        Evaluate a specific property with exact matching (no hierarchy).

        Args:
            target_property: The property URI to evaluate

        Returns:
            dict: Metrics including tp, fp, fn, precision, recall, f1
        """
        subject_map = {test: ref for ref, test in self.subject_alignment}

        ref_spo = [
            str(s) + str(p) + str(o)
            for s, p, o in self.reference_graph
            if str(p) == target_property
        ]
        test_spo = [
            subject_map.get(str(s), str(s)) + str(p) + subject_map.get(str(o), str(o))
            for s, p, o in self.test_graph
            if str(p) == target_property
        ]

        tp = len(overlapping_lists(test_spo, ref_spo))
        fp = len(test_spo) - tp
        fn = len(ref_spo) - tp
        tn = 0

        return {
            'precision': precision_score(tp, fp),
            'recall': recall_score(tp, fn),
            'f1': f1_score(tp, fp, fn),
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    def evaluate_property_direct_with_datatype(self, target_property: str) -> dict:
        """
        Evaluate a specific property including datatype validation.

        Args:
            target_property: The property URI to evaluate

        Returns:
            dict: Metrics including tp, fp, fn, precision, recall, f1
        """
        subject_map = {test: ref for ref, test in self.subject_alignment}

        ref_spo = [
            str(s) + str(p) + str(o) + str(o.datatype)
            for s, p, o in self.reference_graph
            if str(p) == target_property and isinstance(o, Literal)
        ]
        test_spo = [
            subject_map.get(str(s), str(s)) + str(p) + str(o) + str(o.datatype)
            for s, p, o in self.test_graph
            if str(p) == target_property and isinstance(o, Literal)
        ]

        tp = len(overlapping_lists(test_spo, ref_spo))
        fp = len(test_spo) - tp
        fn = len(ref_spo) - tp
        tn = 0

        return {
            'precision': precision_score(tp, fp),
            'recall': recall_score(tp, fn),
            'f1': f1_score(tp, fp, fn),
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    def evaluate_property_inverse(self, target_property: str) -> dict:
        """
        Evaluate if a property is used in inverse direction.

        Args:
            target_property: The property URI to evaluate

        Returns:
            dict: Metrics including precision, recall, f1
        """
        subject_map = {test: ref for ref, test in self.subject_alignment}

        ref_spo = [
            str(s) + str(p) + str(o)
            for s, p, o in self.reference_graph
            if str(p) == target_property
        ]
        # Note: swapping s and o for inverse
        test_ops = [
            subject_map.get(str(o), str(o)) + str(p) + subject_map.get(str(s), str(s))
            for s, p, o in self.test_graph
            if str(p) == target_property
        ]

        tp = len(overlapping_lists(test_ops, ref_spo))
        fp = len(test_ops) - tp
        fn = len(ref_spo) - tp

        return {
            'precision': precision_score(tp, fp),
            'recall': recall_score(tp, fn),
            'f1': f1_score(tp, fp, fn)
        }

    def evaluate_all_properties_direct(self) -> dict:
        """Evaluate all configured properties with direct matching."""
        results = {}
        for property_uri in self.config.predicates_to_evaluate:
            results[property_uri] = self.evaluate_property_direct(property_uri)
        return results

    def evaluate_all_properties_inverse(self) -> dict:
        """Evaluate all configured properties for inverse usage."""
        results = {}
        for property_uri in self.config.predicates_to_evaluate:
            results[property_uri] = self.evaluate_property_inverse(property_uri)
        return results

    def evaluate_single_property_hierarchy(self, target_property: str) -> dict:
        """
        Evaluate a single property considering hierarchy.

        Args:
            target_property: The property URI to evaluate

        Returns:
            dict: Metrics including precision, recall, f1
        """
        scores = []
        s_alignments = self.subject_alignment
        p_alignments = self._align_subject_properties(s_alignments)
        p_alignments = [p for p in p_alignments if p[0] == target_property]

        p_ref = [
            str(p)
            for s, p, o in self.reference_graph
            if str(p) == target_property
        ]

        for ref_prop, test_prop in p_alignments:
            similarity = self.calculate_property_similarity(ref_prop, test_prop)
            scores.append((ref_prop, test_prop, similarity))

        precision = average([score[2] for score in scores])
        recall = len([s for s in scores if s[2] > 0]) / len(p_ref) if p_ref else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def evaluate_multiple_properties_hierarchy(self, target_properties: list) -> dict:
        """
        Evaluate multiple properties considering hierarchy.

        Args:
            target_properties: List of property URIs to evaluate

        Returns:
            dict: Results for each property
        """
        results = {}
        for property_uri in target_properties:
            results[property_uri] = self.evaluate_single_property_hierarchy(property_uri)
        return results
