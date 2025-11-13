"""
Property Evaluation Metrics

This module provides metrics for evaluating RDF properties (predicates).
"""

from rdflib import Literal
from .base import Metrics
from automap.utils import overlapping_lists, calculate_metrics


class PropertyMetrics(Metrics):

    def evaluate_properties(self) -> dict:
        """
        Evaluate properties (predicates) used in the graphs.

        Returns:
            dict: Metrics including tp, fp, fn, tn, precision, recall, f1
        """
        # Optimized: Use RDFLib's predicates() method
        test_properties = list(self.test_graph.predicates())
        reference_properties = list(self.reference_graph.predicates())

        tp = len(overlapping_lists(test_properties, reference_properties))
        fp = len(test_properties) - tp
        fn = len(reference_properties) - tp
        tn = 0

        return calculate_metrics(tp, fp, fn, tn)

    def evaluate_properties_unique(self) -> dict:
        """
        Evaluate unique property-object pairs.

        Returns:
            dict: Metrics and lists of property-object combinations
        """
        test_po = set([str(p) + str(o) for s, p, o in self.test_graph])
        reference_po = set([str(p) + str(o) for s, p, o in self.reference_graph])

        tp = len(test_po.intersection(reference_po))
        fp = len(test_po) - tp
        fn = len(reference_po) - tp
        tn = 0

        return {
            'test_po': list(test_po),
            'reference_po': list(reference_po),
            **calculate_metrics(tp, fp, fn, tn)
        }

    def evaluate_predicate_datatypes(self) -> dict:
        """
        Evaluate property-datatype combinations for literal values.

        Returns:
            dict: Metrics and lists of predicate-datatype pairs
        """
        test_p_datatype = list([str(p) + str(o.datatype) for s, p, o in self.test_graph if isinstance(o, Literal)])
        reference_p_datatype = list([str(p) + str(o.datatype) for s, p, o in self.reference_graph if isinstance(o, Literal)])

        tp = len(overlapping_lists(test_p_datatype, reference_p_datatype))
        fp = len(test_p_datatype) - tp
        fn = len(reference_p_datatype) - tp
        tn = 0

        return {
            'test_p_datatype': test_p_datatype,
            'reference_p_datatype': reference_p_datatype,
            **calculate_metrics(tp, fp, fn, tn)
        }

    def evaluate_predicate_datatypes_unique(self) -> dict:
        """
        Evaluate unique property-datatype combinations.

        Returns:
            dict: Metrics and lists of unique predicate-datatype pairs
        """
        test_p_datatype = set([str(p) + str(o.datatype) for s, p, o in self.test_graph if isinstance(o, Literal)])
        reference_p_datatype = set([str(p) + str(o.datatype) for s, p, o in self.reference_graph if isinstance(o, Literal)])

        tp = len(test_p_datatype.intersection(reference_p_datatype))
        fp = len(test_p_datatype) - tp
        fn = len(reference_p_datatype) - tp
        tn = 0

        return {
            'test_p_datatype': list(test_p_datatype),
            'reference_p_datatype': list(reference_p_datatype),
            **calculate_metrics(tp, fp, fn, tn)
        }

    def count_predicate_usage(self, predicate: str) -> int:
        # [CG]: Not used
        return len([s for s, p, o in self.test_graph if str(p) == predicate])

    def count_predicate_with_literals(self, predicate: str) -> int:
        # [CG]: Not used
        return len([s for s, p, o in self.test_graph if str(p) == predicate and isinstance(o, Literal)])

    def count_predicate_with_objects(self, predicate: str) -> int:
        # [CG]: Not used
        return len([s for s, p, o in self.test_graph if str(p) == predicate and not isinstance(o, Literal)])

    def check_all_reference_predicates_present(self) -> bool:
        # [CG]: Not used
        # Optimized: Use RDFLib's predicates() method
        test_predicates = set(self.test_graph.predicates())
        reference_predicates = set(self.reference_graph.predicates())

        overlap = test_predicates.intersection(reference_predicates)
        return 1 if len(overlap) == len(reference_predicates) else 0

    def check_only_reference_predicates_present(self) -> bool:
        # [CG]: Not used
        # Optimized: Use RDFLib's predicates() method
        test_predicates = set(self.test_graph.predicates())
        reference_predicates = set(self.reference_graph.predicates())

        overlap = test_predicates.intersection(reference_predicates)
        return 1 if len(overlap) == len(test_predicates) else 0
