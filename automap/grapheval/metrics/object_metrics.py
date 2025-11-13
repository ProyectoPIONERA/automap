"""
Object Evaluation Metrics

This module provides metrics for evaluating RDF objects (values).
"""

from rdflib import Literal, URIRef
from .base import Metrics
from automap.utils import overlapping_lists, calculate_metrics


class ObjectMetrics(Metrics):
    def evaluate_objects(self) -> dict:
        """
        Evaluate all objects (both URIs and literals).

        Returns:
            dict: Metrics including tp, fp, fn, tn, precision, recall, f1, and object lists
        """
        # Optimized: Use RDFLib's objects() method
        test_objects = list([str(o) for o in self.test_graph.objects()])
        reference_objects = list([str(o) for o in self.reference_graph.objects()])

        tp = len(overlapping_lists(test_objects, reference_objects))
        fp = len(test_objects) - tp
        fn = len(reference_objects) - tp
        tn = 0

        return {
            'test_objects': test_objects,
            'reference_objects': reference_objects,
            **calculate_metrics(tp, fp, fn, tn)
        }

    def evaluate_object_uris(self) -> dict:
        """
        Evaluate only URI objects (not literals)

        Returns:
            dict: Metrics including tp, fp, fn, tn, precision, recall, f1, and URI lists
        """
        # Optimized: Use RDFLib's objects() method
        test_uris = list([str(o) for o in self.test_graph.objects() if isinstance(o, URIRef)])
        reference_uris = list([str(o) for o in self.reference_graph.objects() if isinstance(o, URIRef)])

        tp = len(overlapping_lists(test_uris, reference_uris))
        fp = len(test_uris) - tp
        fn = len(reference_uris) - tp
        tn = 0

        return {
            'test_uris': test_uris,
            'reference_uris': reference_uris,
            **calculate_metrics(tp, fp, fn, tn)
        }

    def evaluate_object_literals(self) -> dict:
        """
        Evaluate only literal objects (without considering datatype).

        This is a relaxed version that matches literal values regardless of datatype.

        Returns:
            dict: Metrics including tp, fp, fn, tn, precision, recall, f1, and literal lists
        """
        # Optimized: Use RDFLib's objects() method
        test_literals = list([str(o) for o in self.test_graph.objects() if isinstance(o, Literal)])
        reference_literals = list([str(o) for o in self.reference_graph.objects() if isinstance(o, Literal)])

        tp = len(overlapping_lists(test_literals, reference_literals))
        fp = len(test_literals) - tp
        fn = len(reference_literals) - tp
        tn = 0

        return {
            'test_literals': test_literals,
            'reference_literals': reference_literals,
            **calculate_metrics(tp, fp, fn, tn)
        }
