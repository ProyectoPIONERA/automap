"""
Basic Graph Evaluation Metrics

This module provides fundamental metrics for comparing RDF graphs:
- Subject comparison
- Triple comparison  
- Class comparison
"""

from rdflib import URIRef
from automap.utils import overlapping_lists, calculate_metrics
from .base import Metrics


class BasicMetrics(Metrics):
    def evaluate_triples(self) -> dict:
        """
        Evaluate complete triples (s+p+o) with exact matching.

        Returns:
            dict: Metrics including tp, fp, fn, tn, precision, recall, f1
        """
        # Keep Python iteration for full graph scan (optimal for unfiltered)
        test_triples = set([str(s) + str(p) + str(o) for s, p, o in self.test_graph])
        reference_triples = set([str(s) + str(p) + str(o) for s, p, o in self.reference_graph])

        tp = len(test_triples.intersection(reference_triples))
        fp = len(test_triples) - tp
        fn = len(reference_triples) - tp
        tn = 0

        return calculate_metrics(tp, fp, fn, tn)

    def evaluate_subjects_unique(self) -> dict:
        """
        Evaluate unique subject IRIs with exact matching.

        Returns:
            dict: Metrics including tp, fp, fn, tn, precision, recall, f1
        """
        # Optimized: Use RDFLib's subjects() method
        test_subjects = set(self.test_graph.subjects())
        reference_subjects = set(self.reference_graph.subjects())

        tp = len(test_subjects.intersection(reference_subjects))
        fp = len(test_subjects) - tp
        fn = len(reference_subjects) - tp
        tn = 0

        return {
            'test_subjects_unique': list(test_subjects),
            'reference_subjects_unique': list(reference_subjects),
            **calculate_metrics(tp, fp, fn, tn),
        }

    def evaluate_subjects_fuzzy(self) -> dict:
        """
        Evaluate subjects with fuzzy matching based on ID extraction.

        Useful when subjects have different prefixes but same IDs.

        Returns:
            dict: Metrics including tp, fp, fn, tn, precision, recall, f1
        """
        # Optimized: Use RDFLib's subjects() method
        test_subjects = set(self.test_graph.subjects())
        test_ids = [str(s).split("/")[-1] for s in test_subjects]
        reference_subjects = set(self.reference_graph.subjects())
        reference_ids = [str(s).split("/")[-1] for s in reference_subjects]

        tp = 0
        for s in test_ids:
            if any(ref_id in str(s) for ref_id in reference_ids):
                tp += 1

        fp = len(test_ids) - tp
        fn = len(reference_ids) - tp
        tn = 0

        return {
            'test_subjects_fuzzy': list(test_ids),
            'reference_subjects_fuzzy': list(reference_ids),
            **calculate_metrics(tp, fp, fn, tn),
        }

    def evaluate_classes_unique(self) -> dict:
        """
        Evaluate unique classes (rdf:type objects) used in the graphs.

        Returns:
            dict: Metrics and lists of test/reference classes
        """
        # Optimized: Use RDFLib's objects() method with predicate filter (10x faster!)
        rdf_type = URIRef(self.config.rdf_type_uri)
        test_classes = set(self.test_graph.objects(predicate=rdf_type))
        reference_classes = set(self.reference_graph.objects(predicate=rdf_type))

        tp = len(test_classes.intersection(reference_classes))
        fp = len(test_classes) - tp
        fn = len(reference_classes) - tp
        tn = 0

        return {
            'test_classes': list(test_classes),
            'reference_classes': list(reference_classes),
            **calculate_metrics(tp, fp, fn, tn),
        }

    def evaluate_classes(self) -> dict:
        """
        Evaluate class usage (counting duplicates).

        Useful when multiple subjects have the same class.

        Returns:
            dict: Metrics including tp, fp, fn, tn, precision, recall, f1, and class lists
        """
        # Optimized: Use RDFLib's objects() method with predicate filter
        rdf_type = URIRef(self.config.rdf_type_uri)
        test_classes = list(self.test_graph.objects(predicate=rdf_type))
        reference_classes = list(self.reference_graph.objects(predicate=rdf_type))

        tp = len(overlapping_lists(test_classes, reference_classes))
        fp = len(test_classes) - tp
        fn = len(reference_classes) - tp
        tn = 0

        return {
            'test_classes': test_classes,
            'reference_classes': reference_classes,
            **calculate_metrics(tp, fp, fn, tn)
        }
