"""
Graph Evaluator - Main Orchestrator

This module provides the main GraphEvaluator class that orchestrates
all evaluation metrics for RDF graph comparison.
"""

from rdflib import Graph
from pathlib import Path
from typing import Union, Optional
from automap.utils import Config
from .basic_metrics import BasicMetrics
from .property_metrics import PropertyMetrics
from .object_metrics import ObjectMetrics
from .domain_metrics import DomainMetrics
from .hierarchy import HierarchyScorer


class GraphEvaluator:
    """
    Main evaluator class for comprehensive RDF graph comparison.

    This class orchestrates various evaluation metrics including:
    - Basic metrics (subjects, triples, classes)
    - Property metrics (predicates, datatypes)
    - Object metrics (URIs, literals)
    - Domain-specific metrics
    - Hierarchy-based scoring

    """

    def __init__(
            self,
            test_graph: Graph,
            reference_graph: Graph,
            config: Optional[Union[Config, str, Path]] = None
    ):
        """
        Initialize the graph evaluator.

        Args:
            test_graph: The RDF graph to evaluate
            reference_graph: The ground truth RDF graph
            ontology_graph: The ontology RDF graph for hierarchy-based metrics (optional)
            config: Configuration object or path to YAML config file (optional)
                   If string/Path provided, will load Config from that path
                   If None, will use default global configuration for backward compatibility
        """
        self.test_graph = test_graph
        self.reference_graph = reference_graph

        if isinstance(config, (str, Path)):
            self.config = Config(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            raise TypeError(f"config must be Config, str, Path, or None, got {type(config)}")

        with open(self.config.ontology_file, 'r') as f:
            ontology_data = f.read()

        # Auto-detect RDF format from file extension
        ontology_format = self._detect_rdf_format(self.config.ontology_file)
        self.ontology_graph = Graph().parse(data=ontology_data, format=ontology_format)

        # Auto-extract namespaces and predicates from ontology if not provided in config
        # Pass ontology_path for caching
        self.config.extract_from_ontology(self.ontology_graph, ontology_path=self.config.ontology_file)

        # Auto-extract entity IDs from reference graph if not provided in config
        self.config.extract_ids_from_graph(reference_graph)

        self.basic_metrics = BasicMetrics(test_graph, reference_graph, config=self.config)
        self.property_metrics = PropertyMetrics(test_graph, reference_graph)
        self.object_metrics = ObjectMetrics(test_graph, reference_graph)
        self.domain_metrics = DomainMetrics(test_graph, reference_graph, ontology_graph=self.ontology_graph, config=self.config)

        self.hierarchy_scorer = None
        if self.ontology_graph:
            self.hierarchy_scorer = HierarchyScorer(
                test_graph,
                reference_graph,
                ontology_graph=self.ontology_graph,
                config=self.config
            )

    def _detect_rdf_format(self, filepath: Union[str, Path]) -> str:
        extension = Path(filepath).suffix.lower()

        format_map = {
            '.ttl': 'turtle',
            '.rdf': 'xml',
            '.owl': 'xml',
            '.n3': 'n3',
            '.nt': 'nt',
            '.jsonld': 'json-ld',
        }

        detected_format = format_map.get(extension, 'turtle')
        return detected_format

    # ========== Basic Metrics ==========

    def evaluate_subjects_unique(self) -> dict:
        """Evaluate subject IRIs with exact matching."""
        return self.basic_metrics.evaluate_subjects_unique()

    def evaluate_subjects_fuzzy(self) -> dict:
        """Evaluate subjects with fuzzy matching based on ID extraction."""
        return self.basic_metrics.evaluate_subjects_fuzzy()

    def evaluate_triples(self) -> dict:
        """Evaluate complete triples (s+p+o) with exact matching."""
        return self.basic_metrics.evaluate_triples()

    def evaluate_classes(self) -> dict:
        """Evaluate class usage (counting duplicates)."""
        return self.basic_metrics.evaluate_classes()

    def evaluate_classes_unique(self) -> dict:
        """Evaluate unique classes used in the graphs."""
        return self.basic_metrics.evaluate_classes_unique()

    # ========== Property Metrics ==========

    def evaluate_properties(self) -> dict:
        """Evaluate properties (predicates) used in the graphs."""
        return self.property_metrics.evaluate_properties()

    def evaluate_properties_unique(self) -> dict:
        """Evaluate unique property-object pairs."""
        return self.property_metrics.evaluate_properties_unique()

    def evaluate_predicate_datatypes(self) -> dict:
        """Evaluate property-datatype combinations for literal values."""
        return self.property_metrics.evaluate_predicate_datatypes()

    def evaluate_predicate_datatypes_unique(self) -> dict:
        """Evaluate unique property-datatype combinations."""
        return self.property_metrics.evaluate_predicate_datatypes_unique()

    # ========== Object Metrics ==========

    def evaluate_objects(self) -> dict:
        """Evaluate all objects (both URIs and literals)."""
        return self.object_metrics.evaluate_objects()

    def evaluate_object_uris(self) -> dict:
        """Evaluate only URI objects (not literals)."""
        return self.object_metrics.evaluate_object_uris()

    def evaluate_object_literals(self) -> dict:
        """Evaluate only literal objects."""
        return self.object_metrics.evaluate_object_literals()

    # ========== Hierarchy-Based Metrics ==========

    def evaluate_class_hierarchies(self) -> dict:
        """Evaluate class usage considering ontology hierarchy."""
        if not self.hierarchy_scorer:
            return {'error': 'Ontology not loaded, hierarchy scoring unavailable'}
        return self.hierarchy_scorer.evaluate_class_hierarchies()

    def evaluate_property_hierarchies(self) -> dict:
        """Evaluate property usage considering ontology hierarchy."""
        if not self.hierarchy_scorer:
            return {'error': 'Ontology not loaded, hierarchy scoring unavailable'}
        return self.hierarchy_scorer.evaluate_property_hierarchies()

    def evaluate_properties_direct(self) -> dict:
        """Evaluate all configured properties with direct matching."""
        if not self.hierarchy_scorer:
            return {'error': 'Ontology not loaded, hierarchy scoring unavailable'}
        return self.hierarchy_scorer.evaluate_all_properties_direct()

    def evaluate_properties_inverse(self) -> dict:
        """Evaluate all configured properties for inverse usage."""
        if not self.hierarchy_scorer:
            return {'error': 'Ontology not loaded, hierarchy scoring unavailable'}
        return self.hierarchy_scorer.evaluate_all_properties_inverse()

    def evaluate_properties_with_hierarchy(self) -> dict:
        """Evaluate configured properties considering hierarchy."""
        if not self.hierarchy_scorer:
            return {'error': 'Ontology not loaded, hierarchy scoring unavailable'}
        return self.hierarchy_scorer.evaluate_multiple_properties_hierarchy(self.config.predicates_to_evaluate)

    # ========== Domain-Specific Metrics ==========

    def evaluate_entity_coverage(self) -> dict:
        """Summarize coverage of expected entities across all types."""
        return self.domain_metrics.summarize_entity_coverage()

    def evaluate_predicate_details(self) -> dict:
        """Detailed evaluation of all configured predicates."""
        return self.domain_metrics.evaluate_all_predicates_detailed(self.hierarchy_scorer)

    # ========== Comprehensive Evaluation ==========

    def evaluate_all(self) -> dict:
        """
        Run all available evaluation metrics.

        Returns:
            dict: Comprehensive evaluation results with all metrics
        """
        results = {
            # Basic, property and object
            **self.evaluate_common(),

            # Domain-specific metrics
            **self.evaluate_in_domain(),
        }

        return results

    def evaluate_common(self) -> dict:
        """
        Run only basic evaluation metrics (fast, no ontology needed).

        Returns:
            dict: Basic evaluation results
        """
        return {
            # Basic metrics
            'triples': self.evaluate_triples(),
            'subjects_unique': self.evaluate_subjects_unique(),
            'subjects_fuzzy_unique': self.evaluate_subjects_fuzzy(),
            'classes': self.evaluate_classes(),
            'classes_unique': self.evaluate_classes_unique(),

            # Property metrics
            'predicates': self.evaluate_properties(),
            'predicates_unique': self.evaluate_properties_unique(),
            'predicate_datatype_range': self.evaluate_predicate_datatypes(),
            'predicate_datatype_range_unique': self.evaluate_predicate_datatypes_unique(),

            # Object metrics
            'objects': self.evaluate_objects(),
            'objects_uris': self.evaluate_object_uris(),
            'objects_literals': self.evaluate_object_literals(),
        }

    def evaluate_in_domain(self) -> dict:
        # Domain-specific metrics
        metrics = {
            'entity_coverage': self.evaluate_entity_coverage(),
        }

        if self.hierarchy_scorer:
            metrics.update({
                'classes_with_hierarchy': self.evaluate_class_hierarchies(),
                'predicates_with_hierarchy': self.evaluate_property_hierarchies(),
                'single_property_hierarchy_scores': self.evaluate_properties_with_hierarchy(),
                'predicates_direct': self.evaluate_properties_direct(),
                'predicates_inverse': self.evaluate_properties_inverse(),
                'predicate_details': self.evaluate_predicate_details(),
            })

        return metrics
