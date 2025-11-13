from rdflib import Graph
from automap.utils import Config


class Metrics:
    def __init__(self, test_graph: Graph, reference_graph: Graph, ontology_graph: Graph = None, config: Config = None):
        """
        Initialize basic metrics calculator.

        Args:
            test_graph: The RDF graph to evaluate
            reference_graph: The ground truth RDF graph
            ontology_graph: The ontology RDF graph for hierarchy-based metrics (optional)
            config: Configuration object (optional)
        """
        self.test_graph = test_graph
        self.reference_graph = reference_graph
        self.config = config
        self.ontology_graph = ontology_graph
