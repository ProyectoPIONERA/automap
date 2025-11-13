"""
GraphEval - RDF Graph Evaluation Package

A comprehensive package for evaluating and comparing RDF graphs with support for:
- Basic metrics (precision, recall, F1-score)
- Hierarchy-based scoring using ontologies
- Domain-specific metrics
"""

__version__ = '0.1.0'
__author__ = 'Carlos Golvano <carlos.golvano@upm.es>'

from .metrics import GraphEvaluator

__all__ = [
    'GraphEvaluator',
]
