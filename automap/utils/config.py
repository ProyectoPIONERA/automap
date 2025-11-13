"""
Graph Evaluation Configuration

This module contains the Config class for loading configuration from YAML files
and backward-compatible global constants.
"""

import yaml
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Union, Optional


class Config:
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is not valid YAML
            KeyError: If required config keys are missing
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        if not config_data:
            raise ValueError(f"Config file is empty: {config_path}")

        self.ontology_file: str = config_data.get('ontology_file', '')
        self.rdf_type_uri: str = config_data.get('rdf_type_uri', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
        self.base_iri: str = config_data.get('base_iri', '')

        # ids_by_type is optional - can be auto-extracted from reference graph if not provided
        self.ids_by_type: Dict[str, List[str]] = config_data.get('ids_by_type', {})
        self.property_suffixes: List[str] = config_data.get('property_suffixes', [])

        # Namespaces and predicates can be auto-extracted from ontology if not provided
        self.namespaces: Dict[str, str] = config_data.get('namespaces', {})
        self.predicates_to_evaluate_config: Dict[str, List[str]] = config_data.get('predicates_to_evaluate', {})

        # Build full predicate URIs
        self.predicates_to_evaluate = self.build_predicates_list(config_data=config_data)

        queries = config_data.get('sparql_queries', {})
        self.subclass_query: str = queries.get('subclass', self._default_subclass_query())
        self.subproperty_query: str = queries.get('subproperty', self._default_subproperty_query())
        self.subject_class_query: str = queries.get('subject_class', self._default_subject_class_query())
        self.subject_property_query: str = queries.get('subject_property', self._default_subject_property_query())
        self.subject_property_value_query: str = queries.get('subject_property_value',
                                                             self._default_subject_property_value_query())

    def build_predicates_list(self, config_data) -> List[str]:
        predicates = []
        namespaces = config_data.get('namespaces', {})
        predicates_to_evaluate_suffixes = config_data.get('predicates_to_evaluate', {})
        for prefix, suffixes in predicates_to_evaluate_suffixes.items():
            namespace = namespaces.get(prefix, '')
            for suffix in suffixes:
                if namespace:
                    predicates.append(namespace + suffix)
        return predicates

    def extract_ids_from_graph(self, reference_graph) -> None:
        """
        Extract entity IDs by type from the reference graph.

        This automatically populates ids_by_type if it's empty by:
        1. Finding all entities with rdf:type
        2. Extracting the ID from the entity URI (part after base_iri)
        3. Grouping IDs by their entity type

        Args:
            reference_graph: rdflib.Graph containing the reference/gold graph
        """
        if self.ids_by_type:
            # Already populated from config, don't override
            return

        if not self.base_iri:
            # Can't extract IDs without knowing the base IRI
            return

        from rdflib import URIRef

        ids_by_type = {}

        # Query all entities with their types
        for subject, predicate, obj in reference_graph:
            if str(predicate) == self.rdf_type_uri:
                entity_type = str(obj)
                subject_uri = str(subject)

                # Extract ID from subject URI
                if subject_uri.startswith(self.base_iri):
                    # Get the part after base_iri
                    entity_id = subject_uri[len(self.base_iri):]

                    # Handle different URI patterns:
                    # - http://example.org/Venus -> "Venus"
                    # - http://example.org/10 -> "10"
                    # - http://example.org/person/10 -> "10"
                    # - http://example.org/person/Venus -> "Venus"

                    # Take the last segment after the last '/'
                    if '/' in entity_id:
                        entity_id = entity_id.split('/')[-1]

                    # Add to the appropriate type
                    if entity_type not in ids_by_type:
                        ids_by_type[entity_type] = []

                    if entity_id and entity_id not in ids_by_type[entity_type]:
                        ids_by_type[entity_type].append(entity_id)

        self.ids_by_type = ids_by_type

    def _get_ontology_cache_path(self, ontology_path: str) -> Path:
        """
        Get the cache file path for a given ontology.

        Args:
            ontology_path: Path to the ontology file

        Returns:
            Path to the cache file
        """
        # Create cache directory
        cache_dir = Path.home() / '.cache' / 'automap' / 'ontology_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate cache filename based on ontology path and modification time
        ontology_file = Path(ontology_path)
        if ontology_file.exists():
            # Include file mtime in hash to invalidate cache when file changes
            mtime = ontology_file.stat().st_mtime
            cache_key = f"{ontology_path}_{mtime}"
        else:
            cache_key = ontology_path

        # Hash the cache key to create a valid filename
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return cache_dir / f"{cache_hash}.json"

    def _load_ontology_cache(self, ontology_path: str) -> Optional[Dict]:
        """
        Load cached ontology extraction data if available.

        Args:
            ontology_path: Path to the ontology file

        Returns:
            Cached data dict or None if cache doesn't exist or is invalid
        """
        cache_path = self._get_ontology_cache_path(ontology_path)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)

            # Validate cache structure
            if 'namespaces' in cache_data and 'predicates_by_namespace' in cache_data:
                return cache_data
        except (json.JSONDecodeError, IOError):
            # Cache corrupted, ignore
            pass

        return None

    def _save_ontology_cache(self, ontology_path: str, namespaces: Dict[str, str],
                             predicates_by_namespace: Dict[str, List[str]]) -> None:
        """
        Save ontology extraction data to cache.

        Args:
            ontology_path: Path to the ontology file
            namespaces: Extracted namespaces
            predicates_by_namespace: Extracted predicates grouped by namespace
        """
        cache_path = self._get_ontology_cache_path(ontology_path)

        cache_data = {
            'namespaces': namespaces,
            'predicates_by_namespace': predicates_by_namespace,
            'ontology_file': ontology_path
        }

        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except IOError:
            # If caching fails, just continue without cache
            pass

    def extract_from_ontology(self, ontology_graph, ontology_path: Optional[str] = None) -> None:
        """
        Extract namespaces and predicates from the ontology graph.

        This automatically populates:
        - namespaces: Extracted from ontology prefix declarations
        - predicates_to_evaluate: All properties defined in the ontology

        Only extracts if not already provided in config (backward compatible).
        Uses caching to avoid re-parsing the same ontology.

        Args:
            ontology_graph: rdflib.Graph containing the ontology
            ontology_path: Path to the ontology file (for caching)
        """
        from rdflib import OWL, RDF, RDFS
        from rdflib.namespace import NamespaceManager

        # If already provided in config, don't extract (but we might still need to parse for other reasons)
        needs_extraction = not (self.namespaces and self.predicates_to_evaluate_config)

        if not needs_extraction:
            return

        # Try to load from cache first
        cache_data = None
        if ontology_path:
            cache_data = self._load_ontology_cache(ontology_path)

        if cache_data:
            # Use cached data
            if not self.namespaces:
                self.namespaces = cache_data['namespaces']
            if not self.predicates_to_evaluate_config:
                self.predicates_to_evaluate_config = cache_data['predicates_by_namespace']

                # Rebuild the full predicate URIs list
                self.predicates_to_evaluate = []
                for prefix, suffixes in self.predicates_to_evaluate_config.items():
                    namespace = self.namespaces.get(prefix, '')
                    for suffix in suffixes:
                        if namespace:
                            self.predicates_to_evaluate.append(namespace + suffix)
            return

        # Extract namespaces from ontology if not provided in config
        if not self.namespaces:
            self.namespaces = {}
            for prefix, namespace in ontology_graph.namespaces():
                if prefix and prefix not in ['', 'xml', 'rdf', 'rdfs', 'xsd', 'owl']:
                    self.namespaces[prefix] = str(namespace)

        # Extract predicates if not provided in config
        if not self.predicates_to_evaluate_config:
            predicates_by_namespace = {}

            # Query for all DatatypeProperty and ObjectProperty
            property_types = [OWL.DatatypeProperty, OWL.ObjectProperty]

            for prop_type in property_types:
                for prop in ontology_graph.subjects(RDF.type, prop_type):
                    prop_uri = str(prop)

                    # Find which namespace this property belongs to
                    for prefix, namespace in self.namespaces.items():
                        if prop_uri.startswith(namespace):
                            # Extract the local name (suffix)
                            local_name = prop_uri[len(namespace):]

                            if prefix not in predicates_by_namespace:
                                predicates_by_namespace[prefix] = []

                            if local_name not in predicates_by_namespace[prefix]:
                                predicates_by_namespace[prefix].append(local_name)
                            break

            self.predicates_to_evaluate_config = predicates_by_namespace

            # Rebuild the full predicate URIs list
            self.predicates_to_evaluate = []
            for prefix, suffixes in predicates_by_namespace.items():
                namespace = self.namespaces.get(prefix, '')
                for suffix in suffixes:
                    if namespace:
                        self.predicates_to_evaluate.append(namespace + suffix)

            # Save to cache for next time
            if ontology_path:
                self._save_ontology_cache(ontology_path, self.namespaces, predicates_by_namespace)

    @staticmethod
    def _default_subclass_query() -> str:
        return """
SELECT ?superClass ?subClass
WHERE {
    ?subClass rdfs:subClassOf ?superClass .
}
"""

    @staticmethod
    def _default_subproperty_query() -> str:
        return """
SELECT ?superProperty ?subProperty
WHERE {
    ?subProperty rdfs:subPropertyOf ?superProperty .
}
"""

    @staticmethod
    def _default_subject_class_query() -> str:
        return """
SELECT ?class
WHERE {
    ?s a ?class .
}
"""

    @staticmethod
    def _default_subject_property_query() -> str:
        return """
SELECT ?property
WHERE {
    ?s ?property ?o .
}
"""

    @staticmethod
    def _default_subject_property_value_query() -> str:
        return """
SELECT ?property ?value
WHERE {
    ?s ?property ?value .
}
"""

    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"
