#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import sys
import argparse
from typing import Set
from collections import defaultdict
from rdflib import Graph, RDF, RDFS, OWL, BNode, URIRef


class Onto2LightTTL:
    """
    Reduce una ontología OWL/RDFS a lo mínimo necesario para generar mappings.
    """

    def __init__(
        self,
        ordered: bool = True,
    ):
        """
        :param ordered: Si True, devuelve Turtle ordenado por:
                        (1) clases, (2) object properties, (3) data properties.
        """
        self.ordered = ordered

    def __call__(self, data: str, input_format: str = "turtle") -> str:
        """
        :param data: Ontología completa en un string (Turtle, RDF/XML, N3, etc.).
        :param input_format: Formato de entrada entendido por rdflib (por defecto 'turtle').
        :return: String Turtle con la ontología reducida.
        """
        g = Graph()
        g.parse(data=data, format=input_format)

        g_min = self._build_minimal_graph(g)

        if self.ordered:
            light_ontology = self._serialize_ordered(g_min)
        else:
            light_ontology = g_min.serialize(format="turtle")

        light_ontology = light_ontology.replace("rdf:type", "a")
        light_ontology = light_ontology.replace("@prefix", "prefix")

        return light_ontology

    def _get_classes(self, g: Graph) -> Set[URIRef]:
        classes: Set[URIRef] = set()

        for s in g.subjects(RDF.type, OWL.Class):
            if isinstance(s, URIRef):
                classes.add(s)

        for s in g.subjects(RDF.type, RDFS.Class):
            if isinstance(s, URIRef):
                classes.add(s)

        return classes

    def _get_object_properties(self, g: Graph) -> Set[URIRef]:
        props: Set[URIRef] = set()
        for s in g.subjects(RDF.type, OWL.ObjectProperty):
            if isinstance(s, URIRef):
                props.add(s)
        return props

    def _get_datatype_properties(self, g: Graph) -> Set[URIRef]:
        props: Set[URIRef] = set()
        for s in g.subjects(RDF.type, OWL.DatatypeProperty):
            if isinstance(s, URIRef):
                props.add(s)
        return props

    def _build_minimal_graph(self, g: Graph) -> Graph:
        classes = self._get_classes(g)
        obj_props = self._get_object_properties(g)
        data_props = self._get_datatype_properties(g)
        all_props = obj_props | data_props

        g_min = Graph()
        g_min.namespace_manager = g.namespace_manager

        # 1) Declaraciones de clases
        for c in classes:
            for _, p, o in g.triples((c, RDF.type, None)):
                if o in (OWL.Class, RDFS.Class):
                    g_min.add((c, p, o))

        # 2) Declaraciones de propiedades
        for p in all_props:
            for _, pred, o in g.triples((p, RDF.type, None)):
                if o in (OWL.ObjectProperty, OWL.DatatypeProperty):
                    g_min.add((p, pred, o))

        # 3) Jerarquía de clases (sin BNodes → sin restricciones)
        for s, o in g.subject_objects(RDFS.subClassOf):
            if isinstance(s, BNode) or isinstance(o, BNode):
                continue
            if s in classes and o in classes:
                g_min.add((s, RDFS.subClassOf, o))

        # 4) Jerarquía de propiedades (simple)
        for s, o in g.subject_objects(RDFS.subPropertyOf):
            if isinstance(s, BNode) or isinstance(o, BNode):
                continue
            if s in all_props and o in all_props:
                g_min.add((s, RDFS.subPropertyOf, o))

        # 5) Domain / range de propiedades (sin BNodes)
        for p in all_props:
            for _, _, d in g.triples((p, RDFS.domain, None)):
                if not isinstance(d, BNode):
                    g_min.add((p, RDFS.domain, d))
            for _, _, r in g.triples((p, RDFS.range, None)):
                if not isinstance(r, BNode):
                    g_min.add((p, RDFS.range, r))

        # 6) (Opcional) labels
        for s in classes | all_props:
            for _, _, lbl in g.triples((s, RDFS.label, None)):
                g_min.add((s, RDFS.label, lbl))

        # 7) comments
        for s in classes | all_props:
            for _, _, cmt in g.triples((s, RDFS.comment, None)):
                g_min.add((s, RDFS.comment, cmt))

        return g_min

    def _serialize_ordered(self, g: Graph) -> str:
        """
        Serializa el grafo en Turtle ordenando:
        1) Clases
        2) Object properties
        3) Datatype properties
        """

        classes = self._get_classes(g)
        obj_props = self._get_object_properties(g)
        data_props = self._get_datatype_properties(g)

        buff = io.StringIO()
        nm = g.namespace_manager

        # Prefijos
        for prefix, ns in sorted(g.namespaces(), key=lambda x: x[0] or ""):
            if prefix is None or prefix == "":
                # puedes decidir si quieres usar prefix vacío o no
                continue
            buff.write(f"@prefix {prefix}: <{ns}> .\n")
        buff.write("\n")

        # Helper para serializar un grupo de sujetos
        def write_block(subjects: Set[URIRef]):
            for s in sorted(subjects, key=lambda x: str(x)):
                triples = list(g.triples((s, None, None)))
                if not triples:
                    continue
                pred_map = defaultdict(list)
                for _, p, o in triples:
                    pred_map[p].append(o)

                subj_str = s.n3(nm)
                buff.write(subj_str + " ")

                preds_sorted = sorted(pred_map.keys(), key=lambda x: str(x))
                pred_lines = []
                for p in preds_sorted:
                    objs = sorted(pred_map[p], key=lambda x: str(x))
                    p_str = p.n3(nm)
                    objs_str = ", ".join(o.n3(nm) for o in objs)
                    pred_lines.append(f"{p_str} {objs_str}")
                buff.write(" ;\n    ".join(pred_lines))
                buff.write(" .\n\n")

        write_block(classes)
        write_block(obj_props)
        write_block(data_props)

        return buff.getvalue()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Transform RDF Turtle to Turtle Light.',
    )

    parser.add_argument('--format', type=str, default='turtle')
    parser.add_argument('-o', '--output', type=str, help='(default: stdout)')

    args = parser.parse_args()

    return args


def api_test():
    """API test function."""
    sample_turtle_path = "/home/carlos/workspace/automap/datasets/blinkg/data/scenario1/ontology.ttl"
    input_format = "turtle"
    output = None

    with open(sample_turtle_path, 'r', encoding='utf-8') as f:
        ontology = f.read()

    ttl2light = Onto2LightTTL(ordered=True)
    result = ttl2light(ontology, input_format=input_format)

    output_path = output if output else sys.stdout
    print(result, file=output_path)


def main():
    """Main CLI entry point."""
    args = parse_args()

    ontology = sys.stdin.read()
    ttl2light = Onto2LightTTL(ordered=True)
    result = ttl2light(ontology, input_format=args.format)

    output_path = args.output if args.output else sys.stdout
    print(result, file=output_path)


if __name__ == '__main__':
    # api_test()
    main()
