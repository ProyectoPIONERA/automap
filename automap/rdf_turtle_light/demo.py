#!/usr/bin/env python3
"""Quick demo of the RDF Turtle Light module."""

from rdf_turtle_light import simplify_turtle, generate_all_variations

# Sample Turtle input
turtle = """
@prefix ex: <http://example.org/>.
@prefix foaf: <http://xmlns.com/foaf/0.1/>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

ex:homerSimpson a foaf:Person ;
    foaf:name "Homer Simpson"^^xsd:string ;
    ex:birthDate "1987-04-19"^^xsd:date ;
    ex:birthYear "1987"^^xsd:gYear .
"""

print("=" * 70)
print("RDF Turtle Light - Quick Demo")
print("=" * 70)

print("\nOriginal Turtle:")
print("-" * 70)
print(turtle)

print("\nTurtle Ultra Light (inline, factorised, no datatype):")
print("-" * 70)
result = simplify_turtle(turtle, datatype=False, inline=True, factorised=True)
print(result)

print("\n" + "=" * 70)
print("Demo complete!")
