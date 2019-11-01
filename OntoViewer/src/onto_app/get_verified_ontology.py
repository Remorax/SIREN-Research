import sys

from aggregate import generate_final_ontology

name = sys.argv[1]
print (name)
generate_final_ontology(name)