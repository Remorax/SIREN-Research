from ontology import *
from glob import glob
import os
import pitfalls.__init__

class PitfallScanner():
	"""docstring for PitfallScanner"""
	def __init__(self, ontology_path, pitfalls_dir):
		self.ontology = Ontology(ontology_path)
		self.pitfalls_dir = os.path.abspath(pitfalls_dir)

	def scan(self):
		results = []
		for pitfall_module in pitfalls.__init__.__load_all__():
			results.append(pitfall_module.scan(self.ontology))
		return results
