
from learn_word_vecs import *
import numpy as np
import numpy.random as rand
import sys

class VanillaTrainer:
	def run(self):
		outdir = 'models/vanilla'
		infile = ''
		a = Alphabet()
		ve = VanillaEmbeddings(alph, k)

		#r = WindowReader(infile, a)
		r = SubsetWindowReader(infile, a)
		r.set_partition_size(int(1e6))
		for i in range(r.num_partitions()):
			W = r.get_partition(i)


