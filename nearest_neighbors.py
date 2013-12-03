import os, sys
import numpy as np
from util import *

class NearestNeighbors:
	def __init__(self, model_dir, show_vec=True):
		print '[NearestNeighbors] loading from', model_dir
		self.show_vec = show_vec
		self.model_dir = model_dir
		self.alph = Alphabet(os.path.join(model_dir, 'alphabet.txt'))
		self.W = np.load(os.path.join(model_dir, 'W.npy'))

	def nearest_to(self, word, k=15):
		#try:
		i = self.alph.lookup_index(word, add=False)
		v = self.W[i]
		if self.show_vec:
			print v
		print 'sorting vecs...'
		cands = [(w, self.dist(v, wv)) for w, wv in self.words_and_vecs()]
		return sorted(cands, key=lambda wd: wd[1])[:k]
		#except:
		#	print "[nearest_to] \"%s\" was not found in the alphabet (len=%d)" % (word, len(self.alph))
		#	return None

	def dist(self, v1, v2):
		return np.linalg.norm(v1 - v2, ord=2)
	
	def words_and_vecs(self):
		for i in range(len(self.alph)):
			w = self.alph.lookup_value(i)
			yield (w, self.W[i])

if __name__ == '__main__':
	
	model_dir = 'models/additive-initialization/'
	if len(sys.argv) == 2:
		model_dir = sys.argv[1]
	
	nn = NearestNeighbors(model_dir)

	while True:
		word = raw_input('> ')
		g = nn.nearest_to(word)
		if g:
			for syn, dist in g:
				print '\t', syn, '\t', dist
		print

