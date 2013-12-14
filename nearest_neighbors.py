import os, sys
import scipy
import numpy as np
from util import *

class NearestNeighbors:
	def __init__(self, model_dir, show_vec=True):
		print '[NearestNeighbors] loading from', model_dir
		self.show_vec = show_vec
		self.model_dir = model_dir
		self.alph = Alphabet(os.path.join(model_dir, 'alphabet.txt'))
		self.W = np.load(os.path.join(model_dir, 'W.npy'))

		import pickle
		rami_emb_file = '/home/travis/Dropbox/School/event-semantics/final-project/data/polyglot-en.pkl'
		words, embeddings = pickle.load(open(rami_emb_file, 'rb'))
		self.W = embeddings
		self.alph = Alphabet()
		for w in words: self.alph.lookup_index(w, add=True)

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
		#return np.linalg.norm(v1 - v2, ord=2)
		return scipy.spatial.distance.cosine(v1, v2)

	def nearest_to2(self, word1, word2, k=15):
		#try:
		i1 = self.alph.lookup_index(word1, add=False)
		i2 = self.alph.lookup_index(word2, add=False)
		v1 = self.W[i1]
		v2 = self.W[i2]
		if self.show_vec:
			print word1, v1
			print word2, v2
		print 'sorting vecs...'
		cands = [(w, self.dist2(wv, v1, v2)) for w, wv in self.words_and_vecs()]
		return sorted(cands, key=lambda wd: wd[1])[:k]
		#except:
		#	print "[nearest_to] \"%s\" was not found in the alphabet (len=%d)" % (word, len(self.alph))
		#	return None

	def dist2(self, vec, c1, c2, softness=0.1):
		d1 = self.dist(vec, c1)
		d2 = self.dist(vec, c2)
		return (d1 + softness) + (d2 + softness)
	
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
		ar = word.split()
		if len(ar) == 1:
			g = nn.nearest_to(word)
		elif len(ar) == 2:
			g = nn.nearest_to2(ar[0], ar[1])
		else:
			continue
		if g:
			for syn, dist in g:
				print '\t', syn, '\t', dist
		print

