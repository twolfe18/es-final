import numpy as np
import time
import theano

int_type = 'int32'

class Alphabet:

	def __init__(self):
		self._by_key = {}
		self._by_index = []

	def lookup_index(self, key, add=False):
		i = self._by_key.get(key)
		if i:
			return i
		elif add:
			i = len(self._by_key)
			self._by_index.append(key)
			self._by_key[key] = i
			return i
		else:
			raise LookupError('there is no value associated with: ' + str(key))
	
	def __len__(self):
		return len(self._by_index)

class NomlexReader:

	def __init__(self, filename, alph):
		self.word2idx = alph
		self.filename = filename

	def get_pairs(self):
		f = open(self.filename, 'r')
		for line in f:
			nom, verb = line.strip().split()
			n = self.word2idx.lookup_index(nom, add=True)
			v = self.word2idx.lookup_index(verb, add=True)
			yield np.array([n, v], dtype=int_type)
		f.close()

class WindowReader:

	def __init__(self, filename, alph=None):
		self.filename = filename
		if alph is None:
			self.word2idx = Alphabet()
		else:
			self.word2idx = alph

	def get_alphabet(self):
		return self.word2idx

	def get_word_lines(self):
		f = open(self.filename, 'r')
		for line in f:
			yield line.strip().split()
		f.close()

	def get_int_lines(self):
		for words in self.get_word_lines():
			yield np.array([self.word2idx.lookup_index(w, add=True) for w in words], dtype=int_type)

if __name__ == '__main__':
	start = time.clock()
	a = Alphabet()
	r = WindowReader('windows.small', alph=a)
	W = np.array(list(r.get_int_lines()))
	print 'windows', W.shape, 'len(alph)', len(a)
	print "done, took %.1f seconds" % (time.clock()-start)

	# count how many instances have one of the nomlex words in it
	nv = NomlexReader('nomlex.txt', a)
	nomlex_pairs = set(np.array(list(nv.get_pairs())).flatten())
	pos = 0
	for window in W:
		if len(set(window) & nomlex_pairs) > 0:
			pos += 1
	print "%d of %d (%.1f%%) examples have a nomlex word in them" % (pos, len(W), (100.0*pos)/len(W))

	# try to do some training
	d = 128	# number of features per word
	h = 64	# hidden layer size
	k = 5	# how many words in a window
	L = theano.shared(np.zeros((len(a), d), dtype='float32'))	# word vecs
	A = theano.shared(np.zeros((k*d, h), dtype='float32'))		# word vecs => hidden
	b = theano.shared(np.zeros(h, dtype='float32'))				# hidden offset
	p = theano.shared(np.zeros(h, dtype='float32'))				# hidden => output
	t = theano.shared(0.0)										# output offset



