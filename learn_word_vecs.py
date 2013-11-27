import numpy as np
import time
import theano
import theano.tensor as T

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
		start = time.clock()
		f = open(self.filename, 'r')
		for line in f:
			nom, verb = line.strip().split()
			n = self.word2idx.lookup_index(nom, add=True)
			v = self.word2idx.lookup_index(verb, add=True)
			yield np.array([n, v], dtype=int_type)
		f.close()
		print "[NomlexReader] reading pairs from %s took %.1f sec" % (self.filename, time.clock()-start)

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

class VanillaEmbeddings:
	""" Try learning without E+N/V for now """
	
	# TODO needs to be able to read in vectors

	def __init__(self, num_words, k, d=64, h=40, batch_size=30):
		self.d = d	# number of features per word
		self.h = h	# hidden layer size
		self.k = k	# how many words in a window
		self.num_words = num_words
		self.batch_size = batch_size

	def setup_theano(self):

		# 1-hidden layer network
		self.W = theano.shared(np.zeros((self.num_words, self.d), dtype='float32'), name='W')	# word vecs
		self.A = theano.shared(np.zeros((self.k * self.d, self.h), dtype='float32'), name='A')	# word vecs => hidden
		self.b = theano.shared(np.zeros(self.h, dtype='float32'), name='b')						# hidden offset
		self.p = theano.shared(np.zeros(self.h, dtype='float32'), name='p')						# hidden => output
		self.t = theano.shared(0.0, name='t')

		word_indices = T.imatrix('phrases')	# each row is a phrase, should have self.k columns
		n, k = word_indices.shape	# won't know this until runtime
		phrases_tensor = self.W[word_indices]	# shape=(n, k, self.d)
		phrases = phrases_tensor.reshape((n, k * self.d))
		hidden = T.tanh( T.dot(phrases, self.A) + self.b )
		scores = T.tanh( T.dot(hidden, self.p) + self.t )

		# score function
		self.f_score = theano.function([word_indices], [scores])

		# SGD step function
		word_indices_corrupted = T.imatrix('word_indices_corrupted')
		scores_corrupted = theano.clone(scores, replace={word_indices: word_indices_corrupted})
		loss_ = T.ones_like(scores) + scores_corrupted - scores
		loss = loss_ * (loss_ > 0)
		avg_loss = loss.mean()
		self.learning_rates = {self.W : 1e-2, self.A : 1e-3, self.b : 1e-3, self.p : 1e-3, self.t : 1e-3}
		updates = []
		grads = {}
		for param, lr in self.learning_rates.iteritems():
			grad = theano.grad(cost=avg_loss, wrt=param)
			update = param - grad
			print 'lr=', lr
			print 'param=', param
			print 'update=', update
			print
			updates.append( (param, update) )
			grads[param.name] = grad

		self.step = theano.function([word_indices, word_indices_corrupted], [avg_loss], updates=updates)

		dscore = theano.grad(cost=scores.mean(), wrt=self.W)
		self.step_debug = theano.function([word_indices, word_indices_corrupted],
			[avg_loss, scores, scores_corrupted,
			 dscore,
			 grads['W'], grads['A'], grads['b'], grads['p'], grads['t']], updates=updates)


	def train(self, phrases):
		""" phrases should be a matrix of word indices, rows are phrases, should have self.k columns """
		assert phrases.shape[1] == self.k
		epochs = 100
		N = len(phrases)
		phrases_corrupted = self.corrupt(phrases)
		for e in range(epochs):
			print "starting epoch %d" % (e)
			for i in range(0, N, self.batch_size):
				j = min(i + self.batch_size, N)
				avg_loss = self.step(phrases[i:j,], phrases_corrupted[i:j,])[0]
				if np.random.randint(100) == 0:
					print "e=%d i=%d avg_loss=%.5g" % (e, i, avg_loss)
					print 'some vector =', self.W.get_value()[5,]
					print 'p vector =', self.p.get_value()
					print 'W.l1 = ', sum(abs(self.W.get_value()))
					print 'W.l2 = ', sum(self.W.get_value() ** 2)
					print 'A.l1 = ', sum(abs(self.A.get_value()))
					print 'A.l2 = ', sum(self.A.get_value() ** 2)


	def debug_train(self, phrases):
		phrases_corrupted = self.corrupt(phrases)
		for i in range(5):
			j = i+1
			p = phrases[i:j,]
			pc = phrases_corrupted[i:j,]
			avg_loss, s, sc, stupid, dW, dA, db, dp, dt = self.step_debug(p, pc)
			print "original  = %s score=%.5f" % (p, s)
			print "corrupted = %s score=%.5f" % (pc, sc)
			print 'avg_loss =', avg_loss
			print 'dW =', dW
			print 'dA =', dA
			print 'db =', db
			print 'dp =', dp
			print 'dt =', dt
			print 'stupid =', stupid
			print
			print 'W =', self.W.get_value()
			print 'A =', self.A.get_value()
			print 'b =', self.b.get_value()
			print 'p =', self.p.get_value()
			print 't =', self.t.get_value()
			print
			print


	def score(self, words, alph):
		""" gives the NNs score of this phrase
			words should be a list of strings and alph and Alphabet containing those strings
		"""
		assert type(words) == list
		w = [a.lookup_index(x, add=False) for x in words]
		i = np.mat( np.array(w, dtype='int32') )
		return self.f_score(i)[0]

	def init_weights(self):

		def set_rand_value(theano_tensor, scale=1.0):
			old_vals = theano_tensor.get_value()
			new_vals = (np.random.rand(*old_vals.shape) - 0.5) * scale
			if(old_vals.dtype != new_vals.dtype):
				new_vals = np.asfarray(new_vals, dtype=old_vals.dtype)
			theano_tensor.set_value(new_vals)

		def set_unif_value(theano_tensor, value):
			old_vals = theano_tensor.get_value()
			new_vals = np.tile(value, old_vals.shape)
			if(old_vals.dtype != new_vals.dtype):
				new_vals = np.asfarray(new_vals, dtype=old_vals.dtype)
			theano_tensor.set_value(new_vals)

		#set_rand_value(self.W, scale=1e-5)
		set_unif_value(self.A, 1e-3)
		#set_rand_value(self.b, scale=1e-7)
		set_rand_value(self.p, scale=1e-2)
		#self.t.set_value(0.0)

	def corrupt(self, phrases):
		""" phrases should be a matrix of word indices, rows are phrases, should have self.k columns """
		n, k = phrases.shape
		assert k == self.k
		mid = k // 2
		corrupted = phrases.copy()
		corrupted[:,mid] = np.random.random_integers(0, self.num_words-1, n)
		print '[corrupt] orig =', phrases[1:10,]
		print '[corrupt] corrupted =', corrupted[1:10,]
		return corrupted


if __name__ == '__main__':
	start = time.clock()
	a = Alphabet()
	#r = WindowReader('windows.small', alph=a)
	r = WindowReader('fake_data.txt', alph=a)
	W = np.array(list(r.get_int_lines()))
	print 'windows', W.shape, 'len(alph)', len(a)
	print "done, took %.1f seconds" % (time.clock()-start)

	"""
	# count how many instances have one of the nomlex words in it
	nv = NomlexReader('nomlex.txt', a)
	nomlex_pairs = set(np.array(list(nv.get_pairs())).flatten())
	pos = 0
	for window in W:
		if len(set(window) & nomlex_pairs) > 0:
			pos += 1
	print "%d of %d (%.1f%%) examples have a nomlex word in them" % (pos, len(W), (100.0*pos)/len(W))
	"""

	print 'making vanilla embeddings...'
	emb = VanillaEmbeddings(len(a), k=3, d=10, h=3, batch_size=20)
	emb.setup_theano()
	emb.init_weights()

	"""
	phrase = ['the', 'quick', 'brown', 'fox', 'jump']
	score = emb.score(phrase, a)	# these words appear in windows.small
	print "score for %s is %.3f" % (' '.join(phrase), score)
	"""

	emb.debug_train(W)
	emb.train(W)

