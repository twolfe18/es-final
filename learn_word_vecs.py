import sys
import math
import time
import itertools
import numpy as np
import theano
import theano.tensor as T

# TODO upgrade to >=0.6rc5 and switch to float32s
int_type = 'int32'
float_type = 'float64'

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
		self.phrase_mat = None

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
	
	def get_phrase_matrix(self):
		""" return a matrix where rows are phrases, should be ~5 columns, entries are ints taken from alph """
		if self.phrase_mat is None:
			self.phrase_mat = np.array(list(self.get_int_lines()))
		return self.phrase_mat

class Regularizer:

	def __init__(self):
		self.seen = set()
		self.penalties = []

	def l1(self, var, coef):
		self.elastic_net(var, coef, 0.0)

	def l2(self, var, coef):
		self.elastic_net(var, 0.0, coef)
		
	def elastic_net(self, var, l1_coef, l2_coef):
		assert l1_coef >= 0.0 and l2_coef >= 0.0
		if var in self.seen:
			raise 'you should regularize once!'
		else:
			self.seen.add(var)
		if l1_coef > 0.0:
			self.penalties.append(var.norm(1) * -l1_coef)
		if l2_coef > 0.0:
			self.penalties.append(var.norm(2) * -l2_coef)
	
	def regularization_var(self):
		""" returns a theano scalar variable for how much penalty has been incurred """
		if len(self.penalties) > 0:
			return sum(self.penalties)
		return None

class AdaGradParam:
	""" only setup to do minimization """

	def __init__(self, theano_mat, input_vars, cost, learning_rate=1e-2, delta=1e-2):
		""" cost should be a theano variable that this var should take gradients wrt
			input_vars should be a list of variables for whic you'll provide values when you call update()
		"""
		debug = False

		self.tvar = theano_mat	# should be a theano.shared
		self.gg = theano.shared(np.ones_like(self.tvar.get_value(), dtype=float_type) * delta)
		if debug:
			print 'gg.type =', self.gg.type

		# TODO upgrade to >=0.6rc5 and switch to float32s

		# there is a bug that has been fixed by 0.6rc5 where when you
		# multiply a variable with dtype='float32' with a theano.tensor.constant,
		# you get something back with dtype='float64'
		# if you get errors in this code, that is almost certainly why (unless your on >=0.6rc5)
		self.lr = T.constant(learning_rate)

		grad = theano.grad(cost=cost, wrt=self.tvar)
		if debug: print 'grad.type =', grad.type

		gg_update = self.gg + (grad ** 2)
		tvar_update = self.tvar - self.lr * grad / (self.gg ** 0.5)
		if debug:
			print 'gg_update.type =', gg_update.type
			print 'tvar_update.type =', tvar_update.type
		self.updates = [(self.gg, gg_update), (self.tvar, tvar_update)]
		print '[AdaGradParam __init__]', self.tvar.name, 'input_vars =', input_vars, type(input_vars)
		self.f_update = theano.function(input_vars, grad, updates=self.updates)

		if debug:
			print 'f_update ='
			theano.printing.debugprint(self.f_update.maker.fgraph.outputs[0])
			print

	def update(self, args, verbose=False):
		""" args should match input_vars provided to __init__ """
		if verbose:
			print "[AdaGradParam update] args =", args
			print "[AdaGradParam update] %s: before   = %s" % (self.name, self.tvar.get_value())
		g = self.f_update(*args)
		if verbose:
			print "[AdaGradParam update] %s: gradient = %s" % (self.name, g)
			print "[AdaGradParam update] %s: after    = %s" % (self.name, self.tvar.get_value())
		return g
	
	def __str__(self):
		return "<AdaGradParam tvar.shape=%s lr=%g gg.l2=%g>" % \
			(self.tvar.get_value().shape, self.lr.value,
			np.linalg.norm(self.gg.get_value(), ord=2))
	
	@property
	def name(self): return self.tvar.name
	@property
	def shape(self): return self.tvar.get_value().shape
	@property
	def l0(self): return np.linalg.norm(self.tvar.get_value(), ord=0)
	@property
	def l1(self): return np.linalg.norm(self.tvar.get_value(), ord=1)
	@property
	def l2(self): return np.linalg.norm(self.tvar.get_value(), ord=2)
	@property
	def lInf(self): return np.linalg.norm(self.tvar.get_value(), ord=np.inf)
	def contains_bad_values(self):
		vals = self.tvar.get_value().flatten()
		nan = np.isnan(vals).any()
		inf = np.isinf(vals).any()
		return nan or inf

class VanillaEmbeddings:
	""" Try learning without E+N/V for now """
	
	# TODO need to have a dev set for reporting performance
	# TODO needs to be able to read/write state

	def __init__(self, num_words, k, d=64, h=40, batch_size=30):
		assert k >= 3 and k % 2 == 1
		assert num_words > k
		self.d = d	# number of features per word
		self.h = h	# hidden layer size
		self.k = k	# how many words in a window
		self.num_words = num_words
		self.batch_size = batch_size

		# 1-hidden layer network
		self.W = theano.shared(np.zeros((self.num_words, self.d), dtype=float_type), name='W')	# word vecs
		self.A = theano.shared(np.zeros((self.k * self.d, self.h), dtype=float_type), name='A')	# word vecs => hidden
		self.b = theano.shared(np.zeros(self.h, dtype=float_type), name='b')					# hidden offset
		self.p = theano.shared(np.zeros(self.h, dtype=float_type), name='p')					# hidden => output
		self.t = theano.shared(0.0, name='t')

		word_indices = T.imatrix('word_indices')	# each row is a phrase, should have self.k columns
		n, k = word_indices.shape	# won't know this until runtime
		phrases_tensor = self.W[word_indices]	# shape=(n, k, self.d)
		phrases = phrases_tensor.reshape((n, k * self.d))
		hidden = T.tanh( T.dot(phrases, self.A) + self.b )
		scores = T.tanh( T.dot(hidden, self.p) + self.t )

		# score function
		self.f_score = theano.function([word_indices], [scores])

		# regularization
		self.reg = Regularizer()
		#self.reg.l2(self.W, 1e-5)
		#self.reg.l1(self.A, 1e-4)

		# loss
		word_indices_corrupted = T.imatrix('word_indices_corrupted')
		scores_corrupted = theano.clone(scores, replace={word_indices: word_indices_corrupted})
		loss_neg = T.ones_like(scores) + scores_corrupted - scores
		loss = loss_neg * (loss_neg > 0)
		r = self.reg.regularization_var()
		if r is None:
			avg_loss = loss.mean()
		else:
			avg_loss = loss.mean() + r

		args = [word_indices, word_indices_corrupted]
		print 'args =', args
		self.params = {
			'W' : AdaGradParam(self.W, args, avg_loss, learning_rate=1.0),
			'A' : AdaGradParam(self.A, args, avg_loss, learning_rate=1e-1),
			'b' : AdaGradParam(self.b, args, avg_loss, learning_rate=1e-2),
			'p' : AdaGradParam(self.p, args, avg_loss, learning_rate=1e-3),
			't' : AdaGradParam(self.t, args, avg_loss, learning_rate=1e-4) \
		}

		upd = [p.updates for p in self.params.values()]
		upd = list(itertools.chain(*upd))	# flatten list
		print 'updates =', upd
		self.f_step = theano.function([word_indices, word_indices_corrupted], [avg_loss], updates=upd)

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
				avg_loss = self.f_step(phrases[i:j,], phrases_corrupted[i:j,])[0]
				if np.random.randint(100) == 0:
					print "e=%d i=%d avg_loss=%.5g" % (e, i, avg_loss)
					print 'W.l1 =', self.params['W'].l1()
					print 'W.l2 =', self.params['W'].l2()
					print 'A.l1 =', self.params['A'].l1()
					print 'A.l2 =', self.params['A'].l2()

	# user-friendly version
	def score(self, words, alph):
		""" gives the NNs score of this phrase
			words should be a list of strings and alph and Alphabet containing those strings
		"""
		assert type(words) == list
		w = [a.lookup_index(x, add=False) for x in words]
		i = np.mat( np.array(w, dtype='int32') )
		return self.f_score(i)[0]

	def raw_score(self, phrases):
		""" expects a matrix of word indices, rows are phrases """
		N, k = phrases.shape
		assert k == self.k
		return self.f_score(phrases)[0]
	
	def loss(self, phrases, corrupted_phrases, avg=True):
		""" returns the hinge loss on this data """
		assert phrases.shape == corrupted_phrases.shape
		g = self.raw_score(phrases)
		b = self.raw_score(corrupted_phrases)

		debug = False
		if debug:
			print 'uncorrupted scores ='
			print g[:100]
			print 'corrupted scores ='
			print b[:100]

		one = np.ones_like(g)
		hinge = one + b - g
		hinge = hinge * (hinge > 0.0)
		if avg:
			return hinge.mean()
		else:
			return hinge.sum()

	def init_weights(self, scale=1.0):

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

		#set_rand_value(self.W, scale=1e-5*scale)
		set_unif_value(self.A, 1e-3*scale)
		#set_rand_value(self.b, scale=1e-7*scale)
		set_rand_value(self.p, scale=1e-2*scale)
		#self.t.set_value(0.0)
	
	def read_weights(self, filename):
		raise 'imlement me!'
	
	def write_weights(self, filename):
		raise 'imlement me!'

	def corrupt(self, phrases):
		""" phrases should be a matrix of word indices, rows are phrases, should have self.k columns """
		n, k = phrases.shape
		assert k == self.k
		mid = k // 2
		corrupted = phrases.copy()
		corrupted[:,mid] = np.random.random_integers(0, self.num_words-1, n)
		return corrupted

	def check_for_bad_params(self):
		bad = []
		for name, param in self.params.iteritems():
			if param.contains_bad_values():
				print name, 'contains bad values'
				bad.append(param)
		return bad

if __name__ == '__main__':

	# count how many instances have one of the nomlex words in it
	nv = NomlexReader('nomlex.txt', a)
	nomlex_pairs = set(np.array(list(nv.get_pairs())).flatten())
	pos = 0
	for window in W:
		if len(set(window) & nomlex_pairs) > 0:
			pos += 1
	print "%d of %d (%.1f%%) examples have a nomlex word in them" % (pos, len(W), (100.0*pos)/len(W))

	phrase = ['the', 'quick', 'brown', 'fox', 'jump']
	score = emb.score(phrase, a)	# these words appear in windows.small
	print "score for %s is %.3f" % (' '.join(phrase), score)

