import os
import sys
import math
import time
import codecs
import itertools
import numpy as np
import theano
import theano.tensor as T

class Alphabet(object):

	def __init__(self, filename=None):
		self._by_key = {}
		self._by_index = []
		if filename is None: return
		with codecs.open(filename, 'r', 'utf-8') as f:
			for line in f:
				line = line.strip()
				ar = line.split()
				key = ar[0]
				i = int(ar[1])
				assert i == len(self._by_key)
				self._by_index.append(key)
				self._by_key[key] = i

	def lookup_index(self, key, add=False):
		i = self._by_key.get(key)
		if i is not None:
			return i
		elif add:
			i = len(self._by_key)
			self._by_index.append(key)
			self._by_key[key] = i
			return i
		else:
			raise LookupError('there is no value associated with: ' + str(key))

	def lookup_value(self, index):
		return self._by_index[index]
	
	def save(self, filename):
		with codecs.open(filename, 'w', 'utf-8') as f:
			for i, key in enumerate(self._by_index):
				f.write("%s\t%d\n" % (key, i))
	
	def __len__(self):
		return len(self._by_index)

	def __str__(self):
		s = len(self)
		if s < 10:
			kvs = ', '.join([str(k) + ':' + str(v) for k, v in self._by_key.iteritems()])
			return "<Alphabet %s>" % (kvs)
		return "<Alphabet size=%d>" % (s)

class CountAlphabet(Alphabet, object):
	def __init__(self):
		self.counts = []
		super(CountAlphabet, self).__init__()
	def lookup_index(self, key, add=False):
		i = super(CountAlphabet, self).lookup_index(key, add)
		if i is not None:
			if i >= len(self.counts):
				self.counts.extend( [0] * (i - len(self.counts) + 1) )
			self.counts[i] += 1
		return i
	def count(self, key):
		#i = super(CountAlphabet, self).lookup_index(key, add)
		i = self.lookup_index(key, add)
		if i is None:
			return 0
		else:
			return self.counts[i]
	def high_count_keys(self, count):
		#by_index = super(CountAlphabet, self)._by_index
		for i, key in enumerate(self._by_index):
			if self.counts[i] >= count:
				yield key

class NomlexReader:

	def __init__(self, filename, alph):
		self.word2idx = alph
		self.filename = filename

	def get_int_pairs(self, take_from=None):
		if take_from is None:
			take_from = self.get_uniq_word_pairs
		start = time.clock()
		for nom, verb in take_from():
			try:
				n = self.word2idx.lookup_index(nom, add=False)
				v = self.word2idx.lookup_index(verb, add=False)
				yield (n, v)
			except:
				pass
		print "[NomlexReader] reading pairs from %s took %.1f sec" % (self.filename, time.clock()-start)

	def get_word_pairs(self):
		f = open(self.filename, 'r')
		for line in f:
			nom, verb = line.strip().split()
			yield (nom, verb)
		f.close()
	
	def get_uniq_word_pairs(self):
		""" if we see a word used more than once, we'll only take the first occurrence """
		seen = set()
		for nom, verb in self.get_word_pairs():
			if nom in seen: continue
			if verb in seen: continue
			seen.add(nom)
			seen.add(verb)
			yield (nom, verb)

	def get_features(self, phrase_matrix):
		""" pass in a numpy matrix of phrases """
		# features:
		# 0 for not in NOMLEX
		# 1 for nom in NOMLEX
		# 2 for verb in NOMLEX
		N, k = phrase_matrix.shape
		assert phrase_matrix.max() < len(self.word2idx)
		feat_map = np.zeros(len(self.word2idx), dtype=int_type)
		for nom, verb in self.get_int_pairs(take_from=self.get_uniq_word_pairs):
			assert feat_map[nom] == 0 or feat_map[nom] == 1, 'collision for ' + self.word2idx.lookup_value(nom)
			assert feat_map[verb] == 0 or feat_map[verb] == 2, 'collision for ' + self.word2idx.lookup_value(verb)
			#print "[get_features] len(alph)=%d nom=%d verb=%d" % (len(self.word2idx), nom, verb)
			feat_map[nom] = 1
			feat_map[verb] = 2
		features = feat_map[phrase_matrix]
		assert features.shape == phrase_matrix.shape
		return features

	# i think i may need to rethink how i'm doing these nomlex pairs
	# my previous idea was to have
	# line[x] = "nom-word verb-word"
	# replace all occurrences of "nom-word" with [word-x + nom]
	# but this is ambiguous given that "nom-word may appear in NOMLEX more than once

	# option 1: tie-breaking heuristic
	# option 2: min cut in bipartite graph
	# option 3: remove duplicated words

class WindowReader:

	def __init__(self, filename, window_size, alph=None, oov='<OOV>'):
		""" if you give an OOV token, it will be swapped in for things in the alphabet
			if you don't, then OOV tokens will be added to the alphabet
		"""
		self.filename = filename
		self.window_size = window_size
		assert type(window_size) == int
		assert os.path.isfile(filename)
		self.oov = oov
		if alph is None:
			self.word2idx = Alphabet()
			self.word2idx.lookup_index(oov, add=True)
		else:
			self.word2idx = alph
		self.phrase_mat = None

	def get_alphabet(self):
		return self.word2idx

	def get_word_lines(self):
		skipped = 0
		total = 0
		sizes = set()
		f = open(self.filename, 'r')
		for line in f:
			ar = line.strip().split()
			if len(ar) == self.window_size:
				yield ar
			else:
				skipped += 1
				sizes.add(len(ar))
			total += 1
		f.close()
		print "[get_word_lines] skipped %d of %d lines in %s" % (skipped, total, self.filename)
		if skipped > 0:
			print "[get_word_lines] sizes of skipped =", sizes
			print "[get_word_lines] self.window_size =", self.window_size
			assert False

	def get_int_lines(self):
		for words in self.get_word_lines():
			if self.oov is None:
				yield np.array([self.word2idx.lookup_index(w, add=True) for w in words], dtype=int_type)
			else:
				r = np.zeros(self.window_size, dtype=int_type)
				for idx, t in enumerate(words):
					try:
						i = self.word2idx.lookup_index(t, add=False)
					except:
						i = self.word2idx.lookup_index(self.oov, add=False)
					r[idx] = i
				yield r
	
	def get_phrase_matrix(self):
		""" return a matrix where rows are phrases, should be ~5 columns, entries are ints taken from alph """
		if self.phrase_mat is None:
			self.phrase_mat = np.array(list(self.get_int_lines()))
		return self.phrase_mat

class NumpyWindowReader:
	def __init__(self, filename):
		self.filename = filename
		self.cache = None
	
	def get_phrase_matrix(self):
		if self.cache is None:
			self.cache = np.load(self.filename)
		return self.cache

class MultiWindowReader:
	
	def __init__(self, files, oov='<OOV>'):
		self.files = files
		self.partition = 0
		self.cache = None
		self.oov = oov
	
	def set_partition(self, i):
		if i == self.partition:
			return
		self.partition = i
		self.cache = None

	def get_phrase_matrix(self):
		if self.cache is None:
			f = self.files[self.partition]
			wr = WindowReader(f, oov=self.oov)
			self.cache = wr.get_phrase_matrix()
		return self.cache
	
	def num_partitions(self): return len(self.files)


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
	""" only setup to do minimization
		should look like a numpy array (forward calls to self.tvar)
	"""

	def __init__(self, theano_mat, input_vars, cost, learning_rate=1e-2, delta=1e-2):
		""" cost should be a theano variable that this var should take gradients wrt
			input_vars should be a list of variables for whic you'll provide values when you call update()
		"""
		self.tvar = theano_mat	# should be a theano.shared
		self.gg = theano.shared(np.ones_like(self.tvar.get_value(), dtype=theano_mat.dtype) * delta)

		# TODO upgrade to >=0.6rc5 and switch to float32s

		# there is a bug that has been fixed by 0.6rc5 where when you
		# multiply a variable with dtype='float32' with a theano.tensor.constant,
		# you get something back with dtype='float64'
		# if you get errors in this code, that is almost certainly why (unless your on >=0.6rc5)
		self.lr = T.constant(learning_rate)

		grad = theano.grad(cost=cost, wrt=self.tvar)

		gg_update = self.gg + (grad ** 2)
		tvar_update = self.tvar - self.lr * grad / (self.gg ** 0.5)
		self.updates = [(self.gg, gg_update), (self.tvar, tvar_update)]
		self.f_update = theano.function(input_vars, grad, updates=self.updates)

	def update(self, args, verbose=False):
		""" args should match input_vars provided to __init__ """
		return self.f_update(*args)
	
	def __str__(self):
		return "<AdaGradParam tvar.shape=%s lr=%g gg.l2=%g>" % \
			(self.tvar.get_value().shape, self.lr.value,
			np.linalg.norm(self.gg.get_value(), ord=2))
	
	def get_value(self): return self.tvar.get_value()
	def set_value(self, v): return self.tvar.set_value(v)

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

class AdaDeltaParam(AdaGradParam):
	def __init__(self, theano_mat, input_vars, cost, learning_rate=1e-1, rho=0.90, delta=1e-3):
		self.tvar = theano_mat	# should be a theano.shared
		self.rho = T.constant(rho)
		self.delta = T.constant(delta)
		self.lr = T.constant(learning_rate)

		grad = theano.grad(cost=cost, wrt=self.tvar)
		self.gg = theano.shared(np.ones_like(self.tvar.get_value(), dtype=float_type) * delta)
		self.ss = theano.shared(np.ones_like(self.tvar.get_value(), dtype=float_type) * delta)
		step = ((self.ss + self.delta) ** 0.5) / ((self.gg + self.delta) ** 0.5) * grad

		gg_update = self.rho * self.gg + (1.0-self.rho) * (grad ** 2)
		ss_update = self.rho * self.ss + (1.0-self.rho) * (step ** 2)
		tvar_update = self.tvar - self.lr * step
		self.updates = [(self.gg, gg_update), (self.ss, ss_update), (self.tvar, tvar_update)]
		self.f_update = theano.function(input_vars, grad, updates=self.updates)

class NPZipper:
	# TODO writeout metadata like when the files were serialized

	@staticmethod
	def save(params, directory):
		""" params should be a dict with string keys and numpy array values """
		assert os.path.isdir(directory)
		for name, p in params.iteritems():
			f = os.path.join(directory, name)
			np.save(f, p)

	@staticmethod
	def load(directory):
		""" returns a dict with string keys and numpy array values """
		assert os.path.isdir(directory)
		params = {}
		for f in os.listdir(directory):
			if f.endswith('.npy'):
				name = f[:-4]
				p = np.load(os.path.join(directory, f))
				params[name] = p
		return params
		
