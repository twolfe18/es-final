import os
import sys
import math
import time
import codecs
import itertools
import numpy as np
import theano
import theano.tensor as T
from util import *

# TODO upgrade to >=0.6rc5 and switch to float32s
int_type = 'int32'
float_type = 'float64'


class Phrase(object):
	""" represents a phrase (word window), or many of them """

	def as_args_list(self):
		""" returns a list of argument values supplied by this object """
		raise 'subclasses need to implement this'
	
	def new_batch(self, size):
		""" sets a different batch (internally) """
		raise 'subclasses need to implement this'

	def __len__(self):
		""" how many examples are in this phrase
			represents the total, not the batch size
		"""
		raise 'subclasses need to implement this'

	def __str__(self):
		return "<Phrase shape=(%d,%d)>" % (len(self), self.width)
	
	@property
	def width(self):
		""" how many words are in one of this/these phrase(s) ? """
		raise 'subclasses need to implement this'

	@property
	def shape(self):
		return (len(self), self.width)


class FeaturizedPhrase(Phrase):

	def __init__(self, word_indices, feature_indices):
		assert word_indices.shape == feature_indices.shape
		assert len(word_indices.shape) == 2
		assert len(feature_indices.shape) == 2
		self.word_indices = word_indices
		self.feature_indices = feature_indices
		self.batch = None
		self.N = word_indices.shape[0]
		self.k = word_indices.shape[1]

	def as_args_list(self):
		if self.batch is None:
			return [self.word_indices, self.feature_indices]
		return [self.word_indices[self.batch,], self.feature_indices[self.batch,]]
	
	def new_batch(self, size):
		self.batch = np.random.choice(self.N, size)

	@property
	def width(self): return self.k
	def __len__(self): return self.N

	def copy(self):
		w = self.word_indices.copy()
		f = self.feature_indices.copy()
		p = FeaturizedPhrase(w, f)
		if self.batch is not None:
			p.batch = self.batch.copy()
		return p


class VanillaPhrase(Phrase):

	def __init__(self, word_indices):
		self.word_indices = word_indices
		self.batch = None
		self.N = word_indices.shape[0]
		self.k = word_indices.shape[1]
		assert len(word_indices.shape) == 2

	def as_args_list(self):
		if self.batch is None:
			return [self.word_indices]
		return [self.word_indices[self.batch,]]

	def new_batch(self, size):
		self.batch = np.random.choice(self.N, size)

	def copy(self):
		p = VanillaPhrase(self.word_indices.copy())
		if self.batch is not None:
			p.batch = self.batch.copy()
		return p

	@property
	def width(self): return self.k
	def __len__(self): return self.N





class Embedding(object):

	def __init__(self, alph, k, f_score, f_step):
		""" f_score is a theano function that takes phrase.as_args_list and returns a score
			  e.g. for Vanilla/VanillaPhrase, f_score: [word_indices] => score
			  e.g. for AdditiveEmbeddings/FeaturizedPhrase, f_score: [word_indices, feature_indices] => score
			f_step is a theano function that takes phrase.as_args_list + corrupted_phrase.as_args_list and takes a gradient step
			  e.g. for Vanilla/VanillaPhrase, f_step: [word_indices, corrupted_word_indices] => score
			  e.g. for AdditiveEmbeddings/FeaturizedPhrase, f_score: [word_indices, feature_indices, corrutped_word_indices, corrupted_feature_indices] => score
		"""
		assert type(k) == int
		self.k = k	# how many words in a window
		self.alph = alph
		self.num_words = len(alph)
		self.f_score = f_score
		self.f_step = f_step

		# this is used for read/write and a few other things
		# should be a map from strings (names of variables)
		# to things that look like theano shared vaiables (but could be e.g. AdaGradParams)
		#self.params = {}
		# NOTE i'm not setting this here so that it doesn't overwrite a lower classes value


	def word_embeddings(self):
		""" returns a numpy matrix where the rows are indexed by word ids """
		raise 'subclasses need to implement this'


	def corrupt(self, phrase):
		""" corrupts phrase to something that should have a lower score """
		raise 'subclasses need to implement this'

	
	def get_vec(self, word):
		W = self.get_embeddings()
		if type(word) == int:
			return W[word,]
		elif type(word) == str:
			i = self.alph.lookup_index(word, add=False)
			return W[i,]
		else:
			raise '[get_vec] I don\'t know what to do with', word


	def raw_score(self, phrase):
		""" returns the network score of this phrase """
		assert phrase.width == self.k
		args = phrase.as_args_list()
		return self.f_score(*args)


	def loss(self, phrases, corrupted_phrases, avg=True):
		""" returns the hinge loss on this data
			phrases should be of type FeaturizedPhrase or VanillaPhrase
		"""
		assert isinstance(phrases, Phrase)
		assert isinstance(corrupted_phrases, Phrase)
		assert len(phrases.word_indices.shape) == 2
		g = self.raw_score(phrases)
		b = self.raw_score(corrupted_phrases)
		one = np.ones_like(g)
		hinge = one + b - g
		hinge = hinge * (hinge > 0.0)
		if avg:
			return hinge.mean()
		return hinge.sum()


	def train(self, train_phrases, dev_phrases, epochs=10, iterations=30, batch_size=500):
		print 'train_phrases.shape =', train_phrases.shape
		print 'self.k =', self.k
		assert train_phrases.width == dev_phrases.width
		assert train_phrases.width == self.k
		train_phrases_corrupted = self.corrupt(train_phrases)
		dev_phrases_corrupted = self.corrupt(dev_phrases)
		assert train_phrases.shape == train_phrases_corrupted.shape
		assert dev_phrases.shape == dev_phrases_corrupted.shape
		dev_loss = []
		for e in range(epochs):
			print "[train] starting epoch %d" % (e)

			# take a few steps
			t = time.clock()
			for i in range(iterations):
				train_phrases.new_batch(batch_size)
				train_phrases_corrupted.new_batch(batch_size)
				args = train_phrases.as_args_list() + train_phrases_corrupted.as_args_list()
				self.f_step(*args)
			t_time = time.clock() - t

			# compute dev loss
			t = time.clock()
			l = self.loss(dev_phrases, dev_phrases_corrupted)
			dev_loss.append(l)

			print "[train] loss on %d examples is %.5f" % (len(dev_phrases), l)
			d_time = time.clock() - t
			print "[train] train took %.2f sec and dev took %.2f" % (t_time, d_time)
			ex = iterations * batch_size + len(dev_phrases)
			ex_per_sec = ex / (t_time + d_time)
			print "[train] %.1f examples per second" % (ex_per_sec)
			#print '[train] W.l2 = ', self.params['W'].l2	# this is actually noticeably slow
			#print '[train] A.l2 = ', self.params['A'].l2
			#print

		return dev_loss

	
	def read_weights(self, fromdir):
		assert os.path.isdir(fromdir)
		print '[read_weights] reading model from', fromdir
		self.alph = Alphabet(os.path.join(fromdir, 'alphabet.txt'))
		for name, value in NPZipper.load(fromdir).iteritems():
			self.params[name].set_value(value)
	

	def write_weights(self, outdir):
		if not os.path.isdir(outdir):
			assert not os.path.isfile(outdir)
			os.mkdir(outdir)
		print '[write_weights] writing model to', outdir
		np = {k:v.get_value() for k, v in self.params.iteritems()}
		NPZipper.save(np, outdir)
		self.alph.save(os.path.join(outdir, 'alphabet.txt'))


	def check_for_bad_params(self):
		bad = []
		for name, param in self.params.iteritems():
			if param.contains_bad_values():
				print name, 'contains bad values'
				bad.append(param)
		return bad





class VanillaEmbedding(Embedding, object):
	
	def __init__(self, alph, k, d=64, h=40, learning_rate_scale=1.0, initialization_scale = 1.0):
		num_words = len(alph)
		self.k = k	# width of phrase/window
		self.d = d	# how many features per word
		self.h = h	# hidden layer size

		# 1-hidden layer network
		self.W = theano.shared(np.zeros((num_words, d), dtype=float_type), name='W')	# word vecs
		self.A = theano.shared(np.zeros((k * d, h), dtype=float_type), name='A')		# word vecs => hidden
		self.b = theano.shared(np.zeros(h, dtype=float_type), name='b')					# hidden offset
		self.p = theano.shared(np.zeros(h, dtype=float_type), name='p')					# hidden => output

		self.reg = Regularizer()
		self.reg.l1(self.A, 1e-5)
		self.reg.l2(self.W, 1e-6)
		self.reg.l2(self.b, 1e-6)
		self.reg.l1(self.p, 1e-5)

		if int_type == 'int64':
			word_indices = T.lmatrix('word_indices')
		else:
			word_indices = T.imatrix('word_indices')	# each row is a phrase, should have self.k columns
		n, k = word_indices.shape	# won't know this until runtime
		phrases_tensor = self.W[word_indices]	# shape=(n, k, self.d)
		phrases = phrases_tensor.reshape((n, k * d))
		hidden = T.tanh( T.dot(phrases, self.A) + self.b )
		scores = T.dot(hidden, self.p)

		# score function
		f_score = theano.function([word_indices], scores)

		# loss
		if int_type == 'int64':
			word_indices_corrupted = T.lmatrix('word_indices_corrupted')
		else:
			word_indices_corrupted = T.imatrix('word_indices_corrupted')
		scores_corrupted = theano.clone(scores, replace={word_indices: word_indices_corrupted})
		loss_neg = T.ones_like(scores) + scores_corrupted - scores
		loss = loss_neg * (loss_neg > 0)
		avg_loss = loss.mean() + self.reg.regularization_var()

		args = [word_indices, word_indices_corrupted]
		#print 'args =', args
		learning_rate_muting = 2.0	# higher means that only W gets updates, 1 means everything has same learning rate
		lrW = learning_rate_scale
		lrA = math.pow(learning_rate_muting, -1.0) * learning_rate_scale
		lrp = math.pow(learning_rate_muting, -2.0) * learning_rate_scale
		lrb = math.pow(learning_rate_muting, -3.0) * learning_rate_scale
		self.params = {
			'W' : AdaGradParam(self.W, args, avg_loss, learning_rate=lrW),
			'A' : AdaGradParam(self.A, args, avg_loss, learning_rate=lrA),
			'p' : AdaGradParam(self.p, args, avg_loss, learning_rate=lrp),
			'b' : AdaGradParam(self.b, args, avg_loss, learning_rate=lrb) \
		}

		upd = [p.updates for p in self.params.values()]
		upd = list(itertools.chain(*upd))	# flatten list
		#print 'updates =', upd
		f_step = theano.function([word_indices, word_indices_corrupted], avg_loss, updates=upd)

		super(VanillaEmbedding, self).__init__(alph, self.k, f_score, f_step)


		# initialize
		Initializer.set_rand_ball(self.W, initialization_scale)

		# d=64,k=5,h=40 => rA=rb=0.258
		Initializer.set_rand_ball(self.A, initialization_scale * Initializer.compute_r(d * self.k, h))
		Initializer.set_rand_ball(self.b, initialization_scale * Initializer.compute_r(d * self.k, h))

		# h=40 => rp=0.765
		Initializer.set_rand_ball(self.p, initialization_scale * Initializer.compute_r(h, 1))


	# user-friendly version
	# FIXME move up to Embedding? (not obvious because you need List[String] => Phrase)
	def score(self, words):
		""" gives the NNs score of this phrase
			words should be a list of strings and alph and Alphabet containing those strings
		"""
		assert type(words) == list
		w = [self.alph.lookup_index(x, add=False) for x in words]
		i = np.mat( np.array(w, dtype=int_type) )
		return self.f_score(i)[0]

	
	def corrupt(self, phrases):
		""" phrases is a VanillaPhrase """
		n, k = phrases.shape
		assert k == self.k
		mid = k // 2
		corrupted = phrases.copy()
		corrupted.word_indices[:,mid] = np.random.random_integers(0, self.num_words-1, n)
		return corrupted


class AdditiveEmbedding(Embedding, object):
	""" same as vanilla model, but each word may be decomposed (additively)
		for words that appear in NOMLEX, I plan to decompose them a vector for their
		base meaning plus a vector for whether they are in nominal or verbal form
	"""
	
	# how many vectors to learn for features
	# in the NOMLEX case, there will be a feature which can
	# take two different values, 'nom' and 'verb'
	# if i decide to learn with more fine grained features like
	# dependency paths, then this might be the number of paths to a headword
	def __init__(self, alph, num_features, k, d=64, h=40, learning_rate_scale = 1.0, initialization_scale = 1.0):

		self.h = h
		self.k = k
		self.d = d
		num_words = len(alph)
		self.num_features = num_features
		self.corruptor = WFCorruptionPolicy()

		# 1-hidden layer network
		self.Ew = theano.shared(np.zeros((num_words, d), dtype=float_type), name='Ew')		# word embeddings
		self.Ef = theano.shared(np.zeros((num_features, d), dtype=float_type), name='Ef')	# feature embeddings
		self.A = theano.shared(np.zeros((k * d, h), dtype=float_type), name='A')			# word+feat => hidden
		self.b = theano.shared(np.zeros(h, dtype=float_type), name='b')						# hidden offset
		self.p = theano.shared(np.zeros(h, dtype=float_type), name='p')						# hidden => output

		self.reg = Regularizer()
		self.reg.l2(self.Ew, 1e-6)
		self.reg.l2(self.Ef, 1e-6)
		self.reg.l1(self.A, 1e-5)
		self.reg.l2(self.b, 1e-6)
		self.reg.l1(self.p, 1e-5)

		if int_type == 'int64':
			word_indices = T.lmatrix('word_indices')
			feat_indices = T.lmatrix('feat_indices')
		else:
			word_indices = T.imatrix('word_indices')	# each row is a phrase, should have self.k columns
			feat_indices = T.imatrix('feat_indices')
		n, k = word_indices.shape
		phrases_tensor = self.Ew[word_indices]
		phrases = phrases_tensor.reshape((n, k * self.d))
		features_tensor = self.Ef[feat_indices]
		features = features_tensor.reshape((n, k * self.d))
		latent = phrases + features
		hidden = T.tanh( T.dot(latent, self.A) + self.b )
		scores = T.dot(hidden, self.p)

		# score function
		f_score = theano.function([word_indices, feat_indices], scores)

		# loss
		if int_type == 'int64':
			word_indices_corrupted = T.lmatrix('word_indices_corrupted')
			feat_indices_corrupted = T.lmatrix('feat_indices_corrupted')
		else:
			word_indices_corrupted = T.imatrix('word_indices_corrupted')
			feat_indices_corrupted = T.imatrix('feat_indices_corrupted')
		scores_corrupted = theano.clone(scores, replace={
			word_indices: word_indices_corrupted, \
			feat_indices: feat_indices_corrupted,
		})
		loss_neg = T.ones_like(scores) + scores_corrupted - scores
		loss = loss_neg * (loss_neg > 0)
		avg_loss = loss.mean() + self.reg.regularization_var()

		args = [word_indices, feat_indices, word_indices_corrupted, feat_indices_corrupted]
		print 'args =', args
		learning_rate_muting = 5.0	# higher means that only W gets updates, 1 means everything has same learning rate
		lrEw = learning_rate_scale
		lrEf = learning_rate_scale
		#lrEf = math.pow(learning_rate_muting, -0.5) * learning_rate_scale
		lrA = math.pow(learning_rate_muting, -1.0) * learning_rate_scale
		lrp = math.pow(learning_rate_muting, -2.0) * learning_rate_scale
		lrb = math.pow(learning_rate_muting, -3.0) * learning_rate_scale
		self.params = {
			'Ew' : AdaGradParam(self.Ew, args, avg_loss, learning_rate=lrEw),
			'Ef' : AdaGradParam(self.Ef, args, avg_loss, learning_rate=lrEf),
			'A' : AdaGradParam(self.A, args, avg_loss, learning_rate=lrA),
			'p' : AdaGradParam(self.p, args, avg_loss, learning_rate=lrp),
			'b' : AdaGradParam(self.b, args, avg_loss, learning_rate=lrb) \
		}

		upd = [p.updates for p in self.params.values()]
		upd = list(itertools.chain(*upd))	# flatten list
		#print 'updates =', upd
		f_step = theano.function(args, avg_loss, updates=upd)
	
		super(AdditiveEmbedding, self).__init__(alph, self.k, f_score, f_step)

		
		# initialize
		# NOTE this is the uninformed version
		# should really initialize from previous vanilla model
		Initializer.set_rand_ball(self.Ew, initialization_scale)
		Initializer.set_rand_ball(self.Ef, initialization_scale * self.num_features / math.pow(self.num_words, 1.0/3.0))

		# d=64,k=5,h=40 => rA=rb=0.258
		Initializer.set_rand_ball(self.A, initialization_scale * Initializer.compute_r(d * self.k, h))
		Initializer.set_rand_ball(self.b, initialization_scale * Initializer.compute_r(d * self.k, h))

		# h=40 => rp=0.765
		Initializer.set_rand_ball(self.p, initialization_scale * Initializer.compute_r(h, 1))

	
	def corrupt(self, featurized_phrase):
		# how to corrupt?
		# lets never propose mis-matched (word,feature) pairs
		# we can take this as a statement that we are learning a conditional distribution
		# (conditioned on interpretting the words correctly -- and extracting the right feature)

		# if you flip a feature => delta(data) indicates a word vector needs to be updated
		# if you flip a word => delta(data) indicates a word vector needs to be updated
		# if you flip both => the learner will backprop the gradient to both (inefficient)

		t = time.clock()
		c_phrase = featurized_phrase.copy()
		n, k = featurized_phrase.shape
		assert k == self.k
		mid = k // 2
		for i in range(n):
			c = self.corruptor.what_to_corrupt()
			if c == 'word' or c == 'both':
				c_phrase.word_indices[i, mid] = np.random.randint(0, self.num_words-1)
			if c == 'phrase' or c == 'both':
				c_phrase.feature_indices[i, mid] = np.random.randint(0, self.num_features-1)
		return c_phrase

class WFCorruptionPolicy:
	""" chooses what to corrupt for models of words and features, see AdditiveWordVecs
		WF = "word feature"
	"""

	def __init__(self, prob_just_word=0.9, prob_just_feature=0.1, prob_both=0.0):
		z = prob_just_word + prob_just_feature + prob_both
		self.prob_just_word = prob_just_word / z
		self.prob_just_feature = prob_just_feature / z
		self.prob_both = prob_both / z

	def what_to_corrupt(self):
		i = np.random.rand()
		if i < self.prob_just_word:
			return 'word'
		elif i < self.prob_just_word + self.prob_just_feature:
			return 'feature'
		else:
			return 'both'



