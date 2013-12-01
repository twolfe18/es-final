import theano
import theano.tensor as T
import numpy as np

float_type = 'float64'

# same as vanilla model, but each word may be decomposed (additively)
# for words that appear in NOMLEX, I plan to decompose them a vector for their
# base meaning plus a vector for whether they are in nominal or verbal form

class WFCorruptionPolicy
	def __init__(self, prob_just_word=0.5, prob_just_feature=0.4, prob_both=0.1):
		z = prob_just_word + prob_just_feature + prob_both
		self.prob_just_word = prob_just_word / z
		self.prob_just_feature = prob_just_feature / z
		self.prob_both = prob_both / z

	@property
	def prob_just_word(self): return self.prob_just_word
	@property
	def prob_just_feature(self): return self.prob_just_feature
	@property
	def prob_both(self): return self.prob_both

	def what_to_corrupt():
		i = np.random.rand()
		if i < self.prob_just_word:
			return 'word'
		elif i < self.prob_just_word + self.prob_just_feature:
			return 'feature'
		else
			return 'both'

class AdditiveWordVecs:
	
	# how many vectors to learn for features
	# in the NOMLEX case, there will be a feature which can
	# take two different values, 'nom' and 'verb'
	# if i decide to learn with more fine grained features like
	# dependency paths, then this might be the number of paths to a headword
	def __init__(self, alph, num_features, k, d=64, h=40):
		self.d = d	# number of features per word
		self.h = h	# hidden layer size
		self.k = k	# how many words in a window
		self.alph = alph
		self.num_features = num_features
		self.num_words = len(alph)
		self.batch_size = batch_size
		self.corruptor = WFCorruptionPolicy()

		# 1-hidden layer network
		self.Ew = theano.shared(np.zeros((self.num_words, self.d), dtype=float_type), name='Ew')	# word embeddings
		self.Ef = theano.shared(np.zeros((self.num_features, self.d), dytpe=float_type), name='Ef')	# feature embeddings
		self.A = theano.shared(np.zeros((self.k * self.d, self.h), dtype=float_type), name='A')	# word+feat => hidden
		self.b = theano.shared(np.zeros(self.h, dtype=float_type), name='b')					# hidden offset
		self.p = theano.shared(np.zeros(self.h, dtype=float_type), name='p')					# hidden => output
		self.t = theano.shared(0.0, name='t')													# output offset

		word_indices = T.imatrix('word_indices')	# each row is a phrase, should have self.k columns
		feat_indices = T.imatrix('feat_indices')
		n, k = word_indices.shape
		phrases_tensor = self.Ew[word_indices]
		phrases = phrases_tensor.reshape((n, k * self.d))
		features_tensor = self.Ef[feat_indices]
		features = features_tensor.reshape((n, k * self.d))
		latent = phrases + features
		hidden = T.tanh( T.dot(latent, self.A) + self.b )
		scores = T.tanh( T.dot(hidden, self.p) + self.t )

		# score function
		self.f_score = theano.function([word_indices, feat_indices], [scores])

		# loss
		word_indices_corrupted = T.imatrix('word_indices_corrupted')
		feat_indices_corrupted = T.imatrix('feat_indices_corrupted')
		scores_corrupted = theano.clone(scores, replace={
			word_indices: word_indices_corrupted, \
			feat_indices: feat_indices_corrupted,
		})
		loss_neg = T.ones_like(scores) + scores_corrupted - scores
		loss = loss_neg * (loss_neg > 0)
		r = self.reg.regularization_var()
		avg_loss = loss.mean()

		args = [word_indices, feat_indices, word_indices_corrupted, feat_indices_corrupted]
		print 'args =', args
		self.params = {
			'Ew' : AdaGradParam(self.Ew, args, avg_loss, learning_rate=learning_rate_scale),
			'Ef' : AdaGradParam(self.Ef, args, avg_loss, learning_rate=learning_rate_scale),
			'A' : AdaGradParam(self.A, args, avg_loss, learning_rate=1e-1 * learning_rate_scale),
			'b' : AdaGradParam(self.b, args, avg_loss, learning_rate=1e-2 * learning_rate_scale),
			'p' : AdaGradParam(self.p, args, avg_loss, learning_rate=1e-2 * learning_rate_scale),
			't' : AdaGradParam(self.t, args, avg_loss, learning_rate=1e-2 * learning_rate_scale) \
		}

		upd = [p.updates for p in self.params.values()]
		upd = list(itertools.chain(*upd))	# flatten list
		#print 'updates =', upd
		self.f_step = theano.function([word_indices, word_indices_corrupted], [avg_loss], updates=upd)
	
	def raw_score(self, phrase, features):
		""" phrase is a vector of word indices with length self.k
			features is a vector of the same length which indicates a component to add on each word
			indices that are 0 in the feature vector won't have anything added
			(you can lift this statement to matrices were the rows are the vectors describe above)
		"""
		assert features.max() < self.num_features
		assert phrase.max() < self.num_words
		pass
	
	def corrupt(self, phrase, features):
		# how to corrupt?
		# lets never propose mis-matched (word,feature) pairs
		# we can take this as a statement that we are learning a conditional distribution
		# (conditioned on interpretting the words correctly -- and extracting the right feature)

		# if you flip a feature => delta(data) indicates a word vector needs to be updated
		# if you flip a word => delta(data) indicates a word vector needs to be updated
		# if you flip both => the learner will backprop the gradient to both (inefficient)

		c_phrase = phrase.copy()
		c_features = features.copy()
		n, k = phrase.shape
		assert k == self.k
		mid = k // 2
		for i in range(n):
			c = self.corruptor.what_to_corrupt()
			if c == 'word' or c == 'both':
				c_phrase[i, mid] = np.random.randint(0, self.num_words-1)
			if c == 'phrase' or c == 'both':
				c_features[i, mid] = np.random.randint(0, self.num_features-1)

		return c_phrase, c_features






















