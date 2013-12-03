import os, sys, random
from embeddings import *
from util import *
import numpy as np
import theano
import theano.tensor as T

# see if you can predict telic vs atelic events from learned representations
class VerbClasses:

	def __init__(self, model_dir):
		self.model_dir = model_dir
		assert os.path.isdir(model_dir)

		# adapted from http://www.sfu.ca/person/dearmond/322/322.event.class.htm
		self.states = ['burning', 'frozen', 'feel', 'experience', 'wants', 'wanted', \
			'see', 'thought', 'thinks', 'loves', 'hates', 'confused']
		self.processes = ['walk', 'swim', 'fly', 'paint', 'write', 'eat', 'snore', \
			'breathe', 'sleep', 'dream', 'speak', 'sing', 'run', 'watch', 'snow', 'seek', 'sit']
		self.accomplishments = ['arrive', 'reach', 'succeed', 'pass', 'win', 'lose', 'gain', \
			'die', 'happen', 'acquire', 'find', 'say', 'claim', 'declare', 'avert', 'recognize']

		use_ramis = True
		if use_ramis:
			import pickle
			rami_emb_file = '/home/travis/Dropbox/School/event-semantics/final-project/data/polyglot-en.pkl'
			words, embeddings = pickle.load(open(rami_emb_file, 'rb'))
			self.W = embeddings
			self.alph = Alphabet()
			for w in words: self.alph.lookup_index(w, add=True)
		else:
			self.alph = Alphabet(os.path.join(self.model_dir, 'alphabet.txt'))
			self.W = np.load(os.path.join(self.model_dir, 'W.npy'))
		N, d = self.W.shape
		print 'W =', self.W.shape, self.W.dtype
		print 'len(alph) =', len(self.alph)

		use_offsets = False

		offsets = theano.shared(np.zeros((3,d), dtype='float64'), name='offsets')
		weights = theano.shared(np.zeros((3,d), dtype='float64'), name='weights')
		intercepts = theano.shared(np.zeros(3, dtype='float64'), name='intercepts')

		wordvec = T.dvector('wordvec')	# input
		label = T.iscalar('label')
		if use_offsets:
			label_score = T.dot(weights[label,], wordvec - offsets[label,]) + intercepts[label]
		else:
			label_score = T.dot(weights[label,], wordvec) + intercepts[label]

		swordvec = T.stack(wordvec, wordvec, wordvec)
		if use_offsets:
			model_scores = T.sum(weights * (swordvec - offsets), axis=1) + intercepts	# sum_rows( (3,d) elem-prod (3,d) ) = (3,)
		else:
			model_scores = T.sum(weights * swordvec, axis=1) + intercepts
		guess = T.argmax(model_scores)
		
		loss = np.ones(3) - T.eye(3, k=label)
		hinge = T.max(model_scores - label_score + loss)
		reg = 1e-4 * T.sum(offsets ** 2) + 1e-4 * T.sum(weights ** 2)
		obj = hinge + reg
		
		scale = 0.1
		self.weights = AdaGradParam(weights, [wordvec, label], obj, learning_rate=scale)
		self.offsets = AdaGradParam(offsets, [wordvec, label], obj, learning_rate=scale/2.0)
		self.intercepts = AdaGradParam(intercepts, [wordvec, label], obj, learning_rate=scale)

		if use_offsets:
			upd = self.weights.updates + self.offsets.updates + self.intercepts.updates
		else:
			upd = self.weights.updates + self.intercepts.updates
		self.f_step = theano.function([wordvec, label], guess, updates=upd)
		self.f_predict = theano.function([wordvec], guess)

	def train_classifier(self):
		
		examples = []
		lab_counts = [0, 0, 0]
		for words, label in [(self.states,0), (self.processes,1), (self.accomplishments,2)]:
			for s in words:
				try:
					i = self.alph.lookup_index(s)
					lab_counts[label] += 1
					examples.append( (self.W[i,], label, s) )
				except:
					print '[train_classifier] didn\'t find', s, 'in the alphabet'
		random.shuffle(examples)

		wordvecs = np.vstack([x[0] for x in examples])
		labels = np.array([x[1] for x in examples])
		print 'labels =', labels.shape, labels.dtype
		print 'wordvecs =', wordvecs.shape, wordvecs.dtype

		# try a prediction
		print 'first prediction =', self.f_predict(wordvecs[0])
		
		right = 0
		N = len(examples)
		for i in range(N):
			test_wordvec, test_label, test_word = examples[i]
			train_idx = list(range(i)) + list(range(i+1,N))

			# train from scratch
			self.weights.set_value(np.zeros((3,64)))
			self.offsets.set_value(np.zeros((3,64)))
			for j in range(200):
				for jj in range(N):
					if jj == i: continue
					self.f_step(wordvecs[jj], labels[jj])

			# guess
			guess = self.f_predict(test_wordvec)
			sys.stdout.write("word=%s label=%d guess=%d" % (test_word, test_label, guess))
			if guess == test_label:
				print '\tRIGHT'
				right += 1
			else:
				print '\tWRONG'

		print '[train_classifier] accuracy =', float(100*right)/N
		print '[train_classifier] accuracy by chance =', float(100*max(lab_counts))/N


		"""
		mean_weights = self.weights.get_value().mean(axis=0)
		mean_offsets = self.offsets.get_value().mean(axis=0)
		print "[train_classifier] states: weights=%s diff=%s" % \
			(self.weights.get_value()[0] - mean_weights, self.offsets.get_value()[0] - mean_offsets)
		print "[train_classifier] processes: weights=%s diff=%s" % \
			(self.weights.get_value()[1] - mean_weights, self.offsets.get_value()[1] - mean_offsets)
		print "[train_classifier] accomplishments: weights=%s diff=%s" % \
			(self.weights.get_value()[2] - mean_weights, self.offsets.get_value()[2] - mean_offsets)
		"""
		print 'offsets =', self.offsets.get_value()

if __name__ == '__main__':
	vc = VerbClasses('models/verb-classification/')
	vc.train_classifier()

