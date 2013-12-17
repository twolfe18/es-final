from embeddings import *
import numpy as np
import numpy.random as rand
import sys
import random
import codecs, gzip


class Trainer(object):

	def __init__(self, alph, model_dir, data_dir, k):
		self.alph = alph
		self.model_dir = model_dir
		self.data_dir = data_dir
		self.k = k
		print '[Trainer init] alphbet contains', len(self.alph), 'words'
	

	def get_embedding_to_train(self, learning_rate_scale=1.0):
		raise Exception('subclasses need to implement this')


	def dev_phrase(self):
		""" return a Phrase that will serve as dev data """
		raise Exception('subclasses need to implement this')


	def num_train_phrases(self):
		""" how many partitions is the training data broken up into? """
		raise Exception('subclasses need to implement this')

	
	def train_phrase(self, index):
		""" return a Phrase corresponding to the index-th partition """
		raise Exception('subclasses need to implement this')

	
	def params_to_report_on(self):
		""" return a list of strings that are keys for self.params
			every training epoch i'll print some stats about these params
		"""
		return []


	def run(self):

		emb = self.get_embedding_to_train(learning_rate_scale=1.0)

		outer_epochs = 500	# affects the total runtime
		inner_epochs = 6	# make this larger to amortize reporting time
		bs = 100			# batch size
		itr = 50			# how many batch-sized steps to take
		print 'training for', outer_epochs, 'macro-epochs'
		sys.stdout.flush()
		W_dev = self.dev_phrase()
		prev_avg_loss = 1.0
		pi = list(range(self.num_train_phrases()))	# permutation to view train batches through
		for epoch in range(outer_epochs):
			print 'starting epoch', epoch
			sys.stdout.flush()
			improvement = False
			for train_idx in range(self.num_train_phrases()):
				W_train = self.train_phrase(pi[train_idx])
				losses = emb.train(W_train, W_dev, epochs=inner_epochs, iterations=itr, batch_size=bs)
				print 'losses =', losses
				avg_loss = sum(losses) / len(losses)
				if avg_loss < prev_avg_loss - 1e-4 or avg_loss / prev_avg_loss < 0.99:
					if epoch % 4 == 0 or (epoch > 30 and epoch % 2 == 0) or (epoch > 60):
						emb.write_weights(os.path.join(self.model_dir, 'epoch' + str(epoch)))
					improvement = True
					prev_avg_loss = avg_loss
					break
				prev_avg_loss = avg_loss
			random.shuffle(pi)

			for name in self.params_to_report_on():
				param = emb.params[name]
				s = param.shape
				if len(s) == 0:
					print "[train] %s=%s" % (name, param.get_value())
				else:
					print "[train] %s.shape=%s" % (name, s)
					print "[train] %s.l2=%s" % (name, param.l2)
					print "[train] %s.lInf=%s" % (name, param.lInf)
				print
				sys.stdout.flush()

			if not improvement:
				break

		print 'saving final model...'
		emb.write_weights(os.path.join(self.model_dir, 'final'))



class VanillaTrainer(Trainer, object):


	def __init__(self, model_dir, data_dir, k, init_params_with_dir=None):

		self.init_params_with_dir = init_params_with_dir
		alph = Alphabet(os.path.join(data_dir, 'train-dev.alphabet'))
		super(VanillaTrainer, self).__init__(alph, model_dir, data_dir, k)

		train_files = []
		for f in os.listdir(data_dir):
			if not f.endswith('.npy'): continue
			if f.startswith('train'):
				train_files.append(f)
		#print '[Vanilla init] train files =', train_files
		self.train_readers = [NumpyWindowReader(os.path.join(data_dir, f)) for f in train_files]


	def get_embedding_to_train(self, learning_rate_scale=1.0):
		emb = VanillaEmbedding(self.alph, self.k, d=32, h=20, learning_rate_scale=learning_rate_scale)
		if self.init_params_with_dir is not None:
			print '[AdditiveTrainer] initializing embeddings from', self.init_params_with_dir
			W = np.load(os.path.join(self.init_params_with_dir, 'W.npy'))
			N, d = W.shape
			print '[Additive initialize] W.shape =', W.shape
			print '[Additive initialize] self.k =', emb.d
			assert N == len(emb.alph)
			assert d == emb.d

			emb.params['W'].set_value(W)

			A = np.load(os.path.join(self.init_params_with_dir, 'A.npy'))
			emb.params['A'].set_value(A)

			b = np.load(os.path.join(self.init_params_with_dir, 'b.npy'))
			emb.params['b'].set_value(b)

			p = np.load(os.path.join(self.init_params_with_dir, 'p.npy'))
			emb.params['p'].set_value(p)
		return emb

	
	def dev_phrase(self):
		dev_file = os.path.join(self.data_dir, 'dev.tiny.npy')
		print 'dev file =', dev_file
		assert os.path.isfile(dev_file)
		dev_reader = NumpyWindowReader(dev_file)
		return VanillaPhrase(dev_reader.get_phrase_matrix())


	def num_train_phrases(self):
		return len(self.train_readers)
	

	def train_phrase(self, i):
		r = self.train_readers[i]
		return VanillaPhrase(r.get_phrase_matrix())

	def params_to_report_on(self):
		#return ['W', 'A', 'p', 'b']
		return []




class AdditiveTrainer(Trainer, object):

	def __init__(self, model_dir, data_dir, k, init_params_with_dir=None):
		self.init_params_with_dir = init_params_with_dir
		alph = Alphabet(os.path.join(data_dir, 'train-dev.alphabet'))
		super(AdditiveTrainer, self).__init__(alph, model_dir, data_dir, k)

		self.nomlex = NomlexReader('data/nomlex.txt', self.alph)
		self.dev = None

		self.train_files = []
		for f in os.listdir(data_dir):
			if not f.endswith('.npy'): continue
			if f.startswith('train'):
				self.train_files.append(os.path.join(data_dir, f))
		#print '[Additive init] train files =', self.train_files


	def get_embedding_to_train(self, learning_rate_scale=1.0):

		emb = AdditiveEmbedding(self.alph, 3, self.k, learning_rate_scale=learning_rate_scale)

		# - initialize params['Ef'][0,] = mean(params['W'])
		# - initialize params['Ew'] = params['W'] - params['Ef'][0,]
		if self.init_params_with_dir is not None:
			print '[AdditiveTrainer] initializing embeddings from', self.init_params_with_dir
			W = np.load(os.path.join(self.init_params_with_dir, 'W.npy'))
			N, d = W.shape
			print '[Additive initialize] W.shape =', W.shape
			print '[Additive initialize] self.k =', emb.d
			assert N == len(emb.alph)
			assert d == emb.d

			emb.params['Ew'].set_value(W)

			Ef = emb.params['Ef']
			Ef.set_value( np.zeros_like(Ef.get_value(), dtype=Ef.dtype) )

			A = np.load(os.path.join(self.init_params_with_dir, 'A.npy'))
			emb.params['A'].set_value(A)

			b = np.load(os.path.join(self.init_params_with_dir, 'b.npy'))
			emb.params['b'].set_value(b)

			p = np.load(os.path.join(self.init_params_with_dir, 'p.npy'))
			emb.params['p'].set_value(p)

		return emb

	def params_to_report_on(self):
		return ['Ew', 'Ef', 'A', 'p', 'b']

	def dev_phrase(self):
		if self.dev is None:
			print '[Additive dev_phrase] constructing dev phrase...'
			word_indices = np.load(os.path.join(self.data_dir, 'dev.tiny.npy'))
			feat_indices = self.nomlex.get_features(word_indices)
			self.dev = FeaturizedPhrase(word_indices, feat_indices)
		return self.dev


	def num_train_phrases(self):
		return len(self.train_files)

		
	def train_phrase(self, i):

		word_indices = NumpyWindowReader(self.train_files[i]).get_phrase_matrix()
		feat_indices = self.nomlex.get_features(word_indices)

		# about 4.2% last time i checked
		prop_feat = (feat_indices > 0).sum() / float(len(feat_indices) * len(feat_indices[0]))
		print "[AdditiveTrainer partition=%d] have features for %.1f%% of words in windows" % (i, 100.0*prop_feat)

		p = FeaturizedPhrase(word_indices, feat_indices)
		print "[AdditiveTrainer partition=%d] phrases.shape=(%d,%d)" % (i, len(p), p.width)
		return p


class FixedAdditiveTrainer(AdditiveTrainer, object):
	""" this fixes a bug in AdditiveTrainer"""
	
	def __init__(self, model_dir, data_dir, k, init_params_with_dir=None):
		super(FixedAdditiveTrainer, self).__init__(model_dir, data_dir, k, init_params_with_dir=init_params_with_dir)
	#def __init__(self):
		self.wordAlph = Alphabet()
		self.posAlph = Alphabet()
		self.k = 5
		self.f = '/home/travis/Desktop/word-windows-5.txt.gz'

		if os.path.isfile('cache/fadditive/W_train.npy'):
			print 'loading from cache...'
			self.W_train = np.load('cache/fadditive/W_train.npy')
			self.W_dev = np.load('cache/fadditive/W_dev.npy')
			self.P_train = np.load('cache/fadditive/P_train.npy')
			self.P_dev = np.load('cache/fadditive/P_dev.npy')
			self.wordAlph = Alphabet('cache/fadditive/wordAlph.txt')
			self.posAlph = Alphabet('cache/fadditive/posAlph.txt')
		else:
			print 'reading data from', self.f
			W, P = self.get_data(onlyTake=750000)
			n, k = W.shape
			i = n - 80000
			self.W_train = W[:-i,]
			self.W_dev = W[:i,]
			self.P_train = P[:-i,]
			self.P_dev = P[:i,]
			print 'saving to cache...'
			np.save('cache/fadditive/W_train.npy', self.W_train)
			np.save('cache/fadditive/W_dev.npy', self.W_dev)
			np.save('cache/fadditive/P_train.npy', self.P_train)
			np.save('cache/fadditive/P_dev.npy', self.P_dev)
			self.wordAlph.save('cache/fadditive/wordAlph.txt')
			self.posAlph.save('cache/fadditive/posAlph.txt')

		self.alph = self.wordAlph
		for m, name in [(self.W_train, 'W_train'), (self.W_dev, 'W_dev'), (self.P_train, 'P_train'), (self.P_dev, 'P_dev')]:
			print name, 'has shape', m.shape


	def get_embedding_to_train(self, learning_rate_scale=1.0):
		return AdditiveEmbedding(self.alph, len(self.posAlph), self.k, learning_rate_scale=learning_rate_scale)

	def num_train_phrases(self):
		return 1
	
	def dev_phrase(self):
		return FeaturizedPhrase(self.W_dev, self.P_dev)
	
	def train_phrase(self, i):
		return FeaturizedPhrase(self.W_train, self.P_train)

	def get_data(self, onlyTake=None, reservoir=False):
		t = time.time()
		reservoir_cutoff = 75000
		try: reservoir_cutoff = max(reservoir_cutoff, 2*onlyTake)
		except: pass
		words = []
		pos = []
		n = 0
		for w, p in self.get_words_and_pos():
			if onlyTake is not None and n >= onlyTake:
				if reservoir:
					i = np.random.randint(0, n)
					if i < len(words):
						words[i] = w
						pos[i] = p
				else:
					break
			else:
				words.append(w)
				pos.append(p)
			n += 1
			if n > reservoir_cutoff: break	# debugging
		words = np.array(words)
		pos = np.array(pos)
		print 'done reading took', (time.time()-t), 'seconds'
		return (words, pos)

	def get_words_and_pos(self):
		k = self.k
		fd = gzip.open(self.f, 'rb')
		r = codecs.getreader('utf-8')
		for lineNum, line in enumerate(r(fd)):
			try:
				#print 'line =', line
				words = np.zeros(k, dtype='int32')
				pos = np.zeros(k, dtype='int32')
				for i, tok in enumerate(line.strip().split()):
					w, p = tok.split('::')
					#print 'tok =', tok
					#print 'w =', w
					#print 'p =', p
					words[i] = self.wordAlph.lookup_index(w, add=True)
					pos[i] = self.posAlph.lookup_index(p, add=True)
				#print 'words =', words
				#print 'pos =', pos
				yield (words, pos)
			except:
				pass
		fd.close()



if __name__ == '__main__':
	
	init_with = None	# 'models/additive-initialization'
	data_dir = 'data/jumbo/'
	k = 5
	if sys.argv[1] == 'vanilla':
		target = 'models/vanilla/'
		if len(sys.argv) > 2:
			target = sys.argv[2]
		t = VanillaTrainer(target, data_dir, k, init_params_with_dir=init_with)
		t.run()
	elif sys.argv[1] == 'additive':
		target = 'models/additive/'
		if len(sys.argv) > 2:
			target = sys.argv[2]
		t = AdditiveTrainer(target, data_dir, k, init_params_with_dir=init_with)
		t.run()
	elif sys.argv[1] == 'fadditive':
		target = 'models/fadditive/'
		if len(sys.argv) > 2:
			target = sys.argv[2]
		t = FixedAdditiveTrainer(target, data_dir, k, init_params_with_dir=init_with)
		t.run()
	else:
		print 'i don\'t know how to handle these args', sys.argv







