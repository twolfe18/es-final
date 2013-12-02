from embeddings import *
import numpy as np
import numpy.random as rand
import sys
import random


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


	def run(self):

		emb = self.get_embedding_to_train(learning_rate_scale=3.0)

		outer_epochs = 500
		inner_epochs = 5
		bs = 600
		itr = 20
		print 'training for', outer_epochs, 'macro-epochs'
		W_dev = self.dev_phrase()
		prev_avg_loss = 1.0
		pi = list(range(self.num_train_phrases()))	# permutation to view train batches through
		for epoch in range(outer_epochs):
			print 'starting epoch', epoch
			improvement = False
			for train_idx in range(self.num_train_phrases()):
				W_train = self.train_phrase(pi[train_idx])
				losses = emb.train(W_train, W_dev, epochs=inner_epochs, iterations=itr, batch_size=bs)
				print 'losses =', losses
				avg_loss = sum(losses) / len(losses)
				if avg_loss < prev_avg_loss - 1e-4 or avg_loss / prev_avg_loss < 0.99:
					if epoch % 10 == 0 or (epock > 50 and epoch % 5 == 0) or (epoch > 100):
						emb.write_weights(self.model_dir + 'epoch' + str(epoch))
					improvement = True
					prev_avg_loss = avg_loss
					break
				prev_avg_loss = avg_loss
			random.shuffle(pi)
			if not improvement:
				break

		print 'saving model...'
		emb.write_weights(self.model_dir + 'final')



class VanillaTrainer(Trainer, object):


	def __init__(self, model_dir, data_dir, k):

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
		return VanillaEmbedding(self.alph, self.k, learning_rate_scale=learning_rate_scale)

	
	def dev_phrase(self):
		dev_file = os.path.join(self.data_dir, 'dev.small.npy')
		print 'dev file =', dev_file
		assert os.path.isfile(dev_file)
		dev_reader = NumpyWindowReader(dev_file)
		return VanillaPhrase(dev_reader.get_phrase_matrix())


	def num_train_phrases(self):
		return len(self.train_readers)
	

	def train_phrase(self, i):
		r = self.train_readers[i]
		return VanillaPhrase(r.get_phrase_matrix())




class AdditiveTrainer(Trainer, object):

	def __init__(self, model_dir, data_dir, k):
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
		return AdditiveEmbedding(self.alph, 3, k, learning_rate_scale=learning_rate_scale)


	def dev_phrase(self):
		if self.dev is None:
			print '[Additive dev_phrase] constructing dev phrase...'
			word_indices = np.load(os.path.join(self.data_dir, 'dev.small.npy'))
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
		print "[AdditiveTrainer train_phrase[%d]] have features for %.1f%% of words in windows" % (i, 100.0*prop_feat)

		p = FeaturizedPhrase(word_indices, feat_indices)
		print "[AdditiveTrainer train_phrase[%d]] phrases.shape=(%d,%d)" % (i, len(p), p.width)
		return p



if __name__ == '__main__':
	
	data_dir = 'data/jumbo/'
	k = 5
	runners = {
		'vanilla' : VanillaTrainer('models/vanilla/', data_dir, k), \
		'additive' : AdditiveTrainer('models/additive/', data_dir, k) \
	}
	for a in sys.argv[1:]:
		if a in runners:
			runners[a].run()
		else:
			print 'i dont know how to run', a






