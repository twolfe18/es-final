
from embeddings import *
import numpy as np
import numpy.random as rand
import sys
import random

class VanillaTrainer:
	""" train some vanilla windows """

	def run(self):

		#model_dir = 'models/vanilla-testing/'
		#data_dir = 'data/split-for-testing/'
		model_dir = 'models/vanilla/'
		data_dir = 'data/jumbo/'
		k = 5

		a = Alphabet(os.path.join(data_dir, 'train-dev.alphabet'))
		print 'alphbet contains', len(a), 'words'
		ve = VanillaEmbedding(a, k, learning_rate_scale=5.0)

		dev_file = os.path.join(data_dir, 'dev.small.npy')
		assert os.path.isfile(dev_file)
		train_files = []
		for f in os.listdir(data_dir):
			if not f.endswith('.npy'): continue
			if f.startswith('train'):
				train_files.append(f)

		print 'train files =', train_files
		print 'dev file =', dev_file
		train_readers = [NumpyWindowReader(os.path.join(data_dir, f)) for f in train_files]
		dev_reader = NumpyWindowReader(dev_file)
		W_dev = VanillaPhrase(dev_reader.get_phrase_matrix())

		e = 500
		print 'training for', e, 'macro-epochs'
		prev_avg_loss = 1.0
		for epoch in range(e):
			print 'starting epoch', epoch
			improvement = False
			for r in train_readers:
				W_train = VanillaPhrase(r.get_phrase_matrix())
				print 'file', r.filename, '...training...'
				losses = ve.train(W_train, W_dev, epochs=6, iterations=30, batch_size=500)
				print 'losses =', losses
				avg_loss = sum(losses) / len(losses)
				if avg_loss < prev_avg_loss - 1e-4 or avg_loss / prev_avg_loss < 0.99:
					if e % 10 == 0 or (e > 50 and e % 4 == 0) or (e > 100):
						ve.write_weights(model_dir + 'epoch' + str(epoch))
					improvement = True
					prev_avg_loss = avg_loss
					break
				prev_avg_loss = avg_loss
			random.shuffle(train_readers)
			if not improvement:
				break

		print 'saving model...'
		d = model_dir + 'final'
		ve.write_weights(d)

class AdditiveTrainer:
	def run(self):
		# read in phrases
		# read in NOMLEX
		# construct features
		# call train
		
		data_dir = 'data/jumbo-test/'
		model_dir = 'models/additive-test/'
		assert os.path.isdir(data_dir)
		k = 5

		alph = Alphabet(os.path.join(data_dir, 'train-dev.alphabet'))
		train_phrases = np.load(os.path.join(data_dir, 'train0.npy'))
		dev_phrases = np.load(os.path.join(data_dir, 'dev.small.npy'))

		nomlex = NomlexReader('data/nomlex.txt', alph)
		train_features = nomlex.get_features(train_phrases)
		dev_features = nomlex.get_features(dev_phrases)

		# about 4.2% last time i checked
		prop_feat = (train_features > 0).sum() / float(len(train_features) * len(train_features[0]))
		print "[AdditiveTrainer] have features for %.1f%% of words in windows" % (100.0*prop_feat)

		av = AdditiveEmbedding(alph, 3, k)
		for i in range(100):
			av.train(train_phrases, dev_phrases, train_features, dev_features)


if __name__ == '__main__':
	
	runners = { 'vanilla':VanillaTrainer(), 'additive':AdditiveTrainer() }
	for a in sys.argv[1:]:
		if a in runners:
			runners[a].run()
		else:
			print 'i dont know how to run', a






