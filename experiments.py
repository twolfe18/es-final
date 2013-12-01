
from learn_word_vecs import *
import numpy as np
import numpy.random as rand
import sys

class VanillaTrainer:
	""" train some vanilla windows """

	def run(self):

		model_dir = 'models/vanilla-testing/'
		data_dir = 'data/split-for-testing/'
		k = 5

		a = Alphabet(os.path.join(data_dir, 'train-dev.alphabet'))
		print 'alphbet contains', len(a), 'words'
		ve = VanillaEmbeddings(a, k)
		ve.init_weights()

		dev_file = os.path.join(data_dir, 'dev.npy')
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
		W_dev = dev_reader.get_phrase_matrix()

		e = 100
		print 'training for', e, 'epochs'
		prev_avg_loss = 1.0
		for epoch in range(e):
			print 'starting epoch', epoch
			improvement = False
			for r in train_readers:
				W_train = r.get_phrase_matrix()
				print 'file', r.filename, '...training...'
				losses = ve.train(W_train, W_dev, epochs=7, iterations=100)
				print 'losses =', losses
				avg_loss = sum(losses) / len(losses)
				if avg_loss > prev_avg_loss + 1e-3:
					break
				prev_avg_loss = avg_loss
				improvement = True
			d = model_dir + '.epoch' + str(e)
			os.mkdir(d)
			ve.write_weights(d)
			if not improvement:
				break

		print 'saving model...'
		d = model_dir + '.final'
		os.mkdir(d)
		ve.write_weights(d)


if __name__ == '__main__':
	vt = VanillaTrainer()
	vt.run()

