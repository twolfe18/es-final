
from learn_word_vecs import *
import numpy as np
import numpy.random as rand
import sys

class VanillaTrainer:
	""" train some vanilla windows """

	def run(self):

		model_dir = 'models/vanilla/'
		data_dir = 'data/split-for-testing/'

		a = Alphabet(os.path.join(data_dir, 'train-dev.alphabet'))
		print 'alphbet contains', len(a), 'words'
		ve = VanillaEmbeddings(a, k)
		ve.init_weights()

		train_files = []
		dev_file = None
		for f in os.listdir(data_dir):
			if not f.endswith('.npy'): continue
			if f.startswith('train'):
				train_files.append(f)
			elif f.startswith('dev'):
				assert dev_file is None
				dev_file = f

		print 'train files =', train_files
		print 'dev file =', dev_file
		train = MultiWindowReader(train_files, a, add_to_alph=False)
		dev = WindowReader(dev_file, a)
		W_dev = dev.get_phrase_matrix()

		e = 10
		print 'training for', e, 'epochs'
		for epoch in range(e):
			print 'starting epoch', epoch
			for i in range(train.num_partitions()):
				train.set_partition(i)
				W_train = train.get_phrase_matrix()
				print 'file', i, '...training...'
				ve.train(W_train, W_dev)

		print 'saving model...'
		ve.write_weights(model_dir)


if __name__ == '__main__':
	vt = VanillaTrainer()
	vt.run()

