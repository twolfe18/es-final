
from learn_word_vecs import *
import numpy as np
import numpy.random as rand
import sys

class AdaGradParamTest:
	def run(self):
		print 'initializing param...'
		in_var = T.dvector('in_var')
		v = theano.shared(np.ones((2,5)))
		cost = T.dot(v, in_var).norm(2)
		p = AdaGradParam(v, [in_var], cost, learning_rate=0.1)
		print 'before any updates:', p, v.get_value()
		print

		itr = 0

		# step 1
		itr += 1
		in_val = np.random.rand(5)
		print 'in_val =', in_val
		p.update([in_val])
		print 'after 1 update: param =', p, 'v =', v.get_value()
		print 'gg =', p.gg.get_value()
		print

		# step 2
		itr += 1
		in_val = np.random.rand(5)
		print 'in_val =', in_val
		p.update([in_val])
		print 'after 1 update: param =', p, 'v =', v.get_value()
		print 'gg =', p.gg.get_value()
		print

		# limit: v should go to 0
		itr += 100
		for i in range(100):
			in_val = np.random.rand(5)
			p.update([in_val])
		print 'after', itr, 'updates: param =', p, 'v =', v.get_value()
		print 'gg =', p.gg.get_value()
		print

		itr += 1000
		for i in range(1000):
			in_val = np.random.rand(5)
			p.update([in_val])
		print 'after', itr, 'updates: param =', p, 'v =', v.get_value()
		print 'gg =', p.gg.get_value()
		print

		itr += 10000
		for i in range(10000):
			in_val = np.random.rand(5)
			p.update([in_val])
		print 'after', itr, 'updates: param =', p, 'v =', v.get_value()
		print 'gg =', p.gg.get_value()
		print
		
class VanillaEmbeddingsTest:

	def run(self):
		self.alph = Alphabet()
		self.windows = WindowReader('fake_data.txt', alph=self.alph)
		if self.basic():
			self.learn_fake_data()
		else:
			print 'failed the basic test'

	def basic(self):
	
		W = self.windows.get_phrase_matrix()
		N, k = W.shape
		ve = VanillaEmbeddings(len(self.alph), k, d=80, h=40, learning_rate_scale=0.5)
		ve.init_weights(scale=100.0)
		print 'W.shape =', W.shape

		Z = ve.corrupt(W)
		print 'orig      =', W[1:10,]
		print 'corrupted =', Z[1:10,]

		idx = list(range(N))
		rand.shuffle(idx)
		i = int(N * 0.8)
		train_idx = idx[:i]
		dev_idx = idx[i:]
		
		W_train = W[train_idx,]
		Z_train = Z[train_idx,]
		W_dev = W[dev_idx,]
		Z_dev = Z[dev_idx,]

		N_train = W_train.shape[0]
		N_dev = W_dev.shape[0]
		print "there are %d training instances and %d dev" % (N_train, N_dev)

		print 'at start, dev.hinge =', ve.loss(W_dev, Z_dev, avg=False)

		for name, param in ve.params.iteritems():
			print name, 'has shape', param.shape

		# try to make sense of the gradient
		#batch_loss = ve.f_step(W[1:100,], Z[1:100,])
		#print 'batch_loss =', batch_loss
		print 'W.l2 before =', ve.params['W'].l2
		print 'ggW before =', ve.params['W'].gg.get_value()
		r = rand.choice(N_train, 2, replace=False)
		print 'W[r,] =', W[r,]
		print 'Z[r,] =', Z[r,]
		p = [W[r,], Z[r,]]
		print 'p =', p, type(p)
		dW = ve.params['W'].update(p, verbose=True)
		print 'dW =', dW
		print 'W.l2 after =', ve.params['W'].l2
		print 'ggW after =', ve.params['W'].gg.get_value()

		# below this seems to work



		print '*' * 100
		for i in range(50):
			print
		print '*' * 100







		def show_params():
			print 'W.l1 =', ve.params['W'].l1
			print 'W.lInf =', ve.params['W'].lInf
			print 'A.l1 =', ve.params['A'].l1
			print 'A.lInf =', ve.params['A'].lInf
			bad = ve.check_for_bad_params()

		# take one grad setup with this batch size
		def batch(size):
			r = rand.choice(N_train, size)
			W = W_train[r,]
			Z = Z_train[r,]
			ve.f_step(W, Z)

		def dev_accuracy():
			g = ve.raw_score(W_dev)
			b = ve.raw_score(Z_dev)
			right = (g > b).sum()
			return (100.0 * right) / len(g)

		print 'starting...'
		sys.stdout.flush()

		show_params()
		start = time.clock()
		bs = 500
		for i in range(5000):
			for j in range(50):
				batch(bs)
			t = time.clock() - start
			print i, t, 'dev.hinge =', ve.loss(W_dev, Z_dev, avg=False), 'accuracy =', dev_accuracy()
			sys.stdout.flush()


		return False

	def learn_fake_data(self):
		start = time.clock()
		W = self.windows.get_phrase_matrix()
		print 'windows', W.shape, 'len(alph)', len(a)
		print
		print 'making vanilla embeddings...'
		emb = VanillaEmbeddings(len(a), k=3, d=10, h=3, batch_size=20)
		emb.init_weights()

		print 'training...'
		emb.train(W)


if __name__ == '__main__':
	tests = {'adagrad': AdaGradParamTest(), 'vanilla': VanillaEmbeddingsTest()}
	names = tests.keys()
	if len(sys.argv) > 1:
		names = sys.argv[1:]
	for name in names:
		t = tests[name]
		if t is None:
			print 'no test named', name
		else:
			print "====================== Running %s test ======================" % (name)
			t.run()


