
from learn_word_vecs import *

class AdaGradParamTest:
	def run(self):
		print 'initializing param...'
		in_var = T.dvector('in_var')
		v = theano.shared(np.ones((2,5)))
		cost = T.dot(v, in_var).norm(2)
		p = AdaGradParam(v, [in_var], cost, learning_rate=1e-2)
		print 'before any updates:', p, v.get_value()
		print

		# step 1
		in_val = np.random.rand(5)
		print 'in_val =', in_val
		p.update([in_val])
		print 'after 1 update: param =', p, 'v =', v.get_value()
		print 'gg =', p.gg.get_value()
		print

		# limit: v should go to 0
		itr = 300
		for i in range(itr):
			in_val = np.random.rand(5)
			p.update([in_val])
		print 'after', (itr+1), 'updates: param =', p, 'v =', v.get_value()
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
		ve = VanillaEmbeddings(len(self.alph), 3)
		print 'W.shape =', W.shape
		W_bad = ve.corrupt(W)
		print 'orig      =', W[1:10,]
		print 'corrupted =', W_bad[1:10,]

		

		return False

	def learn_fake_data(self):
		start = time.clock()
		W = self.windows.get_phrase_matrix()
		print 'windows', W.shape, 'len(alph)', len(a)
		print
		print 'making vanilla embeddings...'
		emb = VanillaEmbeddings(len(a), k=3, d=10, h=3, batch_size=20)
		emb.setup_theano()
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


