
import sys
import numpy as np

num_words = 80
num_examples = 250000
agreeability = 12
window_size = 3

def adjacency_matrix():
	rows = []
	for i in range(num_words):
		num_neighbors = np.random.poisson(agreeability)
		neighbors = np.random.choice(num_words, num_neighbors, replace=False)
		rows.append(neighbors)
	return rows

def random_window(adjacency):
	s = np.random.randint(num_words)
	for i in range(window_size):
		c = adjacency[s]
		if len(c) == 0:
			s = np.random.randint(num_words)
		else:
			s = np.random.choice(c)
		yield s

if __name__ == '__main__':

	if len(sys.argv) != 2:
		print 'please give an output file'
		sys.exit(-1)

	a = adjacency_matrix()
	#print 'a =', a

	f = open(sys.argv[1], 'w')
	for i in xrange(num_examples):
		w = random_window(a)
		w = [str(x) for x in w]
		f.write('\t'.join(w) + '\n')
	f.close()


