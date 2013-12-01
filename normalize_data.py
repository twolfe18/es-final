import re
import os
import sys
import gzip
import time
import codecs
import random
import tempfile
import subprocess
from learn_word_vecs import *

class Splitter:

	def wc(self, f):
		print 'counting lines in', f
		i = subprocess.check_output(['wc', '-l', f])
		return int(i.split()[0])
	
	def split(self, input_file, output_dir, lines_per_file=500000, \
			filename_prefix='file', filename_suffix='.txt'):
		assert os.path.isdir(output_dir)
		n = self.wc(input_file)
		k = int(float(n) / lines_per_file + 0.5)
		print "%s has %d lines, splitting into %d files with no more than %d lines" % \
			(input_file, n, k, lines_per_file)
		files = []
		for i in range(k):
			f = os.path.join(output_dir, "%s%d%s" % (filename_prefix, i, filename_suffix))
			files.append( codecs.open(f, 'w', 'utf-8') )
		#print '[split] files =', [f.name for f in files]
		with codecs.open(input_file, 'r', 'utf-8') as f:
			c = 0
			for line in f:
				i = random.randint(0, k-1)
				files[i].write(line)
				c += 1
				if c % 100000 == 0:
					sys.stdout.write('*')
			print
		for f in files:
			f.close()
		return [f.name for f in files]

class Normalizer:

	# TODO '1-732-390-4697'
	# TODO '115.7'
	# TODO '1990s'
	# TODO '24th'
	# TODO '5,600'
	# TODO '120-foot-long'
	# TODO '1\/4'
	# TODO just take sort by count and look at what sticks out
	def __init__(self, arity=5):
		""" arity is how many tokens should appear in a phrase """
		self.number = re.compile('\d+')
		self.allcaps = re.compile('[^a-z]{5,}')	# only get words that don't look like acronyms
		self.arity = arity

	def normalize_token(self, tok):
		if self.number.match(tok):
			try:
				if len(tok) == 4:
					n = int(tok)
					if n >= 1800 and n <= 2099:
						return tok[:3] + 'D'
				return tok[0] + ('D' * (len(tok)-1))
			except:
				n = float(tok)
				return 'F' * int(math.log(n))
			finally:
				return tok
		if self.allcaps.match(tok):
			return tok.lower()
		return tok

	def normalize(self, infile, outfile):
		print "[normalize] %s => %s" % (infile, outfile)
		start = time.clock()
		fin = codecs.open(infile, 'r', 'utf-8')
		fout = codecs.open(outfile, 'w', 'utf-8')
		#fin = gzip.open(infile, 'rb')
		#fout = gzip.open(outfile, 'wb')
		#reader = codecs.getreader('utf-8')
		skipped = 0
		total = 0
		for line in fin:
			total += 1
			toks = line.strip().split()
			if len(toks) != self.arity:
				skipped += 1
				continue
			ntoks = [self.normalize_token(x) for x in toks]
			nline = '\t'.join(ntoks) + '\n'
			fout.write(nline)
		fin.close()
		fout.close()
		print "[normalize] done in %.1f seconds, skipped %d of %d lines" % \
			(time.clock()-start, skipped, total)

if __name__ == '__main__':

	input_file = 'data/windows.small'
	output_dir = 'data/split-for-testing/'
	lines_per_file = 20000

	input_file = 'data/windows.med'
	output_dir = 'data/split-for-testing/'
	lines_per_file = 200000

	input_file = '/home/travis/Desktop/word-windows-5.txt'
	output_dir = 'data/jumbo/'
	lines_per_file = 5000000

	# split into managable text files
	splitter = Splitter()
	files = splitter.split(input_file, output_dir, lines_per_file=lines_per_file)
	print '[main] splitter returned', files

	# normalize text (inplace)
	norm = Normalizer(arity=5)
	_, tempfile = tempfile.mkstemp()
	for f in files:
		norm.normalize(f, tempfile)
		subprocess.check_call(['cp', tempfile, f])
	os.unlink(tempfile)
	
	# choose train/dev/test split
	random.shuffle(files)
	dev = files[0]
	test = files[1]
	train = files[2:]
	print 'train =', train
	print 'dev   =', dev
	print 'tet   =', test
	

	def compress_and_rename(input_file, alph, output_file):
		print "[compress_and_rename] %s => %s len(alph)=%d" % (input_file, output_file, len(alph))
		if output_file is None:
			# count
			wr = WindowReader(input_file, alph, oov=None)
		else:
			# get real phrases (with OOVs)
			wr = WindowReader(input_file, alph, oov='<OOV>')

		# numpy format
		phrases = wr.get_phrase_matrix()

		if output_file is not None:
			np.save(output_file, phrases)

	# count
	a = CountAlphabet()
	for f in [dev] + train:
		compress_and_rename(dev, a, None)

	# prune
	a_pruned = Alphabet()
	a_pruned.lookup_index('<OOV>', add=True)
	for key in a.high_count_keys(10):
		a_pruned.lookup_index(key, add=True)
	print "[main] pruned original alphabet (size=%d) to exclude words that appeared fewer than %d times (size=%d)" % \
		(len(a), 10, len(a_pruned))
	a = a_pruned

	# go over text again, only convert what pruned tokens
	compress_and_rename(test, a, os.path.join(output_dir, 'test.npy'))
	compress_and_rename(dev, a, os.path.join(output_dir, 'dev.npy'))
	for i, f in enumerate(train):
		compress_and_rename(f, a, os.path.join(output_dir, 'train'+str(i)+'.npy'))

	# save the alphabet
	a_path = os.path.join(output_dir, 'train-dev.alphabet')
	print '[main] saving alphabet to', a_path
	a.save(a_path)

	# tests
	a2 = Alphabet(a_path)
	print a._by_key == a2._by_key

	W1 = np.load(os.path.join(output_dir, 'dev.npy'))
	W2 = WindowReader(dev, a).get_phrase_matrix()

	print W1
	print W2

	n1, k1 = W1.shape
	assert k1 == 5

	n2, k2 = W1.shape
	assert k2 == 5

	assert n1 == n2

	# they are equal but this fails for some reason
	e = np.equal(W1, W2).all()
	print e
	if not e:
		print 'np: ', W1
		print 'window reader:', W2





