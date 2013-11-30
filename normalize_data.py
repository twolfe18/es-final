import sys, codecs, gzip, re


class Splitter:
	def __init__(self):

class Normalizer:

	# TODO '1-732-390-4697'
	# TODO '115.7'
	# TODO '1990s'
	# TODO '24th'
	# TODO '5,600'
	# TODO '120-foot-long'
	# TODO '1\/4'
	# TODO just take sort by count and look at what sticks out
	def __init__(self):
		self.number = re.compile('\d+')
		self.allcaps = re.compile('[^a-z]{5,}')	# only get words that don't look like acronyms

	def normalize_token(tok):

		if number.match(tok):
			if len(tok) == 4:
				n = int(tok)
				if n >= 1800 && n <= 2099:
					return tok[:3] + 'D'
			return tok[0] + ('D' * (len(tok)-1))

		if allcaps.match(tok):
			return tok.lower()

		return tok

	def normalize_windows(infile, outfile):
		fin = gzip.open(infile, 'rb')
		fout = gzip.open(outfile, 'wb')
		reader = codecs.getreader('utf-8')
		#writer = codecs.getreader('utf-8')
		for line in reader(fin):
			toks = line.strip().split()
			ntoks = [normalize_token(x) for x in toks]
			nline = '\t'.join(ntoks) + '\n'
			#fout
		fin.close()
		fout.close()

if __name__ == '__main__':

	input_file = 'windows.med'
	output_dir = 'data/split-for-testing/'
	#input_file = '/home/travis/Desktop/word-windows-5.txt'
	#output_file = 'data/jumbo/'

	# split into managable text files
	splitter = Splitter()
	files = splitter.split(input_file, output_dir, lines_per_file=int(1e6))

	# normalize text
	norm = Normalizer()
	tempfile = Tempfile()
	for f in files:
		norm.normalize(f, tempfile)
		subprocess.call(['mv', tempfile.name, f.name])
	
	# choose train/dev/test split
	random.shuffle(files)
	dev = files[0]
	test = files[1]
	train = files[2:]

	# build alphabet and compress text to ints
	def compress_and_rename(input_file, alph, output_file, add_to_alph=True):
		wr = WindowReader(input_file, alph)
		phrases = wr.get_phrase_matrix()
		np.save(output_file, phrases)

	a = Alphabet()
	for i, f in enumerate(train):
		compress_and_rename(f, a, os.join(output_dir, 'train'+str(i)+'.npy'))
	compress_and_rename(dev, a, os.join(output_dir, 'dev.npy'))
	compress_and_rename(test, a, os.join(output_dir, 'test.npy'), add=False)

	# save the alphabet
	a.save(os.join(output_dir, 'train-dev.alphabet'))





