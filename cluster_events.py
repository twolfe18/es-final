
from operator import itemgetter
import scipy

class EventCluster:

	def __init__(self, word_id, word_vec, is_verbal, providence='unknown'):
		self.center = (word_id, word_vec)
		self.nom_members = []
		self.verb_members = []
		self.providence = providence
		self.add_word(word_id, word_vec, is_verbal)
	
	def add_word(self, word_id, word_vec, is_verbal):
		if is_verbal:
			self.verb_members.append( (word_id, word_vec) )
		else:
			self.nom_members.append( (word_id, word_vec) )
	
	def __len__(self): return len(self.verb_members) + len(self.nom_members) + 1

	def __str__(self):
		n = ', '.join([str(w_id) for w_id, w_vec in self.nom_members])
		v = ', '.join([str(w_id) for w_id, w_vec in self.verb_members])
		return "<EventCluster center=%d noms=[%s] verbs=[%s]>" % (self.center[0], n, v)

	def get_member_word_ids(self):
		s = []
		for word_id, word_vec in self.nom_members + self.verb_members:
			s.append(word_id)
		return s
			

class EventClusterer:

	def __init__(self, alph, nomlex_file, train_files, dev_file, dist='euclidean'):
		""" dist should be either 'euclidean' or 'cosine' """

		self.dist_method = dist
		self.alph = alph
		self.clusters = []	# list of EventClusters
		self.merged = {}	# keys are word ids, values are EventClusters in self.clusters
		emb = AdditiveEmbeddings()

		# add nomlex pairs as clusters
		nomlex = NomlexReader(nomlex_file, alph)
		for n, v in nomlex.get_int_pairs():
			c = EventCluster(n, emb.get_vec(n), False, providence='nomlex')
			c.add_word(v, emb.get_vec(v), True)
			self.add_cluster(c)

	def dist(self, vec1, vec2):
		if self.dist_method == 'euclidean':
			d = vec1 - vec2
			return np.linalg.norm(d)
		elif self.dist_method == 'cosine':
			return scipy.spatial.distance.cosine(vec1, vec2)
		else:
			raise 'I don\'t know what distance you\'re talking about:', self.dist_method

	def add_cluster(self, c):
		self.clusters.append(c)
		for w_id in c.get_member_word_ids():
			self.merged[w_id] = c
	
	def make_a_merge(self, pairs_to_take):
		""" provide how many pairs should be accepted as merges into event clusters
			returns the distance of the last merge made
		"""
	
		print "[make_a_merge] about to merge %d words" % (pairs_to_take)
		pairs = []
		for n_id, n_vec in emb.get_noms():
			for v_id, v_vec in emb.get_verbs():

				# this means these two words appear in a cluster already
				if v_id in self.merged and n_id in self.merged:
					continue

				d = self.dist(n_vec, v_vec)
				pairs.append( (n_id, v_id, d) )

		print "[make_a_merge] found %d candidate merges, sorting to find the %d best" % \
			(len(pairs), pairs_to_take)
		word_dist = 0
		pairs = sorted(pairs, key=itemgetter(2))
		for i in range(pairs_to_take):
			n_id, v_id, d = pairs[i]
			worst_dist = d

			c = self.merged.get(n_id)
			if c is not None:
				c.add_word(v_id, emb.get_vec(v_id), True)
				print "[make_a_merge] just added VERB: %s to %s" % (self.alph.lookup_value(v_id), c)
				continue

			c = self.merged.get(v_id)
			if c is not None:
				c.add_word(n_id, emb.get_vec(n_id), True)
				print "[make_a_merge] just added NOM: %s to %s" % (self.alph.lookup_value(n_id), c)
				continue

			c = EventCluster(n_id, emb.get_vec(n_id), False, providence='EventClusterer')
			c.add_word(v_id, emb.get_vec(v_id), True)
			print "[make_a_merge] made a new cluster from %s and %s" % \
				(self.alph.lookup_value(n_id), self.alph.lookup_value(v_id))
			self.add_cluster(c)

		return worst_dist


