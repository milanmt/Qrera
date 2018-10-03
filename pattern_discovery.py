#! /usr/bin/env python3

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AffinityPropagation
import custompattern_mining as cpm
from dtw import dtw
import numpy as np
import time


def timing_wrapper(func):
	def wrapper(*args,**kwargs):
		
		t0= time.time()
		func_val = func(*args,**kwargs)
		time_taken = time.time() - t0

		print (str(func),' took: ', time_taken)
		return func_val
	return wrapper


class PatternDiscovery:
	def __init__ (self, sequence, state_attributes,min_len, max_len):
		self.state_attributes = state_attributes
		self.pm = cpm.PatternMining(sequence, state_attributes, min_len, max_len)
		self.all_pattern_sets = self.pm.pattern_sets
		self.patterns = self.__get_patterns()
		self.max_var_label = None
		self.idle_label = None
		self.pattern_dict = None
		self.working_patterns = None
		self.idle_patterns = None
		self.classification_dict = None
		
	def __get_patterns(self):
		seq_support = self.pm.find_patterns()
		# # Saving data for autoacc
		# with open('autoacc_patterns.txt', 'w') as f:
		# 	for seq, freq in seq_support:
		# 		f.write('([')
		# 		for i in range(len(seq)):
		# 			if i!= len(seq)-1:
		# 				f.write('{0}, '.format(seq[i]))
		# 			else:
		# 				f.write('{0}], {1})\n'.format(seq[i],freq))

		## reading data for auto acc
		# seq_support = []
		# with open('fwt_patterns.txt', 'r') as f:
		# 	for line in f:
		# 		to_be_removed = ['[', ']', '(', ')']
		# 		t_l = line
		# 		for s in to_be_removed:
		# 			t_l = t_l.replace(s, '')

		# 		split_t_l = t_l.split(',')
		# 		seq = [int(s.strip()) for s in split_t_l[:-1]]
		# 		freq = int(split_t_l[-1].strip())
		# 		if freq > 1:
		# 			seq_support.append((seq, freq))

		return seq_support

	def __dtw_clustering(self, seq_f):
		### Clustering sequences using affinity propagation, dtw
		### Computing similarity/affinity matrix using dtw
		p_dist = np.zeros((len(seq_f), len(seq_f)))
		if isinstance(seq_f[0],tuple):
			seq = [item[0] for item in seq_f]
			freq = np.array([item[1] for item in seq_f])
		else:
			seq = seq_f

		for i in range(len(seq)):
			for j in range(i,len(seq)):
				p_dist[i][j] = self.__pattern_distance(seq[i],seq[j])
				if i != j:
					p_dist[j][i] = p_dist[i][j]

		p_dist_max = np.max(p_dist)
		if p_dist_max == 0:
			p_dist_max = 2
		p_dist = p_dist_max - p_dist
		
		### Affinity Propagation
		freq = 2*p_dist_max*freq/max(freq)
		ap = AffinityPropagation(affinity='precomputed', preference=freq)
		ap.fit(p_dist)
		cluster_subseqs_exs = [ seq[ind] for ind in ap.cluster_centers_indices_]

		### Arranging sequences by cluster label 
		cluster_subseqs = dict()
		for seq, label in zip(seq_f,ap.labels_):
			if label not in cluster_subseqs:
				cluster_subseqs.update({label : [seq]})
			else:
				cluster_subseqs[label].append(seq)
				
		return cluster_subseqs, cluster_subseqs_exs

	@timing_wrapper
	def cluster_patterns(self, seq_f):
		### Clustering sequences using dtw and affinity propagation
		cluster_subseqs, cluster_subseqs_exs = self.__dtw_clustering(seq_f)
		print ('Number of clusters with DTW: ', len(cluster_subseqs))
		if len(cluster_subseqs) == 1:
			return cluster_subseqs, cluster_subseqs_exs, None, None			
		
		### Getting average variances and means of exemplars for classification
		cluster_mv = np.zeros((len(cluster_subseqs),2))
		cluster_mv_norm = list(np.zeros(len(cluster_subseqs)))
		for label, seq_l in cluster_subseqs.items():
			var_seq_l = []
			avg_seq_l = []
			for seq_supp in seq_l:
				seq = seq_supp[0]
				var = np.std([self.state_attributes[str(s)][0] for s in seq])
				avg = np.mean([self.state_attributes[str(s)][0] for s in seq])
				var_seq_l.append(var)
				avg_seq_l.append(avg)
			cluster_mv[label][1] = np.mean(var_seq_l)
			cluster_mv[label][0] = np.mean(avg_seq_l)
			cluster_mv_norm[label] = np.linalg.norm(cluster_mv[label])

		### Obtaining affinity matrix
		p_dist = squareform(pdist(cluster_mv))
		p_dist_max = np.max(p_dist)
		if p_dist_max == 0:
			p_dist_max = 2
		p_dist = p_dist_max - p_dist

		### Affinity Propagation based on means and variances
		ap_mv = AffinityPropagation(affinity='precomputed', preference=0.85*p_dist_max)
		cl_mv_labels = ap_mv.fit_predict(p_dist)
		cl_v_exs = [ cluster_mv[ind][1] for ind in ap_mv.cluster_centers_indices_]
		idle_label = cl_mv_labels[cluster_mv_norm.index(min(cluster_mv_norm))]
		self.idle_label = idle_label
		max_label = cl_mv_labels[cluster_mv_norm.index(max(cluster_mv_norm))]
		self.max_var_label = max_label

		### Grouping sequences by cluster label -> later inference 
		cluster_seqs = dict()
		for e,label in enumerate(cl_mv_labels):
			if label not in cluster_seqs:
				cluster_seqs.update({label : list(cluster_subseqs[e])})
			else:
				cluster_seqs[label].extend(cluster_subseqs[e])

		labels_to_remove = []
		### Checking if clusters have to be removed
		for l,p_set in cluster_seqs.items():
			max_p = max(p_set, key=lambda x:x[1])[0] 
			for p_set2 in cluster_seqs.values():
				max_p2 = max(p_set2, key=lambda x:x[1])[0]
				if max_p2 != max_p and seq_contains(max_p2,max_p):
					labels_to_remove.append(l)

		for l in labels_to_remove:
			if l != self.max_var_label or l != self.idle_label:
				del cluster_seqs[l]
								
		### Classification based on variance of patterns, min_var -> idle, other exemplars-> working
		working_patterns = []
		idle_patterns = []
		for l in cluster_seqs:
			if l == idle_label:
				idle_patterns.append(max(cluster_seqs[l], key=lambda x:x[1])[0])
			else:
				working_patterns.append(max(cluster_seqs[l], key=lambda x:x[1])[0])

		### Printing values
		print ('Final Number of Clusters: ', len(cluster_seqs))
		print ('Idle Class: ', idle_label)
		print ('Max Var Mean: ', max_label)
		for k in cluster_seqs:
			print (k)
			print (cluster_seqs[k])


		### Pattern dictionary for classification
		new_cluster_dict = dict()
		for e,label in enumerate(cl_mv_labels):
			full_list = []
			for pattern,f in cluster_subseqs[e]:
				for p, p_dict in self.pm.pattern_sets.items():
					if self.__pattern_distance(p,pattern) == 0.0:
						for p_s, f in p_dict.items():
							full_list.append((list(p_s),f))


			if label not in new_cluster_dict:
				new_cluster_dict.update({label : full_list})
			else:
				new_cluster_dict[label].extend(full_list)

		self.classification_dict = new_cluster_dict
		for k in new_cluster_dict:
			print (k)
			print (new_cluster_dict[k])

		return cluster_seqs, working_patterns, idle_patterns 

	@timing_wrapper
	def discover_pattern(self):
		if len(self.patterns) == 1:
			self.pattern_dict = {0: [self.patterns[0]]}
			return self.patterns[0][0]
		
		### Looking for signals which start and stop with minimas. Need to doscover most likely candidate. 
		possible_patterns = []
		for seq in self.patterns:
			if all(s >= seq[0][0] for s in seq[0]):
				possible_patterns.append(seq)

		### Clustering with DTW to find patterns. Exemplars from DTW -> final patterns 
		### These clustered based on mean and variance to identify idle and working patterns
		if len(possible_patterns) > 1:
			self.pattern_dict, self.working_patterns, self.idle_patterns= self.cluster_patterns(possible_patterns)
			final_patterns = []
			if self.working_patterns == None:
				for p_set in self.pattern_dict.values():
					for p in p_set:
						final_patterns.append(p[0])
			else:
				for p in self.working_patterns:
					final_patterns.append(p)

			print (final_patterns)
			return final_patterns
		
		elif len(possible_patterns) == 1:
			self.pattern_dict = {0: [self.patterns[0]]}
			print( possible_patterns[0][0])
			return [possible_patterns[0][0]]
		else:
			return None

	def __pattern_distance(self,a,b):
		val_a = [self.state_attributes[str(s)][0] for s in a]
		val_b = [self.state_attributes[str(s)][0] for s in b]
		dist, _, _, _ = dtw(val_a, val_b, dist=lambda x,y:np.linalg.norm(x-y))
		return dist 


def seq_contains(seq, subseq):
	#### Should make this dtw
	seq_s = str()
	for x in seq:
		seq_s = seq_s + str(x)
	subseq_s = str()
	for x in subseq:
		subseq_s = subseq_s + str(x)
	if subseq_s in seq_s:
		return True
	else:
		return False 