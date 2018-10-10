#! /usr/bin/env python3

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AffinityPropagation
import custompattern_mining as cpm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import peak_detector as pd 
from dtw import dtw
import numpy as np
import plotly
import time
import json
import os

def timing_wrapper(func):
	def wrapper(*args,**kwargs):
		
		t0= time.time()
		func_val = func(*args,**kwargs)
		time_taken = time.time() - t0

		print (str(func),' took: ', time_taken)
		return func_val
	return wrapper


class SegmentDiscovery:
	def __init__ (self, no_segments, sequence, state_attributes,min_len, max_len):
		self.state_attributes = state_attributes
		self.pm = cpm.PatternMining(sequence, state_attributes, min_len, max_len)
		self.all_pattern_sets = self.pm.pattern_sets
		self.patterns = self.__get_patterns()
		self.working_label = None
		self.idle_label = None
		self.pattern_dict = None
		self.no_segments = no_segments
			
	def __get_patterns(self):
		seq_support = self.pm.find_patterns()
		# # Saving data for autoacc
		# with open('pyn_patterns.txt', 'w') as f:
		# 	for seq, freq in seq_support:
		# 		f.write('([')
		# 		for i in range(len(seq)):
		# 			if i!= len(seq)-1:
		# 				f.write('{0}, '.format(seq[i]))
		# 			else:
		# 				f.write('{0}], {1})\n'.format(seq[i],freq))

		# ## reading data for auto acc
		# seq_support = []
		# with open('pyn2_patterns.txt', 'r') as f:
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
		print (cluster_subseqs)
		print (cluster_subseqs_exs)
		print ('Number of clusters with DTW: ', len(cluster_subseqs))
		if len(cluster_subseqs) == 1:
			return cluster_subseqs, cluster_subseqs_exs, None, None			
		
		### Getting average variances and means of exemplars for classification
		cluster_mv = np.zeros((len(cluster_subseqs),2))
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

		### KMeans 
		kmeans = KMeans(self.no_segments, random_state=3).fit(cluster_mv)
		cl_mv_labels = kmeans.labels_
		cluster_norms = [np.linalg.norm(el) for el in kmeans.cluster_centers_]
		idle_label = cluster_norms.index(min(cluster_norms))
		self.idle_label = idle_label
		self.working_label = 1 - self.idle_label
		
		### Grouping sequences by cluster label -> later inference 
		cluster_seqs = dict()
		for e,label in enumerate(cl_mv_labels):
			if label not in cluster_seqs:
				cluster_seqs.update({label : list(cluster_subseqs[e])})
			else:
				cluster_seqs[label].extend(cluster_subseqs[e])

		### Printing values
		print ('Final Number of Clusters: ', len(cluster_seqs))
		print ('Idle Class: ', idle_label)
		print ('Working Class: ', self.working_label)
		for k in cluster_seqs:
			print (k)
			print (cluster_seqs[k])

		return cluster_seqs

	@timing_wrapper
	def discover_segmentation_pattern(self):
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
			self.pattern_dict = self.cluster_patterns(possible_patterns)
		
		elif len(possible_patterns) == 1:
			self.pattern_dict = {0: [self.patterns[0]]}


	def __pattern_distance(self,a,b):
		val_a = [self.state_attributes[str(s)][0] for s in a]
		val_b = [self.state_attributes[str(s)][0] for s in b]
		dist, _, _, _ = dtw(val_a, val_b, dist=lambda x,y:np.linalg.norm(x-y))
		return dist 