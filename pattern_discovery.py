#! /usr/bin/env python3

from scipy.spatial.distance import pdist, squareform
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import AffinityPropagation
import custompattern_mining as cpm
import matplotlib.pyplot as plt 
from dtw import dtw
import numpy as np
import subprocess
import jenkspy 
import pandas 
import json
import time
import math
import os

def timing_wrapper(func):
	def wrapper(*args,**kwargs):
		
		t0= time.time()
		func_val = func(*args,**kwargs)
		time_taken = time.time() - t0

		print (str(func),' took: ', time_taken)

		return func_val

	return wrapper

class DTWClusteringError(Exception):
	### Raised when DTW clustering could not find required clusters.
	pass

class PatternDiscovery:
	def __init__ (self, sequence, state_attributes,min_len, max_len, similarity_constraint=0.9):
		self.state_attributes = state_attributes
		self.pm = cpm.PatternMining(sequence, state_attributes, min_len, max_len)
		self.patterns = self.__get_patterns()
		self.max_var_label = None
		self.idle_label = None
		self.pattern_dict = None
		self.working_patterns = None
		self.similarity_constraint = similarity_constraint  ## No single element can appear more than 100x% of the time

	def __get_patterns(self):
		pattern_sets = self.pm.find_patterns()
		seq_support = [(p_set[0], len(p_set)) for p_set in pattern_sets]
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
		freq = p_dist_max*freq/max(freq)
		ap = AffinityPropagation(affinity='precomputed', preference=freq)
		ap.fit(p_dist)
		cluster_subseqs_exs = [ seq[ind] for ind in ap.cluster_centers_indices_]

		### Arranging sequences by cluster label 
		cluster_subseqs = dict()
		for seq, label in zip(seq_f,ap.labels_):
			if label not in cluster_subseqs:
				cluster_subseqs.update({label : [seq]})
			else:
				seq_list = cluster_subseqs[label]
				seq_list.append(seq)
				cluster_subseqs.update({ label: seq_list})
		
		return cluster_subseqs, cluster_subseqs_exs

	@timing_wrapper
	def cluster_patterns(self):
		seq_f =  self.patterns
		### Clustering sequences using dtw and affinity propagation
		cluster_subseqs, cluster_subseqs_exs = self.__dtw_clustering(seq_f)
		print ('Number of clusters with DTW: ', len(cluster_subseqs))
		if len(cluster_subseqs) == 1:
			raise DTWClusteringError('Incorrect Clustering -> Minimum No. Of Clusters Not Found')
		# print (cluster_subseqs)

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
		
		### Classification based on variance of patterns, min_var -> idle, others-> working
		working_patterns = []
		idle_patterns = []
		for e,l in enumerate(cl_mv_labels):
			if l == idle_label:
				idle_patterns.extend(cluster_subseqs[e])
			else:
				working_patterns.extend(cluster_subseqs[e])

		### Grouping sequences by cluster label -> later inference 
		cluster_seqs = dict()
		for e,label in enumerate(cl_mv_labels):
			if label not in cluster_seqs:
				cluster_seqs.update({label : cluster_subseqs[e]})
			else:
				seq_list = cluster_seqs[label]
				seq_list.extend(cluster_subseqs[e])
				cluster_seqs.update({ label: seq_list})
		
		### Printing values
		print ('Final Number of Clusters: ', len(cluster_seqs))
		print ('Idle Class: ', idle_label)
		print ('Max Var Mean: ', max_label)
		for k in cluster_seqs:
			print (k)
			print (cluster_seqs[k])
		
		return working_patterns, idle_patterns, cluster_seqs 

	@timing_wrapper
	def discover_pattern(self):
		if len(self.patterns) == 1:
			return self.patterns[0][0]
		try:
			working_patterns, idle_patterns, self.pattern_dict = self.cluster_patterns()
			self.working_patterns = working_patterns
		except DTWClusteringError:
			return None
		
		### Looking for patterns that start and stop in the same state, and have less than 90% similarity
		possible_patterns = []
		for seq in working_patterns:
			if all( self.state_attributes[str(s)][0] >= self.state_attributes[str(seq[0][0])][0] for s in seq[0]):
				_, count = np.unique(seq[0], return_counts=True)
				count = count/sum(count)
				if all(c < self.similarity_constraint for c in count):
					possible_patterns.append(seq)
		
		### Clustering with DTW to find patterns. Exemplars -> final patterns 
		if len(possible_patterns) > 1:
			different_patterns, exemplars = self.__dtw_clustering(possible_patterns)
			print (different_patterns)
			print ('Number of clusters with DTW: ', len(different_patterns))
			print (exemplars)
			return exemplars
		elif len(possible_patterns) == 1:
			## If no such pattern exists, extend patterns that gives likely output
			print( possible_patterns[0][0])
			return possible_patterns[0][0]
		else:
			return None

	def __pattern_distance(self,a,b):
		val_a = np.array([self.state_attributes[str(s)][0] for s in a])
		val_b = np.array([self.state_attributes[str(s)][0] for s in b])
		dist, _, _, _ = dtw(val_a.reshape(-1,1), val_b.reshape(-1,1), dist=lambda x,y:np.linalg.norm(x-y))
		return dist 
	
def seq_contains(seq, subseq):
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
