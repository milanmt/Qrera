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


class SequentialPatternMining:
	def __init__(self, sequence, state_attributes, peak_indices):
		### Mining generative patterns only
		self.MAX_LEN = 20
		self.MIN_LEN = 5
		self.MAX_LEN_EXTENSION = 15
		self.MIN_SUPPORT = 0.3
		self.LEN_SEGMENT = 600
		self.sequence = list(sequence) if not isinstance(sequence, list) else sequence
		self.peak_indices = peak_indices
		self.state_attributes = state_attributes
		self.states = [s for s in self.state_attributes.keys()]
		self.db_filename = 'timedb_test.txt'
		self.pattern_filename = 'output.txt'
		self.path_to_spmf = '/media/milan/DATA/Qrera'
		self.similarity_constraint = 0.9  ## No single element can appear more than 100x% of the time
		self.generator_patterns = self.__get_all_freq_seq() 
		self.pattern_dict = None
		self.working_patterns = None
		self.max_var_label = None
		self.idle_label = None
	
	def __get_pref(self, flat_array):
		return jenkspy.jenks_breaks(flat_array, nb_class=2)[1]

	def __pattern_distance(self,a,b):
		val_a = np.array([self.state_attributes[str(s)][0] for s in a])
		val_b = np.array([self.state_attributes[str(s)][0] for s in b])
		dist, _, _, _ = dtw(val_a.reshape(-1,1), val_b.reshape(-1,1), dist=lambda x,y:np.linalg.norm(x-y))
		return dist 

	@timing_wrapper
	def __generate_timeseries_db(self):
		with open(self.db_filename, 'w') as f:
			for i in range(86400//self.LEN_SEGMENT):
				segment_ind = []
				for e,ind in enumerate(self.peak_indices):
					if ind >= i*self.LEN_SEGMENT and ind < (i+1)*self.LEN_SEGMENT:
						segment_ind.append(e)
					elif ind >= (i+1)*self.LEN_SEGMENT:
						break

				if segment_ind:
					segment = self.sequence[segment_ind[0]:segment_ind[-1]+1]
					for s in segment:
						f.write('{0} -1 '.format(s)) 
					f.write('-2\n')

		self.db_filename = os.path.realpath(self.db_filename)

	def __pattern_mining(self):
		self.__generate_timeseries_db()	
		## Mining Generative Patterns
		subprocess.call(('java -jar spmf.jar run VGEN '+self.db_filename+
			' '+self.pattern_filename+' '+str(self.MIN_SUPPORT)+' '+
			str(self.MAX_LEN)+' 1 false'),cwd=self.path_to_spmf,shell=True)
		self.pattern_filename = os.path.join(self.path_to_spmf, self.pattern_filename)

	@timing_wrapper
	def __get_all_freq_seq(self):
		# self.__pattern_mining()
		# seq_support = []
		# with open(self.pattern_filename, 'r') as f:
		# 	for line in f:
		# 		if '-1' in line:
		# 			temp_l = line.split(' -1 ')
		# 			seq = []
		# 			support = 0 
		# 			for s in temp_l:
		# 				if '#SUP' in s:
		# 					support = int(s.split(':')[1].strip())
		# 				else:
		# 					seq.append(int(s))
		# 			if len(seq) >= self.MIN_LEN:
		# 				seq_support.append((seq, support))

		cpm_pm = cpm.PatternMining(self.sequence, self.state_attributes)
		pattern_sets = cpm_pm.find_patterns()
		seq_support = [(p_set[0], len(p_set)) for p_set in pattern_sets]
		return seq_support

	
	def __get_pattern_by_extension(self, working_patterns):
		add_pattern = None
		print ('Case when pattern not found directly')

		init_patterns = [seq for seq in working_patterns if all(seq[0][0] <= s for s in seq[0]) and np.std(seq[0])!=0]
		
		print (init_patterns)

		if not init_patterns:
			return None

		elif len(init_patterns) > 1: 
			_,roots_ex = self.__dtw_clustering(init_patterns, preference='jenks')
			roots = roots_ex
		else:
			roots = [p[0] for p in init_patterns]
		
		print (roots)

		final_patterns = []
		for final_pattern in roots:
			bag_of_patterns = list([seq for seq in working_patterns if seq[0] != final_pattern])
			
			first_element = final_pattern[0]
			look_from_here = len(final_pattern) - list(reversed(final_pattern)).index(first_element)
			max_limit_reached = False
			while(all(first_element < el for el in final_pattern[look_from_here:]) and bag_of_patterns):
				min_len = np.inf
				add_pattern = [] 
				for seq in bag_of_patterns:
					if seq[0][0] == final_pattern[-1] and not seq_contains(final_pattern, seq[0]):
						add_pattern.append(seq)

				if add_pattern:
					if len(add_pattern) > 1:
						clusters, exs = self.__dtw_clustering(add_pattern, preference='jenks')
						print (clusters)
						print (exs)
						ext_ind = max(clusters.keys(), key=lambda x: sum([p[1] for p in clusters[x]]))
						extension = exs[ext_ind]
						print (extension)
						final_pattern.extend(extension[1:])
						print (final_pattern)
						for pattern in clusters[ext_ind]:
							bag_of_patterns.remove(pattern)

					else:
						extension = add_pattern[0]
						print (add_pattern)
						print (extension)
						final_pattern.extend(extension[0][1:])
						print (final_pattern)

				
					last_element = self.state_attributes[str(final_pattern[-1])][0]
				else:
					final_patterns.append(final_pattern)

				if len(final_pattern) >= self.MAX_LEN_EXTENSION:
					max_limit_reached = True
					break

			end_ind = len(final_pattern) - list(reversed(final_pattern)).index(first_element)
			
			if max_limit_reached:
				final_patterns.append(final_pattern)
			else:
				final_patterns.append(final_pattern[:end_ind])

		## Clustering final results to avoid similar patterns
		if len(final_patterns) > 1:
			_, final_patterns_ex = self.__dtw_clustering(final_patterns, 'jenks')
			return final_patterns_ex
		else:
			return final_patterns

	def discover_pattern_by_extension(self):
		try:
			working_patterns, idle_patterns, self.pattern_dict = self.cluster_patterns()
			self.working_patterns = working_patterns
		except DTWClusteringError:
			return None
		return self.__get_pattern_by_extension(working_patterns)


	@timing_wrapper
	def discover_pattern(self):
		try:
			working_patterns, idle_patterns, self.pattern_dict = self.cluster_patterns()
			self.working_patterns = working_patterns
		except DTWClusteringError:
			return None
		
		### Looking for patterns that start and stop in the same state, and have less than 90% similarity
		possible_patterns = []
		for seq in working_patterns:
			if seq[0][0] == seq[0][-1]:
				if all( self.state_attributes[str(s)][0] >= self.state_attributes[str(seq[0][0])][0] for s in seq[0]):
					_, count = np.unique(seq[0], return_counts=True)
					count = count/sum(count)
					if all(c < self.similarity_constraint for c in count):
						possible_patterns.append(seq)
	
		### Clustering with DTW to find patterns. Exemplars -> final patterns 
		if len(possible_patterns) > 1:
			different_patterns, exemplars = self.__dtw_clustering(possible_patterns, preference='jenks')
			
			# print (exemplars)
			print (different_patterns)
			print ('Number of clusters with DTW: ', len(different_patterns))
			
			## Checking if no exemplar is part of any other exemplar
			if len(exemplars) != len(possible_patterns):
				for el in exemplars:
					for p in exemplars:
						if p != el:
							if seq_contains(p,el) or self.__pattern_distance(p,el) == 0:
								del exemplars[exemplars.index(el)]
								break
				print (exemplars)
				return exemplars
			else:
				return self.__get_pattern_by_extension(working_patterns)

		elif (len(possible_patterns) == 1 and len(working_patterns) == 1):
			return possible_patterns[0]
		else:
			## If no such pattern exists, extend patterns that gives likely output
			return self.__get_pattern_by_extension(working_patterns)

	def __dtw_clustering(self, seq_f, preference=None):
		### Clustering sequences using affinity propagation, dtw
		### Computing similarity/affinity matrix using dtw
		p_dist = np.zeros((len(seq_f), len(seq_f)))
		for i in range(len(seq_f)):
			for j in range(i,len(seq_f)):
				if isinstance(seq_f[i], list):
					p_dist[i][j] = self.__pattern_distance(seq_f[i],seq_f[j])
				else:
					p_dist[i][j] = self.__pattern_distance(seq_f[i][0],seq_f[j][0])
				if i != j:
					p_dist[j][i] = p_dist[i][j]
		
		if np.max(p_dist) == 0:
			p_dist_max = 2
		else:
			p_dist_max = np.max(p_dist)

		p_dist = p_dist_max - p_dist
		
		### Affinity Propagation
		if preference == 'percent_max':
			ap = AffinityPropagation(affinity='precomputed',preference=0.9*p_dist_max)
		elif preference == 'jenks':
			ap = AffinityPropagation(affinity='precomputed',preference=self.__get_pref(p_dist.flatten()))
		else:
			ap = AffinityPropagation(affinity='precomputed')
		
		ap.fit(p_dist)
		
		if isinstance(seq_f[0], list):
			cluster_subseqs_exs = [ seq_f[ind] for ind in ap.cluster_centers_indices_]
		else:
			cluster_subseqs_exs = [ seq_f[ind][0] for ind in ap.cluster_centers_indices_]
		
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
		seq_f =  self.generator_patterns
		### Clustering sequences using dtw and affinity propagation
		cluster_subseqs, cluster_subseqs_exs = self.__dtw_clustering(seq_f, preference='jenks')
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
		if np.max(p_dist) == 0:
			p_dist_max = 2
		else:
			p_dist_max = np.max(p_dist)
		
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