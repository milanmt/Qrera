#! /usr/bin/env python3

from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt 
from dtw import dtw
import numpy as np
import subprocess
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

class SequentialPatternMining:
	def __init__(self, sequence, state_attributes):
		### Mining generative patterns only
		self.MAX_LEN =  15
		self.MIN_LEN = 5
		self.MIN_SUPPORT = 0.2
		self.N_SEGMENTS = 48
		self.sequence = list(sequence) if not isinstance(sequence, list) else sequence
		self.state_attributes = state_attributes
		self.states = [s for s in self.state_attributes.keys()]
		self.db_filename = 'timedb_test.txt'
		self.pattern_filename = 'output.txt'
		self.path_to_spmf = '/media/milan/DATA/Qrera'
		self.similarity_constraint = 0.9  ## No single element can appear more than 100x% of the time
		self.generator_patterns = self.__get_all_freq_seq() 
	
	def __pattern_distance(self,a,b,normalised):
		val_a = np.array([self.state_attributes[str(s)][0] for s in a])
		val_b = np.array([self.state_attributes[str(s)][0] for s in b])
		if normalised==True:
			val_a = val_a/np.max(val_a)
			val_b = val_b/np.max(val_b)
		dist, _, _, _ = dtw(val_a.reshape(-1,1), val_b.reshape(-1,1), dist=lambda x,y:np.linalg.norm(x-y))
		return dist 

	@timing_wrapper
	def __generate_timeseries_db(self):
		len_segment = len(self.sequence)//self.N_SEGMENTS
		with open(self.db_filename, 'w') as f:
			for i in range(self.N_SEGMENTS):
				segment = self.sequence[i*len_segment : i*len_segment+len_segment]
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
		self.__pattern_mining()
		seq_support = []
		with open(self.pattern_filename, 'r') as f:
			for line in f:
				if '-1' in line:
					temp_l = line.split(' -1 ')
					seq = []
					support = 0 
					for s in temp_l:
						if '#SUP' in s:
							support = int(s.split(':')[1].strip())
						else:
							seq.append(int(s))
					if len(seq) >= self.MIN_LEN:
						seq_support.append((seq, support))
		return seq_support

	def __get_max_var_pattern(self, list_seq):
		seq_variances = []
		for seq in list_seq:
			var = np.std([self.state_attributes[str(s)][0] for s in seq])
			seq_variances.append(var)
		sorted_seq_variances = sorted(seq_variances, reverse=True)

		for ind in range(len(list_seq)):
			pattern_ind = seq_variances.index(sorted_seq_variances[ind])
			req_pattern = list_seq[pattern_ind]
			min_var = self.state_attributes[str(req_pattern[0])][0]
			if all( self.state_attributes[str(s)][0] >= min_var for s in req_pattern):
				return req_pattern

		return None


	def __get_most_common_subseq(self, possible_patterns):
		subseq_count = list(np.zeros(len(possible_patterns)))
		for e,seq in enumerate(possible_patterns):
			for seql in possible_patterns:
				if seq_contains(seql, seq):
					subseq_count[e] += 1

		max_ind = subseq_count.index(max(subseq_count))

		if all(c == max(subseq_count) for c in subseq_count):
			return None
		else:
			return possible_patterns[max_ind]
	
	def __get_pattern_by_extension(self, working_patterns):
		add_pattern = None
		print ('Case when pattern not found directly')
		max_var_pattern = self.__get_max_var_pattern(working_patterns)
		print(max_var_pattern)
		if max_var_pattern == None:
			return None

		min_len = np.inf
		add_pattern = []
		for seq in working_patterns:
			if seq[0] == max_var_pattern[-1] and seq[-1] == max_var_pattern[0] and not seq_contains(seq, max_var_pattern):
				add_pattern.append(seq)

		if add_pattern:
			extension = self.__get_most_common_subseq(add_pattern)
			if extension == None:
				print (add_pattern)
				extension = max(add_pattern, key=lambda x: np.std([self.state_attributes[str(s)][0] for s in x]))
				print (extension)
				extension = min(add_pattern, key=lambda x: len(x))
				print (extension)
			
			max_var_pattern.extend(extension[1:])
			final_pattern = max_var_pattern
			print (final_pattern)
			return final_pattern
		else:
			return None

	@timing_wrapper
	def discover_pattern(self):
		working_patterns, idle_patterns, pattern_dict = self.cluster_patterns()
		
		### Looking for patterns that start and stop in the same state
		possible_patterns = []
		for seq in working_patterns:
			if seq[0][0] == seq[0][-1]:
				if all( self.state_attributes[str(s)][0] >= self.state_attributes[str(seq[0][0])][0] for s in seq[0]):
					_, count = np.unique(seq, return_counts=True)
					count = count/sum(count)
					if all(c < self.similarity_constraint for c in count):
						possible_patterns.append(seq[0])
		print ('possible_patterns')
		print (possible_patterns)

		### Finding max variance among patterns that start and end the same state
		if possible_patterns:
			different_patterns, exemplars = self.__dtw_clustering(possible_patterns,normalised=True)
			final_patterns = []
			for patterns in different_patterns.values():
				fp = self.__get_most_common_subseq(patterns)
				if fp != None:
					final_patterns.append(fp)

			if final_patterns:
				print(final_patterns) 
				return final_patterns
			else:
				return self.__get_pattern_by_extension(working_patterns)

		else:
			## If no such pattern exists, extend patterns that gives likely output
			return self.__get_pattern_by_extension(working_patterns)

	def __dtw_clustering(self, seq_f, normalised=False):
		### Clustering sequences using affinity propagation, dtw
		### Computing similarity/affinity matrix using dtw
		p_dist = np.zeros((len(seq_f), len(seq_f)))
		for i in range(len(seq_f)):
			for j in range(i,len(seq_f)):
				p_dist[i][j] = self.__pattern_distance(seq_f[i][0],seq_f[j][0],normalised)
				if i != j:
					p_dist[j][i] = p_dist[i][j]
		p_dist = p_dist/np.max(p_dist)
		p_dist = 1 - p_dist

		### Affinity Propagation
		ap = AffinityPropagation(affinity='precomputed')
		ap.fit(p_dist)
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
		cluster_subseqs, cluster_subseqs_exs = self.__dtw_clustering(seq_f)
		# print (cluster_subseqs)
		print ('Number of clusters with non-normal DTW: ', len(cluster_subseqs))

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
		
		### Affinity Propagation based on means and variances
		ap_mv = AffinityPropagation(affinity='euclidean')
		cl_mv_labels = ap_mv.fit_predict(cluster_mv)
		cl_v_exs = [ cluster_mv[ind][1] for ind in ap_mv.cluster_centers_indices_]
		idle_label = cl_mv_labels[cluster_mv_norm.index(min(cluster_mv_norm))]
		
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