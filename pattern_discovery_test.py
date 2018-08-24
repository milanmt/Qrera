#! /usr/bin/env python3

from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt 
import numpy as np
import subprocess
import pandas 
import json
import math
import os


class SequentialPatternMining:
	def __init__(self, sequence, state_attributes, time_based_algorithm=False):
		### After testing, patterns discovered by generative patterns is more suitable to our needs.
		### So keep time_based_algorithm as False. If true, closed patterns will be detected. 
		### The current filtering method is not guaranteed to work on closed patterns with stability.
		self.MAX_LEN = 10
		self.MIN_LEN = 2
		self.MIN_SUPPORT = 0.3
		self.N_SEGMENTS = 48
		self.sequence = list(sequence) if not isinstance(sequence, list) else sequence
		self.state_attributes = state_attributes
		self.states = [s for s in self.state_attributes.keys()]
		self.diff_matrix = self.get_diff_matrix()
		self.db_filename = 'timedb_test.txt'
		self.pattern_filename = 'output.txt'
		self.path_to_spmf = '/media/milan/DATA/Qrera'
		self.time_based_algorithm = time_based_algorithm
		self.similarity_constraint = 1  ## No single element can appear more than 100x% of the time.

	
	def get_diff_matrix(self):
		S = len(self.states)
		diff_matrix = np.zeros((S,S))
		for s in range(S):
			for sn in range(s,S):
				diff_matrix[s][sn] = abs(self.state_attributes[self.states[s]][0] - 
					self.state_attributes[self.states[sn]][0])
				if sn != s:
					diff_matrix[sn][s] = diff_matrix[s][sn]

		diff_matrix = (S*diff_matrix)//np.max(diff_matrix)+1
		for s in range(S):
			diff_matrix[s][s] = 0
		print (diff_matrix)	
		return diff_matrix

	
	def levenshtein_distance(self,a,b):
		lev_m = np.zeros((len(a)+1,len(b)+1))
		a.insert(0,0)
		b.insert(0,0)
		max_score = np.max(self.diff_matrix)+1
		for i in range(len(a)):
			for j in range(len(b)):
				### forming matrix
				if i == 0 and j == 0:
					lev_m[i][j] = 0
				elif i == 0:
					lev_m[i][j] = lev_m[i][j-1]+max_score
				elif j == 0:
					lev_m[i][j] = lev_m[i-1][j]+max_score
				else:
					### getting scores
					score = self.diff_matrix[self.states.index(str(a[i]))][self.states.index(str(b[j]))]
					if any(a_l in b for a_l in a):
						score_ins_a = score//2
					else:
						score_ins_a = max_score
					if any(b_l in a for b_l in b):
						score_ins_b = score//2
					else:
						score_ins_b = max_score
					lev_m[i][j] = min(lev_m[i-1][j]+score_ins_a, lev_m[i][j-1]+score_ins_b, lev_m[i-1][j-1]+score)
		return lev_m[len(a)-1][len(b)-1]


	def __generate_timeseries_db(self):
		len_segment = len(self.sequence)//self.N_SEGMENTS
		with open(self.db_filename, 'w') as f:
			for i in range(self.N_SEGMENTS):
				segment = self.sequence[i*len_segment : i*len_segment+len_segment]
				if self.time_based_algorithm:
					for j in range(len(segment)):
						f.write('<{0}> {1} -1 '.format(j, segment[j]))		
				else:
					for s in segment:
						f.write('{0} -1 '.format(s)) 
				f.write('-2\n')

		self.db_filename = os.path.realpath(self.db_filename)


	def __pattern_mining(self):
		self.__generate_timeseries_db()	
		if self.time_based_algorithm == True:
		## Mining Closed Patterns
			subprocess.call('java -jar spmf.jar run Fournier08-Closed+time '+self.db_filename
			+' '+self.pattern_filename+' '+str(self.MIN_SUPPORT)+' 1 1 '+
			str(self.MIN_LEN)+' '+str(self.MAX_LEN),cwd=self.path_to_spmf,shell=True)

		else:
		## Mining Generative Patterns
			subprocess.call(('java -jar spmf.jar run VGEN '+self.db_filename+
			' '+self.pattern_filename+' '+str(self.MIN_SUPPORT)+' '+
			str(self.MAX_LEN)+' 1 false'),cwd=self.path_to_spmf,shell=True)
		
		self.pattern_filename = os.path.join(self.path_to_spmf, self.pattern_filename)


	def __get_all_freq_seq(self):
		self.__pattern_mining()
		seq_support = []
		with open(self.pattern_filename, 'r') as f:
			if self.time_based_algorithm:
				for line in f:
					temp_l = line.split(' -1 ')
					seq = []
					support = 0 
					for s in temp_l:
						if '<' in s and '>' in s:
							seq.append(int(s.split(' ')[1]))
						elif '#SUP' in s:
							support = int(s.split(':')[1].strip())
					seq_support.append((seq, support))
			else:
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
						seq_support.append((seq, support))
		return seq_support


	def __get_max_var_ind(self, list_seq):
		cluster_variances = []
		for seq in list_seq:
			var = np.std([self.state_attributes[str(s)][0] for s in seq])
			cluster_variances.append(var)
		max_var_ind = cluster_variances.index(max(cluster_variances))
		return max_var_ind


	def __get_maximal_patterns(self, seq_support):
		maximal_patterns = []
		for seq in seq_support:
			for seql in seq_support:
				if seq_contains(seql[0], seq[0]) and seql[0] != seq[0]:
					if seql not in maximal_patterns:
						maximal_patterns.append(seql)
					if seq in maximal_patterns:
						ind = maximal_patterns.index(seq)
						del maximal_patterns[ind]
		return maximal_patterns

	
	def discover_pattern(self):
		seq_support = self.__get_all_freq_seq()
		seq_support_m = self.__get_maximal_patterns(seq_support)

		possible_patterns = []
		for seq,support in seq_support_m:
			if seq[0] == seq[-1]:
				possible_patterns.append(seq)

		### Finding max variance among patterns that start and end the same state
		if possible_patterns:
			max_var_ind = self.__get_max_var_ind(possible_patterns) 
			final_pattern = possible_patterns[max_var_ind] 
		else:
			## If no such pattern exists, extend patterns that gives likely output
			print ('Case when pattern not found directly')
			max_var_ind = self.__get_max_var_ind([seq[0] for seq in seq_support_m])
			max_var_pattern = seq_support_m[max_var_ind][0] 
			min_len = np.inf
			for seq in seq_support_m:
				if seq[0][0] == max_var_pattern[-1] and seq[0][-1] == max_var_pattern[0]:
					if len(seq[0]) < min_len:
							min_len = len(seq[0])
							add_pattern = seq[0]
			max_var_pattern.extend(add_pattern[1:])
			final_pattern = max_var_pattern
		print (final_pattern)

		def cluster_patterns(self):
		seq_support = self.__get_all_freq_seq()
		seq_f =  self.__get_maximal_patterns(seq_support)	

		### Clustering maximal sequences using affinity propagation
		p_dist = np.zeros((len(seq_f), len(seq_f)))
		for i in range(len(seq_f)):
			for j in range(i,len(seq_f)):
				a = list(seq_f[i][0])
				b = list(seq_f[j][0])
				p_dist[i][j] = self.levenshtein_distance(a,b)
				if i != j:
					p_dist[j][i] = p_dist[i][j]
		p_dist = p_dist/np.max(p_dist)
		p_dist = 1 - p_dist

		ap = AffinityPropagation(affinity='precomputed')
		ap.fit(p_dist)
		cluster_subseqs_exs = [ seq_f[ind][0] for ind in ap.cluster_centers_indices_]
		subseq_labels = ap.labels_
	
		print(cluster_subseqs_exs)

		### Arranging sequences by cluster label -> mostly for future use
		cluster_subseqs = dict()
		for seq, label in zip(seq_f,subseq_labels):
			if label not in cluster_subseqs:
				cluster_subseqs.update({label : [seq]})
			else:
				seq_list = cluster_subseqs[label]
				seq_list.append(seq)
				cluster_subseqs.update({ label: seq_list})

		print (cluster_subseqs)

		return cluster_subseqs

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