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
	def __init__(self, sequence, state_attributes):
		### Mining generative patterns only
		self.MAX_LEN = 10
		self.MIN_LEN = 3
		self.MIN_SUPPORT = 0.33
		self.N_SEGMENTS = 48
		self.sequence = list(sequence) if not isinstance(sequence, list) else sequence
		self.state_attributes = state_attributes
		self.states = [s for s in self.state_attributes.keys()]
		self.diff_matrix = self.get_diff_matrix()
		self.db_filename = 'timedb_test.txt'
		self.pattern_filename = 'output.txt'
		self.path_to_spmf = '/media/milan/DATA/Qrera'
		self.similarity_constraint = 1  ## No single element can appear more than 100x% of the time
		self.generator_patterns = self.__get_all_freq_seq() 
			
	
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
		# print (diff_matrix)	
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


	def __get_max_var_ind(self, list_seq):
		cluster_variances = []
		for seq in list_seq:
			var = np.std([self.state_attributes[str(s)][0] for s in seq])
			cluster_variances.append(var)
		max_var_ind = cluster_variances.index(max(cluster_variances))
		return max_var_ind


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
			return possible_patterns[max_in]

	
	def __get_pattern_by_extension(self, working_patterns):
		add_pattern = None
		print ('Case when pattern not found directly')
		max_var_ind = self.__get_max_var_ind([seq for seq in working_patterns])
		max_var_pattern = working_patterns[max_var_ind]
		print(max_var_pattern)
		min_len = np.inf
		for seq in working_patterns:
			if seq[0] == max_var_pattern[-1] and seq[-1] == max_var_pattern[0]:
				if len(seq) < min_len:
						min_len = len(seq)
						add_pattern = seq

		if add_pattern != None:
			max_var_pattern.extend(add_pattern[1:])
			final_pattern = max_var_pattern
			print (final_pattern)
			return final_pattern
		else:
			return None

	def discover_pattern(self):
		working_patterns, idle_patterns, pattern_dict = self.cluster_patterns()
		
		### Looking for patterns that start and stop in the same state
		possible_patterns = []
		for seq in working_patterns:
			_, count = np.unique(seq, return_counts=True)
			count = count/sum(count)
			if seq[0] == seq[-1] and all(c < self.similarity_constraint for c in count):
				possible_patterns.append(seq)
		print ('possible_patterns')
		print (possible_patterns)

		### Finding max variance among patterns that start and end the same state
		if possible_patterns:
			final_pattern = self.__get_most_common_subseq(possible_patterns) 
			if final_pattern != None:
				print(final_pattern) 
				return final_pattern
			else:
				return self.__get_pattern_by_extension(working_patterns)

		else:
			## If no such pattern exists, extend patterns that gives likely output
			return self.__get_pattern_by_extension(working_patterns)
		

	def cluster_patterns(self):
		seq_f =  self.generator_patterns
		
		### Getting means and variances of the patterns
		pattern_mv = list(np.zeros((len(seq_f),2)))
		pattern_variances = []
		for e, seq in enumerate(seq_f):
			var = np.std([self.state_attributes[str(s)][0] for s in seq[0]])
			mean_p = np.mean([self.state_attributes[str(s)][0] for s in seq[0]])
			pattern_mv[e][1] = var
			pattern_variances.append(var)
			pattern_mv[e][0] = mean_p
	
		### Clustering based on variance and means
		ap = AffinityPropagation(affinity='euclidean')
		cl_labels = ap.fit_predict(pattern_mv)
		cl_exs = [ pattern_mv[ind] for ind in ap.cluster_centers_indices_]
		cl_v_exs = [pattern_mv[ind][1] for ind in ap.cluster_centers_indices_]
		idle_label = cl_labels[pattern_variances.index(min(cl_v_exs))]

		### Grouping sequences by cluster label -> later inference 
		cluster_seqs = dict()
		for seq, label in zip(seq_f,cl_labels):
			if label not in cluster_seqs:
				cluster_seqs.update({label : [seq[0]]})
			else:
				seq_list = cluster_seqs[label]
				seq_list.append(seq[0])
				cluster_seqs.update({ label: seq_list})
		print (cluster_seqs)

		### Classification based on variance of patterns, min_var -> idle, rest-> working
		working_patterns = []
		idle_patterns = []
		for e,l in enumerate(cl_labels):
			if l == idle_label:
				idle_patterns.append(seq_f[e][0])
			else:
				working_patterns.append(seq_f[e][0])

		print ('Working Patterns')
		print (working_patterns)
		print ('Idle Patterns')
		print (idle_patterns)
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