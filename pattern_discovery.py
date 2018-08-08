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
	def __init__(self, sequence, state_attributes, time_based_algorithm=True):
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
					if a[i] in b:
						score_ins_a = 1 
					else:
						score_ins_a = score+1
					if b[j] in a:
						score_ins_b = 1
					else:
						score_ins_b = score+1
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
						# if len(seq) > 1:
						seq_support.append((seq, support))
		return seq_support


	def discover_pattern(self):
		seq_support = self.__get_all_freq_seq()
		print (len(seq_support),'-> ALL')
		print (seq_support)

		### Filtering off patterns according to req criterion (amount of similarity tolerated)		
		seq_f = []
		for seq, support in seq_support:
			s_val, s_count = np.unique(seq, return_counts=True)
			s_count = s_count/sum(s_count)
			if all(c < self.similarity_constraint for c in s_count):   
				seq_f.append((seq,support))

		print (len(seq_f), '-> Filtered')
		print (seq_f)

		### Clustering filtered sequences using affinity propagation
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

		### Finding max variance among clusters to get initial pattern
		cluster_variances = []
		for seq in cluster_subseqs_exs:
			var = np.std([self.state_attributes[str(s)][0] for s in seq])
			cluster_variances.append(var)

		print (cluster_variances)
		
		max_var_ind = cluster_variances.index(max(cluster_variances))

		### Finding pattern with maximum support in the highest variance cluster
		max_support = 0
		max_pattern = None
		for subseq in cluster_subseqs[max_var_ind]:
			print (subseq)
			if subseq[1] > max_support:
				max_support = subseq[1]
				max_pattern = subseq[0]

		print (max_pattern)

		### Finding the patterns that include the generator pattern
		probable_list = []
		for seq_list in cluster_subseqs.values():
			for seq in seq_list:
				if seq_contains(seq[0], max_pattern) and seq[0] != max_pattern:
					probable_list.append(seq[0])

		print (probable_list)

		### Selecting pattern of the maxlength of the probable list
		max_len = 0
		for seq in probable_list:
			if max_len <= len(seq):
				max_len = len(seq)

		req_pattern = []
		for seq in probable_list:
			if max_len == len(seq):
				req_pattern.append(seq)

		print (req_pattern)

		### If more than one pattern is selected from above, 
		selected_pattern = None
		if len(req_pattern) > 1:
			var_pattern = []
			for p in req_pattern:
				var = np.std([self.state_attributes[str(s)][0] for s in p])
				var_pattern.append(var)

			max_ind = var_pattern.index(max(var_pattern))
			selected_pattern = req_pattern[max_ind]
		else:
			selected_pattern = req_pattern[0]

		print (selected_pattern)

		final_pattern = None
		min_len = np.inf
		if selected_pattern[0] != selected_pattern[-1]:
			for seq in seq_f:
				if seq[0][0] == selected_pattern[-1] and seq[0][-1] == selected_pattern[0]:
					if len(seq[0]) < min_len:
						min_len = len(seq[0])
						add_pattern = seq[0]
			selected_pattern.extend(add_pattern[1:])
			final_pattern = selected_pattern
		else:
			final_pattern = selected_pattern

		print (final_pattern)

		return final_pattern

###### Add extending pattern for case where generator pattern cannot fit into any other pattern.




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



def get_freq_sequences(state_attributes, time_based_algorithm=True):
	
	seq_support_f = []
	for seq, support in seq_support:
		if len(seq) > 1:
			if np.std(seq) != 0 and seq[0] == seq[-1]:
				seq_support_f.append((seq, support))

	max_subseq = []
	max_count = max(x[1] for x in seq_support_f)

	for subseq, count in seq_support_f:
		if max_count == count:
			max_subseq.append(subseq)


	print (max_subseq)

	if len(max_subseq) < 2:
		max_count2 = max(x[1] for x in seq_support_f if x[1] != max_count)
		for subseq, count in seq_support_f:
			if max_count2 == count:
				max_subseq.append(subseq)

	print (max_subseq)

	p_dist = np.zeros((len(max_subseq), len(max_subseq)))
	for i in range(len(max_subseq)):
		for j in range(len(max_subseq)):
			a = list(max_subseq[i])
			b = list(max_subseq[j])
			p_dist[i][j] = levenshtein_distance(a,b)
	p_dist = p_dist/np.max(p_dist)
	p_dist = 1 - p_dist

	ap = AffinityPropagation(affinity='precomputed')
	ap.fit(p_dist)
	final_subseqs = [ max_subseq[ind] for ind in ap.cluster_centers_indices_]
	
	print(final_subseqs)

	final_variances = []
	for seq in final_subseqs:
		var = 0
		for s in seq:
			var = var + state_attributes[str(s)][1]
		final_variances.append(var)

	print (final_variances)

	max_var_ind = final_variances.index(max(final_variances))
	selected_pattern = final_subseqs[max_var_ind]

	print (selected_pattern)

	return selected_pattern



def test_main():

	with open('state_attributes.json', 'r') as f:
		state_attributes = json.load(f)

	spm = SequentialPatternMining([1,2], state_attributes)
	spm.discover_pattern()


if __name__ == '__main__':
	test_main()



