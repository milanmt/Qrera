#! /usr/bin/env python3

from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt 
import numpy as np
import subprocess
import pandas 
import math
import os


class SequentialPatternMining:
	def __init__(self, sequence, state_attributes, time_based_algorithm=True):
		self.MAX_LEN = 10
		self.MIN_LEN = 2
		self.MIN_SUPPORT = 0.3
		self.N_SEGMENTS = 24
		self.sequence = list(sequence) if not isinstance(sequence, list) else sequence
		self.state_attributes = state_attributes
		self.states = [s for s in self.state_attributes.keys()]
		self.diff_matrix = self.get_diff_matrix()
		self.db_filename = 'timedb_test.txt'
		self.pattern_filename = 'output.txt'
		self.path_to_spmf = '/media/milan/DATA/Qrera'
		self.time_based_algorithm = time_based_algorithm

	
	def get_diff_matrix(self):
		S = len(self.states)
		diff_matrix = np.zeros((S,S))
		for s in range(S):
			for sn in range(S):
				diff_matrix[s][sn] = abs(self.state_attributes[self.states[s]][0] - 
					self.state_attributes[self.states[sn]][0])

		diff_matrix = S*diff_matrix/np.max(diff_matrix)
		for s in range(S):
			for sn in range(S):
				diff_matrix[s][sn] = round(diff_matrix[s][sn])

		print (diff_matrix)	

	
	def levenshtein_distance(self,a,b):
		lev_m = np.zeros((len(a)+1,len(b)+1))
		a.insert(0,0)
		b.insert(0,0)
		for i in range(len(a)):
			for j in range(len(b)):
				score = self.diff_matrix[self.states.index(a[i])][self.states.index(b[j])]
				if i == 0 or j == 0:
					lev_m[i][j] = 0
				else:
					lev_m[i][j] = score + min(lev_m[i-1][j], lev_m[i][j-1], lev_m[i-1][j-1])
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
		## Closed Patterns
		# subprocess.call('java -jar spmf.jar run Fournier08-Closed+time '+
		# 	+self.db_filename+' '+self.pattern_filename+' '+str(self.MIN_SUPPORT)+' 1 1 '+
		# 	str(self.MIN_LEN)+' '+str(self.MAX_LEN),cwd=self.path_to_spmf,shell=True)	
		## Generative patterns
		subprocess.call('java -jar spmf.jar run VGEN '+self.db_filename+
			' '+self.pattern_filename+' '+str(self.MIN_SUPPORT)+' '+
			str(self.MAX_LEN)+' 1 false',cwd=self.path_to_spmf,shell=True)
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


	def discover_pattern(self):
		seq_support = self.__get_all_freq_seq()
		print (len(seq_support),'-> ALL')
		seq_f = []
		for seq, support in seq_support:
			s_val, s_count = np.unique(seq, return_counts=False)
			s_count = s_count/sum(s_count)
			if all(c < 0.5 for c in s_count):   ## Making sure no value takes 
				seq_f.append(seq)

		print (len(seq_f), '-> Filtered')
		print (seq_f)




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
	return ('Test')
if __name__ == '__main__':
	test_main()



