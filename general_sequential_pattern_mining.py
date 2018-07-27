#! /usr/bin/env python3

from sklearn.cluster import AffinityPropagation
import peak_detector as pd 
import numpy as np
import subprocess
import pandas 
import json

def levenshtein_distance(a,b):
	lev_m = np.zeros((len(a)+1,len(b)+1))
	a.insert(0,0)
	b.insert(0,0)

	for i in range(len(a)):
		for j in range(len(b)):
			if i == 0:
				if a[i] != b[j]:
					lev_m[i][j] = lev_m[i][j-1]+1
				else:
					lev_m[i][j] = lev_m[i][j-1]

			elif j == 0:
				if a[i] != b[j]:
					lev_m[i][j] = lev_m[i-1][j]+1
				else:
					lev_m[i][j] = lev_m[i-1][j]
			else:
				if a[i] != b[j]:
					lev_m[i][j] = min(lev_m[i-1][j]+1, lev_m[i][j-1]+1, lev_m[i-1][j-1]+1)
				else:
					lev_m[i][j] = min(lev_m[i-1][j], lev_m[i][j-1], lev_m[i-1][j-1])


	return lev_m[len(a)-1][len(b)-1]


class SequentialPatternMining:

	def __init__(self, sequence, state_attributes):
		self.MAX_LEN = 10
		self.MIN_SUPPORT = 2
		self.N_SEGMENTS = 24
		self.sequence = list(sequence) if not isinstance(sequence, list) else sequence
		self.state_attributes = state_attributes
		# self.subsequences, self.subsequences_count = self.__generate_subsequences(self.sequence)


	def __generate_subsequences(self, sequence):
		subsequence_list = []
		subsequence_count = []
		for pattern_length in range(2,self.MAX_LEN+1):
			for i in range(len(sequence)-pattern_length+1):
				sub_seq = sequence[i:i+pattern_length]
				if sub_seq not in subsequence_list:
					subsequence_list.append(sequence[i:i+pattern_length])
					subsequence_count.append(1)
				else:
					sub_index = subsequence_list.index(sub_seq)
					subsequence_count[sub_index] += 1

		return subsequence_list, subsequence_count


	def get_max_freq_patterns(self):
		max_count = max(self.subsequences_count)
		the_zip = zip(self.subsequences, self.subsequences_count)
		max_subseq = []

		for subseq, count in the_zip:
			if max_count == count:
				max_subseq.append(subseq)

		print(max_subseq)
		print(max_count)


	def generate_timeseries_db(self, time_based_algorithm=True):

		len_segment = len(self.sequence)//self.N_SEGMENTS

		with open('timedb_test.txt', 'w') as f:
			for i in range(self.N_SEGMENTS):
				segment = self.sequence[i*len_segment : i*len_segment+len_segment]

				if time_based_algorithm == True:
					for j in range(len(segment)):
						f.write('<{0}> {1} -1 '.format(j, segment[j]))
				
				else:
					for s in segment:
						f.write('{0} -1 '.format(s)) 

				f.write('-2\n')


		return f




def get_freq_sequences(state_attributes, time_based_algorithm=True):
	output_file_name = '/media/milan/DATA/Qrera/output.txt'
	seq_support = []
	with open(output_file_name, 'r') as f:
		if time_based_algorithm == True:
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



if __name__ == '__main__':
	# array = [1, 2, 3, 3, 1, 2, 3, 5, 1, 2, 3, 1, 2, 3,5, 5, 5, 5, 1, 2, 2, 1, 2, 2, 3, 5, 1, 2, 3, 5, 5, 1, 1, 2, 2, 3, 3, 5, 1, 2, 3, 5]

	# device_path = '/media/milan/DATA/Qrera/FWT/5CCF7FD0C7C0'
	# day = '2018_07_07'
	# file1, file2 = pd.get_required_files(device_path, day)
	# power_d, power_f = pd.preprocess_power(file1, file2)
	# final_peaks, peak_indices = pd.detect_peaks(power_d, power_f)
	# array, state_attributes = pd.peaks_to_discrete_states(final_peaks)

	# with open('state_attributes.json', 'w') as f:
	# 	json.dump(state_attributes, f)

	# pm = SequentialPatternMining(array, state_attributes)

	# timedb_file = pm.generate_timeseries_db(time_based_algorithm=True)

	# subprocess.call('java -jar spmf.jar run Fournier08-Closed+time trials/timedb_test.txt output.txt 0.3 1 1 2 10',cwd='/media/milan/DATA/Qrera',shell=True)
	### subprocess.call('java -jar spmf.jar run VGEN trials/timedb_test.txt output.txt 0.3 10 1 false',cwd='/media/milan/DATA/Qrera',shell=True)
	

	with open('state_attributes.json', 'r') as f:
		state_attributes = json.load(f)
	pattern = get_freq_sequences(state_attributes, time_based_algorithm=True)
