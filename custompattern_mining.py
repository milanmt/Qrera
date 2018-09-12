#! /usr/bin/env python3 

import matplotlib.pyplot as plt
import peak_detector as pd
from dtw import dtw 
import numpy as np
import jenkspy
import os

def get_required_files(device_path, day):
	print ('Obtaining Required Files...')
	file1 = None
	file2 = None
	end_search = False
	for root, dirs, files in os.walk(device_path):
		if files and not end_search:
			files.sort()
			for f in files:
				if day in f and f.endswith('.csv.gz'):
					file1 = os.path.join(root,f)
				if file1 and os.path.join(root,f) > file1:
					file2 =  os.path.join(root,f)
					end_search = True
					break
	return file1, file2


class PatternMining:

	def __init__(self, sequence, state_attributes):
		self.sequence = sequence
		self.MAX_LEN = 10
		self.MIN_LEN = 4
		self.state_attributes = state_attributes
		self.min_states, self.max_states = self.__partition_states()
		self.pattern_dict = dict()


	def __partition_states(self):
		seq_means = [self.state_attributes[str(s)][0] for s in self.sequence]
		plt.plot(seq_means)
		plt.show()
		mean_th = jenkspy.jenks_breaks(seq_means, nb_class=2)[1]
		max_states = []
		min_states = []
		for state, attributes in state_attributes.items():
			if attributes[0] >= mean_th:
				max_states.append(int(state))
			else:
				min_states.append(int(state))
		return min_states, max_states

	def find_patterns(self):
		### Looking for pattterns that start and stop with all possible min states.
		for init_ind in range(len(self.sequence)):
			print (init_ind)
			if self.sequence[init_ind] in self.min_states:
				p_temp = self.sequence[init_ind:init_ind+self.MAX_LEN]
				
				# end_ind = len(p_temp) - list(reversed(p_temp)).index(p_temp[0])
				# if end_ind == 1:
				# 	end_ind = len(p_temp)
				
				try:
					end_ind = self.MIN_LEN + p_temp[self.MIN_LEN:].index(p_temp[0])
				except ValueError:
					end_ind = len(p_temp)

				p = self.sequence[init_ind : init_ind+end_ind]

				p_set = self.__pattern_dict_contains(p)
				if p_set == None:
					p_set = PatternSet(p, self.state_attributes)
					self.pattern_dict.update({p_set:p_set.head})
				else:
					p_set.pattern_list.append(p)

		for p_set, p in self.pattern_dict.items():
			print (p, len(p_set.pattern_list))

		print ('Total Unique Patterns: ', len(self.pattern_dict))

	def __pattern_dict_contains(self, pattern):
		for p_set in self.pattern_dict.keys():
			if p_set.set_contains(pattern):
				return p_set
		return None

class PatternSet:
	def __init__(self,first_pattern, state_attributes):
		self.head = first_pattern
		self.pattern_list = [first_pattern]
		self.state_attributes = state_attributes

	def __pattern_distance(self, head, pattern):
		val_a = np.array([self.state_attributes[str(s)][0] for s in head])
		val_b = np.array([self.state_attributes[str(s)][0] for s in pattern])
		dist, _, _, _ = dtw(val_a.reshape(-1,1), val_b.reshape(-1,1), dist=lambda x,y:np.linalg.norm(x-y))
		return dist

	def set_contains(self,pattern):
		if self.__pattern_distance(pattern, self.head) == 0.0:
			return True
		else:
			return False






if __name__ == '__main__':

	# device_path = '/media/milan/DATA/Qrera/HiraAutomation/B4E62D388226'
	# day = '2018_06_25'
	device_path = '/media/milan/DATA/Qrera/FWT/5CCF7FD0C7C0'
	day = '2018_07_07'
	file1, file2 = get_required_files(device_path, day)
	power_d, power_f = pd.preprocess_power(file1, file2)
	final_peaks, peak_indices = pd.detect_peaks(power_d, power_f)
	array, state_attributes = pd.peaks_to_discrete_states(final_peaks)

	pm = PatternMining(array,state_attributes)
	print (pm.min_states, 'min')
	print (pm.max_states, 'max')
	pm.find_patterns()
	