#! /usr/bin/env python3 

import matplotlib.pyplot as plt
import peak_detector as pd
from dtw import dtw 
import numpy as np
import jenkspy
import time
import os

def timing_wrapper(func):
	def wrapper(*args,**kwargs):
		
		t0= time.time()
		func_val = func(*args,**kwargs)
		time_taken = time.time() - t0

		print (str(func),' took: ', time_taken)

		return func_val

	return wrapper

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
		self.MIN_LEN = 3
		self.state_attributes = state_attributes
		self.min_states, self.max_states = self.__partition_states()
		self.pattern_sets = []


	def __partition_states(self):
		seq_means = [self.state_attributes[str(s)][0] for s in self.sequence]
		mean_th = jenkspy.jenks_breaks(seq_means, nb_class=2)[1]
		max_states = []
		min_states = []
		for state, attributes in self.state_attributes.items():
			if attributes[0] >= mean_th:
				max_states.append(int(state))
			else:
				min_states.append(int(state))
		return min_states, max_states

	@timing_wrapper
	def find_patterns(self):
		### Looking for pattterns that start and stop with all possible min states.
		print ('Mining Required Patterns...')
		for init_ind in range(len(self.sequence)):
			# print (init_ind)
			if self.sequence[init_ind] in self.min_states:
				p_temp = self.sequence[init_ind:init_ind+self.MAX_LEN]
				
				try:
					end_ind = self.MIN_LEN + p_temp[self.MIN_LEN:].index(p_temp[0])+1
				except ValueError:
					end_ind = len(p_temp)

				p = self.sequence[init_ind : init_ind+end_ind]

				if len(p) < len(p_temp):
					p_set = self.__patternset_list_contains(p)
					if p_set == None:
						p_set = [p]
						self.pattern_sets.append(p_set)
					else:
						p_set.append(p)

		# for p_set in self.pattern_sets:
		# 	print (p_set[0], len(p_set))

		print ('Total Unique Patterns: ', len(self.pattern_sets))
		return self.pattern_sets

	def __patternset_list_contains(self, pattern):
		for p_set in self.pattern_sets:
			if self.__pattern_distance(pattern, p_set[0]) == 0.0:
				return p_set
		return None

	def __pattern_distance(self, head, pattern):
		val_a = np.array([self.state_attributes[str(s)][0] for s in head])
		val_b = np.array([self.state_attributes[str(s)][0] for s in pattern])
		dist, _, _, _ = dtw(val_a.reshape(-1,1), val_b.reshape(-1,1), dist=lambda x,y:np.linalg.norm(x-y))
		return dist

if __name__ == '__main__':

	# device_path = '/media/milan/DATA/Qrera/HiraAutomation/B4E62D388226'
	# day = '2018_06_25'
	device_path = '/media/milan/DATA/Qrera/FWT/5CCF7FD0C7C0'
	day = '2018_07_06'
	file1, file2 = get_required_files(device_path, day)
	power_d, power_f = pd.preprocess_power(file1, file2)
	final_peaks, peak_indices = pd.detect_peaks(power_d, power_f)
	array, state_attributes = pd.peaks_to_discrete_states(final_peaks)

	pm = PatternMining(array,state_attributes)
	print (pm.min_states, 'min')
	print (pm.max_states, 'max')
	pm.find_patterns()