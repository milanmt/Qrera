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

class PatternMining:
	def __init__(self, sequence, state_attributes, max_len):
		self.sequence = sequence
		self.MAX_LEN = max_len   ## Max len should be greater than longest possible pattern
		self.state_attributes = state_attributes
		self.min_states, self.max_states = self.__partition_states()
		self.pattern_sets = []

	def __partition_states(self):
		seq_means = [s[0] for s in self.state_attributes.values()]
		print (seq_means)
		mean_th = jenkspy.jenks_breaks(seq_means, nb_class=2)[1]
		max_states = []
		min_states = []
		for state, attributes in self.state_attributes.items():
			if attributes[0] > mean_th:
				max_states.append(int(state))
			else:
				min_states.append(int(state))
		print (max_states)
		print (min_states)
		return min_states, max_states

	
	def __get_end_ind(self, p_temp):
		start = 0
		for i in range(1,len(p_temp)):
			if p_temp[0] != p_temp[i]:
				start = i
				break

		if start == 0:
			return len(p_temp)
		else:
			try:
				end = start + p_temp[start:].index(p_temp[0]) +1
			except ValueError:
				if start >= 3:
					end = start
				else:
					end = len(p_temp)
			return end

	@timing_wrapper
	def find_patterns(self):
		### Looking for patterns that start and stop with all possible min states.
		print ('Mining Required Patterns...')
		for init_ind in range(len(self.sequence)-2):
			# print (init_ind)
			if self.sequence[init_ind] in self.min_states:
				p_temp = self.sequence[init_ind:init_ind+self.MAX_LEN]
				end_ind = self.__get_end_ind(p_temp)
				p = self.sequence[init_ind : init_ind+end_ind]

				if len(p) == 3 :
					print (p)
				if p[0] == p[-1]:
					p_set = self.__patternset_list_contains(p)
					if p_set == None:
						p_set = [p]
						self.pattern_sets.append(p_set)
					else:
						p_set.append(p)

		print ('Total Unique Patterns: ', len(self.pattern_sets))
		
		if len(self.pattern_sets) == 0:
			raise ValueError('No Patterns Found')
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