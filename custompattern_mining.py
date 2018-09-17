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
	def __init__(self, sequence, state_attributes, min_len, max_len):
		self.sequence = sequence
		self.MAX_LEN = max_len
		self.MIN_LEN = min_len
		if min_len > max_len:
			raise ValueError('Incorrect values for minimum and maximum length of required pattern')
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
		### Looking for patterns that start and stop with all possible min states.
		print ('Mining Required Patterns...')
		for init_ind in range(len(self.sequence)):
			
			if self.sequence[init_ind] in self.min_states:
				print (init_ind)
				p_temp = self.sequence[init_ind:init_ind+self.MAX_LEN]
				print (p_temp)
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

		print ('Total Unique Patterns: ', len(self.pattern_sets))
		if len(self.pattern_sets) == 0:
			raise 
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