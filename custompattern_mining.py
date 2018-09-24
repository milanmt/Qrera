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
	def __init__(self, sequence, state_attributes, max_len, min_len):
		self.sequence = sequence
		self.MAX_LEN = max_len   ## Max len should be greater than longest possible pattern
		self.MIN_LEN = min_len
		self.state_attributes = state_attributes
		self.min_states, self.max_states = self.__partition_states()
		self.__pattern_sets = dict()
		self.patterns_unique = []

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
		start = self.MIN_LEN
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
				p = tuple(self.sequence[init_ind : init_ind+end_ind])

				if p[0] == p[-1]:
					p_set = self.__patternset_list_contains(p)
					if p_set == None:
						self.__pattern_sets.update({p : {p :1}})
					else:
						if p not in p_set:
							p_set.update({p:1})
						else:
							p_set[p] +=1

		### Finding most frequent pattern
		for p_set in self.__pattern_sets.values():
			total_freq = sum(p_set.values())
			max_item = max([z for z in p_set.items()], key= lambda x:x[1])
			self.patterns_unique.append((list(max_item[0]),total_freq))

		print ('Total Unique Patterns: ', len(self.patterns_unique))
		print (self.patterns_unique)
		
		if len(self.__pattern_sets) == 0:
			raise ValueError('No Patterns Found')
		
		return self.patterns_unique

	def __patternset_list_contains(self, pattern):
		for p_head in self.__pattern_sets:
			if self.__pattern_distance(pattern, p_head) == 0.0:
				return self.__pattern_sets[p_head]
		return None

	def __pattern_distance(self, head, pattern):
		val_a = np.array([self.state_attributes[str(s)][0] for s in head])
		val_b = np.array([self.state_attributes[str(s)][0] for s in pattern])
		dist, _, _, _ = dtw(val_a.reshape(-1,1), val_b.reshape(-1,1), dist=lambda x,y:np.linalg.norm(x-y))
		return dist