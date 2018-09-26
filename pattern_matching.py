#! /usr/bin/env python3
from dtw import dtw
import numpy as np
import time

def timing_wrapper(func):
	def wrapper(*args,**kwargs):
		t0= time.time()
		func_val = func(*args,**kwargs)
		time_taken = time.time() - t0
		print (str(func),' took: ', time_taken)
		return func_val
	return wrapper

class PatternMatching:
	def __init__(self, pattern_dict, state_attributes, sequence, min_len,  max_len):
		self.pattern_dict = pattern_dict
		self.sequence = sequence
		self.state_attributes = state_attributes
		self.pattern_sequence = None
		self.pattern_sequence_indices = None
		self.max_len = max_len
		self.min_len = min_len

	@timing_wrapper
	def find_matches(self):
		print ('Matching Discovered Patterns...')
		start_ind = 0
		end_ind = 0
		pattern_sequence = []
		pattern_sequence_indices = []
		while start_ind < len(self.sequence)-1:
			min_pdist = np.inf
			for label, p_set in self.pattern_dict.items():

				p_set_d	= [max(p_set, key=lambda x:x[1])]
				for pattern,freq in p_set_d:
					dists = []
					ends = []
					end_ind_t = start_ind+self.min_len-1
					while end_ind_t < start_ind+self.max_len:
						p_temp = self.sequence[start_ind:end_ind_t+1]
						dist = self.__pattern_distance(p_temp,pattern)
						dists.append(dist)
						ends.append(end_ind_t)
						end_ind_t +=1

					### preferring longer patterns rather than shorter ones 
					min_dist = min(dists)
					for e,d in enumerate(dists):
						if d == min_dist:
							end_ind_f = ends[e]

					if min_pdist > min_dist:
						min_pdist = min_dist
						end_ind = end_ind_f
						req_label = label

			pattern_sequence.append(req_label)
			if end_ind < len(self.sequence):
				pattern_sequence_indices.append(end_ind)
			start_ind = end_ind
		
		self.pattern_sequence = pattern_sequence
		self.pattern_sequence_indices = pattern_sequence_indices
		vals, counts = np.unique(pattern_sequence, return_counts=True) ### Counting unique values
		print (vals)
		print (counts)
		return pattern_sequence, pattern_sequence_indices

	def __pattern_distance(self,a,b):
		val_a = [self.state_attributes[str(s)][0] for s in a]
		val_b = [self.state_attributes[str(s)][0] for s in b]
		dist, _, _, _ = dtw(val_a, val_b, dist=lambda x,y:np.linalg.norm(x-y))
		return dist 