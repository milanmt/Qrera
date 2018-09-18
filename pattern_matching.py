#! /usr/bin/env python3


from dtw import dtw
import numpy as np



def timing_wrapper(func):
	def wrapper(*args,**kwargs):
		t0= time.time()
		func_val = func(*args,**kwargs)
		time_taken = time.time() - t0
		print (str(func),' took: ', time_taken)
		return func_val
	return wrapper

class PatternMatching:
	def __init__(self, pattern_dict, state_attributes, sequence, max_len):
		self.pattern_dict = pattern_dict
		self.sequence = sequence
		self.state_attributes = state_attributes
		self.pattern_sequence = None
		self.max_len = max_len

	@timing_wrapper
	def find_matches(self):
		start_ind = 0
		end_ind = 0
		pattern_sequence = []
		while start_ind < len(self.sequence)-1:
			min_pdist = np.inf
			for label, p_set in self.pattern_dict.items():
				for pattern,freq in p_set:
					dists = []
					ends = []
					end_ind_t = start_ind+len(pattern)
					while end_ind_t <= start_ind+self.max_len:
						p_temp = self.sequence[start_ind:end_ind_t]
						dist = self.__pattern_distance(p_temp,pattern)
						dists.append(dist)
						ends.append(end_ind_t)
						end_ind_t +=1

					min_dist = min(dists)
					max_end = -1
					for e,dist in enumerate(dists):
						if dist == min_dist and ends[e] > max_end:
							end_ind_f = ends[e]

					if min_pdist > min_dist:
						min_pdist = min_dist
						end_ind = end_ind_f
						req_label = label

			pattern_sequence.append(req_label)
			start_ind = end_ind - 1

		self.pattern_sequence = pattern_sequence
		### Counting unique values
		vals, counts = np.unique(pattern_sequence, return_counts=True)
		print (vals)
		print (counts)
		return pattern_sequence

	def __pattern_distance(self,a,b):
		val_a = np.array([self.state_attributes[str(s)][0] for s in a])
		val_b = np.array([self.state_attributes[str(s)][0] for s in b])
		dist, _, _, _ = dtw(val_a.reshape(-1,1), val_b.reshape(-1,1), dist=lambda x,y:np.linalg.norm(x-y))
		return dist 