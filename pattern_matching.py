#! /usr/bin/env python3

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly
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
	def __init__(self, pattern_dict, state_attributes, sequence, min_len,  max_len, peak_indices):
		self.pattern_dict = pattern_dict
		self.sequence = sequence
		self.state_attributes = state_attributes
		self.pattern_sequence = None
		self.pattern_sequence_indices = None
		self.max_len = max_len
		self.min_len = min_len
		self.peak_indices = peak_indices

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

					# ### preferring longer patterns rather than shorter ones 
					# min_dist = min(dists)
					# for e,d in enumerate(dists):
					# 	if d == min_dist:
					# 		end_ind_f = ends[e]

					### preferring shorter patterns over long ones 
					min_dist = min(dists)
					end_ind_f = ends[dists.index(min_dist)]

					if min_pdist > min_dist:
						min_pdist = min_dist
						end_ind = end_ind_f
						req_label = label

			# if np.std(self.sequence[start_ind:end_ind+1]) != 0:
			# 	print (self.sequence[start_ind:end_ind+1], end_ind)
			pattern_sequence.append(req_label)
			if end_ind < len(self.sequence):
				pattern_sequence_indices.append(end_ind)
			start_ind = end_ind
		
		self.pattern_sequence = pattern_sequence
		self.pattern_sequence_indices = pattern_sequence_indices
		vals, counts = np.unique(pattern_sequence, return_counts=True) ### Counting unique values
		print (vals)
		print (counts)

		# print ('Mapping time indices...')
		# simplified_seq = np.zeros((len(self.sequence)))
		# start_ind = 0
		# for e,i in enumerate(pattern_sequence_indices):
		# 	simplified_seq[start_ind:i+1] = pattern_sequence[e]
		# 	start_ind = i
	
		# print ('Plotting...')
		# unique_labels = list(np.unique(simplified_seq))
		# y_plot = np.zeros((len(unique_labels),len(simplified_seq)))
		# for e,el in enumerate(simplified_seq):
		# 	y_plot[unique_labels.index(int(el)),e] = self.sequence[e]
		# 	time = np.arange(len(self.sequence))
	
		# plotly.tools.set_credentials_file(username='MilanMariyaTomy', api_key= '8HntwF4rtsUwPvjW3Sl4')
		# data = [go.Scattergl(x=time, y=y_plot[i,:]) for i in range(len(unique_labels))]
		# pattern_edges = len(time)*[None]
		# for ind in pattern_sequence_indices:
		# 	pattern_edges[ind] = self.sequence[ind]
		# data.append(go.Scattergl(x=time,y=pattern_edges,mode='markers'))
		# fig = go.Figure(data = data)
		# plotly.plotly.plot(fig, filename='fwtc_pattern_counting')
		return pattern_sequence, pattern_sequence_indices

	def __pattern_distance(self,a,b):
		val_a = [self.state_attributes[str(s)][0] for s in a]
		val_b = [self.state_attributes[str(s)][0] for s in b]
		dist, _, _, _ = dtw(val_a, val_b, dist=lambda x,y:np.linalg.norm(x-y))
		return dist 