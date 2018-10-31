#! /usr/bin/env python3

from sklearn.cluster import AffinityPropagation, KMeans
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.mixture import BayesianGaussianMixture
from dtw import dtw
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


class SinglePatternError(Exception):
	pass

class PatternLength:
	def __init__(self, raw_dataframe, min_len, max_len, order, n_states=10):
		##### Parameters to be set 
		self.min_len = min_len
		self.max_len = max_len
		if self.max_len <= self.min_len:
			raise ValueError('Incorrect values for length of pattern')
		self.order = order
		self.n_states = n_states
		##### Need not be initialised. Initialised as code runs
		final_peaks, self.__peak_indices = self.__preprocess_power(raw_dataframe)
		self.__off_regions = [e for e,p in enumerate(self.power) if p == 0]
		self.__pattern_dict = None
		no_iter = 1
		while self.__pattern_dict == None:
			self.__sequence, self.__state_attributes = self.__discretise_power(final_peaks)
			self.__pattern_dict = self.__discover_patterns()
			no_iter += 1
			if no_iter >= 5:
				raise ValueError('Could not find segments for signal. Try again! Or-> Check if min_length of pattern is too small. Check if number of segments are  suitable for data.')	

	def __partition_states(self):
		seq_means = np.array([self.__state_attributes[str(s)][0] for s in self.__sequence]).reshape(-1,1)
		kmeans = KMeans(2).fit(seq_means)
		cl_centers = [cl[0] for cl in kmeans.cluster_centers_] 
		if cl_centers[0] > cl_centers[1]:
			max_id = 0
		else:
			max_id = 1
		max_states = []
		min_states = []
		for state, attributes in self.__state_attributes.items():
			predicted_output = kmeans.predict([[attributes[0]]])
			if int(predicted_output[0]) == max_id:
				max_states.append(int(state))
			else:
				min_states.append(int(state))
		return min_states, max_states

	def __mine_patterns(self):
		### Looking for patterns that start and stop with all possible min states.
		pattern_sets = dict()
		patterns_unique = []
		min_states, max_states = self.__partition_states()
		for init_ind in range(len(self.__sequence)-2):
			if self.__sequence[init_ind] in min_states:
				p_temp = self.__sequence[init_ind:init_ind+self.max_len]
				try:
					end_ind = self.min_len-1+p_temp[self.min_len-1:].index(p_temp[0])+1
				except ValueError:
					end_ind = len(p_temp)
				
				if end_ind < len(p_temp):
					p = tuple(self.__sequence[init_ind : init_ind+end_ind])
					for p_head in pattern_sets:
						if self.__pattern_distance(p, p_head) == 0.0:
							p_set = pattern_sets[p_head]
							if p not in p_set:
								p_set.update({p:1})
							else:
								p_set[p] +=1
							break
					pattern_sets.update({p : {p :1}})

		### Finding most frequent pattern
		for p_set in pattern_sets.values():
			total_freq = sum(p_set.values())
			max_item = max([z for z in p_set.items()], key= lambda x:x[1])
			if total_freq > 1:
				patterns_unique.append((list(max_item[0]),total_freq))

		if len(pattern_sets) == 0:
			raise ValueError('No Patterns Found')
		if len(patterns_unique) == 1:
			raise SinglePatternError('Only one pattern found. Check raw signal. Signal too small for min and max length of patterns.')

		return patterns_unique
		
	def __discover_patterns(self):
		print ('Discovering Patterns...')
		patterns = self.__mine_patterns()	
		### Looking for signals which start and stop with minimas. Need to discover most likely candidate. 
		possible_patterns = []
		for seq in patterns:
			if all(s >= seq[0][0] for s in seq[0]):
				possible_patterns.append(seq)
		### Clustering with DTW to find patterns. Exemplars from DTW -> final patterns 
		### These clustered based on mean and variance to identify idle and working patterns
		if len(possible_patterns) > 1:
			pattern_dict = self.__cluster_patterns(possible_patterns)
		elif len(possible_patterns) == 1:
			pattern_dict = {0: [patterns[0]]}
		else:
			pattern_dict = None
		return pattern_dict

	def __dtw_clustering(self, seq_f):
		### Clustering sequences using affinity propagation, dtw
		### Computing similarity/affinity matrix using dtw
		p_dist = np.zeros((len(seq_f), len(seq_f)))
		if isinstance(seq_f[0],tuple):
			seq = [item[0] for item in seq_f]
			freq = np.array([item[1] for item in seq_f])
		else:
			seq = seq_f

		for i in range(len(seq)):
			for j in range(i,len(seq)):
				p_dist[i][j] = self.__pattern_distance(seq[i],seq[j])
				if i != j:
					p_dist[j][i] = p_dist[i][j]

		p_dist_max = np.max(p_dist)
		if p_dist_max == 0:
			p_dist_max = 2
		p_dist = p_dist_max - p_dist
		
		### Affinity Propagation
		freq = 2*p_dist_max*freq/max(freq)
		ap = AffinityPropagation(affinity='precomputed', preference=freq)
		ap.fit(p_dist)
		
		### Arranging sequences by cluster label 
		cluster_subseqs = dict()
		for seq, label in zip(seq_f,ap.labels_):
			if label not in cluster_subseqs:
				cluster_subseqs.update({label : [seq]})
			else:
				cluster_subseqs[label].append(seq)
				
		return cluster_subseqs
	
	def __cluster_patterns(self, seq_f):
		### Clustering sequences using dtw and affinity propagation
		cluster_subseqs = self.__dtw_clustering(seq_f)
		if len(cluster_subseqs) == 1:
			return cluster_subseqs				
		### Getting average variances and means of exemplars for classification
		cluster_mv = np.zeros((len(cluster_subseqs),2))
		for label, seq_l in cluster_subseqs.items():
			var_seq_l = []
			avg_seq_l = []
			for seq_supp in seq_l:
				seq = seq_supp[0]
				var = np.std([self.__state_attributes[str(s)][0] for s in seq])
				avg = np.mean([self.__state_attributes[str(s)][0] for s in seq])
				var_seq_l.append(var)
				avg_seq_l.append(avg)
			cluster_mv[label][1] = np.mean(var_seq_l)
			cluster_mv[label][0] = np.mean(avg_seq_l)

		### KMeans 
		kmeans = KMeans(2, random_state=3).fit(cluster_mv)
		self.__pattern_predictor = kmeans
		cl_mv_labels = kmeans.labels_
		cluster_norms = [np.linalg.norm(el) for el in kmeans.cluster_centers_]
		idle_label = cluster_norms.index(min(cluster_norms))
		self.idle_label = idle_label
		self.working_label = 1 - self.idle_label

		cluster_seqs = dict()
		for e,label in enumerate(cl_mv_labels):
			if label not in cluster_seqs:
				cluster_seqs.update({label : list(cluster_subseqs[e])})
			else:
				cluster_seqs[label].extend(cluster_subseqs[e])
		for label in cluster_seqs:
			possible_patterns = cluster_seqs[label]
			for seq in possible_patterns:
				for subseq in possible_patterns:
					if seq != subseq and self.__seq_contains(seq[0],subseq[0]):
						del possible_patterns[possible_patterns.index(subseq)]
		return cluster_seqs

	def __preprocess_power(self, df):
		print ('Preprocessing power...')
		### Preprocessing
		df['TS'] = df['TS'].apply(lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))))
		self.power = np.zeros((86400))
		self.power[0] = df.iloc[0,0]
		offset = int(df.iloc[0,1])
		t = offset
		for i in range(1,df.shape[0]):
			if int(df.iloc[i,1]) != t:
				if round(df.iloc[i,1]-t) == 1.0:
					self.power[t+1-offset] = df.iloc[i,0]
					t+=1			
				elif int(df.iloc[i,1])-t < 11.0:
					orig_t = t
					req_offset = orig_t+1-offset
					for j in range(int(df.iloc[i,1]-orig_t)):
						self.power[req_offset+j] = (df.iloc[i,0]+df.iloc[i-1,0])/2
						t+=1
				else:
					orig_t = t
					req_offset = orig_t+1-offset
					for j in range(int(df.iloc[i,1]-orig_t)):
						self.power[req_offset+j] = 0
						t+=1
			else: 
				self.power[t-offset] = (self.power[t-offset]+df.iloc[i,0])/2
		
		### Filtering
		# b, a = butter(3, 0.6)
		# power_f = filtfilt(b, a, self.power)
		# min_power = np.min(power_f)
		# if min_power < 0:
		# 	power_f = power_f + abs(min_power)
		
		# plt.plot(self.power, color='g')
		# plt.plot(power_f)
		# plt.show()
		power_f = self.power
		### Detecting Peaks
		peak_indices_list = []
		power_fi = power_f
		for i in range(self.order):
			peak_indicesi, _ = find_peaks(power_fi)
			power_fi = power_fi[peak_indicesi]
			peak_indices_list.append(peak_indicesi)
		peak_indices = peak_indices_list[0]
		
		for j in range(1,self.order):
			peak_indices = peak_indices[peak_indices_list[j]]
		final_peaks = power_f[peak_indices]

		return final_peaks, peak_indices
		
	def __discretise_power(self, final_peaks):
		print ('Discretising power...')
		### Discretising Values
		X = np.array(final_peaks).reshape(-1,1)
		gamma = np.std(final_peaks)/(len(final_peaks))
		dpgmm = BayesianGaussianMixture(n_components=self.n_states,max_iter= 500,covariance_type='spherical',random_state=0).fit(X)
		unordered_labels = dpgmm.predict(X)
		original_means = [x[0] for x in dpgmm.means_]
		sorted_means = sorted(dpgmm.means_, key=lambda x:x[0])
		labels = [sorted_means.index(dpgmm.means_[l][0]) for l in unordered_labels]
		
		states = np.unique(labels)
		state_attributes = dict()
		for s in states:
			mean = sorted_means[s][0]
			state_attributes.update({ str(s) : (mean, dpgmm.covariances_[original_means.index(mean)])}) # key should be string for json 
		
		return labels, state_attributes

	def __pattern_distance(self, head, pattern):
		val_a = [self.__state_attributes[str(s)][0] for s in head]
		val_b = [self.__state_attributes[str(s)][0] for s in pattern]
		dist, _, _, _ = dtw(val_a, val_b, dist=lambda x,y:np.linalg.norm(x-y))
		return dist

	def __get_end_limits(self, start_ind):
		max_limit = start_ind+self.max_len
		if max_limit-1 >= len(self.__peak_indices):
			max_limit = len(self.__peak_indices)
			return max_limit
		if self.__off_regions:
			for i in range(start_ind+1,max_limit-1):
				if any(point in self.__off_regions for point in range(self.__peak_indices[i],self.__peak_indices[i+1]+1)):
					return i+1
			return max_limit
		else:
			return max_limit

	def __find_matches(self):
		print ('Finding matches for patterns...')
		start_ind = 0
		pattern_sequence = []
		pattern_sequence_indices = []
		while start_ind < len(self.__sequence)-self.min_len:
			dist_end_pattern = []
			end_limit = self.__get_end_limits(start_ind)
			for label, p_set in self.__pattern_dict.items():
				pattern, freq = max(p_set, key=lambda x:x[1])
				end_t = start_ind + self.min_len
				dist_t_pattern = []
				if end_t > end_limit:
					end_t = end_limit
				while end_t <= end_limit:
					p_temp = self.__sequence[start_ind:end_t]
					dist_t_pattern.append(self.__pattern_distance(p_temp, pattern))
					end_t +=1
				min_dist_t = min(dist_t_pattern)
				end_ind_p  = len(dist_t_pattern)-1 - dist_t_pattern[::-1].index(min_dist_t) ## Longest length within same pattern
				dist_end_pattern.append((min_dist_t, end_ind_p))
				
			dist_end_pattern.sort(key=lambda x:x[1])
			min_dist, end = min(dist_end_pattern, key=lambda x:x[0]) 
			end_ind = start_ind + self.min_len + end
			p_mean = np.mean([self.__state_attributes[str(s)][0] for s in self.__sequence[start_ind:end_ind]])
			p_var = np.std([self.__state_attributes[str(s)][0] for s in self.__sequence[start_ind:end_ind]])
			req_label = self.__pattern_predictor.predict(np.array([[p_mean, p_var]]))
			pattern_sequence.append(req_label[0])
			if end_ind <= len(self.__sequence):
				pattern_sequence_indices.append(end_ind-1)
			start_ind = end_ind-1
		return pattern_sequence, pattern_sequence_indices

	def __seq_contains(self, seq, subseq):
		if len(subseq) > len(seq):
			return False
		start_p = []
		for e,s in enumerate(seq):
			if s == subseq[0]:
				start_p.append(e)
		end_p = []
		for e,s in enumerate(seq):
			if s == subseq[-1]:
				end_p.append(e)
		if not (start_p and end_p):
			return False

		for start_ind in start_p:
			for end_ind in end_p:
				if end_ind > start_ind:
					dist,_,_,_=  dtw(subseq, seq[start_ind:end_ind+1], dist=lambda x,y:np.linalg.norm(x-y))
					if dist == 0:
						return True
		return False
 
 	
	def get_average_cycle_time(self):
		print ('Getting Average Cycle Time...')
		p_array, p_indices = self.__find_matches()
		unique_labels, counts = np.unique(p_array, return_counts=True)
		p_l = 0
		for e,p in enumerate(p_array):
			if p == self.working_label:
				if e == 0:
					p_l += self.__peak_indices[p_indices[e]] - self.__peak_indices[p_indices[0]]
				else:
					p_l += self.__peak_indices[p_indices[e]] - self.__peak_indices[p_indices[e-1]]
		cycle_time = p_l/counts[list(unique_labels).index(self.working_label)]
		print (p_l/counts[list(unique_labels).index(self.working_label)],'s -> Working Pattern')
		return cycle_time



#### plot and check whats going wrong
