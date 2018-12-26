#! /usr/bin/env python3

from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.mixture import BayesianGaussianMixture
from datetime import datetime
from dtw import dtw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly

def timing_wrapper(func):
	def wrapper(*args,**kwargs):
		t = datetime.now()
		func_val = func(*args,**kwargs)
		time_taken = datetime.now() -t
		print (str(func),' took: ', time_taken)
		return func_val
	return wrapper


class SinglePatternError(Exception):
	pass



class PatternLength:
	def __init__(self, raw_dataframe, total_time, min_len, max_len, order, n_states=10):
		##### Parameters to be set 
		self.total_time = total_time
		self.min_len = min_len
		self.max_len = max_len
		if self.max_len <= self.min_len:
			raise ValueError('Incorrect values for length of pattern')
		self.order = order
		self.n_states = n_states
		self.uni_min = 3   ## Pattern has to be atleast this long to be considered a pattern
		self.no_max_freq = 3 ## No.of patterns from each region to consider for matching and learning
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
		self.p_array, self.p_indices = self.__find_matches()
	
	@ timing_wrapper
	def __preprocess_power(self, df):
		print ('Preprocessing power...')
		### Preprocessing
		df['TS'] = df['TS'].apply(lambda x: int(datetime.timestamp(datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))))
		# print (df.shape[0], 'orig_signal length')
		self.power = np.zeros((self.total_time))
		self.power[0] = df.iloc[0,0]
		self.offset = int(df.iloc[0,1])
		t = self.offset
		for i in range(1,df.shape[0]):
			if int(df.iloc[i,1]) != t:
				if round(df.iloc[i,1]-t) == 1.0:
					self.power[t+1-self.offset] = df.iloc[i,0]
					t+=1			
				elif int(df.iloc[i,1])-t < 21.0:
					orig_t = t
					req_offset = orig_t+1-self.offset
					avg = (df.iloc[i,0]+df.iloc[i-1,0])/2
					for j in range(int(df.iloc[i,1]-orig_t)):
						self.power[req_offset+j] = avg
						t+=1
				else:
					orig_t = t
					req_offset = orig_t+1-self.offset
					for j in range(int(df.iloc[i,1]-orig_t)):
						self.power[req_offset+j] = 0
						t+=1
			else: 
				self.power[t-self.offset] = (self.power[t-self.offset]+df.iloc[i,0])/2
		
		### Filtering
		if self.order == 1:
			# power_f = savgol_filter(self.power, 5,2,mode='nearest')
			### butterworth
			a,b = butter(3, 0.5)
			power_nf = filtfilt(a, b, self.power)
			min_power_nf
			if min_power_nf < 0:  
				power_f = power_nf + abs(min_power_nf)
			else:
				power_f = power_nf

		else:
			power_f = self.power   ## No filtering

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

		### Adding intial and final power values to maintain completeness
		np.insert(final_peaks,0, power_f[0])
		np.insert(peak_indices,0,0)
		np.append(final_peaks,power_f[-1])
		np.append(peak_indices,len(power_f)-1)
		
		# print (len(final_peaks), 'peaks')
		return final_peaks, peak_indices
		
	@timing_wrapper
	def __discretise_power(self, final_peaks):
		print ('Discretising power...')
		### Discretising Values
		X = np.array(final_peaks).reshape(-1,1)
		gamma = np.std(final_peaks)/(len(final_peaks))
		# print (gamma, 'gamma')
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
		
		# print ('states ->', state_attributes)
		return labels[:], state_attributes

	def __partition_states(self):
		print (self.__state_attributes)
		seq_means = np.array([self.__state_attributes[str(s)][0] for s in self.__sequence]).reshape(-1,1)
		kmeans = KMeans(2, random_state=2).fit(seq_means)
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
		# print ('min and max states', min_states, max_states)
		return min_states, max_states

	def __mine_patterns(self):
		### Looking for patterns that start and stop with all possible min states.
		pattern_sets = dict()
		patterns_unique = []
		self.min_states, self.max_states = self.__partition_states()
		# print (self.min_states, self.max_states)
		# print ('Printing unique patterns')
		for init_ind in range(len(self.__sequence)-self.min_len):
			if self.__sequence[init_ind] in self.min_states:
				p_temp = self.__sequence[init_ind:init_ind+self.max_len]
				try:
					end_ind = self.min_len-1+p_temp[self.min_len-1:].index(p_temp[0])+1
				except ValueError:
					end_ind = len(p_temp)
				
				if end_ind < len(p_temp):
					p = tuple(self.__sequence[init_ind : init_ind+end_ind])
					new_pattern = True
					for p_head in pattern_sets:
						if self.__pattern_distance(p, p_head) == 0.0:
							new_pattern = False
							p_set = pattern_sets[p_head]
							if p not in p_set:
								p_set.update({p:1})
							else:
								p_set[p] +=1
							break
					if new_pattern:
						# print (p)
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

		# print ('len unique patterns', len(patterns_unique))
		return patterns_unique
		
	@timing_wrapper
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
		# print ('--------------------')
		# for l in cluster_subseqs:
		# 	print (l)
		# 	print (cluster_subseqs[l])
		# print ('--------------------')
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
		# print (cluster_mv, 'cluster_mv')

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
			if label == self.working_label:
				possible_patterns = cluster_seqs[label]
				for seq in possible_patterns:
					for subseq in possible_patterns:
						if seq != subseq and self.__seq_contains(seq[0],subseq[0]):
							del possible_patterns[possible_patterns.index(subseq)]

		### Printing values
		print ('Final Number of Clusters: ', len(cluster_seqs))
		print ('Idle Class: ', idle_label)
		print ('Working Class: ', self.working_label)
		for k in cluster_seqs:
			print (k)
			print (cluster_seqs[k])
		return cluster_seqs

	def __pattern_distance(self, head, pattern):
		val_a = [self.__state_attributes[str(s)][0] for s in head]
		val_b = [self.__state_attributes[str(s)][0] for s in pattern]
		dist, _, _, _ = dtw(val_a, val_b, dist=lambda x,y:np.linalg.norm(x-y))
		return dist

	def __get_end_limits(self, start_ind):
		max_limit = start_ind+self.max_len
		if max_limit > len(self.__sequence):
			max_limit = len(self.__sequence)
		if self.__off_regions:
			for i in range(start_ind+1,max_limit+1):
				if i+1 >= len(self.__peak_indices):
					return len(self.__peak_indices), False
				if any(point in range(self.__peak_indices[i],self.__peak_indices[i+1]+1) for point in self.__off_regions):
					return i+2, True
			return max_limit, False
		else:
			return max_limit, False

	@timing_wrapper
	def __find_matches(self):
		print ('Matching Discovered Patterns...')
		# print (len(self.__sequence))
		start_ind = 0
		pattern_sequence = []
		pattern_sequence_indices = []

		self.working_mean = self.__pattern_predictor.cluster_centers_[self.working_label][0]
		self.working_var = self.__pattern_predictor.cluster_centers_[self.working_label][1]
		self.idle_mean = self.__pattern_predictor.cluster_centers_[self.idle_label][0]
		self.idle_var = self.__pattern_predictor.cluster_centers_[self.idle_label][1]

		idle_states = []
		for pt,freq in self.__pattern_dict[self.idle_label]:
			for s in pt:
				if s not in idle_states:
					idle_states.append(s)
		
		min_ws = np.inf
		for pt,freq in self.__pattern_dict[self.working_label]:
			for s in pt:
				if s < min_ws and s not in self.max_states and s not in idle_states:
					min_ws = s

		if max(idle_states) >= max(self.min_states):
			idle_equal_min = True
		else:
			idle_equal_min = False

		while start_ind < len(self.__sequence)-1:
			min_pdist = []
			end_ind_l = []
			req_labels = []
			max_limit, off_region_present = self.__get_end_limits(start_ind)
			
			# print (start_ind, max_limit)
			# print (self.__sequence[start_ind:max_limit])
			contains_idle = any(s in self.__sequence[start_ind+self.uni_min-1:max_limit] for s in idle_states)
			
			if min_ws != np.inf:
				if min_ws in self.__sequence[start_ind+self.uni_min-1:max_limit]:
					contains_min = True
				else:
					contains_min = False
			else:
				contains_min = False
			
			for label, p_set in self.__pattern_dict.items():
				sorted_patterns	= sorted(p_set, key=lambda x:x[1], reverse=True)
				for pattern,freq in sorted_patterns[:self.no_max_freq]:	
					dists = []
					ends = []
					end_ind_t = start_ind+self.uni_min
					if end_ind_t > max_limit:
						end_ind_t = max_limit
					starting_end_ind_t = end_ind_t
			
					while end_ind_t <= max_limit:
						if not idle_equal_min:
							if contains_idle:
								if self.__sequence[end_ind_t-1] in idle_states:
									p_temp = self.__sequence[start_ind:end_ind_t]
									if self.__sequence[end_ind_t-1] != self.__sequence[end_ind_t-2] or starting_end_ind_t == end_ind_t:
										dist = self.__pattern_distance(p_temp,pattern)
										# print (p_temp, pattern, dist)
										dists.append(dist)
										ends.append(end_ind_t)

							elif contains_min:
								if self.__sequence[end_ind_t-1] == min_ws:
									p_temp = self.__sequence[start_ind:end_ind_t]
									if self.__sequence[end_ind_t-1] != self.__sequence[end_ind_t-2] or starting_end_ind_t == end_ind_t:
										dist = self.__pattern_distance(p_temp,pattern)
										# print (p_temp, pattern, dist)
										dists.append(dist)
										ends.append(end_ind_t)

							else:
								p_temp = self.__sequence[start_ind:end_ind_t]
								dist = self.__pattern_distance(p_temp,pattern)
								# print (p_temp, pattern, dist)
								dists.append(dist)
								ends.append(end_ind_t)

						else:
							p_temp = self.__sequence[start_ind:end_ind_t]
							dist = self.__pattern_distance(p_temp,pattern)
							# print (p_temp, pattern, dist)
							dists.append(dist)
							ends.append(end_ind_t)

						end_ind_t +=1

					### preferring short patterns rather than longer ones intra pattern
					min_dist = min(dists)
					for e,d in enumerate(dists):
						if d == min_dist:
							end_ind_f = ends[e]
							break                ## break included for finding short patterns
				
					min_pdist.append(min_dist)
					end_ind_l.append(end_ind_f)
					req_labels.append(label)
			
			req_pdist = min(min_pdist)
			end_ind = np.inf 
			for e, d in enumerate(min_pdist):   
				if end_ind_l[e] < end_ind and d == req_pdist:   ## shorter inter patterns
					end_ind = end_ind_l[e]
					final_label = req_labels[e]

			#### Check if pattern has a minimum within itself
			if any(s in idle_states for s in self.__sequence[start_ind+self.uni_min-1:end_ind-1]) and final_label == self.working_label:
				dist_n = np.inf
				for e, s in enumerate(self.__sequence[start_ind+self.uni_min-1:end_ind-1]):
					if s in idle_states:
						idle_ind = start_ind+self.uni_min-1+e
					
						for label, p_set in self.__pattern_dict.items():
							sorted_patterns	= sorted(p_set, key=lambda x:x[1], reverse=True)
							for pattern,freq in sorted_patterns[:self.no_max_freq]:
								temp_dist = self.__pattern_distance(pattern, self.__sequence[idle_ind:end_ind])
								# print (self.__sequence[idle_ind:end_ind],pattern, temp_dist)
												
								if dist_n > temp_dist:
									dist_n = temp_dist
									new_end_ind = idle_ind

				if dist_n <= req_pdist:
					end_ind = new_end_ind+1
 
			p_mean = np.mean([self.__state_attributes[str(s)][0] for s in self.__sequence[start_ind:end_ind]])
			p_var = np.std([self.__state_attributes[str(s)][0] for s in self.__sequence[start_ind:end_ind]])
			req_label = self.__pattern_predictor.predict(np.array([[p_mean, p_var]]))
			if np.std(self.__sequence[start_ind:end_ind]) != 0:
				print (self.__sequence[start_ind:end_ind])
				print (final_label, req_label)

			### Increasing resolution
			real_power = self.power[self.__peak_indices[start_ind]:self.__peak_indices[end_ind-1]+2]
				
			if any(p in self.__off_regions for p in range(self.__peak_indices[start_ind],self.__peak_indices[end_ind-1]+2)):
				print ('off region case')
				# print (self.__peak_indices[start_ind], self.__peak_indices[end_ind-1]+1)
				# print (real_power)
				for e, v in enumerate(real_power):
					if v == 0:
						start_o = self.__peak_indices[start_ind]+e
						# print (start_o)
						rp_o_s = e
						# print ('rpos', rp_o_s)
						break
				for e, v in enumerate(real_power[::-1]):
					if v == 0:
						end_o = self.__peak_indices[start_ind]+len(real_power)-1-e
						# print (end_o)
						rp_o_e = len(real_power)-1-e
						# print ('rpoe', rp_o_e)
						break

				rp_before_off = real_power[:rp_o_s]
				# print (rp_before_off)
				start_w = None
				end_w = None 
				for e, v in enumerate(rp_before_off):
					if v > self.working_mean-self.working_var:
						start_w = self.__peak_indices[start_ind]+e
						# print ('start_w', start_w)
						break
				for e, v in enumerate(rp_before_off[::-1]):
					if v > self.working_mean-self.working_var:
						end_w = self.__peak_indices[start_ind]+len(rp_before_off)-1-e
						# print ('end_w', end_w)
						break 

				if start_w != None and end_w != None:
					if start_w != self.__peak_indices[start_ind]:
						pattern_sequence.append(self.idle_label)
						# print (self.idle_label)
						pattern_sequence_indices.append(start_w-1)	
						# print (start_w-1)					
					pattern_sequence.append(req_label[0])
					# print (req_label[0])
					pattern_sequence_indices.append(end_w+1)
					# print (end_w+1)
					if end_w != start_o-1:
						pattern_sequence.append(self.idle_label)
						# print (self.idle_label)
						pattern_sequence_indices.append(start_o)
						# print (start_o)
				else:
					pattern_sequence.append(self.idle_label)
					# print (self.idle_label)
					if end_ind <= len(self.power):
						pattern_sequence_indices.append(start_o)
						# print (start_o)
					else:
						pattern_sequence_indices.append(len(self.power)-1)
						# print (len(self.power)-1)

				pattern_sequence.append(2)
				# print (2)
				pattern_sequence_indices.append(end_o)
				# print (end_o)

				rp_after_off = real_power[rp_o_e+1:]
				start_w = None
				end_w = None 
				for e, v in enumerate(rp_after_off):
					if v > self.working_mean-self.working_var:
						start_w = self.__peak_indices[start_ind]+rp_o_e+1+e
						break
				for e, v in enumerate(rp_after_off[::-1]):
					if v > self.working_mean-self.working_var:
						end_w = self.__peak_indices[start_ind]+rp_o_e+1+len(rp_after_off)-1-e
						break 

				if start_w != None and end_w != None:
					if start_w != self.__peak_indices[start_ind]+rp_o_e+1:
						pattern_sequence.append(self.idle_label)
						# print (self.idle_label)
						pattern_sequence_indices.append(start_w-1)
						# print (start_w-1)
					pattern_sequence.append(3)
					# print (3)
					pattern_sequence_indices.append(end_w+1)
					# print (end_w+1)
					if end_w != self.__peak_indices[end_ind-1]+1:
						pattern_sequence.append(self.idle_label)
						# print (self.idle_label)
						pattern_sequence_indices.append(self.__peak_indices[end_ind-1]+1)
						# print (self.__peak_indices[end_ind-1]+1)

				else:
					pattern_sequence.append(self.idle_label)
					# print (self.idle_label)
					if end_ind <= len(self.power):
						# print(self.__peak_indices[end_ind-1]+1)
						pattern_sequence_indices.append(self.__peak_indices[end_ind-1]+1)
					else:
						# print(len(self.power)-1)
						pattern_sequence_indices.append(len(self.power)-1)

			else:
				start_w = None
				end_w = None 
				for e, v in enumerate(real_power):
					if v > self.working_mean-self.working_var:
						start_w = self.__peak_indices[start_ind]+e
						break
				for e, v in enumerate(real_power[::-1]):
					if v > self.working_mean-self.working_var:
						end_w = self.__peak_indices[start_ind]+len(real_power)-1-e
						break 

				if start_w != None and end_w != None:
					if start_w != self.__peak_indices[start_ind]:
						# print (self.idle_label)
						pattern_sequence.append(self.idle_label)
						# print (start_w-1)
						pattern_sequence_indices.append(start_w-1)						
					pattern_sequence.append(req_label[0])
					# print (req_label[0])
					pattern_sequence_indices.append(end_w+1)
					# print (end_w+1)
					if end_w != self.__peak_indices[end_ind-1]+1:
						pattern_sequence.append(self.idle_label)
						# print (self.idle_label)
						pattern_sequence_indices.append(self.__peak_indices[end_ind-1]+1)
						# print (self.__peak_indices[end_ind-1]+1)

				else:
					pattern_sequence.append(req_label[0])
					# print (req_label[0])
					if end_ind <= len(self.__sequence):
						pattern_sequence_indices.append(self.__peak_indices[end_ind-1]+1)
						# print (self.__peak_indices[end_ind-1]+1)
					else:
						pattern_sequence_indices.append(len(self.power)-1)
						# print (len(self.power)-1)
		
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
		
		p_l = 0
		count_avg = 0
		for e,p in enumerate(self.p_array):
			if p == self.working_label:
				if e == 0:
					if all(point not in range(0,self.p_indices[e]+1) for point in self.__off_regions):
						p_l += self.p_indices[e]+1
						count_avg += 1
				elif e == len(self.p_array)-1:
					if all(point not in range(self.p_indices[e-1],self.p_indices[e]+1) for point in self.__off_regions):
						p_l += self.p_indices[e] - self.p_indices[e-1] + 1
						count_avg += 1
				else:
					if self.p_array[e-1]!= 2 and self.p_array[e+1] !=2:
						p_l += self.p_indices[e] - self.p_indices[e-1] + 1
						count_avg += 1

		cycle_time = p_l/count_avg
		print ('Avg cycle time ->', cycle_time)

		print ('Mapping time indices...')  ## This is not needed for average length
		simplified_seq = np.zeros((len(self.power)))
		start_ind = 0
		for e,i in enumerate(self.p_indices):
			simplified_seq[start_ind:i+1] = self.p_array[e]
			start_ind = i+1
		simplified_seq[self.__off_regions] = 2

		print ('Plotting...')
		unique_labels = list(np.unique(simplified_seq))
		y_plot = np.zeros((len(unique_labels),len(simplified_seq)))
		for e,el in enumerate(simplified_seq):
			# print (e,el)
			if el == 2:
				y_plot[unique_labels.index(el),e] = 1500
			else:
				y_plot[unique_labels.index(el),e] = self.power[e]
		time = np.arange(len(self.power))
	
		plotly.tools.set_credentials_file(username='MilanMariyaTomy', api_key= '8HntwF4rtsUwPvjW3Sl4')
		data = [go.Scattergl(x=time, y=y_plot[i,:]) for i in range(len(unique_labels))]
		pattern_edges = len(time)*[None]
		for ind in self.p_indices:
			pattern_edges[ind] = self.power[ind]
		# print (len([l for l in pattern_edges if l != None]))
		data.append(go.Scattergl(x=time,y=pattern_edges,mode='markers'))
		fig = go.Figure(data = data)
		plotly.plotly.plot(fig, filename='fwtc_pattern_counting')
		return cycle_time

	def get_estimate_count(self):
		unique_labels, counts = np.unique(self.p_array, return_counts=True)
		working_ind = list(unique_labels).index(self.working_label)
		estimate_count = counts[working_ind]
		print ('No. of working patterns found : ' , estimate_count)
		return estimate_count

	def __get_load_signals(self):
		idle_patterns =  self.__pattern_dict[self.idle_label]
		sorted_idle_patterns = sorted(idle_patterns, key=lambda x:x[1], reverse=True)
		idle_pf = sorted_idle_patterns[:self.no_max_freq]
		if len(idle_pf) == 1:
			req_idle = idle_pf	
			self.__idle_predictor = None
			print (req_idle)
			return req_idle, None
	
		else:
			### Getting average variances and means of idles for classification
			cluster_mv = np.zeros((len(idle_pf),1))
			for e, seq in enumerate(idle_pf):
				# var = np.std([self.__state_attributes[str(s)][0] for s in seq[0]])
				avg = np.mean([self.__state_attributes[str(s)][0] for s in seq[0]])
				# cluster_mv[e][1] = var
				cluster_mv[e][0] = avg
			# print (cluster_mv, 'cluster_mv')

			seq_fs = [seq for seq in idle_pf]
			seq_fs.sort(key=lambda x: x[1])
			print (seq_fs)
			c1 = np.mean([self.__state_attributes[str(s)][0] for s in seq_fs[-1][0]])
			c2 = np.mean([self.__state_attributes[str(s)][0] for s in seq_fs[-2][0]])
			
			### KMeans 
			kmeans_i = KMeans(2, init=np.array([[c1], [c2]]), random_state=7).fit(cluster_mv)
			self.__idle_predictor = kmeans_i
			cl_mv_labels = kmeans_i.labels_
			cl_mean = [el[0] for el in kmeans_i.cluster_centers_]
			self.off_label = cl_mean.index(min(cl_mean))
			self.load_label = 1 - self.off_label

			idle_clusters = dict()
			for e,label in enumerate(cl_mv_labels):
				if label not in idle_clusters:
					idle_clusters.update({label : [idle_pf[e]]})
				else: 
					idle_clusters[label].append(idle_pf[e])
			print (idle_clusters)

			req_idle = idle_clusters[self.load_label]
			return req_idle, idle_clusters[self.off_label]
		
	def get_average_uloading_time(self):
		uload_patterns, idleoff_patterns = self.__get_load_signals()
		segmented_signal, end_points, uload_present = self.__segment_with_uload()
	
		ul = []
		if uload_present:
			for e,p in enumerate(segmented_signal):
				if p == 4:  ## 4 is the uloadlabel
					if e == 0:
						if all(point not in range(0,end_points[e]+1) for point in self.__off_regions):
							ul.append(end_points[e])

					elif e == len(segmented_signal)-1:
						if all(point not in range(end_points[e-1],end_points[e]+1) for point in self.__off_regions):
							ul.append(end_points[e] - end_points[e-1] +1)
					else:
						if segmented_signal[e-1]!= 2 and segmented_signal[e+1] !=2:
							ul.append(end_points[e] - end_points[e-1] +1)

		else:
			for e,p in enumerate(segmented_signal):
				if p == self.idle_label:  ## when no uload, only idle
					if e == 0:
						if all(point not in range(0,end_points[e]+1) for point in self.__off_regions):
							ul.append(end_points[e])

					elif e == len(segmented_signal)-1:
						if all(point not in range(end_points[e-1],end_points[e]+1) for point in self.__off_regions):
							ul.append(end_points[e] - end_points[e-1] +1)
					else:
						if segmented_signal[e-1]!= 2 and segmented_signal[e+1] !=2:
							ul.append(end_points[e] - end_points[e-1] +1)

		avg_ul_time = np.mean(ul)
		max_ul_time = max(ul)
		min_ul_time = min(ul)
		print ('Avg Loading Unloading Time', avg_ul_time)
		print ('Max Loading Unloading Time', max_ul_time)
		print ('Min Loading Unloading Time', min_ul_time)

		return avg_ul_time

	def __segment_with_uload(self):
		p_array_w_uload = self.p_array
		if self.__idle_predictor != None:
			uload_present = True
		else:
			uload_present = False

		for i,p in enumerate(self.p_array):
			if p == self.idle_label:
				if i == 0:
					real_idle = self.power[:self.p_indices[i]+1]
				else:
					real_idle = self.power[self.p_indices[i-1]:self.p_indices[i]+1]
				
				ri_m = np.mean(real_idle)
				# ri_v = np.std(real_idle)
				if self.__idle_predictor != None:
					l_id = self.__idle_predictor.predict(np.array([[ri_m]]))
					if l_id == self.load_label:
						p_array_w_uload[i] = 4  # 0,1, 2- off, 3-ambiguous, 4-uload
				else:
					if ri_m > self.idle_mean - self.idle_var:
						uload_present = True
						p_array_w_uload[i] = 4

		print ('Mapping time indices...')  ## This is not needed for average length
		simplified_seq = np.zeros((len(self.power)))
		start_ind = 0
		for e,i in enumerate(self.p_indices):
			simplified_seq[start_ind:i+1] = p_array_w_uload[e]
			start_ind = i+1
		simplified_seq[self.__off_regions] = 2

		print ('Plotting...')
		unique_labels = list(np.unique(simplified_seq))
		y_plot = np.zeros((len(unique_labels),len(simplified_seq)))
		for e,el in enumerate(simplified_seq):
			# print (e,el)
			if el == 2:
				y_plot[unique_labels.index(el),e] = 1500
			else:
				y_plot[unique_labels.index(el),e] = self.power[e]
		time = np.arange(len(self.power))
	
		plotly.tools.set_credentials_file(username='MilanMariyaTomy', api_key= '8HntwF4rtsUwPvjW3Sl4')
		data = [go.Scattergl(x=time, y=y_plot[i,:]) for i in range(len(unique_labels))]
		pattern_edges = len(time)*[None]
		for ind in self.p_indices:
			pattern_edges[ind] = self.power[ind]
		# print (len([l for l in pattern_edges if l != None]))
		data.append(go.Scattergl(x=time,y=pattern_edges,mode='markers'))
		fig = go.Figure(data = data)
		plotly.plotly.plot(fig, filename='fwtc_pattern_counting')

		segmented_signal, end_points = self.__segment_signal(p_array_w_uload)
		
		return segmented_signal, end_points, uload_present

	def __segment_signal(self, pattern_array):
		sig = np.array(pattern_array)
		change_points = np.where(sig[:-1] != sig[1:])[0]
		segmented_signal = []
		end_points = []
		for e, cp in enumerate(change_points):
			segmented_signal.append(sig[int(cp)])
			end_points.append(self.p_indices[int(cp)])
			if e == len(change_points)-1:
				segmented_signal.append(sig[int(cp)+1])
				end_points.append(self.p_indices[int(cp)+1])
		
		return segmented_signal, end_points

	def get_switch_segment(self, time_stamp):
		print ('Obtaining Required Operator Region...')
		req_ts =  datetime.timestamp(datetime.strptime(time_stamp, '%Y-%m-%d %H:%M:%S'))
		segmented_signal, end_points = self.__segment_signal(self.p_array)
		for i, boundary in enumerate(end_points):
			if i != 0 and req_ts-self.offset >= end_points[i-1] and req_ts-self.offset <= boundary:
				if segmented_signal[i] == self.idle_label:
					return datetime.fromtimestamp(self.offset+end_points[i-1]), datetime.fromtimestamp(self.offset+boundary)
				else:
					if abs(req_ts-self.offset - end_points[i-1]) < abs(req_ts-self.offset - boundary):
						if segmented_signal[i-1] == self.idle_label and i != 1: 
							return datetime.fromtimestamp(self.offset+end_points[i-2]), datetime.fromtimestamp(self.offset+end_points[i-1])
						else:
							return None, None
					else:
						if i+1 == len(end_points):
							return None, None
						elif segmented_signal[i+1] == self.idle_label:
							return datetime.fromtimestamp(self.offset+end_points[i]), datetime.fromtimestamp(self.offset+end_points[i+1])
						else:
							return None, None

		return None, None 

	def get_mean_dictionary(self):
		mean_dict = dict()
		working_means = []
		uload_means = []
		idleoff_means = []

		working_patterns = self.__pattern_dict[self.working_label]
		sorted_patterns	= sorted(working_patterns, key=lambda x:x[1], reverse=True)
		for seq,f in sorted_patterns[:self.no_max_freq]:
			working_means.append(int(np.mean([self.__state_attributes[str(s)][0] for s in seq])))
		mean_dict.update({ 'Working': working_means})

		uload_patterns, idleoff_patterns = self.__get_load_signals()	

		if uload_patterns is not None:
			for seq,f in uload_patterns:
				uload_means.append(int(np.mean([self.__state_attributes[str(s)][0] for s in seq])))
			mean_dict.update({'ULoad': uload_means})

		if idleoff_patterns is not None:
			for seq,f in idleoff_patterns:
				idleoff_means.append(int(np.mean([self.__state_attributes[str(s)][0] for s in seq])))
			mean_dict.update({'Idle' : idleoff_means})

		if len(mean_dict) == 3:
			return mean_dict
		else:
			return None


	def get_mean_variance_dictionary(self):
		mv_dict = dict()
		working_means = []
		working_vars = []
		uload_means = []
		uload_vars = []
		idleoff_means = []
		idleoff_vars = []

		working_patterns = self.__pattern_dict[self.working_label]
		sorted_patterns	= sorted(working_patterns, key=lambda x:x[1], reverse=True)
		for seq,f in sorted_patterns[:self.no_max_freq]:
			working_means.append(int(np.mean([self.__state_attributes[str(s)][0] for s in seq])))
			working_vars.append(int(np.std([self.__state_attributes[str(s)][0] for s in seq])))
		mv_dict.update({ 'Working': [mv for mv in zip(working_means,working_vars)]})

		uload_patterns, idleoff_patterns = self.__get_load_signals()	

		if uload_patterns is not None:
			for seq,f in uload_patterns:
				uload_means.append(int(np.mean([self.__state_attributes[str(s)][0] for s in seq])))
				uload_vars.append(int(np.std([self.__state_attributes[str(s)][0] for s in seq])))
			mv_dict.update({'ULoad': [mv for mv in zip(uload_means,uload_vars)]})

		if idleoff_patterns is not None:
			for seq,f in idleoff_patterns:
				idleoff_means.append(int(np.mean([self.__state_attributes[str(s)][0] for s in seq])))
				idleoff_vars.append(int(np.std([self.__state_attributes[str(s)][0] for s in seq])))
			mv_dict.update({'Idle' : [mv for mv in zip(idleoff_means, idleoff_vars)]})

		if len(mv_dict) == 3:
			return mv_dict
		else:
			return None

