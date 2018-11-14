#! /usr/bin/env python3

from sklearn.cluster import AffinityPropagation, KMeans
import custompattern_mining as cpm
import peak_detector as pd 
import matplotlib.pyplot as plt
from dtw import dtw
import numpy as np
import datetime


def timing_wrapper(func):
	def wrapper(*args,**kwargs):
		t = datetime.datetime.now()
		func_val = func(*args,**kwargs)
		time_taken = datetime.datetime.now() -t
		print (str(func),' took: ', time_taken)
		return func_val
	return wrapper


class SignalSegmentation:
	def __init__ (self,min_len, max_len, derivative_order):
		self.min_len = min_len
		self.max_len = max_len
		self.order = derivative_order
		self.state_attributes = None
		self.peak_indices = None
		self.sequence = None
		self.pattern_sequence = None
		self.pattern_sequence_indices = None
		self.working_label = None
		self.idle_label = None
		self.pattern_dict = None
		self.patterns = None
		self.predictor = None
		self.off_regions = None
		self.uni_min = 3
		self.no_max_freq = 3

	def __get_patterns(self):
		self.pm = cpm.PatternMining(self.sequence, self.state_attributes, self.min_len, self.max_len)
		self.patterns = self.pm.find_patterns()
		if len(self.patterns) == 1:
			raise ValueError('Required number of patterns not found')

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

	@timing_wrapper
	def __cluster_patterns(self, seq_f):
		### Clustering sequences using dtw and affinity propagation
		cluster_subseqs = self.__dtw_clustering(seq_f)
		print ('Number of clusters with DTW: ', len(cluster_subseqs))
		if len(cluster_subseqs) == 1:
			return cluster_subseqs			
		
		### Getting average variances and means of exemplars for classification
		cluster_mv = np.zeros((len(cluster_subseqs),2))
		for label, seq_l in cluster_subseqs.items():
			var_seq_l = []
			avg_seq_l = []
			for seq_supp in seq_l:
				seq = seq_supp[0]
				var = np.std([self.state_attributes[str(s)][0] for s in seq])
				avg = np.mean([self.state_attributes[str(s)][0] for s in seq])
				var_seq_l.append(var)
				avg_seq_l.append(avg)
			cluster_mv[label][1] = np.mean(var_seq_l)
			cluster_mv[label][0] = np.mean(avg_seq_l)

		### KMeans 
		kmeans = KMeans(2, random_state=3).fit(cluster_mv)
		self.predictor = kmeans
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
					if seq != subseq and seq_contains(seq[0],subseq[0]):
						del possible_patterns[possible_patterns.index(subseq)]

		### Printing values
		print ('Final Number of Clusters: ', len(cluster_seqs))
		print ('Idle Class: ', idle_label)
		print ('Working Class: ', self.working_label)
		for k in cluster_seqs:
			print (k)
			print (cluster_seqs[k])

		return cluster_seqs

	@timing_wrapper
	def __discover_segmentation_pattern(self):
		self.__get_patterns()
		if len(self.patterns) == 1:
			self.pattern_dict = {0: [self.patterns[0]]}
			return self.patterns[0][0]
		
		### Looking for signals which start and stop with minimas. Need to discover most likely candidate. 
		possible_patterns = []
		for seq in self.patterns:
			if all(s >= seq[0][0] for s in seq[0]):
				possible_patterns.append(seq)

		### Clustering with DTW to find patterns. Exemplars from DTW -> final patterns 
		### These clustered based on mean and variance to identify idle and working patterns
		if len(possible_patterns) > 1:
			self.pattern_dict = self.__cluster_patterns(possible_patterns)
		
		elif len(possible_patterns) == 1:
			self.pattern_dict = {0: [self.patterns[0]]}


	def __pattern_distance(self,a,b):
		val_a = [self.state_attributes[str(s)][0] for s in a]
		val_b = [self.state_attributes[str(s)][0] for s in b]
		dist, _, _, _ = dtw(val_a, val_b, dist=lambda x,y:np.linalg.norm(x-y))
		return dist 

	def __get_end_limits(self, start_ind):
		max_limit = start_ind+self.max_len
		if max_limit > len(self.sequence):
			max_limit = len(self.sequence)
		if self.off_regions:
			for i in range(start_ind, max_limit+1):
				if i+1 >= len(self.peak_indices):
					return len(self.peak_indices)
				if any(point in range(self.peak_indices[i],self.peak_indices[i+1]+1) for point in self.off_regions):
					return i+2     ## to include the point after off region, to allow smooth transition to next pattern
			return max_limit
		else:
			return max_limit


	@timing_wrapper
	def __find_matches(self):
		print ('Matching Discovered Patterns...')
		start_ind = 0
		pattern_sequence = []
		pattern_sequence_indices = []
		idle_states = []
		for pt,freq in self.pattern_dict[self.idle_label]:
			for s in pt:
				if s not in idle_states:
					idle_states.append(s)
		
		min_ws = np.inf
		for pt,freq in self.pattern_dict[self.working_label]:
			for s in pt:
				if s < min_ws and s not in self.pm.max_states and s not in idle_states:
					min_ws = s
		# print (min_ws, 'min working')

		if max(idle_states) == max(self.pm.min_states):
				idle_equal_min = True
		else:
			idle_equal_min = False

		# print (max(idle_states),  max(self.pm.min_states))


		while start_ind < len(self.sequence)-1:
			min_pdist = []
			max_limit = self.__get_end_limits(start_ind)
			end_ind_l = []
			req_labels = []
			# print (start_ind, max_limit)
			# print (self.sequence[start_ind:max_limit])
			contains_idle = any(s in self.sequence[start_ind+self.uni_min:max_limit] for s in idle_states)
	
			if min_ws != np.inf:
				if min_ws in self.sequence[start_ind+self.uni_min:max_limit]:
					contains_min = True
				else:
					contains_min = False
			else:
				contains_min = False
						
			for label, p_set in self.pattern_dict.items():
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
								if self.sequence[end_ind_t-1] in idle_states:
									p_temp = self.sequence[start_ind:end_ind_t]
									if self.sequence[end_ind_t-1] != self.sequence[end_ind_t-2] or starting_end_ind_t == end_ind_t:
										dist = self.__pattern_distance(p_temp,pattern)
										# print (p_temp, pattern, dist)
										dists.append(dist)
										ends.append(end_ind_t)

							elif contains_min:
								if self.sequence[end_ind_t-1] == min_ws:
									p_temp = self.sequence[start_ind:end_ind_t]
									if self.sequence[end_ind_t-1] != self.sequence[end_ind_t-2] or starting_end_ind_t == end_ind_t:
										dist = self.__pattern_distance(p_temp,pattern)
										# print (p_temp, pattern, dist)
										dists.append(dist)
										ends.append(end_ind_t)

							else:
								p_temp = self.sequence[start_ind:end_ind_t]
								dist = self.__pattern_distance(p_temp,pattern)
								# print (p_temp, pattern, dist)
								dists.append(dist)
								ends.append(end_ind_t)

						else:
							p_temp = self.sequence[start_ind:end_ind_t]
							dist = self.__pattern_distance(p_temp,pattern)
							# print (p_temp, pattern, dist)
							dists.append(dist)
							ends.append(end_ind_t)

						end_ind_t +=1
							
					### preferring shorter patterns rather than longer ones intra pattern
					min_dist = min(dists)
					# print (min_dist)
					for e,d in enumerate(dists):
						if d == min_dist:
							end_ind_f = ends[e]
							break
				
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
			if any(s in idle_states for s in self.sequence[start_ind+self.uni_min:end_ind-1]) and final_label == self.working_label:
				dist_n = np.inf
				for e, s in enumerate(self.sequence[start_ind+self.uni_min:end_ind-1]):
					if s in idle_states:
						idle_ind = start_ind+self.uni_min+e
					
						for label, p_set in self.pattern_dict.items():
							sorted_patterns	= sorted(p_set, key=lambda x:x[1], reverse=True)
							for pattern,freq in sorted_patterns[:self.no_max_freq]:
								temp_dist = self.__pattern_distance(pattern, self.sequence[idle_ind:end_ind])
								# print (self.sequence[idle_ind:end_ind],pattern, temp_dist)
												
								if dist_n > temp_dist:
									dist_n = temp_dist
									new_end_ind = idle_ind

				if dist_n <= req_pdist:
					end_ind = new_end_ind+1
			
			p_mean = np.mean([self.state_attributes[str(s)][0] for s in self.sequence[start_ind:end_ind]])
			p_var = np.std([self.state_attributes[str(s)][0] for s in self.sequence[start_ind:end_ind]])
			req_label = self.predictor.predict(np.array([[p_mean, p_var]]))
			# if np.std(self.sequence[start_ind:end_ind]) !=0 :
			# print (self.sequence[start_ind:end_ind])
			# print (final_label, req_label)
			# print ('#################################')
			
			if end_ind <= len(self.sequence):
				pattern_sequence_indices.append(end_ind-1)
				pattern_sequence.append(req_label[0])

			start_ind = end_ind-1
		
		self.pattern_sequence = pattern_sequence
		self.pattern_sequence_indices = pattern_sequence_indices
			
		return pattern_sequence, pattern_sequence_indices

	@timing_wrapper
	def get_average_working_pattern_length(self):
		unique_labels, counts = np.unique(self.pattern_sequence, return_counts=True)
		working_ind = list(unique_labels).index(self.working_label)
		print ('No. of working patterns found : ' , counts[working_ind])
		p_l = 0
		for e,p in enumerate(self.pattern_sequence):
			if p == self.working_label:
				if e == 0:
					if all(point not in range(self.peak_indices[self.pattern_sequence_indices[0]],self.peak_indices[self.pattern_sequence_indices[e]]+1) for point in self.off_regions):
						p_l += self.peak_indices[self.pattern_sequence_indices[e]] - self.peak_indices[self.pattern_sequence_indices[0]]
				else:
					if all(point not in range(self.peak_indices[self.pattern_sequence_indices[e-1]],self.peak_indices[self.pattern_sequence_indices[e]]+1) for point in self.off_regions):
						p_l += self.peak_indices[self.pattern_sequence_indices[e]] - self.peak_indices[self.pattern_sequence_indices[e-1]]
		cycle_time = p_l/counts[list(unique_labels).index(self.working_label)]
		print (p_l/counts[list(unique_labels).index(self.working_label)],'s -> Working Pattern')
		return cycle_time

	@timing_wrapper
	def segment_signal(self, power_signal):
		self.power = power_signal
		self.off_regions = [e for e,p in enumerate(power_signal) if p == 0]
		power_f = pd.filter_signal(power_signal)
		final_peaks, self.peak_indices = pd.detect_peaks(power_f,self.order) ## Order of the derivative
		no_iter = 1
		while self.pattern_dict == None:
			self.sequence, self.state_attributes, self.state_predictor= pd.signal_to_discrete_states(final_peaks)
			self.__discover_segmentation_pattern()
			no_iter += 1
			if no_iter >= 5:
				raise ValueError('Could not find segments for signal. Try again! Or-> Check if min_length of pattern is too small. Check if number of segments are  suitable for data.')	
		p_array, p_indices = self.__find_matches()

		print ('Mapping time indices...')
		self.simplified_seq = np.zeros((len(power_signal)))
		start_ind = 0
		for e,i in enumerate(p_indices):
			self.simplified_seq[start_ind:self.peak_indices[i]+2] = p_array[e]
			start_ind = self.peak_indices[i]+2
		self.simplified_seq[self.off_regions] = 2

		print ('Segmenting regions based on time...')
		unique_labels = list(np.unique(self.simplified_seq))
		segmented_regions = dict()
		for r in unique_labels:
			start_stop = []
			started = False
			for e,s in enumerate(self.simplified_seq):
				if r == s and started == False:
					start = e
					started = True
				elif r != s and started == True:
					stop = e
					start_stop.append((start,stop))
					started = False
			
			if r == self.working_label:
				segmented_regions.update({'working_regions':start_stop})
			elif r == self.idle_label:
				segmented_regions.update({'idle_regions': start_stop})
			else:
				segmented_regions.update({'off_regions': start_stop})
		return self.simplified_seq, segmented_regions


	def get_accurate_average_working_length(self):
		unique_labels, counts = np.unique(self.pattern_sequence, return_counts=True)
		working_ind = list(unique_labels).index(self.working_label)
		print ('No. of working patterns found : ' , counts[working_ind])
		p_l = 0
		for e,p in enumerate(self.pattern_sequence):
			if p == self.working_label:
				if e == 0:
					if all(point not in range(self.peak_indices[self.pattern_sequence_indices[0]],self.peak_indices[self.pattern_sequence_indices[e]]+1) for point in self.off_regions):
						p_current = np.array(self.power[:self.peak_indices[self.pattern_sequence_indices[e]]+2])
						print (np.mean(p_current))
						p_current = p_current.reshape(-1,1)
						pc_labels = self.state_predictor.predict(p_current)

						color=['navy', 'cornflowerblue', 'gold', 'c', 'darkorange', 'r', 'g', 'm', 'y', 'k', 'teal', 'chocolate', 'crimson', 'dimgray', 'purple']
						color_labels = []
						for label in pc_labels:
							color_labels.append(color[int(label)])
						# print ([ color[s] for s in states])
						plt.scatter(range(len(pc_labels)), p_current, color= color_labels)
						plt.show()

				else:
					if all(point not in range(self.peak_indices[self.pattern_sequence_indices[e-1]],self.peak_indices[self.pattern_sequence_indices[e]]+1) for point in self.off_regions):
						p_current = np.array(self.power[self.peak_indices[self.pattern_sequence_indices[e-1]+1]:self.peak_indices[self.pattern_sequence_indices[e]]+2])
						print (np.mean(p_current))
						p_current = p_current.reshape(-1,1)
						pc_labels = self.state_predictor.predict(p_current)

						color=['navy', 'cornflowerblue', 'gold', 'c', 'darkorange', 'r', 'g', 'm', 'y', 'k', 'teal', 'chocolate', 'crimson', 'dimgray', 'purple']
						color_labels = []
						for label in pc_labels:
							color_labels.append(color[int(label)])
						# print ([ color[s] for s in states])
						plt.scatter(range(len(pc_labels)), p_current, color= color_labels)
						plt.show()
		
		print (p_l/counts[list(unique_labels).index(self.working_label)],'s -> Working Pattern')
		cycle_time = p_l/counts[list(unique_labels).index(self.working_label)]
		return cycle_time


	
def seq_contains(seq, subseq):
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

