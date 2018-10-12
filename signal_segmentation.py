#! /usr/bin/env python3

from sklearn.cluster import AffinityPropagation, KMeans
import custompattern_mining as cpm
import peak_detector as pd 
import matplotlib.pyplot as plt
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

	def __get_patterns(self):
		pm = cpm.PatternMining(self.sequence, self.state_attributes, self.min_len, self.max_len)
		self.patterns = pm.find_patterns()

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
		cluster_subseqs_exs = [ seq[ind] for ind in ap.cluster_centers_indices_]

		### Arranging sequences by cluster label 
		cluster_subseqs = dict()
		for seq, label in zip(seq_f,ap.labels_):
			if label not in cluster_subseqs:
				cluster_subseqs.update({label : [seq]})
			else:
				cluster_subseqs[label].append(seq)
				
		return cluster_subseqs, cluster_subseqs_exs

	@timing_wrapper
	def __cluster_patterns(self, seq_f):
		### Clustering sequences using dtw and affinity propagation
		cluster_subseqs, cluster_subseqs_exs = self.__dtw_clustering(seq_f)
		print ('Number of clusters with DTW: ', len(cluster_subseqs))
		if len(cluster_subseqs) == 1:
			return cluster_subseqs, cluster_subseqs_exs, None, None			
		
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


	@timing_wrapper
	def __find_matches(self):
		print ('Matching Discovered Patterns...')
		start_ind = 0
		end_ind = 0
		pattern_sequence = []
		pattern_sequence_indices = []
		while start_ind < len(self.sequence)-1:
			min_pdist = np.inf
			end_ind_l = []
			for label, p_set in self.pattern_dict.items():
				pattern, freq	= max(p_set, key=lambda x:x[1])
					
				dists = []
				ends = []
				end_ind_t = start_ind+self.min_len-1
				while end_ind_t < start_ind+self.max_len:
					p_temp = self.sequence[start_ind:end_ind_t+1]
					dist = self.__pattern_distance(p_temp,pattern)
					dists.append(dist)
					ends.append(end_ind_t)
					end_ind_t +=1

				### preferring longer patterns rather than shorter ones intra pattern
				min_dist = min(dists)
				for e,d in enumerate(dists):
					if d == min_dist:
						end_ind_f = ends[e]

				if min_pdist >= min_dist:
					min_pdist = min_dist
					end_ind_l.append(end_ind_f)
		
			end_ind = min(end_ind_l)  ## Shorter patterns inter pattern
			p_mean = np.mean([self.state_attributes[str(s)][0] for s in self.sequence[start_ind:end_ind+1]])
			p_var = np.std([self.state_attributes[str(s)][0] for s in self.sequence[start_ind:end_ind+1]])
			req_label = self.predictor.predict(np.array([[p_mean, p_var]]))
			pattern_sequence.append(req_label[0])
			if end_ind < len(self.sequence):
				pattern_sequence_indices.append(end_ind)
			start_ind = end_ind
		
		self.pattern_sequence = pattern_sequence
		self.pattern_sequence_indices = pattern_sequence_indices
		return pattern_sequence, pattern_sequence_indices

	@timing_wrapper
	def segment_signal(self, power_signal):
		off_regions = [e for e,p in enumerate(power_signal) if p == 0]
		power_f = pd.filter_signal(power_signal)
		final_peaks, self.peak_indices = pd.detect_peaks(power_f,self.order) ## Order of the derivative
		no_iter = 1
		while self.pattern_dict == None:
			self.sequence, self.state_attributes = pd.signal_to_discrete_states(final_peaks)
			self.__discover_segmentation_pattern()
			no_iter += 1
			if no_iter >= 5:
				raise ValueError('Could not find segments for signal. Try again! Or-> Check if min_length of pattern is too small. Check if number of segments are  suitable for data.')	
		p_array, p_indices = self.__find_matches()

		print ('Mapping time indices...')
		simplified_seq = np.zeros((len(power_signal)))
		start_ind = 0
		for e,i in enumerate(p_indices):
			simplified_seq[start_ind:self.peak_indices[i]+2] = p_array[e]
			start_ind = self.peak_indices[i]+2
		simplified_seq[off_regions] = 2

		print ('Segmenting regions based on time...')
		unique_labels = list(np.unique(simplified_seq))
		segmented_regions = dict()
		for r in unique_labels:
			
			start_stop = []
			started = False
			for e,s in enumerate(simplified_seq):
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

		return simplified_seq, segmented_regions