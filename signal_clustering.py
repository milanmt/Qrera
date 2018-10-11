#! /usr/bin/env python3

from sklearn.cluster import KMeans
import peak_detector as pd 
import numpy as np

class SignalClustering:
	def __init__ (self,resolution=60):
		self.window = resolution

	def __partition_sequence(self,no_segments,sequence,state_attributes):
		seq = np.array([state_attributes[str(s)][0] for s in sequence]).reshape(-1,1)
		kmeans = KMeans(no_segments).fit(seq)
		return kmeans.labels_

	def segment_signal(self, no_segments, power_signal):
		print ('Segmenting Signal...')
		power_pa = pd.piecewise_approximation(power_signal,self.window)
		off_regions = [e for e,p in enumerate(power_pa) if p == 0]
		no_segments = no_segments-1
		sequence, state_attributes = pd.signal_to_discrete_states(power_pa)
		simplified_seq = self.__partition_sequence(no_segments, sequence,state_attributes)

		print ('Mapping time indices...')
		power_segmented = np.zeros((len(power_signal)))
		for e,s in enumerate(simplified_seq):
			if e in off_regions:
				power_segmented[e*self.window:(e+1)*self.window] = no_segments
			else:
				power_segmented[e*self.window:(e+1)*self.window] = s
		return power_segmented