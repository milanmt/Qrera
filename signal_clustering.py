#! /usr/bin/env python3
import fnmatch
from sklearn.cluster import KMeans
import peak_detector as pd 
import numpy as np
import matplotlib.pyplot as plt
import os

class SignalClustering:
	def __init__ (self,resolution=60):
		self.window = resolution

	def __partition_sequence(self,no_segments,sequence,state_attributes):
		seq = np.array([state_attributes[str(s)][0] for s in sequence]).reshape(-1,1)
		kmeans = KMeans(no_segments).fit(seq)
		return kmeans.labels_, kmeans.cluster_centers_

	def segment_signal(self, no_segments, power_signal):
		print ('Segmenting Signal...')
		# power_pa = pd.piecewise_approximation(power_signal,self.window)
		off_regions = [e for e,p in enumerate(power_signal) if p == 0]
		# no_segments = no_segments -1  ## inclusing off r
		sequence, state_attributes = pd.signal_to_discrete_states(power_signal)
		simplified_seq, segment_means = self.__partition_sequence(no_segments, sequence,state_attributes)

		# print ('Mapping time indices...')
		# power_segmented = np.zeros((len(power_signal)))
		# for e,s in enumerate(simplified_seq):
		# 	if e in off_regions:
		# 		power_segmented[e*self.window:(e+1)*self.window] = no_segments
		# 	else:
		# 		power_segmented[e*self.window:(e+1)*self.window] = s

		return simplified_seq, segment_means

	def get_threshold(self, simplified_seq, segment_means):
		states = np.unique(simplified_seq)
	
		color=['navy', 'cornflowerblue', 'gold', 'c', 'darkorange', 'r', 'g', 'm', 'y', 'k', 'teal', 'chocolate', 'crimson', 'dimgray', 'purple']
	
		color_labels = []
		for label in simplified_seq:
		 	color_labels.append(color[int(label)])

		print ([ color[s] for s in states])

		plt.scatter(range(len(power)), power, color= color_labels)
		plt.show()

		state_dic = dict()
		for e,s in enumerate(simplified_seq):
			if s not in state_dic:
				state_dic.update({s: [power[e]]})
			else:
				state_dic[s].append(power[e])

		segment_means_sorted = sorted([e for e,m in enumerate(segment_means)],key=lambda x:segment_means[x])
		seg_001 = segment_means_sorted[1]
		seg_010 = segment_means_sorted[2]
		# print (seg_001)
		# print (seg_010)
		max_001 = max(state_dic[seg_001])
		min_010 = min(state_dic[seg_010])
		exp_thresh = (max_001+min_010)/2
		print (day, exp_thresh)
		return exp_thresh



if __name__ == '__main__':

	###### ONE DAY

	# device_path = '/media/milan/DATA/Qrera/KakadeLaser/B4E62D38855E'
	# day = '2018_07_31'
	# no_segments = 8
	# file1, file2 = pd.get_required_files(device_path, day)
	# power = pd.preprocess_power(file1, file2)
	# # plt.plot(power)
	# # plt.show()
	
	# ss = SignalClustering()
	# simplified_seq, segment_means = ss.segment_signal(no_segments,power)
	# threshold = ss.get_threshold(simplified_seq, segment_means)

	####### all days 

	device_path = '/media/milan/DATA/Qrera/KakadeLaser/B4E62D38855E'
	files = []
	no_segments = 8 

	for root, dirs, fs in os.walk(device_path):
		if fs:
			files.extend(os.path.join(root,f) for f in fs if f.endswith('.csv.gz') and fnmatch.fnmatch(f,"*_*_*"))

	threshold_days = []
	days = []
	
	for file in files:
		day = file[-17:-7]
		file1, file2 = pd.get_required_files(device_path, day)
		power = pd.preprocess_power(file1, file2)
		ss = SignalClustering()
		simplified_seq, segment_means = ss.segment_signal(no_segments,power)
		threshold = ss.get_threshold(simplified_seq, segment_means)
		threshold_days.append(threshold)
		days.append(file[-17:-7])

	df = pandas.DataFrame( data = list(zip(threshold_days, days)), columns = ['Threshold', 'Day'])
	df.to_csv('csv_files/KakadeLaser_thresh.csv', index= True, header=True)