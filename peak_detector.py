#! /usr/bin/env python3

from scipy.signal import find_peaks, butter, filtfilt
from sklearn.mixture import BayesianGaussianMixture
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pattern_discovery
import find_threshold 
import pandas as pd 
import numpy as np 
import time
import os


THRESHOLD = 2610

def timing_wrapper(func):
	def wrapper(*args,**kwargs):
		
		t0= time.time()
		func_val = func(*args,**kwargs)
		time_taken = time.time() - t0

		print (str(func),' took: ', time_taken)

		return func_val

	return wrapper

def get_required_files(device_path, day):
	print ('Obtaining Required Files...')
	file1 = None
	file2 = None
	end_search = False
	for root, dirs, files in os.walk(device_path):
		if files and not end_search:
			files.sort()
			for f in files:
				if day in f and f.endswith('.csv.gz'):
					file1 = os.path.join(root,f)
				if file1 and os.path.join(root,f) > file1:
					file2 =  os.path.join(root,f)
					end_search = True
					break
	return file1, file2


def zero_cross_detector(sig):
	indices = []
	for i in range(len(sig)-1):
		if sig[i] > 0 and sig[i+1] < 0:
			indices.append(i)
	return np.array(indices)

def lpf(data):
	if len(data) < 13:
		data.extend((13-len(data))*[data[-1]])

	b, a = butter(3, 0.5)
	y = filtfilt(b, a, data)
	
	return y

@timing_wrapper
def preprocess_power(f1, f2):
	print ('Preprocessing files to extract data...')

	df1 = pd.read_csv(f1)
	df2 = pd.read_csv(f2)
	df1.sort_values(by='TS')
	df2.sort_values(by='TS')
	df = pd.concat([df1,df2])
	start_time = datetime.isoformat(datetime.strptime(f1[-17:-7]+' 08:00:00', '%Y_%m_%d %H:%M:%S'), sep=' ')
	end_time = datetime.isoformat(timedelta(days=1) + datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S'), sep=' ')

	df = df[(df['TS'] >= start_time) & (df['TS'] < end_time)]

	df.to_csv('checking.csv')

	df['TS'] = df['TS'].apply(lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))))

	power = []
	for i in range(df.shape[0]):
		if not power:
			power.append(df.iloc[i,0])
		else:
			if df.iloc[i,1]-df.iloc[i-1,1] == 1.0:
				power.append(df.iloc[i,0])
		
			elif df.iloc[i,1] == df.iloc[i-1,1]:
				power[-1] = (power[-1]+df.iloc[i,0])/2

			elif df.iloc[i,1]-df.iloc[i-1,1] > 1.0:
				for j in range(int(df.iloc[i,1]-df.iloc[i-1,1])):
					power.append(df.iloc[i-1,0])


	## Thresholding Signal   ## This is necessary if you plan to distinguish cycles based on threshold.
	# power = pd.Series(power).apply(lambda x: x if x > THRESHOLD else THRESHOLD)

	## Smoothing 
	print ('Filtering signal ...')
	power_f = lpf(power)

	### Differencing filter
	print ('Differentiating signal ...')
	p_detrend = []
	for i in range(len(power_f)-1):
		p_detrend.append(power_f[i+1]-power_f[i])

	## Smoothing derivative
	print ('Smoothing derivative of signal ...')
	power_d = lpf(p_detrend)

	return power_d, power_f

@timing_wrapper
def detect_peaks(power_d, power_f):
	print ('Detecting Peaks of Signal....')
	# power_d - smoothed derivative of signal;
	# power_f - filtered signal
	# peaks, _ = find_peaks(power_d)  # Find peaks returns the actual peaks of derivative
	
	peaks = zero_cross_detector(power_d)  # Returns indices of peaks in signal 
	peaks = peaks+1
	
	## For obtaining actual signal with peaks.
	# peak_p = np.zeros((len(power_d)))
	# peak_p[peaks] = power_f[peaks]

	final_peaks = power_f[peaks]

	return final_peaks, peaks

@timing_wrapper
def peaks_to_discrete_states(final_peaks):
	#### BayesianGaussianMixture

	total_peaks = np.array(final_peaks)
	X = total_peaks.reshape(-1,1)

	dpgmm = BayesianGaussianMixture(n_components=10,max_iter= 500,covariance_type='spherical').fit(X)
	labels = dpgmm.predict(X)
	states = np.unique(labels)

	state_attributes = dict()
	for s in states:
		state_attributes.update({ str(s) : (dpgmm.means_[s][0], dpgmm.covariances_[s])}) # key should be string for json 
	
	print ('Number of states: ', len(np.unique(labels)), states)
	# print (dpgmm.means_)
	# print (dpgmm.covariances_)

	color=['navy', 'c', 'cornflowerblue', 'gold','darkorange', 'r', 'g', 'm', 'y', 'k']
	
	color_labels = []
	for label in labels:
		color_labels.append(color[int(label)])

	print ([ color[s] for s in states])

	plt.scatter(range(len(final_peaks)), final_peaks, color= color_labels)
	plt.show()


	return labels, state_attributes


if __name__ == '__main__':

	device_path = '/media/milan/DATA/Qrera/FWT/5CCF7FD0C7C0'
	day = '2018_07_07'

	file1, file2 = get_required_files(device_path, day)

	power_d, power_f = preprocess_power(file1, file2)

	final_peaks, peak_indices = detect_peaks(power_d, power_f)

	labels = peaks_to_discrete_states(final_peaks)


	################ Visualization

	# total_peaks_len = len(final_peaks)
	# print (total_peaks_len)
	# plt.plot(range(total_peaks_len),final_peaks)
	# plt.show()

	# color=['navy', 'c', 'cornflowerblue', 'gold','darkorange', 'r', 'g', 'm', 'y', 'k']
	
	# color_labels = []
	# for label in labels:
	# 	color_labels.append(color[int(label)])


	# plt.scatter(range(total_peaks_len), final_peaks, color= color_labels)
	# plt.show()

	# print (dpgmm.means_)
	# print (dpgmm.covariances_)
	# print (dpgmm.weights_)


	#################################################################################################

	##################### Finding patterns where correlation was maximum.

	# n_power_f = (power_f - np.mean(power_f))/(np.std(power_f)*len(power_f))

	# pattern = np.array(power_f[180:200])
	# n_pattern = (pattern - np.mean(pattern))/(np.std(pattern))
	
	# correlation = np.correlate(n_power_f, n_pattern,'full')

	# peaks, _ = find_peaks(correlation, height=0.0005)
	# print ('Total Signals Detected: ', len(peaks))

	# plt.plot(correlation[0:1500])
	# plt.show()

	######################
	

	####################### Filtering peaks with a threshold
	# peak_pr = [round(p) for p in final_peaks]
	# peak_threshold = find_threshold.get_otsus_threshold(peak_pr)
	# print ('peak_threshold', peak_threshold)

	# for p in range(len(final_peaks)):
	# 	if final_peaks[p] < peak_threshold:
	# 		final_peaks[p] = 0
	
	# total_peaks = [p for p in final_peaks if p != 0]
	# print( 'Total Peaks Count: ', len(total_peaks))
	
	########################