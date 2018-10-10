#! /usr/bin/env python3

from scipy.signal import find_peaks, butter, filtfilt
from sklearn.mixture import BayesianGaussianMixture
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pattern_discovery
import pandas as pd 
import numpy as np 
import time
import os

N_MAX = 10

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
				if file1 and os.path.join(root,f) > file1 and f.endswith('.csv.gz'):
					file2 =  os.path.join(root,f)
					end_search = True
					break
	return file1, file2

def lpf(data):
	if len(data) < 13:
		data.extend((13-len(data))*[data[-1]])

	b, a = butter(3, 0.5)
	y = filtfilt(b, a, data)
	
	return y

@timing_wrapper
def preprocess_power(f1, f2):
	print ('Preprocessing files to extract data...')

	if f2 == None:
		df = pd.read_csv(f1)
		df.sort_values(by='TS')
		start_time = datetime.isoformat(datetime.strptime(f1[-17:-7]+' 08:00:00', '%Y_%m_%d %H:%M:%S'), sep=' ')
		end_time = datetime.isoformat(datetime.strptime(f1[-17:-7]+' 11:59:59', '%Y_%m_%d %H:%M:%S'), sep=' ')
	else:
		df1 = pd.read_csv(f1)
		df2 = pd.read_csv(f2)
		df1.sort_values(by='TS')
		df2.sort_values(by='TS')
		df = pd.concat([df1,df2])
		start_time = datetime.isoformat(datetime.strptime(f1[-17:-7]+' 08:00:00', '%Y_%m_%d %H:%M:%S'), sep=' ')
		end_time = datetime.isoformat(timedelta(days=1) + datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S'), sep=' ')

	df = df[(df['TS'] >= start_time) & (df['TS'] < end_time)]
	df['TS'] = df['TS'].apply(lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))))

	power = np.zeros((86400))
	power[0] = df.iloc[0,0]
	offset = int(df.iloc[0,1])
	t = offset
	for i in range(1,df.shape[0]):
		if df.iloc[i,1] != t:
			if df.iloc[i,1]-t == 1.0:
				power[t+1-offset] = df.iloc[i,0]
				t+=1
			else:
				orig_t = t
				req_offset = orig_t+1-offset
				for j in range(int(df.iloc[i,1]-orig_t)):
					power[req_offset+j] = 0
					t+=1
		else: 
			power[t-offset] = (power[t-offset]+df.iloc[i,0])/2

	

def filter_signal(power):
	## Smoothing 
	print ('Filtering signal ...')
	power_f = lpf(power)
	min_power = np.min(power_f)
	if min_power < 0:
		power_f = power_f + abs(min_power)
	return power_f

def piecewise_approximation(power, WINDOW):
	print ('Finding piece wise approcimation of signal....')
	samples = []
	sample_indices = []
	samples = np.array([power[i] for i in range(0,len(power),WINDOW)])
	return samples

@timing_wrapper
def detect_peaks(power_f, order):   ## Order of the derivative required
	print ('Detecting Peaks of Signal....')
	peak_indices_list = []
	power_fi = power_f
	for i in range(order):
		peak_indicesi, _ = find_peaks(power_fi)
		power_fi = power_fi[peak_indicesi]
		peak_indices_list.append(peak_indicesi)

	peak_indices = peak_indices_list[0]
	for j in range(1,order):
		peak_indices = peak_indices[peak_indices_list[j]]

	final_peaks = power_f[peak_indices]
	return final_peaks, peak_indices 

@timing_wrapper
def signal_to_discrete_states(final_peaks):
	print ('Discretising Values...')
	total_peaks = np.array(final_peaks)
	X = total_peaks.reshape(-1,1)
	
	#### BayesianGaussianMixture
	gamma = np.std(final_peaks)/(len(final_peaks))
	print (gamma)
	dpgmm = BayesianGaussianMixture(n_components=N_MAX,max_iter= 500,covariance_type='spherical',random_state=0).fit(X)
	unordered_labels = dpgmm.predict(X)
	original_means = [x[0] for x in dpgmm.means_]
	sorted_means = sorted(dpgmm.means_, key=lambda x:x[0])
	labels = [sorted_means.index(dpgmm.means_[l][0]) for l in unordered_labels]
	states = np.unique(labels)

	state_attributes = dict()
	for s in states:
		mean = sorted_means[s][0]
		state_attributes.update({ str(s) : (mean, dpgmm.covariances_[original_means.index(mean)])}) # key should be string for json 
	
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
	# device_path = '/media/milan/DATA/Qrera/FWT/5CCF7FD0C7C0'
	# day = '2018_07_07'

	device_path = '/media/milan/DATA/Qrera/PYN/B4E62D'
	day = '2018_09_18'
	
	# device_path = '/media/milan/DATA/Qrera/AutoAcc/39FFBE'
	# day = '2018_04_27' #'2017_12_09'
	
	file1, file2 = get_required_files(device_path, day)
	power_f = preprocess_power(file1, file2)
	final_peaks, peak_indices = detect_peaks(power_f,3)
	plt.plot(final_peaks)
	plt.show()
	print (len(final_peaks))
	print (len(power_f))
	labels = signal_to_discrete_states(final_peaks)