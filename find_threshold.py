#! /usr/bin/env python3

import os
import time
import jenkspy 
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.signal import argrelmax


FILTER_WINDOW = 30 

def timing_wrapper(func):
	def wrapper(*args,**kwargs):
		
		t0= time.time()
		func_val = func(*args,**kwargs)
		time_taken = time.time() - t0

		print (str(func),' took: ', time_taken)

		return func_val

	return wrapper


@timing_wrapper
def get_otsus_threshold(power):
	print ('Calculating threshold...')
	power_val, power_count = np.unique(power, return_counts=True)
	prob = power_count/float(np.sum(power_count))

	max_sigma = 0
		
	for i in range(len(power_val)):
		w0 = np.sum(prob[:i+1])
		w1 = np.sum(prob[i+1:])

		u0t = 0
		for j in range(i+1):
			u0t = u0t + power_val[j]*prob[j]/w0

		u1t = 0
		for j in range(i+1,len(power_val)):
			u1t  = u1t + power_val[j]*prob[j]/w1
	
		sigma = w0*w1*(u0t-u1t)*(u0t-u1t)

		if sigma >= max_sigma:
			max_sigma = sigma
			threshold = power_val[i]

	print (threshold)

	return threshold

@timing_wrapper
def get_jenks_threshold(power, no_thresholds_required):
	print ('Calculating threshold...')
	return jenkspy.jenks_breaks(power, nb_class=int(no_thresholds_required)+1)[1:-1]


def var_round(number):
	number = float(number)
	
	if number/10 <= 10:
		return round(number)
	elif number/10 <= 1000:
		return round(number, -1)
	else:
		return round(number, -2)

def filter_data(power_sig):

	if FILTER_WINDOW == 0:
		## Simple rounding
		power_rounded = [var_round(p) for p in power_sig]
		return power_rounded
	
	## Smoothing Filter 
	power_smoothed = [var_round(sum(power_sig[i:i+FILTER_WINDOW])/FILTER_WINDOW) 
	for i in range(0,power_sig.shape[0],FILTER_WINDOW)]
	if not power_smoothed:
		return [0]
	else:
		return power_smoothed



@timing_wrapper
def process_data(path_to_device, day):
	#### Change depending on where data is available 
	#### Accesses complete data available or till date specified, from location provided
	print ('Processing files...')
	power_f = []
	power = []
	got_file = False
	for root, dirs, fs in os.walk(path_to_device):
		if got_file:
			break
		if fs:
			for file in fs:
				if file.endswith('.csv.gz'):
					filename = os.path.join(root, file)
					pd_entries = pd.read_csv(filename, engine="python")
					try:
						power_sig = filter_data(pd_entries['POWER'])
						power.extend(pd_entries['POWER'])
					except KeyError:
						power_sig = filter_data(pd_entries['VALUE'])
						power.extend(pd_entries['VALUE'])

					if not all(p == 0 for p in power_sig):
						power_f.extend(power_sig)

					if day in file:
						print (file)
						got_file = True
						break
			
	if not power_f:
		print ('Cannot Access Files')
		raise IOError

	print(len(power_f))
	print(len(power))

	return pd.Series(power_f), pd.Series(power)

@timing_wrapper
def fit_line(power, threshold, p_th_ov, amount_under_th):	
	if amount_under_th == 0.5:
		print ('fitting line to values below orig threshold and around first histogram peak')
		hist_vals, hist_bins = np.histogram(power, bins='fd')
		peaks_ind = argrelmax(hist_vals, order=10)[0]

		th_bin_no = 0
		for i in range(len((hist_bins))):
			if threshold >= hist_bins[i] and threshold <= hist_bins[i+1]:
				th_bin_no = i
				break
			
		for i in range(len(peaks_ind)):
			if peaks_ind[i] >= th_bin_no:
				peaks_ind_th = peaks_ind[:i]
				second_peak_ind = peaks_ind[i-2]+2
				break

		if len(peaks_ind_th) == 1:
			power_th = p_th_ov
		else:	
			second_peak_val = hist_bins[second_peak_ind]
			power_th = power[(power>=second_peak_val) & (power <= threshold)]

	else:
		print ('fitting line to values below orig threshold')
		power_th = p_th_ov
				
	x = np.array([[i] for i in range(0,power_th.shape[0])])
				
	print ('RANSAC')
	#### RANSAC
	threshold_temp = 0
	for iteration in range(3):
		try:
			ransac = linear_model.RANSACRegressor()						
			ransac.fit(x, power_th)
			new_y = ransac.predict(x)

			# plt.scatter(x, power_th)
			# plt.plot(x,new_y)
			# plt.show()

		except ValueError:
			new_y = [threshold]

		threshold_temp = threshold_temp + max(new_y[-1],new_y[0])
		print (max(new_y[-1],new_y[0]))
	
	threshold = var_round(threshold_temp/3)
	return threshold

@timing_wrapper
def threshold_of_device(path_to_device, no_thresholds_required, day):
	power_f, power = process_data(path_to_device, day)

	if no_thresholds_required == 1:
		threshold = var_round(get_otsus_threshold(power_f))
		# threshold = var_round(get_jenks_threshold(power_f, 1)[0])
		print (threshold, 'overall threshold')
	else:
		threshold = get_jenks_threshold(power_f, no_thresholds_required)

	if no_thresholds_required == 1:

		p_th_ov = power[power <= threshold]
		amount_under_th = round(p_th_ov.shape[0]/power.shape[0],1)
		print (p_th_ov.shape[0]/power.shape[0])

		if amount_under_th >= 0.5:
			threshold = fit_line(power, threshold, p_th_ov, amount_under_th)

	print (threshold)
	return threshold


if __name__ == '__main__':

	#### Change depending on where data is available 
	#### Accesses complete data available, or till date specified from location provided
	th = threshold_of_device('/media/milan/DATA/Qrera/AutoAcc', 1, '2018_07_02')