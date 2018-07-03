#! /usr/bin/env python3

import os
import time
import pandas
import jenkspy 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

FILTER_WINDOW = 300 

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
	
		sigma = 1*(w0*w1*(u0t-u1t)*(u0t-u1t))

		if sigma >= max_sigma:
			max_sigma = sigma
			threshold = power_val[i]

	return threshold

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
	
	## Smoothing Filter
	power_smoothed = [var_round(np.mean(power_sig[i:i+FILTER_WINDOW])) 
	for i in range(power_sig.shape[0]-FILTER_WINDOW+1) if i%FILTER_WINDOW == 0]
	
	return power_smoothed


def process_data(path_to_device, day):
	#### Change depending on where data is available 
	#### Accesses complete data available from location provided

	files = []
	for root, dirs, fs in os.walk(path_to_device):
		if fs:
			files.extend(os.path.join(root,f) for f in fs)

	if not files:
		print ('Cannot Access Files')
		raise IOError
	
	print ('Processing files...')

	files.sort()

	power = []
	for file in files:
		if file.endswith('.csv.gz'):
			pd_entries = pandas.read_csv(file, engine="python")
			try:
				power_sig = filter_data(pd_entries['POWER'])
			except KeyError:
				power_sig = filter_data(pd_entries['VALUE'])

			power.extend(power_sig)

			if day in file:
				current_power = power_sig
				break

	return power, current_power

def threshold_of_device(path_to_device, no_thresholds_required, day):

	t0 = time.clock()
	power, current_power = process_data(path_to_device, day)

	if no_thresholds_required == 1:
		threshold = var_round(get_otsus_threshold(power))
	else:
		threshold = get_jenks_threshold(power, no_thresholds_required)

	if no_thresholds_required == 1:
		
		p_th_ov = [p for p in power if p <= threshold]
		
		if len(p_th_ov)/len(power) > 0.55:
			threshold_day = var_round(get_otsus_threshold(current_power))

			if np.average(p_th_ov) < threshold_day or np.absolute(threshold-threshold_day)/threshold > 0.05:
				power_th = np.array([p for p in power if p <= threshold_day])
				x = np.array([[i] for i in range(0,len(power_th))])
				print ('RANSAC')
				threshold_temp = 0
				for iteration in range(5):
					#### ransac
					ransac = linear_model.RANSACRegressor()
					ransac.fit(x, power_th)
					new_y = ransac.predict(x)

					#### lse
					min_lse = np.inf
					for y in new_y:
						lse = np.average((power_th-y)**2)
						if min_lse > lse:
							min_lse = lse
							threshold = y

					threshold_temp = threshold_temp + threshold
				threshold = var_round(threshold_temp/5)
		
	print (threshold)
	print ('Took ', time.clock()-t0, 's')
	return threshold


if __name__ == '__main__':

	th = threshold_of_device('/media/milan/DATA/Qrera/Cannula', 1, '2018_04_26')
	
