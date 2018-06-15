#! /usr/bin/env python

import os
import time
import pandas
import jenkspy 
import numpy as np


def get_otsus_threshold(power):
	print 'Calculating threshold...'
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
	print 'Calculating threshold...'
	return jenkspy.jenks_breaks(power, nb_class=int(no_thresholds_required)+1)[1:-1]


def var_round(number):
	number = float(number)
	
	if number/10 <= 10:
		return number
	elif number/10 <= 1000:
		return round(number, -1)
	else:
		return round(number, -2)


def filter_data(power_sig):
	print 'Filtering Data...'
	
	## Smoothing Filter
	power_smoothed = []
	for i in range(len(power_sig)-299):
		if i%300 == 0:
			power_smoothed.append(var_round(np.mean(power_sig[i:i+300])))

	return power_smoothed


def access_data(path_to_device):
	#### Change depending on where data is available 
	#### Accesses complete data available from location provided

	files = []
	for root, dirs, fs in os.walk(path_to_device):
		if fs:
			files.extend(os.path.join(root,f) for f in fs)
	
	print 'Reading files...'

	power_raw = []
	for file in files:
		if file.endswith('.csv.gz'):
			pd_entries = pandas.read_csv(file, engine="python")
			try:
				power_sig = pd_entries['POWER']
			except KeyError:
				power_sig = pd_entries['VALUE']

			power_raw.extend(power_sig)

	return power_raw

def threshold_of_device(path_to_device, no_thresholds_required):

	t0 = time.clock()
	power_raw = access_data(path_to_device)
	power = filter_data(power_raw)

	if no_thresholds_required == 1:
		threshold = get_otsus_threshold(power)
	else:
		threshold = get_jenks_threshold(power, no_thresholds_required)
				
	print threshold
	print 'Took ', time.clock()-t0, 's'
	return threshold


if __name__ == '__main__':

	th = threshold_of_device('/media/milan/DATA/Qrera/AutoAcc', 1)
	
	