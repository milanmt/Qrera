#! /usr/bin/env python3

import os
import time
import pandas as pd
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
	power_smoothed = [var_round(sum(power_sig[i:i+FILTER_WINDOW])/FILTER_WINDOW) 
	for i in range(0,power_sig.shape[0],FILTER_WINDOW)]
	
	if not power_smoothed:
		return [0]
	else:
		return pd.Series(power_smoothed)


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

	power_f = []
	power = []
	for file in files:
		if file.endswith('.csv.gz'):
			pd_entries = pd.read_csv(file, engine="python")
			try:
				power_sig = filter_data(pd_entries['POWER'])
				power.extend([p for p in pd_entries['POWER']])
			except KeyError:
				power_sig = filter_data(pd_entries['VALUE'])
				power.extend([p for p in pd_entries['VALUE']])

			if not all(p == 0 for p in power_sig):
				power_f.extend(power_sig)

			if day in file:
				current_power_f = power_sig
				break

	return pd.Series(power_f), pd.Series(power), pd.Series(current_power_f)

def threshold_of_device(path_to_device, no_thresholds_required, day):

	t0 = time.clock()
	power_f, power, current_power_f = process_data(path_to_device, day)

	if no_thresholds_required == 1:
		threshold = var_round(get_otsus_threshold(power_f))
		print (threshold, 'overall threshold')
	else:
		threshold = get_jenks_threshold(power_f, no_thresholds_required)

	if no_thresholds_required == 1:
		
		p_th_ov = power[power <= threshold]

		# print (p_th_ov.shape[0]/power.shape[0])
		if round(p_th_ov.shape[0]/power.shape[0]) >= 0.5:
			
			p_th_ov_avg = np.mean(p_th_ov)
			threshold_day = var_round(get_otsus_threshold(current_power_f))
			# print(threshold_day, 'day threshold')
			# print(p_th_ov_avg, 'overall average')

			if round(p_th_ov.shape[0]/power.shape[0], 1) == 0.5:
				print ('fitting line to values below orig threshold and above avg')
				power_th = p_th_ov[p_th_ov > p_th_ov_avg]

			elif p_th_ov_avg < threshold_day and (threshold_day-p_th_ov_avg)/threshold_day > 0.1:
				print ('fitting line to values below avg of two thresholds')
				power_th = power[power <= (threshold_day+threshold)/2]
		
			else:
				print ('fitting line to values below orig threshold')
				power_th = p_th_ov
				
			x = np.array([[i] for i in range(0,power_th.shape[0])])
				

			print ('RANSAC')
			threshold_temp = 0
			for iteration in range(3):
				#### ransac
				try:
					ransac = linear_model.RANSACRegressor()						
					ransac.fit(x, power_th)
					new_y = ransac.predict(x)

					# plt.plot(x, new_y)
					# plt.scatter(x,power_th)
					# plt.show()

				except ValueError:
					new_y = [threshold]

				threshold_temp = threshold_temp + (new_y[0]+new_y[-1])/2
	
			threshold = var_round(threshold_temp/3)

		
	print (threshold)
	print ('Took ', time.clock()-t0, 's')
	return threshold


# if __name__ == '__main__':

	# th = threshold_of_device('/media/milan/DATA/Qrera/Aureate/0E59BE', 1, '2018_04_13')

	# th = threshold_of_device('/media/milan/DATA/Qrera/Cannula', 1, '2018_05_21')
	
