#! /usr/bin/env python3

import os
import time
import pandas
import jenkspy 
import numpy as np
import find_threshold


if __name__ == '__main__':
	
	company_path = '/media/milan/DATA/Qrera/AutoAcc'
	files = []

	for root, dirs, fs in os.walk(company_path):
		if fs:
			files.extend(os.path.join(root,f) for f in fs if f.endswith('.csv.gz'))

	threshold_days = []
	days = []
	
	for file in files:
		print (file[-17:-7])
		t0 = time.clock()
		threshold, RANSAC = find_threshold.threshold_of_device(company_path, 1, file[-17:-7])

		## Repeat once to supress errors due to randomness
		if np.absolute(threshold - np.mean(threshold_days[-5:]))/np.mean(threshold_days[-5:]) > 0.1:
			threshold = find_threshold.threshold_of_device(company_path, 1, file[-17:-7])
		
		threshold_days.append(threshold)
		days.append(file[:-7])

	df = pandas.DataFrame( data = list(zip(threshold_days, days, ransac)), columns = ['Threshold', 'Day', 'RANSAC'])
	df.to_csv('autoacc_filtered_otsus.csv', index= True, header=True)




