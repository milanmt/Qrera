#! /usr/bin/env python3

import os
import time
import pandas
import jenkspy 
import numpy as np
import find_threshold


if __name__ == '__main__':
	
	company_path = '/media/milan/DATA/Qrera/Cannula'
	files = []

	for root, dirs, fs in os.walk(company_path):
		if fs:
			files.extend(os.path.join(root,f) for f in fs)

	threshold_days = []
	days = []
	
	
	for file in files:
		print (file[-17:-7])
		t0 = time.clock()
		threshold = find_threshold.threshold_of_device(company_path, 1, file[-17:-7])
		print (threshold)

		threshold_days.append(threshold)
		days.append(file[:-7])
		print ('Time in secs: ', time.clock()- t0)


	df = pandas.DataFrame( data = list(zip(threshold_days, days)), columns = ['Threshold', 'Day'])
	df.to_csv('cannula_filtered_otsus.csv', index= True, header=True)




