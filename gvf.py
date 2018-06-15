#! /usr/bin/env python

import os
import time
import pandas
import jenkspy 
import numpy as np
import matplotlib.pyplot as plt


def get_jenks_threshold(power, no_thresholds_required):
	print 'Calculating threshold...'
	return jenkspy.jenks_breaks(power, nb_class=int(no_thresholds_required)+1)

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
	## Zero Filter
	power_sig_nz = [] 
	for p in power_sig:
		if p > 0:
			power_sig_nz.append(p)

	## Smoothing Filter
	power_smoothed = []
	for i in range(len(power_sig_nz)-299):
		if i%300 == 0:
			power_smoothed.append(var_round(np.mean(power_sig_nz[i:i+300])))

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

	print 'Calculating threshold...'
	threshold = get_jenks_threshold(power, no_thresholds_required)
				
	print threshold
	print 'Took ', time.clock()-t0, 's'

	return threshold


if __name__ == '__main__':

	device_path = '/media/milan/DATA/Qrera/AutoAcc'

	power_raw = access_data(device_path)
	power = filter_data(power_raw)
	mean = np.mean(power_raw)

	SDAM = 0
	for p in power:
		SDAM = SDAM + (p -mean)*(p-mean)


	gvf = []
	for i in range(1, 21):
		print i 
		threshold = get_jenks_threshold(power, i)
		print threshold
		threshold[0] = 0
		num_of_classes = i+1
		p_dict = dict()

		for p in power:
			for c in range(num_of_classes):
				if p >= threshold[c] and p <= threshold[c+1]:
					if c not in p_dict:
						p_list =[p]
					else:
						p_list = p_dict[c]
						p_list.append(p)

					p_dict.update({ c : p_list})
					break


		SDBC = 0 
		for c in range(num_of_classes):
			c_mean = np.mean(p_dict[c])
			SDBC = SDBC + (c_mean - mean)*(c_mean - mean)
	    
		gvf.append(SDBC/SDAM)
		print SDBC/SDAM 


	plt.plot(range(1,21), gvf)
	plt.show()

