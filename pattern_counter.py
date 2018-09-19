#! /usr/bin/env python3

import custompattern_mining as cpm
import pattern_discovery as ptd
import matplotlib.pyplot as plt
import pattern_matching as ptm
import peak_detector as pd 
from dtw import dtw
import numpy as np
import subprocess
import pandas 
import json
import os


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



if __name__ == '__main__':
	
	# array = [1, 2, 3, 3, 1, 2, 3, 5, 1, 2, 3, 1, 2, 3,5, 5, 5, 5, 1, 2, 2, 1, 2, 2, 3, 5, 1, 2, 3, 5, 5, 1, 1, 2, 2, 3, 3, 5, 1, 2, 3, 5]

	device_path = '/media/milan/DATA/Qrera/FWT/5CCF7FD0C7C0'
	day = '2018_07_18'
	# device_path = '/media/milan/DATA/Qrera/HiraAutomation/B4E62D388226'
	# day = '2018_06_25'
	file1, file2 = get_required_files(device_path, day)

	# file1 = 'test_data.csv'
	# file2 = None

	power_d, power_f = pd.preprocess_power(file1, file2)
	final_peaks, peak_indices = pd.detect_peaks(power_d, power_f)

	final_pattern = None
	no_iter = 1
	while final_pattern == None:
		array, state_attributes = pd.peaks_to_discrete_states(final_peaks)
		with open('state_attributes.json', 'w') as f:
			json.dump(state_attributes, f)

		pm = ptd.PatternDiscovery(array, state_attributes, 3,6)
		final_pattern = pm.discover_pattern()
		print (final_pattern)
		no_iter += 1
		if no_iter >= 5:
			raise ValueError('Could not find valid pattern. Try again! Or-> Check if min_length of pattern is too small. Check if number of segments are  suitable for data.')

	p_m = ptm.PatternMatching(pm.pattern_dict, state_attributes, array, 10)
	p_array = p_m.find_matches()

	# with open('state_attributes.json', 'r') as f:
	# 	state_attributes = json.load(f)
	
	# pattern = [1,2,3,5,1]

	# pattern_recognition(array, pattern, state_attributes)