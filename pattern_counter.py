#! /usr/bin/env python3

import pattern_classification as ptm
import custompattern_mining as cpm
import pattern_discovery as ptd
import matplotlib.pyplot as plt
# import pattern_matching as ptm
import plotly.graph_objs as go
import peak_detector as pd 
import numpy as np
import plotly
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
	device_path = '/media/milan/DATA/Qrera/FWT/5CCF7FD0C7C0'
	day = '2018_07_07'

	# device_path = '/media/milan/DATA/Qrera/PYN/B4E62D'
	# day = '2018_09_18'
	
	# device_path = '/media/milan/DATA/Qrera/AutoAcc/39FFBE'
	# day = '2018_04_27' #'2017_12_09'
	
	file1, file2 = get_required_files(device_path, day)

	# file1 = 'test_data.csv'
	# file2 = None

	power_f = pd.preprocess_power(file1, file2)
	final_peaks, peak_indices = pd.detect_peaks(power_f)

	final_pattern = None
	no_iter = 1
	while final_pattern == None:
		array, state_attributes = pd.peaks_to_discrete_states(final_peaks)
		with open('state_attributes.json', 'w') as f:
			json.dump(state_attributes, f)

		pm = ptd.PatternDiscovery(array, state_attributes, 3,7)
		final_pattern = pm.discover_pattern()
		print (final_pattern)
		no_iter += 1
		if no_iter >= 5:
			raise ValueError('Could not find valid pattern. Try again! Or-> Check if min_length of pattern is too small. Check if number of segments are  suitable for data.')

	# p_m = ptm.PatternMatching(pm.pattern_dict, state_attributes, array,3,7, peak_indices)
	p_m = ptm.PatternClassification(pm.classification_dict,state_attributes,array,3,7)
	p_array, p_indices = p_m.find_matches()

	print (len(p_indices))
	print (len(peak_indices))

	print ('Mapping time indices...')
	simplified_seq = np.zeros((len(power_f)))
	start_ind = 0
	for e,i in enumerate(p_indices):
		simplified_seq[start_ind:peak_indices[i+1]] = p_array[e]
		start_ind = peak_indices[i]
	
	print ('Plotting...')
	unique_labels = list(np.unique(simplified_seq))
	y_plot = np.zeros((len(unique_labels),len(simplified_seq)))
	for e,el in enumerate(simplified_seq):
		y_plot[unique_labels.index(int(el)),e] = power_f[e]
	time = np.arange(len(power_f))
	
	plotly.tools.set_credentials_file(username='MilanMariyaTomy', api_key= '8HntwF4rtsUwPvjW3Sl4')
	data = [go.Scattergl(x=time, y=y_plot[i,:]) for i in range(len(unique_labels))]
	pattern_edges = len(time)*[None]
	for ind in peak_indices[p_indices]:
		pattern_edges[ind] = power_f[ind]
	print (len([l for l in pattern_edges if l != None]))
	data.append(go.Scattergl(x=time,y=pattern_edges,mode='markers'))
	fig = go.Figure(data = data)
	plotly.plotly.plot(fig, filename='fwtc_pattern_counting')