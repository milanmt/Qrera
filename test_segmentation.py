#! /usr/bin/env python3

import signal_segmentation as SS
import signal_clustering as SC
import matplotlib.pyplot as plt
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

	no_segments = 3

	device_path = '/media/milan/DATA/Qrera/FWT/5CCF7FD0C7C0'
	day = '2018_07_09'

	# device_path = '/media/milan/DATA/Qrera/PYN/B4E62D'
	# day = '2018_09_18'
	
	# device_path = '/media/milan/DATA/Qrera/AutoAcc/39FFBE'
	# day = '2018_04_27' #'2017_12_09'
	
	file1, file2 = get_required_files(device_path, day)

	# file1 = 'test_data.csv'
	# file2 = None

	power = pd.preprocess_power(file1, file2)

	ss = SS.SignalSegmentation(3,7,1)  ### min_len, max_len, derivative order
	simplified_seq = ss.segment_signal(no_segments, power)
	
	# ss = SC.SignalClustering()
	# simplified_seq = ss.segment_signal(no_segments,power)
	
	print ('Plotting...')
	unique_labels = np.unique(simplified_seq)
	y_plot = np.zeros((len(unique_labels),len(simplified_seq)))
	for e,el in enumerate(simplified_seq):
		if el == no_segments-1:
			y_plot[int(el),e] = 1500
		else:
			y_plot[int(el),e] = power[e]
	time = np.arange(len(power))
	
	plotly.tools.set_credentials_file(username='MilanMariyaTomy', api_key= '8HntwF4rtsUwPvjW3Sl4')
	data = [go.Scattergl(x=time, y=y_plot[i,:]) for i in range(len(unique_labels))]
	fig = go.Figure(data = data)
	plotly.plotly.plot(fig, filename='fwtc_pattern_counting')