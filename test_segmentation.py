#! /usr/bin/env python3

import signal_segmentation as SS
import matplotlib.pyplot as plt
import plotly.graph_objs as go
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
	# device_path = '/media/milan/DATA/Qrera/FWT/5CCF7FD0C7C0'
	# day = '2018_07_09'

	device_path = '/media/milan/DATA/Qrera/PYN/B4E62D'
	day = '2018_09_18'
	
	# device_path = '/media/milan/DATA/Qrera/AutoAcc/39FFBE'
	# day = '2018_04_27' #'2017_12_09'
	
	file1, file2 = get_required_files(device_path, day)

	# file1 = 'test_data.csv'
	# file2 = None

	ss = SS.SignalSegmentation(5,15,3)
	simplified_seq = ss.segment_signal(3, file1, file2)
	power_f = ss.power_f
	
	print ('Plotting...')
	unique_labels = list(np.unique(simplified_seq))
	y_plot = np.zeros((len(unique_labels),len(simplified_seq)))
	for e,el in enumerate(simplified_seq):
		y_plot[unique_labels.index(int(el)),e] = power_f[e]
	time = np.arange(len(power_f))
	
	plotly.tools.set_credentials_file(username='MilanMariyaTomy', api_key= '8HntwF4rtsUwPvjW3Sl4')
	data = [go.Scattergl(x=time, y=y_plot[i,:]) for i in range(len(unique_labels))]
	fig = go.Figure(data = data)
	plotly.plotly.plot(fig, filename='fwtc_pattern_counting')