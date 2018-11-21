#! /usr/bin/env python3

from datetime import datetime, timedelta
import signal_segmentation as SS
import signal_clustering as SC
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import peak_detector as pd
import pattern_length 
import numpy as np
import pandas
import plotly
import json
import os

def initial_processing(f1,f2):
	print ('Preprocessing files to extract data...')

	if f2 == None:
		df = pandas.read_csv(f1)
		df.sort_values(by='TS')
		start_time = datetime.isoformat(datetime.strptime(f1[-17:-7]+' 08:30:00', '%Y_%m_%d %H:%M:%S'), sep=' ')
		end_time = datetime.isoformat(datetime.strptime(f1[-17:-7]+' 11:59:59', '%Y_%m_%d %H:%M:%S'), sep=' ')
	else:
		df1 = pandas.read_csv(f1)
		df2 = pandas.read_csv(f2)
		df1.sort_values(by='TS')
		df2.sort_values(by='TS')
		df = pandas.concat([df1,df2])
		start_time = datetime.isoformat(datetime.strptime(f1[-17:-7]+' 08:30:00', '%Y_%m_%d %H:%M:%S'), sep=' ')
		end_time = datetime.isoformat(timedelta(days=1) + datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S'), sep=' ')

	df = df[(df['TS'] >= start_time) & (df['TS'] < end_time)]
	df = df.drop_duplicates(subset=['TS'], keep='first')
	return df


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

	# device_path = '/media/milan/DATA/Qrera/FWT/5CCF7FD0C7C0'
	# day = '2018_07_07'

	device_path = '/media/milan/DATA/Qrera/PYN/B4E62D388561'
	day = '2018_10_15'
	# day = '2018_10_10'
	
	# device_path = '/media/milan/DATA/Qrera/AutoAcc/39FFBE'
	# day = '2018_04_27' #'2017_12_09'

	# device_path = '/media/milan/DATA/Qrera/UltimateAppliances/B4E62D38861F'
	# day = '2018_10_31'

	# device_path = '/media/milan/DATA/Qrera/UltimateAppliances/B4E62D3885EB'
	# day = '2018_11_02'
	
	file1, file2 = get_required_files(device_path, day)

	# file1 = 'test_data.csv'
	# file2 = None

################# using pattern_length
	power_df = initial_processing(file1, file2)
	pl = pattern_length.PatternLength(power_df, 5, 30, 3)
	cycle_time = pl.get_average_cycle_time()
	estimate_count = pl.get_estimate_count()

################# using signal segmentation class
	# power = pd.preprocess_power(file1, file2)
	# # plt.plot(power)
	# # plt.show()
	# ss = SS.SignalSegmentation(5,30,3)  ### min_len, max_len, derivative order
	# simplified_seq, segmented_regions = ss.segment_signal(power)
	# # ss.get_accurate_average_working_length()
	# ss.get_average_working_pattern_length()
	
	# print ('Plotting...')
	# unique_labels = list(np.unique(simplified_seq))
	# y_plot = np.zeros((len(unique_labels),len(simplified_seq)))
	# for e,el in enumerate(simplified_seq):
	# 	if el == no_segments-1:
	# 		y_plot[unique_labels.index(el),e] = 1500
	# 	else:
	# 		y_plot[int(el),e] = power[e]
	# time = np.arange(len(power))
	
	# plotly.tools.set_credentials_file(username='MilanMariyaTomy', api_key= '8HntwF4rtsUwPvjW3Sl4')
	# data = [go.Scattergl(x=time, y=y_plot[i,:]) for i in range(len(unique_labels))]
	# pattern_edges = len(time)*[None]
	# for ind in ss.peak_indices[ss.pattern_sequence_indices]:
	# 	pattern_edges[ind] = power[ind]
	# # print (len([l for l in pattern_edges if l != None]))
	# data.append(go.Scattergl(x=time,y=pattern_edges,mode='markers'))
	# fig = go.Figure(data = data)
	# plotly.plotly.plot(fig, filename='fwtc_pattern_counting')
	

########## Clustering 

	# ss = SC.SignalClustering()
	# simplified_seq = ss.segment_signal(no_segments,power)
	
######### Finding over all data 

	# DAY_ARR = []
	# TIME_ARR = []
	# for root, dirs, files in os.walk(device_path):
	# 	if files:
	# 		files.sort()
	# 		for f in files:
	# 			day = f[:10]
	# 			print ('##########################', day)
	# 			file1, file2 = get_required_files(device_path, day)
	# 			power_df = initial_processing(file1, file2)
	# 			try:
	# 				pl = pattern_length.PatternLength(power_df, 5, 15, 3)
	# 				cycle_time = pl.get_average_cycle_time()
	# 				DAY_ARR.append(day)
	# 				TIME_ARR.append(cycle_time)
	# 			except pattern_length.SinglePatternError:
	# 				continue
	# 			except ValueError:
	# 				continue
	
	# with open('pyn_avg_time2.txt', 'w') as wf:
	# 	for day,cycle_time in zip(DAY_ARR,TIME_ARR):
	# 		wf.write('{0} {1}mins \n'.format(day, cycle_time/60))
