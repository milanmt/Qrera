#! /usr/bin/env python3

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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

def preprocess_power(f1, f2):
	print ('Preprocessing files to extract data...')

	if f2 == None:
		df = pd.read_csv(f1)
		df.sort_values(by='TS')
		start_time = datetime.isoformat(datetime.strptime(f1[-17:-7]+' 08:30:00', '%Y_%m_%d %H:%M:%S'), sep=' ')
		end_time = datetime.isoformat(datetime.strptime(f1[-17:-7]+' 11:59:59', '%Y_%m_%d %H:%M:%S'), sep=' ')
	else:
		df1 = pd.read_csv(f1)
		df2 = pd.read_csv(f2)
		df1.sort_values(by='TS')
		df2.sort_values(by='TS')
		df = pd.concat([df1,df2])
		start_time = datetime.isoformat(datetime.strptime(f1[-17:-7]+' 08:30:00', '%Y_%m_%d %H:%M:%S'), sep=' ')
		end_time = datetime.isoformat(timedelta(days=1) + datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S'), sep=' ')

	df = df[(df['TS'] >= start_time) & (df['TS'] < end_time)]
	df['TS'] = df['TS'].apply(lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))))

	power = np.zeros((86400))
	power[0] = df.iloc[0,0]
	offset = int(df.iloc[0,1])
	t = offset
	for i in range(1,df.shape[0]):
		if int(df.iloc[i,1]) != t:
			if round(df.iloc[i,1]-t) == 1.0:
				power[t+1-offset] = df.iloc[i,0]
				t+=1

			else:
				orig_t = t
				req_offset = orig_t+1-offset
				for j in range(int(df.iloc[i,1]-orig_t)):
					power[req_offset+j] = 0
					t+=1
	return power	

def calculate_efficiency(power, threshold):
	W = 0
	for p in power:
		if p >= threshold:
			W+=1
	return W/86400





if __name__ == '__main__':
	day = '2018_07_31'
	device_path = '/media/milan/DATA/Qrera/KakadeLaser/B4E62D38855E'
	threshold = 28000
	f1, f2 = get_required_files(device_path, day)
	power = preprocess_power(f1,f2)
	efficiency = calculate_efficiency(power, threshold)
	print (efficiency)
