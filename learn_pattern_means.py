#! /usr/bin/env python3

from sklearn.naive_bayes import GaussianNB
from datetime import datetime, timedelta
import pattern_length
import fnmatch
import pandas
import json
import os


def initial_processing(f1,f2):
	print ('Preprocessing files to extract data...')

	if f2 == None:
		df = pandas.read_csv(f1)
		df.sort_values(by='TS')
		start_time = datetime.isoformat(datetime.strptime(f1[-17:-7]+' 07:00:00', '%Y_%m_%d %H:%M:%S'), sep=' ')
		end_time = datetime.isoformat(datetime.strptime(f1[-17:-7]+' 23:59:59', '%Y_%m_%d %H:%M:%S'), sep=' ')
	else:
		df1 = pandas.read_csv(f1)
		df2 = pandas.read_csv(f2)
		df1.sort_values(by='TS')
		df2.sort_values(by='TS')
		df = pandas.concat([df1,df2])
		start_time = datetime.isoformat(datetime.strptime(f1[-17:-7]+' 07:00:00', '%Y_%m_%d %H:%M:%S'), sep=' ')
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

	device_path = '/media/milan/DATA/Qrera/PYN/B4E62D388561'
	training_file = '/media/milan/DATA/Qrera/trials/data/mean_dict.json'

	### One day
	day = '2018_11_02'
	file1, file2 = get_required_files(device_path, day)
	power_df = initial_processing(file1, file2)
	pl = pattern_length.PatternLength(power_df, 86400, 5, 30, 3)
	mean_dict = pl.get_mean_dictionary()
	print (mean_dict)

	with open (training_file, 'w') as f:
		json.dump(mean_dict,f)

	print ('reading dictionary')
	
	print (new_dic)

	#### Over multiple days 
	for root, dirs, fs in os.walk(device_path):
		if fs:
			files.extend(os.path.join(root,f) for f in fs if f.endswith('.csv.gz') and fnmatch.fnmatch(f,"*_*_*"))

	i = 0
	for file in files:
		print (file[-17:-7])
		i +=1
		if os.path.isfile(training_file):
			with open(training_file, 'r') as f:
				mean_dict = json.load(f)

			file1, file2 = get_required_files(device_path, file[-17:-7])
			power_df = initial_processing(file1, file2)
			pl = pattern_length.PatternLength(power_df, 86400, 5, 30, 3)
			mean_dict_day = pl.get_mean_dictionary()
			
			if mean_dict_day != None:
				for region in mean_dict:
					mean_dict[region].extend(mean_dict_day[region])
		
		else:
			file1, file2 = get_required_files(device_path, day)
			power_df = initial_processing(file1, file2)
			pl = pattern_length.PatternLength(power_df, 86400, 5, 30, 3)
			mean_dict = pl.get_mean_dictionary()
			print (mean_dict)

		with open(training_file, 'w') as f:
			json.dump(mean_dict, f)

		if i > 3:
			break