#! /usr/bin/env python3

from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.naive_bayes import GaussianNB
from datetime import datetime, timedelta
import plotly.graph_objs as go
import numpy as np
import pandas
import plotly
import json
import os

def timing_wrapper(func):
	def wrapper(*args,**kwargs):
		t = datetime.now()
		func_val = func(*args,**kwargs)
		time_taken = datetime.now() -t
		print (str(func),' took: ', time_taken)
		return func_val
	return wrapper

@timing_wrapper
def preprocess_power(df):
	print ('Preprocessing power...')
	### Preprocessing
	df['TS'] = df['TS'].apply(lambda x: int(datetime.timestamp(datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))))
	# print (df.shape[0], 'orig_signal length')
	power = np.zeros((86400))
	power[0] = df.iloc[0,0]
	offset = int(df.iloc[0,1])
	t = offset
	for i in range(1,df.shape[0]):
		if int(df.iloc[i,1]) != t:
			if round(df.iloc[i,1]-t) == 1.0:
				power[t+1-offset] = df.iloc[i,0]
				t+=1			
			elif int(df.iloc[i,1])-t < 21.0:
				orig_t = t
				req_offset = orig_t+1-offset
				avg = (df.iloc[i,0]+df.iloc[i-1,0])/2
				for j in range(int(df.iloc[i,1]-orig_t)):
					power[req_offset+j] = avg
					t+=1
			else:
				orig_t = t
				req_offset = orig_t+1-offset
				for j in range(int(df.iloc[i,1]-orig_t)):
					power[req_offset+j] = 0
					t+=1
		else: 
			power[t-offset] = (power[t-offset]+df.iloc[i,0])/2
	return power

@timing_wrapper
def get_classifier(training_file):
	print ('Extracting classfier from json file...')
	with open(training_file, 'r') as f:
		mean_dict = json.load(f)

	target_vals = []
	input_vals = []

	targets = [region for region in mean_dict.keys()]

	for region in mean_dict:
		region_ind = targets.index(region)
		for val in mean_dict[region]:
			target_vals.append(region_ind)
			input_vals.append(val)

	X = np.array(input_vals).reshape(-1,1)
	Y = np.array(target_vals)
	# classifier = GaussianNB()
	classifier = KNeighborsClassifier(n_neighbors=1)
	classifier.fit(X,Y)
	return classifier, targets

@timing_wrapper
def get_mv_classifier(training_file):
	print ('Extracting classfier from json file...')
	with open(training_file, 'r') as f:
		mv_dict = json.load(f)

	target_vals = []
	input_vals = []

	targets = [region for region in mv_dict.keys()]

	for region in mv_dict:
		region_ind = targets.index(region)
		for val in mv_dict[region]:
			target_vals.append(region_ind)
			input_vals.append(val)

	X = np.array(input_vals)
	Y = np.array(target_vals)
	# classifier = GaussianNB()
	classifier = KNeighborsClassifier(n_neighbors=5)
	classifier.fit(X,Y)
	# print (classifier.theta_)
	# print (targets, 'targets')
	return classifier, targets

def initial_processing(f1):
	print ('Preprocessing files to extract data...')
	df = pandas.read_csv(f1)
	df.sort_values(by='TS')
	df = df.drop_duplicates(subset=['TS'], keep='first')
	return df

def get_required_file(device_path, day):
	print ('Obtaining Required Files...')
	date_str = day.split('_')
	file1 = device_path+'/'+date_str[0]+'/'+date_str[1]+'/'+day+'.csv.gz'
	if os.path.isfile(file1):
		return file1
	else:
		raise ValueError("File not available")

@timing_wrapper
def get_state_means(device_path, day):
	print ('Discretisings states...')
	start_day = (datetime.strptime(day + ' 00:00:00', '%Y_%m_%d %H:%M:%S') - timedelta(weeks=1))
	req_day = start_day
	for i in range(7):
		req_day_str = req_day.strftime('%Y_%m_%d')
		fname = device_path+'/'+str(req_day.year)+'/'+str(req_day.month)+'/'+req_day_str+'.csv.gz'
		if not os.path.isfile(fname):
			raise IOError('File does not exist. Cannot train classifier without data.')

		df = pandas.read_csv(fname)
		try:
			df_power = df['POWER']
		except KeyError:
			df_power = df['VALUE']

		if req_day_str == start_day.strftime('%Y_%m_%d'):
			power_array = np.array(df_power)
		else:
			power_array = np.append(power_array, np.array(df_power))

		req_day = req_day + timedelta(days=1)

	X = power_array.reshape(-1,1)
	dpgmm = BayesianGaussianMixture(n_components=10,max_iter= 100,random_state=0).fit(X)
	state_means = [x[0] for x in dpgmm.means_]
	return dpgmm, state_means
	

def get_current_status(log_path):
	if os.path.isfile(log_path):
		with open(log_path, 'r') as f:
			status_dict = json.load(f)
	else:
		status_dict = dict()
		status_dict['current_status'] = 0
		status_dict['Working'] = 0
		status_dict['Loading'] = 0
		status_dict['Idle'] = 0
		status_dict['Off'] = 0
		status_dict['notification_sent'] = False
		with open(log_path, 'w') as f:
			json.dump(status_dict, f, indent=4)
	return status_dict

def update_status_log(log_path, detected_region):
	status_dict = get_current_status(log_path)
	if status_dict['current_status'] == 0:
		status_dict['current_status'] = detected_region
		status_dict[detected_region] = 1
	else:
		prev_stat = status_dict['current_status']
		if prev_stat == detected_region:
			status_dict[detected_region] += 1
		else:
			status_dict[prev_stat] = 0
			status_dict['current_status'] = detected_region
			status_dict[detected_region] = 1
			status_dict['notification_sent'] = False
	return status_dict


if __name__ == '__main__':

	
	device_path = '/media/milan/DATA/Qrera/PYN/B4E62D388561'
	day = '2018_11_02'
	file1 = get_required_file(device_path, day)
	power_df = initial_processing(file1)


	## THRESHOLD
	##################################################################################################################################################################################################################
	Th1 = 2000   # Distinguish between idle and working
	Th2 = 1000   # Distinguish between loading and idle
	expected_loading_time = 4 # Minutes 
	expected_idle_time = 4    # Minutes
	notifier_log_path = 'data/notifier_log_path.json'

	### Classification over 60 sec intervals
	print ('Classifying signal over 60 sec intevrals...')
	classified_signal_id = []
	region_labels = ['Working', 'Loading', 'Idle', 'Off']
	start_int = datetime.strptime(day+' 00:00:00', '%Y_%m_%d %H:%M:%S')
	for i in range(1440):
		# print (i,end='\r')
		end_int = start_int + timedelta(minutes=1)   ## Window Required
		start_str = datetime.isoformat(start_int, sep=' ')
		end_str = datetime.isoformat(end_int, sep=' ')
		req_df = power_df[(power_df['TS']>=start_str) & (power_df['TS']<end_str)]
		if req_df.empty:
			label = 3
		else:
			try:
				avg_power = req_df['POWER'].mean()
			except KeyError:
				avg_power = req_df['VALUE'].mean()
			
			if avg_power == 0.0:
				label = 3
			elif avg_power >= Th1:
				label = 0
			else:
				if avg_power >= Th2:
					label = 1
				else:
					label = 2

			status_dict = update_status_log(notifier_log_path, region_labels[label])
			# print (status_dict)
			if status_dict['Loading'] > expected_loading_time and not status_dict['notification_sent']:
				print ('NOTIFICATION FOR EXCEEDING EXPECTED LOADING TIME')
				status_dict['notification_sent'] = True
			if status_dict['Idle'] > expected_idle_time and not status_dict['notification_sent']:
				print ('NOTIFICATION FOR EXCEEDING EXPECTED IDLE TIME')
				status_dict['notification_sent'] = True

			with open(notifier_log_path, 'w') as f:
				json.dump(status_dict, f, indent=4)

		classified_signal_id.append(label)
		start_int = end_int

	### ML 
	##################################################################################################################################################################################################################
	# training_file = '/media/milan/DATA/Qrera/trials/data/mean_dict.json'

	# ### Discretise over trained power signals (One weeks data)
	# dpgmm, state_means = get_state_means(device_path, day)

	# ### Get required classification from training data
	# # region_classifier, region_labels = get_mv_classifier(training_file)
	# region_classifier, region_labels = get_classifier(training_file)
	
	# ### Classification over 60 sec intervals
	# print ('Classifying signal over 60 sec intevrals...')
	# classified_signal = []
	# classified_signal_id = []
	# start_int = datetime.strptime(day+' 00:00:00', '%Y_%m_%d %H:%M:%S')
	# for i in range(1440):
	# 	print (i,end='\r')
	# 	end_int = start_int + timedelta(minutes=1)   ## Window Required
	# 	start_str = datetime.isoformat(start_int, sep=' ')
	# 	end_str = datetime.isoformat(end_int, sep=' ')
	# 	req_df = power_df[(power_df['TS']>=start_str) & (power_df['TS']<end_str)]
	# 	if req_df.empty:
	# 		region = 'Off'
	# 		label = len(region_labels)
	# 	else:
	# 		try:
	# 			state_power = dpgmm.predict(np.array(req_df['POWER']).reshape(-1,1))
	# 			sum_power = req_df['POWER'].sum()
	# 		except KeyError:
	# 			state_power = dpgmm.predict(np.array(req_df['VALUE']).reshape(-1,1))
	# 			sum_power = req_df['VALUE'].sum()

	# 		avg_power = np.mean([state_means[s] for s in state_power])
	# 		# print (avg_power, 'avg')
	# 		var_power = np.std([state_means[s] for s in state_power])
	# 		# print (var_power, 'std')
			
	# 		if sum_power == 0.0:
	# 			region = 'Off'
	# 			label = len(region_labels)
	# 		else:
	# 			# region_id = region_classifier.predict(np.array([[avg_power,var_power]]))
	# 			region_id = region_classifier.predict(np.array([[avg_power]]))
	# 			region = region_labels[int(region_id)]
	# 			label = int(region_id)
	# 			# print (region, 'classified')

	# 	classified_signal.append(region)
	# 	classified_signal_id.append(label)
	# 	start_int = end_int

	# print ('Plotting...')
	# unique_labels = list(np.unique(classified_signal_id))
	# print (unique_labels, region_labels)
	# power = preprocess_power(power_df)
	# y_plot = np.zeros((len(unique_labels),len(power)))
	# for e,el in enumerate(classified_signal_id):
	# 	y_plot[unique_labels.index(el),e*60:(e+1)*60] = power[e*60:(e+1)*60]
	# time = np.arange(86400)
	
	# plotly.tools.set_credentials_file(username='MilanMariyaTomy', api_key= '8HntwF4rtsUwPvjW3Sl4')
	# data = [go.Scattergl(x=time, y=y_plot[i,:]) for i in range(len(unique_labels))]
	# fig = go.Figure(data = data)
	# plotly.plotly.plot(fig, filename='fwtc_pattern_counting')