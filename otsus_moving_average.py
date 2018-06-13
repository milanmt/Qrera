#! /usr/bin/env python

import os
import time
import pandas as pd
import numpy as np
import scipy.signal as sp


if __name__ == '__main__':
	
	file = 'paragon_filtered_otsus.csv'

	
	mod_threshold = []
	
	df = pd.read_csv(file)
	otsus_thresholds = df['Threshold']
	fname = df['Day']
	moving_avg = 0
	days = []

	
	for i in range(len(fname)):
		moving_avg = moving_avg + otsus_thresholds[i]
		mod_threshold.append(round(moving_avg/(i+1)))
		days.append(fname[i][-10:])



############ COMPARISON
	hard_thresh = 30000.0
	error_percent = []
	fp_list = []
	fn_list = []
	tp_l = []
	tn_l = []
	fp_l = []
	fn_l = []

	for i in range(len(fname)):
		print i 
		file = fname[i]+'.csv.gz'
		df2 = pd.read_csv(file)
		power = df2['POWER']
		
		error_count = []
		fp = 0
		tp = 0
		fn = 0
		tn = 0

		for p in power:
			if p <= hard_thresh:
				ht_out = 0
			else:
				ht_out = 1

			if p <= mod_threshold[i]:
				mot_out = 0
			else:
				mot_out = 1

		
			if ht_out != mot_out:
				error_count.append(1)
				if ht_out == 0 and mot_out == 1:
					fp = fp +1
				elif ht_out == 1 and mot_out == 0:
					fn = fn+1

			else:
				error_count.append(0)
				if ht_out == 0 and mot_out ==0:
					tn = tn+1
				elif ht_out == 1 and mot_out == 1:
					tp = tp+1 

		fn_l.append(fn)
		fp_l.append(fp)
		tn_l.append(tn)
		tp_l.append(tp)

		print days[i]
		percent = float(np.sum(error_count))/len(error_count)
		error_percent.append(error_count)
		fp_percent = float(fp)/(fp+tn)*100
		fp_list.append(fp_percent)
		fn_percent = float(fn)/(fn+tp)*100
		fn_list.append(fn_percent) 

		
		print 'Mod Threshold', percent, mod_threshold[i]
		print 'fp', fp_percent
		print 'fn', fn_percent
		
	
	for error in error_percent:
		total_error = sum(error)
		total = len(error)
	print 'Avg Individiual Threshold Error', float(total_error)/total*100
	print 'Misclassified working category %', float(sum(fn_l))/(sum(fn_l)+sum(tp_l))*100
	print 'Misclassified idle category %', float(sum(fp_l))/(sum(fp_l)+sum(tn_l))*100
	# max_val = max(error_percent)
	# max_ind = error_percent.index(max_val)
	# day = days[max_ind]
	# print mod_threshold[max_ind]
	# print max_val, day

	df = pd.DataFrame(data = zip(days, mod_threshold, fp_list, fn_list), columns=['Day', 'Mod_OtsusTh', 'FP', 'FN'])
	df.to_csv('paragon_filtered_avg_otsus_thresh_analysis.csv', header=True, index=False) 