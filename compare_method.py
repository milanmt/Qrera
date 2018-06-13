#! /usr/bin/env python

import os
import pandas as pd
import numpy as np

if __name__ == '__main__':

	df1 = pd.read_csv('paragon_filtered_otsus.csv', header=None, names=['Threshold', 'Day'])
	otsus_thresh = df1['Threshold']
	otsus_fname = df1['Day']
	hard_thresh = 30000
	error_count = []
	days = []
	fp_list = []
	fn_list = []
	tp_l = []
	tn_l = []

	for i in range(len(otsus_fname)):
		file = otsus_fname[i]+'.csv.gz'
		df2 = pd.read_csv(file)
		power = df2['POWER']

		error = []
		fp = 0
		tp = 0
		fn = 0
		tn = 0

		for p in power:
			if p < hard_thresh:
				ht_out = 0
			else:
				ht_out = 1

			if p < otsus_thresh[i]:
				ot_out = 0
			else:
				ot_out = 1

		
			if ht_out != ot_out:
				error_count.append(1)
				error.append(1)
				if ht_out == 0 and ot_out == 1:
					fp = fp +1
				elif ht_out == 1 and ot_out == 0:
					fn = fn+1

			else:
				error_count.append(0)
				error.append(0)
				if ht_out == 0 and ot_out ==0:
					tn = tn+1
				elif ht_out == 1 and ot_out == 1:
					tp = tp+1 


		tp_l.append(tp)
		tn_l.append(tn)
		fp_l.append(fp)
		fn_l.append(fn)

		fp_rate = float(fp)/(fp+tn)*100
		fn_rate = float(fn)/(fn+tp)*100
		percent = float(np.mean(error))*100

		fp_list.append(fp_rate)
		fn_list.append(fn_rate) 
		days.append(file[-17:-7])

		print file[-17:-7]		
		print 'Threshold', otsus_thresh[i], percent
		print 'fp', fp_rate
		print 'fn', fn_rate


	print 'Overall Error Percent', np.mean(error_count)
	print 'Overall Misclassified Working', float(sum(fn_l))/(sum(fn_l)+sum(tp_l))*100
	print 'Overall Misclassified Idle', float(sum(fp_l))/(sum(fp_l)+sum(tn_l))*100

	df = pd.DataFrame(data = zip(days, otsus_thresh, fp_list, fn_list), columns=['Day', 'OtsusTh', 'FP', 'FN'])
	df.to_csv('paragon_otsus_thresh_analysis.csv', header=True, index=False)






