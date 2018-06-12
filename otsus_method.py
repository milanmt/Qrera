#! /usr/bin/env python

import os
import time
import pandas 
import numpy as np
import scipy.signal as sp
from sklearn.mixture import GaussianMixture
from scipy.cluster.vq import kmeans2, ClusterError


def var_round(number):
	number = float(number)
	
	if number/10 <= 10:
		return number
	elif number/10 <= 100:
		return round(number, -1)
	else:
		return round(number, -2)



if __name__ == '__main__':
	
	company_path = '/media/milan/DATA/Qrera/trials/Paragon'
	files = []
	MAX = 0
	MIN = np.inf
	removed_files = []

	for root, dirs, fs in os.walk(company_path):
		if fs:
			files.extend(os.path.join(root,f) for f in fs)

	power = []
	threshold_days = []
	days = []
	
	print 'Reading files...'
	ig_count = 0
	for file in files:
		t0 = time.clock()
		power_day = []
		if file.endswith('.csv.gz'):
			print file
			pd_entries = pandas.read_csv(file, engine="python")
			try:
				power_sig = pd_entries['POWER']
			except KeyError:
				power_sig = pd_entries['VALUE']

			# b,a = sp.butter(2, 0.0167)
			# filtered_power = sp.lfilter(b,a, power_sig)   ##### PUTTING IN FILTERED SIGNAL INSTEAD OF ORIG
			for i in range(len(power_sig)-299):
				if i%300 == 0:
					power_day.append(var_round(np.mean(power_sig[i:i+300])))


			# zero_padding = (86400-len(power_day))*[0]
			# power_day.extend(zero_padding)

			power.extend(power_day)
			power_day = np.array(power_day)
			power_val, power_count = np.unique(power_day, return_counts=True)
			prob = power_count/float(np.sum(power_count))
			data_validity = True

			# sig_in = np.array(map((lambda x : [x]), power_day))
			
			# n_components = np.arange(1, 3)
			# models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(sig_in) for n in n_components]
 
			# if models[0].bic(sig_in) <= models[1].bic(sig_in):
			# 	data_validity = False
			# 	print 'FILE BEING IGNORED'
			# 	print file[-17:-7]
			# else:
			# 	data_validity = True
			
			# new_max = max(power_val)
			# if new_max >= MAX:
			# 	MAX = new_max

			# new_min = min(power_val)
			# if new_min <= MIN:
			# 	MIN = new_min
		
			# k_in = np.array(map((lambda x : [x]), power_day))
			# try:
			# 	clusters, labels = kmeans2(k_in, k=np.array([[MAX], [MIN]]), minit='matrix', missing='raise')
			# 	data_validity = True

			# except ClusterError:
			# 	data_validity = False
			# 	print 'FILE BEING IGNORED'
			# 	print file[-17:-7]
			# 	ig_count = ig_count+1

			# 	removed_files.append(file)





	# df = pandas.DataFrame(removed_files, columns=['f_path'])
	# df.to_csv('deleted_files.csv')

	# print ig_count, 'Files ignored'	
	# print 'Total: ' , len(files)
			if data_validity == True:
				print 'Calculating threshold for day...'
				max_sigma = 0
				sigma_vals = []
				ind = []

				for i in range(len(power_val)):
					w0 = np.sum(prob[:i+1])
					w1 = np.sum(prob[i+1:])

					u0t = 0
					for j in range(i+1):
						u0t = u0t + power_val[j]*prob[j]/w0

					u1t = 0
					for j in range(i+1,len(power_val)):
						u1t  = u1t + power_val[j]*prob[j]/w1

					sigma = 1*(w0*w1*(u0t-u1t)*(u0t-u1t))

					if sigma >= max_sigma:
						max_sigma = sigma
						threshold = power_val[i]
				# 	sigma_vals.append(sigma)

				
				
				# sigma_ind = zip(sigma_vals,range(len(sigma_vals)))
				# # print sigma_ind
				# sigma_ind.sort(key= lambda x: x[0], reverse=True)
				# # print sigma_ind
				# max_sigma = sigma_ind[0][0]
				# for i in range(len(sigma_ind)):
				# 	c_sigma = sigma_ind[i][0]
				# 	if i !=0 :
				# 		change =  (max_sigma - c_sigma)/max_sigma
				# 		if change > 0.5:
				# 			break
				# 		else:
				# 			last_ind = i

				# sigma_ind[:i+1].sort(key= lambda x: prob[x[1]], reverse=True)
				# threshold = power_val[sigma_ind[0][1]]

				threshold_days.append(threshold)
				days.append(file[:-7])
				print file[-17:-7]
				print threshold
				print 'Time: ', (time.clock()- t0)/60

	

	df = pandas.DataFrame( data = list(zip(threshold_days, days)), columns = ['Threshold', 'Day'])
	df.to_csv('paragon_filtered_otsus.csv', index= True, header=True)

	t0 = time.clock()
	power = np.array(power)
	power_val, power_count = np.unique(power, return_counts=True)
	prob = power_count/float(np.sum(power_count))
	
	print 'Calculating overall threshold...'
	max_sigma = 0
	for i in range(len(power_val)):
		w0 = np.sum(prob[:i+1])
		w1 = np.sum(prob[i+1:])

		u0t = 0
		for j in range(i+1):
			u0t = u0t + power_val[j]*prob[j]/w0

		u1t = 0
		for j in range(i+1,len(power_val)):
			u1t  = u1t + power_val[j]*prob[j]/w1

		sigma = 1*(w0*w1*(u0t-u1t)*(u0t-u1t))

		if sigma >= max_sigma:
			max_sigma = sigma
			threshold = power_val[i]


	print threshold
	print 'Time: ', (time.clock()- t0)/60 