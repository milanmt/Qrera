#! /usr/bin/env python

import os
import time
import pandas
import jenkspy 
import numpy as np
import scipy.signal as sp
from scipy.spatial.distance import pdist
from sklearn.mixture import GaussianMixture
from scipy.cluster.vq import kmeans2, ClusterError
from scipy.cluster.hierarchy import fcluster, linkage

def get_otsus_threshold(power):
	print 'Calculating threshold...'
	power_val, power_count = np.unique(power, return_counts=True)
	prob = power_count/float(np.sum(power_count))

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

	return threshold


def var_round(number):
	number = float(number)
	
	if number/10 <= 10:
		return number
	elif number/10 <= 1000:
		return round(number, -1)
	else:
		return round(number, -2)



if __name__ == '__main__':
	
	company_path = '/media/milan/DATA/Qrera/trials/AutoAcc'
	files = []
	MAX = 0
	MIN = np.inf
	removed_files = []
	m_c = 0
	z_c = 0

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
			pd_entries = pandas.read_csv(file, engine="python")
			try:
				power_sig = pd_entries['POWER']
			except KeyError:
				power_sig = pd_entries['VALUE']



			##### PUTTING IN FILTERED SIGNAL INSTEAD OF ORIG

			## Zero Filter
			power_sig_nz = [] 
			for p in power_sig:
				if p > 0:
					power_sig_nz.append(p)

			## Smoothing Filter
			for i in range(len(power_sig_nz)-299):
				if i%300 == 0:
					power_day.append(var_round(np.mean(power_sig_nz[i:i+300])))

			power.extend(power_day)
			data_validity = True

			#### CHECKING VALIDITY OF DATA

			## std way 

	# 		mean = np.mean(power_day)
	# 		std = np.std(power_day)
	
	# 		extrema = list(sp.argrelextrema(power_count, np.greater, order=3))
	# 		peaks =  power_val[extrema[0]]
	# 		# print peaks 
			
	# 		if len(peaks) < 1:
	# 			z_c = z_c + 1
	# 			print  'zero peak', file

	# 		elif all(x >= (mean-std) and x<= (mean+std) for x in peaks):
	# 			m_c = m_c + 1
	# 			print file


	# print m_c, 'No.of files with single peaks'
	# print z_c, 'No.of files with zero peaks'

			## Gaussian Mixture

			# sig_in = np.array(map((lambda x : [x]), power_day))
			
			# n_components = np.arange(1, 3)
			# models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(sig_in) for n in n_components]
 
			# if models[0].bic(sig_in) <= models[1].bic(sig_in):
			# 	data_validity = False
			# 	print 'FILE BEING IGNORED'
			# 	print file[-17:-7]
			# else:
			# 	data_validity = True


			### KMeans Clustering
			
	# 		new_max = max(power_sig_nz)
	# 		if new_max >= MAX:
	# 			MAX = new_max
	# 		print 'MAX', MAX

	# 		new_min = min(power_sig_nz)
	# 		if new_min <= MIN:
	# 			MIN = new_min
	# 		print 'MIN', MIN
		
	# 		k_in = np.array(map((lambda x : [x]), power_sig_nz))
	# 		try:
	# 			clusters, labels = kmeans2(k_in, k=np.array([[MAX], [MIN]]), minit='matrix', missing='raise')
	# 			data_validity = True

	# 		except ClusterError:
	# 			data_validity = False
	# 			print 'FILE BEING IGNORED'
	# 			print file[-17:-7]
	# 			ig_count = ig_count+1

	# 			removed_files.append(file)


	# df = pandas.DataFrame(removed_files, columns=['f_path'])
	# df.to_csv('deleted_files.csv')

	# print ig_count, 'Files ignored'	
	# print 'Total: ' , len(files)

    ###### OTSUS THRESHOLD CALCULATION

			if data_validity == True:
				#### OTSUS
				threshold = get_otsus_threshold(power)
				print threshold

				threshold_days.append(threshold)
				days.append(file[:-7])

				print file[-17:-7]
				print 'Time: ', (time.clock()- t0)/60


				### JENKS
				# threshold = jenkspy.jenks_breaks(power, nb_class=3)
				# print threshold


				# th = 0
				# tc = 0
				# power_val, power_count = np.unique(power, return_counts=True)
				# for i in range(len(power_val)):
				# 	if power_val[i] >= threshold[1] and power_val[i] <= threshold[2]:
				# 		th = th + power_val[i]*power_count[i]
				# 		tc = tc + power_count[i]

				# threshold_days.append(var_round(th/tc))
				# days.append(file[:-7])

				# print file[-17:-7]
				# print var_round(th/tc)
				# print 'Time: ', (time.clock()- t0)/60

				#### checking extrema on either side of threshold
				# extrema = list(sp.argrelextrema(power_count, np.greater, order=3))
				# peaks =  power_val[extrema[0]]

				# if len(peaks) < 1:
				# 	print file 

				# elif all(p )


	

	df = pandas.DataFrame( data = list(zip(threshold_days, days)), columns = ['Threshold', 'Day'])
	df.to_csv('autoacc_filtered_otsus.csv', index= True, header=True)




