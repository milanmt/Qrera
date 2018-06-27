#! /usr/bin/env python3

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import jenkspy 
import time
import math

THRESHOLD = 2000
MIN_IDLE_TIME = 74  # seconds
MIN_WORK_TIME = 74  # seconds 
Fs = 1 				# Sampling Frequency
min_zeros = 5       # FT grouping 


def var_round(number):
	number = float(number)
	
	if number/10 <= 10:
		return number
	elif number/10 <= 1000:
		return round(number, -1)
	else:
		return round(number, -2)

def preprocess_power(power_df, time_df):
	print ('preprocess_power')
	
	pt = list(zip(power_df, time_df))
	pt.sort(key=lambda x: x[1])

	## Sorting and removing time anomalies
	time_raw = []
	power = []
	for i in range(len(pt)):
		if not time_raw and not power:
			time_raw.append(pt[i][1])
			power.append(pt[i][0])
		else:
			if pt[i][1]-pt[i-1][1] == 1.0:
				time_raw.append(pt[i][1])
				power.append(pt[i][0])
		
			elif pt[i][1] == pt[i-1][1]:
				power[i-1] = (pt[i-1][0]+pt[i][0])/2

			elif pt[i][1]-pt[i-1][1] > 1.0:
				for j in range(int(pt[i][1]-pt[i-1][1])):
					time_raw.append(pt[i-1][1]+j+1)
					power.append(pt[i-1][0])

	## Splitting signal into working regions 
	# working = []
	# last_index = -1
	# first_index = -1
	# for j in range(len(power)):
	# 	if power[j] > THRESHOLD: 
	# 		if first_index == -1 :
	# 			first_index = j
	# 		last_index = j 
	# 	else:
	# 		if j-last_index > MIN_IDLE_TIME and last_index != -1 and first_index != -1:
	# 			if len(power[first_index:last_index+1]) >= MIN_WORK_TIME:
	# 				working.append(power[first_index:last_index+1])
	# 			first_index = -1
	# 	if j == len(power)-1 and j - last_index < MIN_IDLE_TIME:
	# 		working.append(power[first_index:])

	print ('work_filter')		

	## Without splitting 
	working = []
	last_index = -1
	first_index = -1
	work_filter = []
	for j in range(len(power)):
		if power[j] > THRESHOLD: 
			work_filter.append(power[j])
		else:
			work_filter.append(0)
	working.append(work_filter)
			
	
	print ('Smoothing Filter')
	### Smoothing Filter 
	power_f_set = []
	for p_wrk in working:
		power_f = []
		for i in range(len(p_wrk)):
			if i >= len(p_wrk) - 9:
				power_f.append(var_round(np.mean(p_wrk[i:])))
			else:
				power_f.append(var_round(np.mean(p_wrk[i:i+10])))
		power_f_set.append(power_f)
	

	print ('Mean removing')
	## Removing the mean of the signal
	power_dmf_set = []
	for power_f in power_f_set:
		power_dmf = map(lambda x: x-np.mean(power_f), power_f)
		print ('list conversion')
		power_dmf_set.append(list(power_dmf))

	return power_dmf_set


def LCM(numbers):
	print ('finding lcm')
	if isinstance(numbers[0], tuple):
		common_den = LCM([n[1] for n in numbers])
		mod_numbers = [(n[0]*(common_den/n[1]), common_den) for n in numbers]
		lcm = float(LCM([n[0] for n in mod_numbers]))/common_den

	else:
		lcm = numbers[0]
		for i in numbers[1:]:
			lcm = i*lcm/math.gcd(int(lcm),int(i))
	return round(lcm)

# def HCF(numbers):



if __name__ == '__main__':


	file = '/media/milan/DATA/Qrera/AutoAcc/39FFBE/2017/12/2017_12_10.csv.gz'

	df = pd.read_csv(file)
	power_df = np.array(df['POWER'])
	time_df = map(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')), df['TS'])	
	
	power_f = preprocess_power(power_df, time_df)
	
	
	fund_freqs = dict()

	print ('FFt')

	for p in power_f:
		# plt.plot(np.arange(0,len(p)), p)
		# plt.show()

		# autocorr = np.correlate(p, p, mode='full')
		# plt.plot(np.arange(0,len(autocorr)), autocorr)
		# plt.show()

		N = len(p)   ## No.of samples 
		nyquist = int(N//2)

		ft = np.absolute(np.fft.fft(p)[0:nyquist])
		# plt.stem(range(len(ft)), ft)
		# plt.show()
		
		th = jenkspy.jenks_breaks(ft, nb_class=2)[1]

		ft_f = []
		for val in ft:
			if val > th:
				ft_f.append(val)
			else:
				ft_f.append(0)

		# plt.stem(range(len(ft_f)), ft_f)
		# plt.show()


		##### crude grouping 


		final_ft = np.zeros((len(ft_f)))
		end_ind = 0
		for i in range(len(ft_f)):
			if ft_f[i] == 0 and i >= end_ind:
				try:
					ind_nz = ft_f[i+1:].index(0) +i+1
				except ValueError:
					break

				if abs(ind_nz-i) != 1:
					
					check_zero = ft_f[ind_nz: ind_nz+min_zeros]
					if all(f == 0 for f in check_zero):
						end_ind = ind_nz
					else:
						for k in range(ind_nz,len(ft_f[ind_nz:])):
							if np.sum(ft_f[k:k+min_zeros]) == 0:
								end_ind = k
								break

					weighted_avg = 0
					for j in range(i+1,end_ind):
						weighted_avg = weighted_avg + j*ft_f[j]
					f = int(round(weighted_avg/np.sum(ft_f[i+1:end_ind])))

					if ft_f[f] == 0 :
						f = max(range(i+1,end_ind), key=lambda x: ft_f[x])
					
					final_ft[f] = ft_f[f]
							
		comp_periods = [(N,list(final_ft).index(x)) for x in final_ft if x != 0]
		print (comp_periods)
		fundamental_period = Fs*LCM(comp_periods)

		print ('Time Period: ', fundamental_period)

		# plt.stem(range(len(ft_f)), final_ft)
		# plt.show()

	# 	freqs = np.linspace(0, float(Fs)/2, N/2)
	# 	freq_ind = [list(final_ft).index(x) for x in final_ft if x != 0]
	# 	comp_freqs = [round(freqs[y],3) for y in freq_ind]
	# 	print (comp_freqs)

	# 	for c_f in comp_freqs:
	# 		if c_f not in fund_freqs:
	# 			count = 1
	# 		else:
	# 			count = fund_freqs[c_f]
	# 			count = count + 1
	# 		fund_freqs.update({ c_f : count})

	# print (fund_freqs)




		



	
