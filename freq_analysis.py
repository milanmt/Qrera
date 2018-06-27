#! /usr/bin/env python3

# from scipy.cluster.hierarchy import fcluster, linkage
# from scipy.spatial.distance import pdist
from scipy.signal import periodogram
from scipy.fftpack import fft 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import jenkspy 
import time
import math

SMOOTHING_FILTER_WINDOW = 10
THRESHOLD = 2000
MIN_IDLE_TIME = 74  # seconds
MIN_WORK_TIME = 74  # seconds 
Fs = 1 				# Sampling Frequency
min_zeros = 2       # FT grouping 


def var_round(number):
	number = float(number)
	
	if number/10 <= 10:
		return number
	elif number/10 <= 1000:
		return round(number, -1)
	else:
		return round(number, -2)

def preprocess_power(df):
	print ('Preprocessing...')
	df.sort_values(by='TS')	
	df['TS'] = df['TS'].apply(lambda x: int(time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))))

	power = []
	for i in range(df.shape[0]):
		if not power:
			power.append(df.iloc[i,0])
		else:
			if df.iloc[i,1]-df.iloc[i-1,1] == 1.0:
				power.append(df.iloc[i,0])
		
			elif df.iloc[i,1] == df.iloc[i-1,1]:
				power[i-1] = (df.iloc[i-1,0]+df.iloc[i,0])/2

			elif df.iloc[i,1]-df.iloc[i-1,1] > 1.0:
				for j in range(int(df.iloc[i,1]-df.iloc[i-1,1])):
					power.append(df.iloc[i-1,0])


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


	## Without splitting 
	working = dict()
	power = pd.Series(power).apply(lambda x: x if x > THRESHOLD else 0)
	working.update({0:power})

	
	## Removing the mean of the signal
	for ind in working.keys():
		mean_pdf = np.mean(working[ind])
		p = working[ind] - mean_pdf
		working.update({ ind : p})

	### Smoothing Filter 
	for ind in working.keys():
		p_wrk = working[ind]
		for x in range(p_wrk.shape[0]):
			if x < p_wrk.shape[0]-SMOOTHING_FILTER_WINDOW-1 :
				p_wrk[x] = var_round(np.mean(p_wrk[x:x+SMOOTHING_FILTER_WINDOW])) 
			else:
				var_round(np.mean(p_wrk[x:]))
		working.update({ ind : p})

	return working


def LCM(numbers):
	if isinstance(numbers[0], tuple):
		common_den = LCM([n[1] for n in numbers])
		mod_numbers = [(n[0]*(common_den/n[1]), common_den) for n in numbers]
		lcm = float(LCM([n[0] for n in mod_numbers]))/common_den

	else:
		lcm = numbers[0]
		for i in numbers[1:]:
			lcm = i*lcm/math.gcd(int(lcm),int(i))
	return round(lcm)




if __name__ == '__main__':


	file = '/media/milan/DATA/Qrera/AutoAcc/39FFBE/2017/12/2017_12_10.csv.gz'

	df = pd.read_csv(file)
	power_f = preprocess_power(df)

	fund_freqs = dict()

	for p in power_f.values():
		# plt.plot(np.arange(0,len(p)), p)
		# plt.show()

		# autocorr = np.correlate(p, p, mode='full')
		# plt.plot(np.arange(0,len(autocorr)), autocorr)
		# plt.show()
		print ('FFT')
		N = len(p)   ## No.of samples 
		nyquist = int(N//2)

		## Fourier Transform
		ft = pd.Series(np.absolute(np.fft.fft(p)[0:nyquist])) 

		## Filtering FT - 1	
		print ('FFT thresholding')
		th = jenkspy.jenks_breaks(ft, nb_class=2)[1]
		ft_f = ft.apply(lambda x: x if x > th else 0)

		## Filtering FT - 2
		print ('FFT grouping')
		ft_f = list(ft_f)
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
							
		# comp_periods = [(N,list(final_ft).index(x)) for x in final_ft if x != 0]
		# print (comp_periods)
		# fundamental_period = Fs*LCM(comp_periods)

		# print ('Time Period: ', fundamental_period)

		
		freqs = np.linspace(0, float(Fs)/2, N/2)
		freq_ind = [list(final_ft).index(x) for x in final_ft if x != 0]
		comp_freqs = [round(freqs[y],4) for y in freq_ind]
		print (comp_freqs)

		plt.figure(4)
		plt.stem(freqs, final_ft)
		plt.show()


	# 	for c_f in comp_freqs:
	# 		if c_f not in fund_freqs:
	# 			count = 1
	# 		else:
	# 			count = fund_freqs[c_f]
	# 			count = count + 1
	# 		fund_freqs.update({ c_f : count})

	# print (fund_freqs)

	