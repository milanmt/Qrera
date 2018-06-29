#! /usr/bin/env python3

# from scipy.cluster.hierarchy import fcluster, linkage
# from scipy.spatial.distance import pdist
from scipy.signal import find_peaks, periodogram, welch
from fractions import Fraction
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import jenkspy 
import time
import math

SMOOTHING_FILTER_WINDOW = 10
FT_FILTER = 3
THRESHOLD = 2000
MIN_IDLE_TIME = 60  # seconds
MIN_WORK_TIME = 76  # seconds 
Fs = 1 				# Sampling Frequency
min_zeros = 2       # FT grouping 


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


	# Splitting signal into working regions 
	working = dict()
	last_index = -1
	first_index = -1
	count = 0
	for j in range(len(power)):
		if power[j] > THRESHOLD: 
			if first_index == -1 :
				first_index = j
			last_index = j 
		else:
			if j-last_index > MIN_IDLE_TIME and last_index != -1 and first_index != -1:
				if len(power[first_index:last_index+1]) >= MIN_WORK_TIME:
					working.update({ count : pd.Series(power[first_index:last_index+1])})
					count = count + 1
				first_index = -1
		if j == len(power)-1 and j - last_index < MIN_IDLE_TIME:
			working.update({ count : pd.Series(power[first_index:])})


	## Without splitting 
	# working = dict()
	# power = pd.Series(power).apply(lambda x: x if x > THRESHOLD else 0)
	# working.update({0:power})

	### Smoothing Filter 
	for ind in working.keys():
		p_wrk = working[ind]
		for x in range(p_wrk.shape[0]):
			if x < p_wrk.shape[0]-SMOOTHING_FILTER_WINDOW-1 :
				p_wrk[x] = np.mean(p_wrk[x:x+SMOOTHING_FILTER_WINDOW])
			else:
				p_wrk[x] = np.mean(p_wrk[x:])
		working.update({ ind : p_wrk})

	#### Differencing filtere 
	for ind,p_wrk in working.items():
		p_detrend = []
		for i in range(p_wrk.shape[0]-1):
			p_detrend.append(p_wrk[i+1]-p_wrk[i])
		working.update({ind : pd.Series(p_detrend)})


	return working


def LCM(numbers):
	print ('Finding LCM ...')
	if isinstance(numbers[0], tuple):
		common_den = LCM([n[1] for n in numbers])
		mod_numbers = [(n[0]*(common_den/n[1]), common_den) for n in numbers]
		lcm = float(LCM([n[0] for n in mod_numbers]))/common_den

	
	elif isinstance(numbers[0], Fraction):
		common_den = LCM([n.denominator for n in numbers])
		mod_numbers = [(n.numerator*(common_den/n.denominator), common_den) for n in numbers]
		lcm = float(LCM([n[0] for n in mod_numbers]))/common_den

	else:
		lcm = numbers[0]
		for i in numbers[1:]:
			lcm = i*lcm/math.gcd(int(lcm),int(i))
	return lcm


if __name__ == '__main__':

	file = '/media/milan/DATA/Qrera/AutoAcc/39FFBE/2017/12/2017_12_10.csv.gz'

	df = pd.read_csv(file)
	power_f = preprocess_power(df)

	# fund_freqs = dict()

	f_a = []
	psd_a = []

	for p in power_f.values():

		print ('periodogram')
		f, psd = welch(p, scaling='density')
		f_a.append(f)
		psd_a.append(psd)

		### Smoothing Filter 
		for x in range(f.shape[0]):
			start = x - (FT_FILTER - FT_FILTER//2)
			if start < 0:
				start = 0 

			stop = x + (FT_FILTER - FT_FILTER//2) + 1
			if stop > f.shape[0]:
				stop = f.shape[0]

			psd[x] = np.mean(psd[start:stop])
	
		plt.stem(f, psd)
		plt.show()

		####################################


	# 	print ('FFT finding peaks...')
	# 	## Filtering FT - 1	
	# 	th = jenkspy.jenks_breaks(ft, nb_class=2)[1]
	# 	# ft_f = ft.apply(lambda x: x if x > th else 0)

	# 	peaks = find_peaks(ft, height=[th])[0]
	# 	final_ft = np.zeros(len(ft))
	# 	final_ft[peaks] = ft[peaks]
	

	# 	# comp_periods = [(N, p) for p in peaks]
	# 	# print (comp_periods)
	# 	# fundamental_period = Fs*LCM(comp_periods)

	# 	# print ('Time Period: ', fundamental_period)


	# 	freqs = np.linspace(0, float(Fs)/2, N//2)
	# 	comp_freqs = [round(freqs[y],4) for y in peaks]
	# 	print (comp_freqs)

	# 	# plt.stem(freqs, final_ft)
	# 	# plt.show()


	# 	for c_f in comp_freqs:
	# 		if c_f not in fund_freqs:
	# 			count = 1
	# 		else:
	# 			count = fund_freqs[c_f]
	# 			count = count + 1
	# 		fund_freqs.update({ c_f : count})

	# print (fund_freqs)

	# freq_count_max = max(fund_freqs.values())
	# print ('Max Frequencies Count')
	# max_freqs = [f for f in fund_freqs.keys() if fund_freqs[f] == freq_count_max]
	# print (max_freqs)

	# print ('Smallest Frequency', min(fund_freqs.keys()))

	# ## Time period calc from most frequent frequencies 
	# # possible_periods = [Fraction(1/x).limit_denominator(1000) for x in fund_freqs.keys() if fund_freqs[x] == freq_count_max]
	# # print (possible_periods)
	# # print (float(Fs*LCM(possible_periods)))

	# print ('GCD')
	# gcd = int(max_freqs[0]*10000)
	# for i in range(1,len(max_freqs)):
	# 	gcd = math.gcd(gcd, int(max_freqs[i]*10000)) ###10000 because rounding of 4 is used

	# print (gcd/10000)


	








