#! /usr/bin/env python3

from scipy.signal import welch, argrelmax, periodogram
from fractions import Fraction
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import jenkspy 
import time
import math

SMOOTHING_FILTER_WINDOW = 9  ## odd numbers
FT_FILTER = 5       # odd numbers
THRESHOLD = 2000
Fs = 1 				# Sampling Frequency

def smoothing_filter(signal, window_size):
	sig_f = np.zeros((signal.shape[0])) 
	for x in range(signal.shape[0]):
		start = x - (window_size - window_size//2)
		if start < 0:
			start = 0 

		stop = x + (window_size - window_size//2) + 1
		if stop > signal.shape[0]:
			stop = signal.shape[0]

		sig_f[x] = np.mean(signal[start:stop])

	return sig_f


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

	## Thresholding Signal 
	power = pd.Series(power).apply(lambda x: x if x > THRESHOLD else 0)

	### Smoothing Filter 
	power = smoothing_filter(power, SMOOTHING_FILTER_WINDOW)

	#### Differencing filter
	p_detrend = []
	for i in range(power.shape[0]-1):
		p_detrend.append(power[i+1]-power[i])
	
	return pd.Series(p_detrend)


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
	print (len(power_f))
	f, psd = welch(power_f, scaling='density')
	print (len(psd))

	plt.plot(f, psd)
	plt.show()
	
	### Smoothing Filter
	psd_f = smoothing_filter(psd, FT_FILTER)

	### Find Peaks
	peaks = argrelmax(psd_f, order=3)[0]
	comp_freqs = f[peaks]
	psd_peaks = np.zeros((psd.shape[0]))
	psd_peaks[peaks] = psd_f[peaks]
	print (comp_freqs)

	plt.plot(f, psd_f)
	plt.show()

	plt.stem(f, psd_peaks)
	plt.show()

	comp_periods = [Fraction.from_float(1/x).limit_denominator() for x in comp_freqs]
	print (comp_periods)
	fundamental_period = LCM(comp_periods)
	print ('fundamental_period: ', fundamental_period)



	t = np.linspace(0,120)
	sig = np.sin(2*np.pi*comp_freqs[0]*t)
	for f in comp_freqs[1:]:
		sig = sig +  np.sin(2*np.pi*f*t)

	plt.plot(t, sig)
	plt.show()


