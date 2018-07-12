#! /usr/bin/env python3

from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
import find_threshold 
import pandas as pd 
import numpy as np 
import time


SMOOTHING_FILTER_WINDOW = 10
THRESHOLD = 2610
MIN_IDLE_TIME = 10  # seconds
MIN_WORK_TIME = 10  # seconds 


def var_round(number):
	number = float(number)
	
	if number/10 <= 10:
		return round(number)
	elif number/10 <= 1000:
		return round(number, -1)
	else:
		return round(number, -2)


def lpf(data):
	if len(data) < 13:
		data.extend((13-len(data))*[data[-1]])

	b, a = butter(3, 0.5)
	y = filtfilt(b, a, data)
	
	return y


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
				power[-1] = (power[-1]+df.iloc[i,0])/2

			elif df.iloc[i,1]-df.iloc[i-1,1] > 1.0:
				for j in range(int(df.iloc[i,1]-df.iloc[i-1,1])):
					power.append(df.iloc[i-1,0])


	## Thresholding Signal 
	power = pd.Series(power).apply(lambda x: x if x > THRESHOLD else THRESHOLD)

	### Differencing filter
	p_detrend = []
	for i in range(len(power)-1):
		p_detrend.append(power[i+1]-power[i])

	## lpf
	power_f = lpf(p_detrend)

	# plt.plot(power)
	# plt.plot(p_detrend)
	# plt.plot(power_f)
	# plt.show()

	return power_f


if __name__ == '__main__':
	
	# file = '/media/milan/DATA/Qrera/AutoAcc/39FFBE/2018/05/2018_05_08.csv.gz'
	file = '/media/milan/DATA/Qrera/FWT/5CCF7FD0C7C0/2018/07/2018_07_05.csv.gz'

	df = pd.read_csv(file)

	power_f = preprocess_power(df)

	peaks, _ = find_peaks(power_f)
	peak_p = power_f[peaks]
	
	peak_pr = [var_round(p) for p in peak_p]
	peak_threshold = find_threshold.get_otsus_threshold(peak_pr)
	print ('peak_threshold', peak_threshold)

	# peak_threshold = 3000
	final_peaks = np.zeros((len(power_f)))
	for p in peaks:
		if power_f[p] > peak_threshold:
			final_peaks[p] = power_f[p]
	
	plt.plot(power_f)
	plt.stem(final_peaks)
	plt.show()
	total_peaks = len([p for p in final_peaks if p != 0])

	print ('Total Peaks:', total_peaks)




