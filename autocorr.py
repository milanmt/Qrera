#! /usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

if __name__ == '__main__':

	file = '/media/milan/DATA/Qrera/AutoAcc/39FFBE/2018/01/2018_01_02.csv.gz'

	df = pd.read_csv(file)
	power = np.array(df['POWER'])
	time = np.array(range(len(power)))
	plt.plot(time,power)
	plt.show()

	power_f = []
	for i in range(len(power)):
		if i >= len(power) - 9:
			power_f.append(np.mean(power[i:]))
		else:
			power_f.append(np.mean(power[i:i+10]))
	plt.plot(time, power_f)
	plt.show()


	autocorr = np.correlate(power, power, mode='full')

	plt.plot(autocorr)
	plt.show()


	autocorr = np.correlate(power_f, power_f, mode='full')

	plt.plot(autocorr)
	plt.show()


