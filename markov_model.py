#! /usr/bin/env python

import os
import math
import pandas
import numpy as np 

	
if __name__ == '__main__':
	
	company_path = '/media/milan/DATA/Qrera/AutoAcc/39FFBE/2018/01/2018_01_02.csv.gz'
	files = []
	files.append(company_path)

	# for root, dirs, fs in os.walk(company_path):
	# 	if fs:
	# 		files.extend(os.path.join(root,f) for f in fs)


	
	N_STATES = 0     # difference between two adjacent values
	STATES = dict()


	for file in files:
		if file.endswith('.csv.gz'):
			print file

			pd_entries = pandas.read_csv(file, engine="python")
			try:
				power_sig = pd_entries['POWER']
			except KeyError:
				power_sig = pd_entries['VALUE']

			power = map(lambda x: round(x), power_sig)
			
			print len(power), 'Total Length'

			for i in range(len(power)-2):

				state0 = power[i+1] - power[i]
				state1 = power[i+2] - power[i+1]

				if state0 not in STATES:
					N_STATES = N_STATES+1
					STATES.update({state0 : dict()}) 
				
				to_states = STATES[state0]
				if state1 not in to_states:
					to_states.update({state1 :1.0})
				else:
					count = to_states[state1]
					to_states.update({state1 : count+1.0})

				STATES.update({state0 : to_states})

				if state1 not in STATES:
					N_STATES = N_STATES + 1
					STATES.update({ state1 : dict()})
					if i == len(power)-3:
						if 0 not in STATES[state1]:							
							STATES.update({ state1 : {0 : 1}})
						else:
							val = STATES[state1][0]
							STATES.update({state1 : { 0 : val+1}})


	TRANS_MATRIX = np.zeros((N_STATES, N_STATES))
	state_inds = STATES.keys()
	for i in range(len(state_inds)):
		to_states =  STATES[state_inds[i]]
		sum_counts = np.sum(to_states.values())

		for state in to_states:
			new_val = float(to_states[state])/sum_counts
			TRANS_MATRIX[i][state_inds.index(state)] = new_val




	TRANS_MATRIX.reshape(N_STATES*N_STATES, 1)










