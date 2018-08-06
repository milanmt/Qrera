#! /usr/bin/env python3

from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
import peak_detector as pd 
import numpy as np
import subprocess
import pandas 
import json

class PatternRecognition:

	def __init__(self, sequence, pattern, state_attributes):
		self.sequence = list(sequence)
		self.sequence.insert(0,0)
		self.pattern = list(pattern)
		self.pattern.insert(0,0)
		self.state_attributes = state_attributes
		self.similarity_matrix = self.get_similarity_matrix()


	def get_similarity_matrix(self):
		S = len(self.state_attributes)
		states = [s for s in state_attributes.keys()]

		similarity_matrix = np.zeros((S,S))

		for s in range(S):
			for sn in range(S):
				similarity_matrix[s][sn] = round(abs(state_attributes[states[s]][0] - state_attributes[states[sn]][0]))

		similarity_matrix = -S*similarity_matrix//np.max(similarity_matrix)

		for s in range(S):
			similarity_matrix[s][s] =  S


		print (similarity_matrix)
		return similarity_matrix



	def similarity_score(self,i,j):
		return self.similarity_matrix[i][j]
		


	def gap_penalty(from_index, to_index, gap_length):

		i_1 = from_index[0]
		j_1 = from_index[1]

		i = to_index[0]
		j = to_index[1]

		mean_vals = [val[0] for val in self.state_attributes.values()]
		min_mean = min(mean_vals)
		max_mean = max(max_vals)

		if i_1  < i:

			mean_i =  self.state_attributes[str(self.sequence[i])][0]

			if (self.sequence[i_1] == self.sequence[i]) and (self.sequence[i_1] in self.pattern):
				penalty = -1*gap_length

			elif mean_i < max_mean and mean_i > min_mean:
				penalty = -2*gap_length

			else:
				penalty = -S*gap_length

		elif j_1 < j:
			if sequence[i-1] == pattern[j-1] and sequence[i+1] == pattern[j+1]:
				penalty = -1*gap_length

			else:
				penalty = -S*gap_length

		return penalty



	def local_alignment():
		scoring_matrix = np.zeros((len(sequence), len(pattern), 3))  # 3 levels for score,i,j
		for i in range(1,len(sequence)):
			for j in range(1,len(pattern)):		
				scoring_matrix[i][j][0] = max(scoring_matrix[i-1][j][0]+similarity_score(i,j),
												scoring_matrix[i][j-1][0]+score,
												scoring_matrix[i-1][j-1][0]+score, 0)
				




def pattern_recognition(array, pattern, state_attributes):

	S = len(state_attributes)
	states = [s for s in state_attributes.keys()]

	similarity_matrix = np.zeros((S,S))

	for s in range(S):
		for sn in range(S):
			similarity_matrix[s][sn] = abs(state_attributes[states[s]][0] - state_attributes[states[sn]][0])

	similarity_matrix = S-S*similarity_matrix//np.max(similarity_matrix)

	for s in range(S):
		similarity_matrix[s][s] += 1


	print (similarity_matrix)



	alignment_matrix = np.zeros((len(array), len(pattern)))

	for i in range(len(array)):
		for j in range(len(pattern)):
			alignment_matrix[i][j] = similarity_matrix[states.index(str(array[i]))][states.index(str(pattern[j]))]


	alignment_matrix = alignment_matrix/np.max(alignment_matrix)

	plt.matshow(alignment_matrix)
	plt.colorbar()
	plt.show()

	print (alignment_matrix)



