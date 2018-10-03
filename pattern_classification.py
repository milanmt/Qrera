#! /usr/bin/env python3

from sklearn.gaussian_process import GaussianProcessClassifier
import numpy as np 
import time

def timing_wrapper(func):
	def wrapper(*args,**kwargs):
		t0= time.time()
		func_val = func(*args,**kwargs)
		time_taken = time.time() - t0
		print (str(func),' took: ', time_taken)
		return func_val
	return wrapper

class PatternClassification:
	def __init__(self, pattern_dict, state_attributes, sequence, min_len,  max_len):
		self.transform_state_labels = [int(l) for l in state_attributes.keys()]
		self.number_classifiers = len(pattern_dict)
		self.feature_dict = self.__get_feature_dict(pattern_dict)
		self.sequence = sequence
		self.min_len = min_len
		self.max_len = max_len
		self.classifiers = []
		for label in self.feature_dict.keys():
			classifier = self.__train_classifier(label)
			self.classifiers.append(classifier)


	def __get_feature_vector(self, p):
		vector = np.zeros((len(self.transform_state_labels)))
		p_unique, p_count = np.unique(p, return_counts=True)
		for el,c in zip(p_unique, p_count):
			vector[self.transform_state_labels.index(int(el))] = c
		return vector

	def __get_feature_dict(self, pattern_dict):
		feature_dict = dict()
		for label, pattern_set in pattern_dict.items():
			feature_set = []
			for p,f in pattern_set:
				feature_set.append(self.__get_feature_vector(p))
			feature_dict.update({ label : feature_set})
		return feature_dict

	def __train_classifier(self, req_label):
		X = []
		y = []
		for l, fv in self.feature_dict.items():
			X.extend(list(fv))
			if l == req_label:
				y.extend(len(fv)*[1])
			else:
				y.extend(len(fv)*[0])

		gpc = GaussianProcessClassifier().fit(np.array(X), np.array(y))
		return gpc

	def get_probabilities(self, text_x):
		fv_x = self.__get_feature_vector(text_x).reshape(1,-1)
		probs = dict()
		for label,classifier in zip(self.feature_dict.keys(), self.classifiers):
			probs.update({label : classifier.predict_proba(fv_x)[0][1]})
		return probs

	@timing_wrapper
	def find_matches(self):
		print ('Matching Discovered Patterns...')
		start_ind = 0
		end_ind = 0
		pattern_sequence = []
		pattern_sequence_indices = []
		while start_ind < len(self.sequence)-1:
			max_prob = 0
			req_label = None
			end_ind_t = start_ind+self.min_len-1
			while end_ind_t < start_ind+self.max_len:
				p_temp = self.sequence[start_ind:end_ind_t+1]
				for label, prob in self.get_probabilities(p_temp).items():
					if prob >= max_prob:
						max_prob = prob
						req_label = label
						end_ind = end_ind_t
					
				end_ind_t +=1

			pattern_sequence.append(req_label)
			if end_ind < len(self.sequence):
				pattern_sequence_indices.append(end_ind)
			start_ind = end_ind
		
		self.pattern_sequence = pattern_sequence
		self.pattern_sequence_indices = pattern_sequence_indices
		vals, counts = np.unique(pattern_sequence, return_counts=True) ### Counting unique values
		print (vals)
		print (counts)
		return pattern_sequence, pattern_sequence_indices