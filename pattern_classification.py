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
		self.classifier =  self.__train_classifier()

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

	def __train_classifier(self):
		X = []
		y = []
		for l, fv in self.feature_dict.items():
			X.extend(list(fv))
			fv_o = len(fv)*[bin(l)]
			y.extend(fv_o)
		gpc = GaussianProcessClassifier(multi_class='one_vs_rest').fit(np.array(X), np.array(y))
		self.transform_pattern_labels = [int(l,2) for l in gpc.classes_]
		return gpc

	def get_probabilities(self, text_x):
		fv_x = self.__get_feature_vector(text_x).reshape(1,-1)
		probs = list(self.classifier.predict_proba(fv_x)[0])
		prob_t = np.max(self.classifier)
		label_t = self.transform_pattern_labels[probs.index(prob_t)]
		return prob_t, label_t

	@timing_wrapper
	def find_matches(self):
		print ('Matching Discovered Patterns...')
		start_ind = 0
		end_ind = 0
		pattern_sequence = []
		pattern_sequence_indices = []
		while start_ind < len(self.sequence)-1:
			max_probs = []
			req_labels = []
			end_ind_list = []
			end_ind_t = start_ind+self.min_len-1
			while end_ind_t < start_ind+self.max_len:
				p_temp = self.sequence[start_ind:end_ind_t+1]
				prob_t, label_t = self.get_probabilities(p_temp).items()
				max_probs.append(prob_t)
				req_labels.append(label_t)
				end_ind_list = end_ind_t
				end_ind_t +=1

			max_prob = max(max_probs)
			req_ind = list(reversed(max_probs)).index(max_prob)
			req_label = req_labels[len(max_probs)-1-req_ind]
			end_ind = end_ind_list[len(max_probs)-1-req_ind]

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