# Author: Genevieve Chikwanha
# Date: 01/05/2026
# Purpose: Baseline
# Code Adapted From: 
		# 1. https://github.com/the-utkarshjain/Speech-and-Language-Processing-3rd-Edition-Solutions/blob/main/Chapter%203/3.8.py
		# 2. Jurafsky, D. and Martin, J. 2026. Speech and Language Processing: An Introduction to Natural Language Processing, 
		     # Computational Linguistics, and Speech Recognition with Language Models, 3rd edition.		
		# 3. https://github.com/daandouwe/char-lm/blob/master/ngram.py

import re
from collections import Counter
from data_utils import load_json
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class UnigramModel:
	def __init__(self):
		"""Constructor to initialise the counters, uique vocab set, and total word count"""

		self.unigram_counts = {}
		self.total_words = 0
		self.vocabulary = []
		self.word_to_index = {}

	def text_preprocessing(self, sentence):
		""" Lowercase and remove punctuation, then split into tokens"""

		lower_sentence = sentence.lower()
		words = re.findall(r'\b\w+\b', lower_sentence)
		
		return words

	def calculate_ngram(self, sentences):
		"""Calculate the unigrams for the sentences, for training specifically"""

		for sentence in sentences:
			words = self.text_preprocessing(sentence)
			for word in words:
				self.total_words += 1
				self.unigram_counts[word] = self.unigram_counts.get(word, 0) + 1

		self.vocabulary = list(self.unigram_counts.keys())
		self.word_to_index = {word: i for i, word in enumerate(self.vocabulary)}

	def count_vectoriser(self, sentences):
		"""Represents each utterance as a bag-of-words count vector."""
		count_vector = []

		for sentence in sentences:
			words = self.text_preprocessing(sentence)
			vector = [0] * len(self.vocabulary)
			for word in words:
				if word in self.word_to_index:
					col_index = self.word_to_index[word]
					vector[col_index] += 1
			
			count_vector.append(vector)

		return count_vector


class CharNgramModel:

	def __init__(self, n_range=(2, 3)):
		"""A constructor to create a character n-gram feature extractor to slide a 
		window of size n (which is a range) over the characters"""
		self.n_range = n_range
		self.ngram_counts = {}
		self.total_ngrams = 0
		self.vocabulary = []
		self.word_to_index = {}
	
	def text_preprocessing(self, sentence):
		"""Lower case the text and tokenise into ngrams of len n"""
		
		lower_sentence = sentence.lower()
		ngrams = []

		for n in range(self.n_range[0], self.n_range[1] + 1):
			for i in range(len(lower_sentence) - n + 1):
				ngram = lower_sentence[i : i + n]
				ngrams.append(ngram)

		return ngrams

	def calculate_ngram(self, sentences):
		"""Calculate the unigrams for the sentences, for training specifically"""
		
		for sentence in sentences:
			ngrams = self.text_preprocessing(sentence)
			for ngram in ngrams:
				self.total_ngrams += 1
				self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1

		self.vocabulary = list(self.ngram_counts.keys())
		self.word_to_index = {word: i for i, word in enumerate(self.vocabulary)}

	def count_vectoriser(self, sentences):
		"""Represents each utterance as a bag-of-words count vector."""

		count_vector = []

		for sentence in sentences:
			ngrams = self.text_preprocessing(sentence)
			vector = [0] * len(self.vocabulary)
			for ngram in ngrams:
				if ngram in self.word_to_index:
					col_index = self.word_to_index[ngram]
					vector[col_index] += 1

			count_vector.append(vector)

		return count_vector

def tune_and_evaluate(train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels, is_ngram=True):
	"""Train the Unigram and NGram Baselines on A Logisic Regression Model"""

	# Hyperparameters for c-regularisation and different ngram combinations
	n_values = [(1, 3), (2, 2), (2, 3), (2, 4), (3, 5), (3, 6)] if is_ngram else [None]
	c_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

	best_acc = 0
	best_c = None
	best_n = None
	best_curve = []
	best_model = None
	best_test_vec = None

	# Do a grid search over the n values and the c values
	for n in n_values:
		feat_extr = CharNgramModel(n_range=n) if is_ngram else UnigramModel()
		feat_extr.calculate_ngram(train_texts)

		train_vec = feat_extr.count_vectoriser(train_texts)
		test_vec = feat_extr.count_vectoriser(test_texts)
		dev_vec = feat_extr.count_vectoriser(dev_texts)

		current_curve = []
		n_is_best = False

		for c in c_values:
			classifier = LogisticRegression(C=c, max_iter=1000, random_state=42)
			classifier.fit(train_vec, train_labels)

			# Check the dev set
			prediction = classifier.predict(dev_vec)
			accuracy = accuracy_score(dev_labels, prediction)
			current_curve.append(accuracy)

			if accuracy > best_acc:
				best_acc = accuracy
				best_c = c
				best_n = n
				best_model = classifier
				best_test_vec = test_vec
				n_is_best = True

		if n_is_best:
			best_curve = current_curve.copy()

	# Get test prediction and accuracy
	test_pred = best_model.predict(best_test_vec)
	test_accuracy = accuracy_score(test_labels, test_pred)

	print(f"best n: {best_n}, best C: {best_c}")
	print(f"dev acc:  {best_acc:.4f}")
	print(f"test acc: {test_accuracy:.4f}\n")

	return best_curve, c_values


# Load the data for the Swahili dataset
swa_train_labels, swa_train_texts = load_json('data/swa/train.jsonl')
swa_dev_labels, swa_dev_texts = load_json('data/swa/dev.jsonl')
swa_test_labels, swa_test_texts = load_json('data/swa/test.jsonl')

# Load the data for the Twi dataset
twi_train_labels, twi_train_texts = load_json('data/twi/train.jsonl')
twi_dev_labels, twi_dev_texts = load_json('data/twi/dev.jsonl')
twi_test_labels, twi_test_texts = load_json('data/twi/test.jsonl')

### Unigram Model ###

print("Swahili Unigram Model")
curve_1, c_vals = tune_and_evaluate(swa_train_texts, swa_train_labels, swa_dev_texts, swa_dev_labels, swa_test_texts, swa_test_labels, is_ngram=False)

print("Twi Unigram Model")
curve_2, _ = tune_and_evaluate(twi_train_texts, twi_train_labels, twi_dev_texts, twi_dev_labels, twi_test_texts, twi_test_labels, is_ngram=False)

### Ngram Model ###

print("Swahili Ngram Model")
curve_3, _ = tune_and_evaluate(swa_train_texts, swa_train_labels, swa_dev_texts, swa_dev_labels, swa_test_texts, swa_test_labels)

print("Twi Ngram Model")
curve_4, _ = tune_and_evaluate(twi_train_texts, twi_train_labels, twi_dev_texts, twi_dev_labels, twi_test_texts, twi_test_labels)

# plot the results
plt.figure(figsize=(8, 5))
plt.plot(c_vals, curve_1, label='Swahili Unigram', marker='o')
plt.plot(c_vals, curve_2, label='Twi Unigram', marker='s')
plt.plot(c_vals, curve_3, label='Swahili Ngram', marker='^')
plt.plot(c_vals, curve_4, label='Twi Ngram', marker='*')

# C is usually plotted on a log scale because the values jump by factors of 10
plt.xscale('log') 
plt.xlabel('C (Inverse Regularization Strength)')
plt.ylabel('Dev Accuracy')
plt.title('Dev Accuracy vs Hyperparameter C')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

