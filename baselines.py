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
	n_values = [(2, 3), (2, 4), (3, 5)] if is_ngram else [None]
	c_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

	best_acc = 0
	best_c = None
	best_n = None
	best_curve = []
	best_model = None
	best_test_vec = None
	best_feat_extr = None

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
				best_feat_extr = feat_extr
				n_is_best = True

		if n_is_best:
			best_curve = current_curve.copy()

	# Get test prediction and accuracy
	test_pred = best_model.predict(best_test_vec)
	test_accuracy = accuracy_score(test_labels, test_pred)

	print(f"best n: {best_n}, best C: {best_c}")
	print(f"dev acc:  {best_acc:.4f}")
	print(f"test acc: {test_accuracy:.4f}\n")

	return best_curve, c_values, test_pred, best_model, best_feat_extr

def show_top_features(model, feat_extr, class_a, class_b, top_n=8):
	"""Print the most discriminative features for two intents."""
	classes    = list(model.classes_)
	vocab      = feat_extr.vocabulary
	idx_a      = classes.index(class_a)
	idx_b      = classes.index(class_b)
	coef_a     = model.coef_[idx_a]
	coef_b     = model.coef_[idx_b]

	# Top features strongly associated with each intent
	top_a_idx  = sorted(range(len(coef_a)), key=lambda i: coef_a[i], reverse=True)[:top_n]
	top_b_idx  = sorted(range(len(coef_b)), key=lambda i: coef_b[i], reverse=True)[:top_n]
	top_a = [vocab[i] for i in top_a_idx]
	top_b = [vocab[i] for i in top_b_idx]

	print(f"  Top features for '{class_a}': {top_a}")
	print(f"  Top features for '{class_b}': {top_b}")

	# Highlight any overlap — shared features are a likely source of confusion
	overlap = set(top_a) & set(top_b)
	if overlap:
		print(f"  Overlapping features (source of confusion): {overlap}")


def error_analysis(test_texts, test_labels, test_preds, model, feat_extr, 
                   language_name, examples_per_pair=3):
	print(f"\n{'='*55}")
	print(f"  {language_name} Error Analysis")
	print(f"{'='*55}")

    # Collect all misclassified examples
	mistakes = [
		(true, pred, text)
		for text, true, pred in zip(test_texts, test_labels, test_preds)
		if true != pred
	]

    # Count confused pairs and sort by frequency
	pair_counts = {}
	for true, pred, text in mistakes:
		pair = (true, pred)
		pair_counts[pair] = pair_counts.get(pair, 0) + 1

	top3 = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:3]

	print(f"\nTop 3 confused pairs:\n")

	for (true, pred), count in top3:
		print(f"  TRUE: '{true}'  →  PREDICTED: '{pred}'  ({count} times)")

		# (i) Quote 2-3 misclassified examples
		examples = [
			text for t, p, text in mistakes
			if t == true and p == pred
		][:examples_per_pair]

		print(f"\n  Misclassified examples:")
		for ex in examples:
			print(f"    - \"{ex}\"")

		# (ii) Token distributions the model learned
		print(f"\n  Token distributions (model coefficients):")
		show_top_features(model, feat_extr, true, pred)
		print()


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
curve_1, c_vals, swa_uni_preds, swa_uni_model, swa_uni_feat = tune_and_evaluate(swa_train_texts, swa_train_labels, swa_dev_texts, swa_dev_labels,swa_test_texts, swa_test_labels, is_ngram=False)

print("Twi Unigram Model")
curve_2, _, twi_uni_preds, twi_uni_model, twi_uni_feat = tune_and_evaluate(twi_train_texts, twi_train_labels, twi_dev_texts, twi_dev_labels, twi_test_texts, twi_test_labels, is_ngram=False)

### N-gram Model ###
print("Swahili N-gram Model")
curve_3, _, swa_ngram_preds, swa_ngram_model, swa_ngram_feat = tune_and_evaluate(swa_train_texts, swa_train_labels, swa_dev_texts, swa_dev_labels, swa_test_texts, swa_test_labels)

print("Twi N-gram Model")
curve_4, _, twi_ngram_preds, twi_ngram_model, twi_ngram_feat = tune_and_evaluate(twi_train_texts, twi_train_labels, twi_dev_texts, twi_dev_labels,twi_test_texts, twi_test_labels)

### Error Analysis — run on your best model per language ###
error_analysis(swa_test_texts, swa_test_labels, swa_ngram_preds,swa_ngram_model, swa_ngram_feat, "Swahili")

error_analysis(twi_test_texts, twi_test_labels, twi_ngram_preds, twi_ngram_model, twi_ngram_feat, "Twi")

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

