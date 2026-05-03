# Author: Genevieve Chikwanha
# Date: 02/05/2026
# Purpose: A BPE tokeniser that includes training, encoding, an end-of-word marker (e.g. _), and a configurable vocabulary size.
# Code adapted from: 
		# 1. https://github.com/sumony2j/Simple-BPE-Tokenizer/blob/main/Tokenizer.py
		# 2. https://sebastianraschka.com/blog/2025/bpe-from-scratch.html
		# 3. Jurafsky, D and Martin, J. 2026. Speech and Language Processing: 
			# An Introduction to Natural Language Processing, Computational Linguistics, 
			# and Speech Recognition with Language Models, 3rd edition. 

from collections import Counter, deque

class BPETokeniser():

	def __init__(self):
		"""A constructor to initialise dictionaries for the base vocab with all byte values and merges"""

		self.vocabulary = {i: bytes([i]) for i in range(256)}
		self.inverse_vocab = {bytes([i]): i for i in range(256)}
		self.merged = {}


	def preprocess(self, text):
		"""Split the text into words and add an end-of-word marker to the end"""
		if isinstance(text, list):
			text = " ".join(text)
		
		preprocessed_text = []
		for word in text.split():
			marked_word = word + "</w>"
			preprocessed_text.append(marked_word)

		processed_text = " ".join(preprocessed_text)

		return processed_text


	def train(self, sentences, num_merges):
		"""Learn merge operations from a corpus by iteratively merging the most frequent adjacent symbol pair"""

		# Convert the corpus into bytes
		processed_corpus = self.preprocess(sentences)
		corpus_as_bytes = processed_corpus.encode("utf-8")
		token_ids = list(corpus_as_bytes)

		# BPE Loop
		for new_id in range(256, 256 + num_merges):
			most_freq_pair = self.find_freq_pair(token_ids)
			if most_freq_pair is None:
				break
			token_ids = self.replace_pair(token_ids, most_freq_pair, new_id)
			self.merged[most_freq_pair] = new_id
			merged_bytes = self.vocabulary[most_freq_pair[0]] + self.vocabulary[most_freq_pair[1]]
			self.vocabulary[new_id] = merged_bytes
			self.inverse_vocab[merged_bytes] = new_id


	def encode(self, text):
		"""Apply the learned merges (in order) to tokenise new text."""
		
		processed = self.preprocess(text)
		tokens = []

		for word in processed.split():
			tokens.extend(self.tokenise_with_bpe(word))

		return tokens


	def decode(self, token_ids):
		"""Reverse the encode operation"""

		decoded_bytes = b"".join(self.vocabulary[id] for id in token_ids)
		decoded_string = decoded_bytes.decode("utf-8", errors="replace")
		decoded_string = decoded_string.replace("</w>", " ").strip()

		return decoded_string


	def tokenise_with_bpe(self, token):
		"""Create token ids to be passed to the encoder"""

		token_ids = list(token.encode("utf-8"))

		for pair, new_id in self.merged.items():
			token_ids = self.replace_pair(token_ids, pair, new_id)

		return token_ids


	@staticmethod
	def find_freq_pair(token_ids):
		"""Finds the most frequently occuring adjacent pair"""

		pairs = zip(token_ids, token_ids[1:])
		pair_counter = Counter(pairs)

		if not pair_counter:
			return None

		return pair_counter.most_common(1)[0][0]


	@staticmethod
	def replace_pair(token_ids, pair_id, new_id):
		"""Uses a double-ended queue"""

		d_queue = deque(token_ids)
		replaced = []

		while d_queue:
			current = d_queue.popleft()
			if d_queue and ((current, d_queue[0]) == pair_id):
				replaced.append(new_id)
				d_queue.popleft()
			else:
				replaced.append(current)

		return replaced

