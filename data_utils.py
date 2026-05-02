# Author: Genevieve Chikwanha
# Date: 01/05/2026
# Purpose: Loads in JSON data, and does train/dev/test splitting

import json

def load_json(filepath):
	"""
	Loads in a JSON file to extract the intent and text
	"""

	intents = []
	texts = []

	with open(filepath, 'r', encoding='utf-8') as file:

		for line in file:
			item = json.loads(line)
			intent = item.get("intent", "")
			text = item.get("text", "")

			intents.append(intent)
			texts.append(text)

	return intents, texts
