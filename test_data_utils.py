# Author: Genevieve Chikwanha
# Date: 01/05/2026
# Purpose: Test the load_json method in the data_utils file

import json
from data_utils import load_json

# Expected length: 2240 training examples
train_labels, train_texts = load_json('data/swa/train.jsonl')
print(f"Loaded {len(train_texts)} training examples.")

# Expected sample: "tafadhali seti kengele ya saa 6:48 asubuhi inikumbushe ninywe dawa"
print("A sample:")
print(train_texts[0])
