# Author: Genevieve Chikwanha
# Date: 02/05/2026
# Purpose: This code runs the bpe tokeniser on a number of merges to get the best merge number
# AI Declaration: AI was used to format the output printed in a more readable way

from data_utils import load_json
from bpe import BPETokeniser

# Load the data for the Swahili dataset
swa_train_labels, swa_train_texts = load_json('data/swa/train.jsonl')
swa_dev_labels, swa_dev_texts = load_json('data/swa/dev.jsonl')
swa_test_labels, swa_test_texts = load_json('data/swa/test.jsonl')

# Load the data for the Twi dataset
twi_train_labels, twi_train_texts = load_json('data/twi/train.jsonl')
twi_dev_labels, twi_dev_texts = load_json('data/twi/dev.jsonl')
twi_test_labels, twi_test_texts = load_json('data/twi/test.jsonl')

## Training ##

# Store tokenisers in a dict
tokenisers = {}
k_values = [100, 300, 500]

for lang, train_texts in [("swa", swa_train_texts), ("twi", twi_train_texts)]:
    tokenisers[lang] = {}
    for k in k_values:
        print(f"\nTraining {lang} tokeniser with k={k} merges...")
        t = BPETokeniser()
        t.train(train_texts, num_merges=k)
        tokenisers[lang][k] = t
        print(f"Done. Vocab size: {len(t.vocabulary)}")    

# Show 5 sample sentences per language
samples = {
    "swa": swa_train_texts[:5],
    "twi": twi_train_texts[:5]
}

for lang in ["swa", "twi"]:
    print(f"\n{'='*60}")
    print(f"Language: {lang.upper()}")
    print(f"{'='*60}")
    for k in k_values:
        print(f"\n--- k={k} merges ---")
        t = tokenisers[lang][k]
        for sentence in samples[lang]:
            token_ids = t.encode(sentence)
            readable = [t.vocabulary[id] for id in token_ids]
            print(f"Input:  {sentence}")
            print(f"Tokens: {readable}")
            print(f"Count:  {len(token_ids)} tokens")
            print()


