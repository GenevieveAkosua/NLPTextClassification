import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack, csr_matrix
from bpe import BPETokeniser
from data_utils import load_json
#https://github.com/Bashkeel/BoW-TFIDF/blob/master/TF-IDF%20and%20N-Grams.ipynb
#https://www.kaggle.com/code/abhisheksinha28/logistic-regression-with-n-grams
# used for tfidfvectoriser -fit/transform code and hstack functions
random.seed(42)
np.random.seed(42)

# fixed k from NB tuning
SWA_K = 1500
TWI_K = 1000

# hyperparameters
C_VALUES = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
NGRAM_RANGES = [(1, 1), (2, 3), (2, 4), (3, 5)]

# load data
swa_train_labels, swa_train_texts = load_json('data/swa/train.jsonl')
swa_dev_labels, swa_dev_texts = load_json('data/swa/dev.jsonl')
swa_test_labels, swa_test_texts = load_json('data/swa/test.jsonl')

twi_train_labels, twi_train_texts = load_json('data/twi/train.jsonl')
twi_dev_labels, twi_dev_texts = load_json('data/twi/dev.jsonl')
twi_test_labels, twi_test_texts = load_json('data/twi/test.jsonl')


# feature extraction

def tokens_to_vector(token_list, vocab_size):
    """Convert a list of BPE token IDs into a count vector."""
    vector = [0] * vocab_size
    for token in token_list:
        if token < vocab_size:
            vector[token] += 1
    return vector


def get_bpe_count_features(train_tokens, dev_tokens, test_tokens, vocab_size):
    """
    BPE token counts. Each position says how many times that token ID appeared in the sentence.
    
    """
    train_vecs = csr_matrix([tokens_to_vector(t, vocab_size) for t in train_tokens])
    dev_vecs = csr_matrix([tokens_to_vector(t, vocab_size) for t in dev_tokens])
    test_vecs = csr_matrix([tokens_to_vector(t, vocab_size) for t in test_tokens])
    return train_vecs, dev_vecs, test_vecs


def encode_as_string(texts, bpe):
    """Convert raw sentences into strings of BPE token IDs for TF-IDF. Current BPE outputs IDs"""
    return [" ".join(str(t) for t in bpe.encode(text)) for text in texts]


def get_tfidf_features(train_texts, dev_texts, test_texts, bpe):
    """
    TF-IDF weighting on BPE tokens. Downweights common tokens across documents in Swahili and Twi, give weights to frequent words   
    """
    train_strings = encode_as_string(train_texts, bpe)
    dev_strings = encode_as_string(dev_texts, bpe)
    test_strings = encode_as_string(test_texts, bpe)

    vectorizer = TfidfVectorizer()
    train_vecs = vectorizer.fit_transform(train_strings)
    dev_vecs = vectorizer.transform(dev_strings)
    test_vecs = vectorizer.transform(test_strings)
    return train_vecs, dev_vecs, test_vecs

def get_char_ngram_features(train_texts, dev_texts, test_texts, ngram_range):
    """
    character n-gram features.
    """
    vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range)
    train_vecs = vectorizer.fit_transform(train_texts)
    dev_vecs = vectorizer.transform(dev_texts)
    test_vecs = vectorizer.transform(test_texts)
    return train_vecs, dev_vecs, test_vecs


def build_full_features(train_texts, dev_texts, test_texts, bpe, ngram_range, vocab_size):
    """Combines all three feature types into one matrix."""
    train_tokens = [bpe.encode(t) for t in train_texts]
    dev_tokens = [bpe.encode(t) for t in dev_texts]
    test_tokens = [bpe.encode(t) for t in test_texts]

    tr_bpe, dv_bpe, te_bpe = get_bpe_count_features(train_tokens, dev_tokens, test_tokens, vocab_size)
	# build and keep the vectorizers
    train_strings = encode_as_string(train_texts, bpe)
    tfidf_vec = TfidfVectorizer()
    tr_tfidf = tfidf_vec.fit_transform(train_strings)
    dv_tfidf = tfidf_vec.transform(encode_as_string(dev_texts,  bpe))
    te_tfidf = tfidf_vec.transform(encode_as_string(test_texts, bpe))

    char_vec = CountVectorizer(analyzer='char', ngram_range=ngram_range)
    tr_char  = char_vec.fit_transform(train_texts)
    dv_char  = char_vec.transform(dev_texts)
    te_char  = char_vec.transform(test_texts)

    train_combined = hstack([tr_bpe, tr_tfidf, tr_char])
    dev_combined   = hstack([dv_bpe, dv_tfidf, dv_char])
    test_combined  = hstack([te_bpe, te_tfidf, te_char])
    return train_combined, dev_combined, test_combined, tfidf_vec, char_vec


# ablation

def run_ablation(train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels, best_k, best_c, best_ngram, language_name, bpe, vocab_size):
    """
    Trains 4 models on best c, k and ngrams. Swaps out features to see how they improve performance
    """
    print(f"\n{language_name} ablation results:")

    train_tokens = [bpe.encode(t) for t in train_texts]
    dev_tokens = [bpe.encode(t) for t in dev_texts]
    test_tokens = [bpe.encode(t) for t in test_texts]

    tr_bpe, dv_bpe, te_bpe = get_bpe_count_features(train_tokens, dev_tokens, test_tokens, vocab_size)
    tr_tfidf, dv_tfidf, te_tfidf = get_tfidf_features(train_texts, dev_texts, test_texts, bpe)
    tr_char, dv_char, te_char = get_char_ngram_features(train_texts, dev_texts, test_texts, best_ngram)

    ablation_results = {}

    # model 1 - BPE counts only
    lr = LogisticRegression(C=best_c, max_iter=1000, random_state=42)
    lr.fit(tr_bpe, train_labels)
    dev_acc = accuracy_score(dev_labels, lr.predict(dv_bpe))
    test_acc = accuracy_score(test_labels, lr.predict(te_bpe))
    ablation_results["BPE only"] = (dev_acc, test_acc)
    print(f"BPE only  dev={dev_acc:.4f}  test={test_acc:.4f}")

    # model 2 - BPE + TF-IDF
    lr = LogisticRegression(C=best_c, max_iter=1000, random_state=42)
    lr.fit(hstack([tr_bpe, tr_tfidf]), train_labels)
    dev_acc = accuracy_score(dev_labels, lr.predict(hstack([dv_bpe, dv_tfidf])))
    test_acc = accuracy_score(test_labels, lr.predict(hstack([te_bpe, te_tfidf])))
    ablation_results["BPE + TF-IDF"] = (dev_acc, test_acc)
    print(f"BPE + TF-IDF  dev={dev_acc:.4f}  test={test_acc:.4f}")

    # model 3 - BPE + char ngrams
    lr = LogisticRegression(C=best_c, max_iter=1000, random_state=42)
    lr.fit(hstack([tr_bpe, tr_char]), train_labels)
    dev_acc = accuracy_score(dev_labels, lr.predict(hstack([dv_bpe, dv_char])))
    test_acc = accuracy_score(test_labels, lr.predict(hstack([te_bpe, te_char])))
    ablation_results["BPE + char ngrams"] = (dev_acc, test_acc)
    print(f"BPE + char ngrams  dev={dev_acc:.4f}  test={test_acc:.4f}")

    # model 4 - all features
    lr = LogisticRegression(C=best_c, max_iter=1000, random_state=42)
    lr.fit(hstack([tr_bpe, tr_tfidf, tr_char]), train_labels)
    dev_acc = accuracy_score(dev_labels, lr.predict(hstack([dv_bpe, dv_tfidf, dv_char])))
    test_acc = accuracy_score(test_labels, lr.predict(hstack([te_bpe, te_tfidf, te_char])))
    ablation_results["BPE + TF-IDF + char ngrams"] = (dev_acc, test_acc)
    print(f"BPE + TF-IDF + char ngrams  dev={dev_acc:.4f}  test={test_acc:.4f}")

    return ablation_results


# tuning

def tune_lr(train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels, best_k, language_name):
    """
    Tunes C and ngram_range on dev set. k is fixed from NB tuning.
    Evaluates best model on test once then runs ablation.
    """
    print(f"\nTraining BPE with k={best_k}")
    bpe = BPETokeniser()
    bpe.train(train_texts, num_merges=best_k)
    vocab_size = 256 + best_k

    all_runs = []
    best_acc = 0
    best_c = None
    best_ngram = None
    best_tfidf_vec = None
    best_char_vec = None

    for ngram_range in NGRAM_RANGES:
        train_vecs, dev_vecs, test_vecs, tfidf_vec, char_vec = build_full_features(train_texts, dev_texts, test_texts, bpe, ngram_range, vocab_size)

        for c in C_VALUES:
            lr = LogisticRegression(C=c, max_iter=1000, random_state=42)
            lr.fit(train_vecs, train_labels)
            dev_acc = accuracy_score(dev_labels, lr.predict(dev_vecs))
            all_runs.append((ngram_range, c, dev_acc))
            print(f"C={c} dev_acc={dev_acc:.4f}")

            if dev_acc > best_acc:
                best_acc = dev_acc
                best_c = c
                best_ngram = ngram_range
                best_tfidf_vec = tfidf_vec
                best_char_vec = char_vec

    # build best lr model with best settings and evaluate on test once
    train_vecs, dev_vecs, test_vecs, _, _ = build_full_features(train_texts, dev_texts, test_texts, bpe, best_ngram, vocab_size)
    best_lr = LogisticRegression(C=best_c, max_iter=1000, random_state=42)
    best_lr.fit(train_vecs, train_labels)

    
    test_preds = best_lr.predict(test_vecs)
    test_acc = accuracy_score(test_labels, test_preds)

    print(f"\n{language_name} best settings: ngram={best_ngram}, C={best_c}")
    print(f"{language_name} dev acc: {best_acc:.4f}")
    print(f"{language_name} test acc: {test_acc:.4f}")

    ablation = run_ablation(train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels, best_k, best_c, best_ngram, language_name, bpe, vocab_size)
    return {
        "language": language_name,
        "all_runs": all_runs,
        "best_k": best_k,
        "best_c": best_c,
        "best_ngram": best_ngram,
        "best_dev_acc": best_acc,
        "best_test_acc": test_acc,
        "test_preds": test_preds,
        "ablation": ablation,
        "model": best_lr,
        "tfidf_vec": best_tfidf_vec,
        "char_vec": best_char_vec,
        "vocab_size": vocab_size
    }


# run everything

print("===SWAHILI===")
swa_lr= tune_lr(swa_train_texts, swa_train_labels, swa_dev_texts,swa_dev_labels, swa_test_texts, swa_test_labels, best_k=SWA_K, language_name="Swahili")

print("===TWI===")
twi_lr = tune_lr(twi_train_texts, twi_train_labels,twi_dev_texts, twi_dev_labels, twi_test_texts, twi_test_labels, best_k=TWI_K, language_name="Twi")

# final summary
print("\n---Final results---")
print(f"{'Language':<10} {'k':<6} {'ngram':<10} {'C':<6} {'Dev Acc':<10} {'Test Acc'}")
print("-" * 55)
for r in [swa_lr, twi_lr]:
    print(f"{r['language']:<10} {r['best_k']:<6} {str(r['best_ngram']):<10} {r['best_c']:<6} {r['best_dev_acc']:<10.4f} {r['best_test_acc']:.4f}")

print("\n--Ablation summary--")
for r in [swa_lr, twi_lr]:
    print(f"\n{r['language']}:")
    for name, (dev_acc, test_acc) in r['ablation'].items():
        print(f"  {name:<35} dev={dev_acc:.4f}  test={test_acc:.4f}")


import matplotlib.pyplot as plt
def plot_lr_results(swa, twi):
    """Two plots: C curve and ngram curve for both languages."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # SWA data
    swa_cs = []
    swa_accs_c = []

    swa_ngrams = []
    swa_accs_n = []

    for ngram, c, acc in swa["all_runs"]:
        if ngram == swa["best_ngram"]:
            swa_cs.append(c)
            swa_accs_c.append(acc)
        if c == swa["best_c"]:
            swa_ngrams.append(str(ngram))
            swa_accs_n.append(acc)

    # TWI data
    twi_cs = []
    twi_accs_c = []

    twi_ngrams = []
    twi_accs_n = []

    for ngram, c, acc in twi["all_runs"]:
        if ngram == twi["best_ngram"]:
            twi_cs.append(c)
            twi_accs_c.append(acc)
        if c == twi["best_c"]:
            twi_ngrams.append(str(ngram))
            twi_accs_n.append(acc)

    # plot 1 C vs dev accuracy  (ngram at best)
    ax = axes[0]
    ax.plot(swa_cs, swa_accs_c, marker='o', label=f"Swahili (ngram={swa['best_ngram']})")
    ax.plot(twi_cs, twi_accs_c, marker='s', label=f"Twi (ngram={twi['best_ngram']})")
    ax.set_xscale('log')
    ax.set_xlabel('C (regularisation strength)')
    ax.set_ylabel('Dev Accuracy')
    ax.set_title('Dev Accuracy vs C')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # plot 2 ngram range vs dev accuracy (C at best)
    ax = axes[1]
    x = range(len(swa_ngrams))
    ax.plot(x, swa_accs_n, marker='o', label=f"Swahili (C={swa['best_c']})")
    ax.plot(x, twi_accs_n, marker='s', label=f"Twi (C={twi['best_c']})")
    ax.set_xticks(x)
    ax.set_xticklabels(swa_ngrams)
    ax.set_xlabel('Character n-gram range')
    ax.set_ylabel('Dev Accuracy')
    ax.set_title('Dev Accuracy vs n-gram range')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('lr_hyperparameter_curves.png', dpi=150)
    plt.show()


plot_lr_results(swa_lr, twi_lr)
#Swahili :k= 1500   n=(2, 4)    c= 0.5   
#Twi  : k= 1000   n=(2, 4)    c=0.5       
# helper to sort by count
def get_count(item):
    return item[1]

#ERROR ANALYSIS 
def show_top_features_combined(model, tfidf_vec, char_vec, vocab_size, class_a, class_b, top_n=8):
    """Show the top character ngram features for two intents"""

    classes = list(model.classes_)
    idx_a = classes.index(class_a)
    idx_b = classes.index(class_b)

    # Character ngram features start after the bpe and tfidf blocks
    char_offset = vocab_size + len(tfidf_vec.vocabulary_)
    char_vocab = {v: k for k, v in char_vec.vocabulary_.items()}
    char_coef_a = model.coef_[idx_a][char_offset:]
    char_coef_b = model.coef_[idx_b][char_offset:]
    top_a = [char_vocab[i] for i in sorted(range(len(char_coef_a)), key=lambda i: char_coef_a[i], reverse=True)[:top_n]]
    top_b = [char_vocab[i] for i in sorted(range(len(char_coef_b)), key=lambda i: char_coef_b[i], reverse=True)[:top_n]]

    print(f"Top character ngrams for '{class_a}': {top_a}")
    print(f"Top character ngrams for '{class_b}: {top_b}")

    overlap = set(top_a) & set(top_b)
    if overlap:
        print(f"Overlapping features that are the course of confusion: {overlap}")

def error_analysis(test_texts, test_labels, result_dict, examples_per_pair=3):
    """ Identify the three intent pairs most frequently confused by your best model
on each language"""

    language_name = result_dict["langauge"]
    test_preds = result_dict["test_preds"]
    model = result_dict["model"]
    tfidf_vec = result_dict["tfidf_vec"]
    char_vec = result_dict["char_vec"]
    vocab_vec = result_dict["vocab_size"]

    print(f"\n{'='*55}")
    print(f"  {language_name} Error Analysis")
    print(f"{'='*55}")

    # find all mistakes
    mistakes = [
    (true, pred, text)
        for text, true, pred in zip(test_texts, test_labels, test_preds)
        if true != pred
    ]
    # count which pairs are most confused
    pair_counts = {}
    for true, pred, _ in mistakes:
        pair = (true, pred)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # sort by count
    pair_counts_list = list(pair_counts.items())
    pair_counts_list.sort(key=get_count, reverse=True)
    top3 = pair_counts_list[:3]

    # print results
    print(f"\nTop 3 confused pairs:\n")
    for (true, pred), count in top3:
        print(f"  TRUE: '{true}'  →  PREDICTED: '{pred}'  ({count} times)")

        examples = [text for t, p, text in mistakes if t == true and p == pred][:examples_per_pair]
        print(f"\n  Misclassified examples:")
        for ex in examples:
            print(f"    - \"{ex}\"")

        print(f"\n  Token distributions (char n-gram coefficients):")
        show_top_features_combined(model, tfidf_vec, char_vec, vocab_size, true, pred)
        print()

error_analysis(swa_test_texts, swa_test_labels, swa_lr)
error_analysis(twi_test_texts, twi_test_labels, twi_lr)


