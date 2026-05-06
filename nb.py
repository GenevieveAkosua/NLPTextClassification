import numpy as np
import random
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from bpe import BPETokeniser
from data_utils import load_json

# set seeds
random.seed(42)
np.random.seed(42)

# load all data 
swa_train_labels,swa_train_texts = load_json('data/swa/train.jsonl')
swa_dev_labels,swa_dev_texts = load_json('data/swa/dev.jsonl')
swa_test_labels,swa_test_texts = load_json('data/swa/test.jsonl')

twi_train_labels,twi_train_texts = load_json('data/twi/train.jsonl')
twi_dev_labels,twi_dev_texts = load_json('data/twi/dev.jsonl')
twi_test_labels,twi_test_texts = load_json('data/twi/test.jsonl')

# naive bayes class, allows you to set alpha
class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.log_priors = {}
        self.log_likelihoods = {}
        self.vocabulary = set()
        self.classes = []

    def fit(self, token_lists, labels):
        # vocab and classes 
        self.classes = list(set(labels))
        for tokens in token_lists:
            self.vocabulary.update(tokens)
        vocab_size = len(self.vocabulary)

        #set up counting dictionaries and set counts to 0
        sentences_per_class = {}
        total_tokens_per_class = {}
        token_counts_per_class = {}

        for c in self.classes:
            sentences_per_class[c] = 0
            total_tokens_per_class[c] = 0
            token_counts_per_class[c] = {}

        #loop through every training sentence and count
        for tokens, label in zip(token_lists, labels):
            sentences_per_class[label] += 1
            total_tokens_per_class[label] += len(tokens)
            for token in tokens:
                token_counts_per_class[label][token] = token_counts_per_class[label].get(token, 0) + 1
                    

        #convert counts to log probabilities
        total_sentences = len(labels)

        for c in self.classes:
            # log prior
            self.log_priors[c] = np.log(
                sentences_per_class[c] / total_sentences
            )
            # log likelihoods
            self.log_likelihoods[c] = {}
            for token in self.vocabulary:
                count = token_counts_per_class[c].get(token, 0)
                self.log_likelihoods[c][token] = np.log(
                    (count + self.alpha) /
                    (total_tokens_per_class[c] + self.alpha * vocab_size)
                )
            # unk score
            self.log_likelihoods[c]["<UNK>"] = np.log(
                self.alpha /
                (total_tokens_per_class[c] + self.alpha * vocab_size)
            )

    def predict(self, token_lists):
        predictions = []
        for tokens in token_lists:
            best_intent = None
            best_score  = float('-inf')
            for c in self.classes:
                score = self.log_priors[c]
                for token in tokens:
                    if token in self.log_likelihoods[c]:
                        score += self.log_likelihoods[c][token]
                    else:
                        score += self.log_likelihoods[c]["<UNK>"]
                if score > best_score:
                    best_score = score
                    best_intent = c
            predictions.append(best_intent)
        return predictions


def tune_nb(train_texts, train_labels,dev_texts,dev_labels,test_texts, test_labels,language_name):
    """
    Grid search over k and alpha.
    Stores all 56 runs so we can use the best them for plotting.
    Evaluates best model on test once at the end.
    """
    k_values = [100, 200, 300, 500, 750, 1000, 1500]
    alpha_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

    all_runs = []   # every (k, alpha, dev_acc) needed for plotting, but only use the best
    best_acc = 0
    best_k = None
    best_alpha = None
    best_nb = None
    best_bpe = None

    for k in k_values:
        bpe = BPETokeniser()
        bpe.train(train_texts, num_merges=k)

        train_tokens = [bpe.encode(t) for t in train_texts]
        dev_tokens = [bpe.encode(t) for t in dev_texts]

        for alpha in alpha_values:
            nb = NaiveBayes(alpha=alpha)
            nb.fit(train_tokens, train_labels)

            preds = nb.predict(dev_tokens)
            dev_acc = accuracy_score(dev_labels, preds)

            all_runs.append((k, alpha, dev_acc))
            print(f"k={k:4d}, alpha={alpha:6.3f}, dev_acc={dev_acc:.4f}")

            if dev_acc > best_acc:
                best_acc = dev_acc
                best_k = k
                best_alpha = alpha
                best_nb = nb
                best_bpe = bpe

    # evaluate best model on test once after choosing best hyperparameters based on dev accuracy
    test_tokens = [best_bpe.encode(t) for t in test_texts]
    test_preds = best_nb.predict(test_tokens)
    test_acc = accuracy_score(test_labels, test_preds)

    print(f"\n{'='*10}{language_name}{'='*10}")
    print(f" BEST: k={best_k}, alpha={best_alpha}")
    print(f" dev acc:  {best_acc:.4f}")
    print(f" test acc: {test_acc:.4f}")
    print(f"{'='*30}\n")

    return {
        "language": language_name,
        "all_runs": all_runs,       
        "best_k": best_k,
        "best_alpha": best_alpha,
        "best_dev_acc": best_acc,
        "best_test_acc": test_acc,
    }




#run everything
print("=== SWAHILI ===")
swa_results = tune_nb(
    swa_train_texts, swa_train_labels,
    swa_dev_texts,swa_dev_labels,
    swa_test_texts,swa_test_labels,
    "Swahili"
)

print("=== TWI ===")
twi_results = tune_nb(
    twi_train_texts,twi_train_labels,
    twi_dev_texts,twi_dev_labels,
    twi_test_texts,twi_test_labels,
    "Twi"
)
def plot_results(swa, twi):
    """Two plots: alpha curve and k curve for both languages."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # SWA data
    swa_alphas = []
    swa_accs = []

    swa_ks = []
    swa_accs_k = []
    for k, alpha, acc in swa["all_runs"]:
        if k == swa["best_k"]:
            swa_alphas.append(alpha)
            swa_accs.append(acc)
        if alpha == swa["best_alpha"]:
            swa_ks.append(k)
            swa_accs_k.append(acc)
    

    # TWI data
    twi_alphas = []
    twi_accs = []

    twi_ks = []
    twi_accs_k = []
    for k, alpha, acc in twi["all_runs"]:
        if k == twi["best_k"]:
            twi_alphas.append(alpha)
            twi_accs.append(acc)
        if alpha == twi["best_alpha"]:
            twi_ks.append(k)
            twi_accs_k.append(acc)
    #Plotting
    ax = axes[0]
    ax.plot(swa_alphas, swa_accs, marker='o',label=f"Swahili (k={swa['best_k']})")
    ax.plot(twi_alphas,twi_accs, marker='s',label=f"Twi (k={twi['best_k']})")
    ax.set_xscale('log')
    ax.set_xlabel('Alpha (smoothing)')
    ax.set_ylabel('Dev Accuracy')
    ax.set_title('Dev Accuracy vs Alpha')
    ax.legend()
    ax.grid(True, linestyle='--',alpha=0.6)


    ax = axes[1]
    ax.plot(swa_ks, swa_accs_k, marker='o', label=f"Swahili (α={swa['best_alpha']})")
    ax.plot(twi_ks,twi_accs_k, marker='s',label=f"Twi (α={twi['best_alpha']})")

    ax.set_xlabel('k (BPE merges)')
    ax.set_ylabel('Dev Accuracy')
    ax.set_title('Dev Accuracy vs k')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('hyperparameter_curves.png', dpi=150)
    plt.show()

    # print tables
    print(f"\n=== Swahili Alpha Accuracies (at best k={swa['best_k']}) ===")
    print(f"{'Alpha':<10} | {'Dev Acc'}")
    print("-" * 25)
    for alpha, acc in zip(swa_alphas, swa_accs):
        print(f"{alpha:<10.3f} | {acc:.4f}")

    print(f"\n=== Twi Alpha Accuracies (at best k={twi['best_k']}) ===")
    print(f"{'Alpha':<10} | {'Dev Acc'}")
    print("-" * 25)
    for alpha, acc in zip(twi_alphas, twi_accs):
        print(f"{alpha:<10.3f} | {acc:.4f}")

    print(f"\n=== Swahili k Accuracies (at best alpha={swa['best_alpha']}) ===")
    print(f"{'k':<10} | {'Dev Acc'}")
    print("-" * 25)
    for k_val, acc in zip(swa_ks, swa_accs_k):
        print(f"{k_val:<10d} | {acc:.4f}")

    print(f"\n=== Twi k Accuracies (at best alpha={twi['best_alpha']}) ===")
    print(f"{'k':<10} | {'Dev Acc'}")
    print("-" * 25)
    for k_val, acc in zip(twi_ks, twi_accs_k):
        print(f"{k_val:<10d} | {acc:.4f}")

# plot both curves
plot_results(swa_results, twi_results)

# final summary table
print("\n=== FINAL RESULTS ===")
print(f"{'Language':<10} {'Best k':<8} {'Best α':<8} {'Dev Acc':<10} {'Test Acc'}")
print("-" * 50)
for r in [swa_results, twi_results]:
    print(f"{r['language']:<10} {r['best_k']:<8} {r['best_alpha']:<8} {r['best_dev_acc']:<10.4f} {r['best_test_acc']:.4f}")
