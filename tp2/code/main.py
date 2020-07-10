import json
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import islice
from collections import Counter
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity as cosine


def get_windows(seq, n):
    """
    Returns a sliding window (of width n) over data from the iterable
    taken from: https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator/6822773#6822773
    :param seq:
    :param n:
    :return:
    """

    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def sample_examples(docs, max_window_size, n_windows, sampling_freqs, n_negs=5, plot_dist=False):
    """
    Generate target, context pairs and negative examples
    :param docs:
    :param max_window_size:
    :param n_windows:
    :param sampling_freqs:
    :param n_negs:
    :param plot_dist:
    :return:
    """

    windows = [get_windows(doc, np.random.randint(1, max_window_size) + 1) for doc in docs]
    windows = [elt for sublist in windows for elt in sublist]  # flatten
    windows = list(np.random.choice(windows, size=n_windows))  # select a subset

    # fill the gap (sample n_negs*len(windows) negatives according to some probability distribution

    all_negs = np.random.choice(np.arange(1, len(sampling_freqs)+1),
                                p=sampling_freqs,
                                size=(n_windows, n_negs))
    return windows, all_negs


def compute_dot_products(pos, negs, target, Wc, Wt):
    # (n_pos + n_negs, d) x (d) -> (n_pos + n_negs)
    prods = Wc[pos + negs, ] @ Wt[target, ]
    return prods


def compute_loss(prodpos, prodnegs):
    """
    prodpos and prodnegs are numpy vectors containing the dot products of the context word vectors with the target word vector
    :param prodpos:
    :param prodnegs:
    :return:
    """
    term_pos = np.log(1 + np.exp(-prodpos))
    term_negs = np.log(1 + np.exp(prodnegs))
    return np.sum(term_pos) + np.sum(term_negs)


def compute_gradients(pos, negs, target, prodpos, prodnegs, Wc, Wt):
    factors_pos = 1 / (np.exp(prodpos) + 1)
    factors_negs = 1 / (np.exp(-prodnegs) + 1)

    # print('Grad :')
    # print(negs)
    # print(pos)
    # print(target)

    # fill the gaps

    partials_pos = [- Wt[target] * fp for fp in factors_pos]
    partials_negs = [Wt[target] * fn for fn in factors_negs]

    term_pos = [- Wc[p] * fp for p, fp in zip(pos, factors_pos)]
    term_negs = [Wc[neg] * fn for neg, fn in zip(negs, factors_negs)]
    partial_target = np.sum(term_pos, axis=0) + np.sum(term_negs, axis=0)

    return partials_pos, partials_negs, partial_target


def my_cos_similarity(word1, word2):
    sim = cosine(Wt[vocab[word1], ].reshape(1, -1),
                 Wt[vocab[word2], ].reshape(1, -1))
    return round(float(sim), 4)

# = = = = = = = = = = = = = = = = = = = = =


path_read = "for_moodle/output/"
path_write = path_read

stpwds = set(stopwords.words('english'))

MAX_WINDOW_SIZE = 5  # extends on both sides of the target word
N_WINDOWS = int(1e6)  # number of windows to sample at each epoch
N_NEGS = 5  # number of negative examples to sample for each positive
d = 30  # dimension of the embedding space
n_epochs = 15
lr_0 = 0.025
decay = 1e-6

train = False

with open(path_read + 'doc_ints.txt', 'r') as file:
    docs = file.read().splitlines()

docs = [[int(eltt) for eltt in elt.split()] for elt in docs]

with open(path_read + 'vocab.json', 'r') as file:
    vocab = json.load(file)

vocab_inv = {v: k for k, v in vocab.items()}

with open(path_read + 'counts.json', 'r') as file:
    counts = json.load(file)

token_ints = range(1, len(vocab) + 1)
neg_distr = [counts[vocab_inv[elt]] for elt in token_ints]
neg_distr = np.sqrt(neg_distr)
neg_distr = neg_distr / sum(neg_distr)  # normalize

# ========== train model ==========

docs = docs

if train:

    total_its = 0

    Wt = np.random.normal(size=(len(vocab) + 1, d))  # + 1 is for the OOV token
    Wc = np.random.normal(size=(len(vocab) + 1, d))

    print('Training ... \n')

    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}/{n_epochs}')

        windows, all_negs = sample_examples(docs, MAX_WINDOW_SIZE, N_WINDOWS, neg_distr)
        print('Training examples sampled')

        np.random.shuffle(windows)
        total_loss = 0

        with tqdm(total=len(windows), unit_scale=True, postfix={'loss': 0.0, 'lr': lr_0},
                  desc="Epoch : %i/%i" % (epoch + 1, n_epochs), ncols=50) as pbar:
            for i, w in enumerate(windows):

                target = w[int(len(w) / 2)]  # elt at the center
                pos = list(w)
                del pos[int(len(w) / 2)]  # all elts but the center one

                negs = all_negs[i].tolist()

                prods = compute_dot_products(pos, negs, target, Wc, Wt)
                prodpos = prods[0:len(pos), ]
                prodnegs = prods[len(pos):(len(pos) + len(negs)), ]

                partials_pos, partials_negs, partial_target = compute_gradients(pos, negs, target, prodpos,
                                                                                prodnegs, Wc, Wt)

                lr = lr_0 * 1 / (1 + decay * total_its)
                total_its += 1

                # fill the gaps (perform the updates) ###
                Wt[target, ] -= lr * partial_target
                Wc[pos, ] -= [lr * pp for pp in partials_pos]
                Wc[negs, ] -= [lr * pn for pn in partials_negs]

                total_loss += compute_loss(prodpos, prodnegs)
                pbar.set_postfix({'loss': total_loss / (i + 1), 'lr': lr})
                pbar.update(1)

    # pickle disabled for portability reasons
    np.save(path_write + 'input_vecs', Wt, allow_pickle=False)
    np.save(path_write + 'output_vecs', Wc, allow_pickle=False)

    print('word vectors saved to disk')

else:
    Wt = np.load(path_write + 'input_vecs.npy')
    Wc = np.load(path_write + 'output_vecs.npy')


# ========== sanity checks ==========

if not train:

    # = = some similarities = =
    # fill the gaps (compute the cosine similarity between some (un)related

    print("similar words:")
    print(my_cos_similarity("movie","film"))
    print("different words:")
    print(my_cos_similarity("movie", "banana"))

    # = = visualization of most frequent tokens = =

    n_plot = 500
    mft = [vocab_inv[elt] for elt in range(1, n_plot + 1)]

    # exclude stopwords and punctuation
    keep_idxs = [idx for idx, elt in enumerate(
        mft) if len(elt) > 3 and elt not in stpwds]
    mft = [mft[idx] for idx in keep_idxs]
    keep_ints = [list(range(1, n_plot + 1))[idx] for idx in keep_idxs]
    Wt_freq = Wt[keep_ints, ]

    # fill the gaps (perfom PCA (10D) followed by t-SNE (2D). For t-SNE, you can use a perplexity of 5.) ###
    # for t-SNE, see https://lvdmaaten.github.io/tsne/#faq ###
    my_pca = PCA(n_components=10)
    my_tsne = TSNE(n_components=2, perplexity=5)

    my_pca_fit = my_pca.fit_transform(Wt_freq)
    my_tsne_fit = my_tsne.fit_transform(my_pca_fit)

    fig, ax = plt.subplots()
    ax.scatter(
        my_tsne_fit[:, 0], my_tsne_fit[:, 1], s=3)
    for x, y, token in zip(my_tsne_fit[:, 0], my_tsne_fit[:, 1], mft):
        ax.annotate(token, xy=(x, y), size=8)

    fig.suptitle('t-SNE visualization of word embeddings', fontsize=20)
    fig.set_size_inches(11, 7)
    fig.savefig(path_write + 'word_embeddings.pdf', dpi=300)
    fig.show()
