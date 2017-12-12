import collections
import itertools
import random

import numpy as np

from online_cognacy_ident.align import needleman_wunsch



def sigmoid(x):
    """
    Implementation of the common sigmoid function, the logistic function with
    standard parameters (L=1, k=1, x₀=0).
    """
    return 1.0 / (1.0 + np.exp(-x))



def calc_pmi(alignments, scores=None):
    """
    Calculate a pointwise mutual information dictionary from alignments.

    Given a sequence of pairwise alignments and their relative weights,
    calculate the logarithmic pairwise mutual information encoded for the
    character pairs in the alignments.

    This function is sourced from PhyloStar's CogDetect library.
    """
    if scores is None:
        scores = itertools.cycle([1])

    sound_dict = collections.defaultdict(float)
    count_dict = collections.defaultdict(float)

    for alignment, score in zip(alignments, scores):
        for a1, a2 in alignment:
            if a1 == "" or a2 == "":
                continue
            count_dict[a1, a2] += 1.0*score
            count_dict[a2, a1] += 1.0*score
            sound_dict[a1] += 2.0*score
            sound_dict[a2] += 2.0*score

    log_weight = 2 * np.log(sum(list(
        sound_dict.values()))) - np.log(sum(list(
            count_dict.values())))

    for (c1, c2) in count_dict.keys():
        m = count_dict[c1, c2]
        #assert m > 0

        num = np.log(m)
        denom = np.log(sound_dict[c1]) + np.log(sound_dict[c2])
        val = num - denom + log_weight
        count_dict[c1, c2] = val

    return count_dict



def train_pmi(word_pairs, alpha=0.75, margin=1.0, max_iter=15, batch_size=256):
    """
    Train the PMI dict mapping pairs of ASJP sounds to their PMI scores on word
    pairs using the EM algorithm with the specified parameters.

    This function is mostly sourced from PhyloStar's OnlinePMI repository.
    """
    pmidict = collections.defaultdict(float)
    num_updates = 0

    for curr_iter in range(max_iter):
        random.shuffle(word_pairs)
        pruned_word_pairs = []

        for index in range(0, len(word_pairs), batch_size):
            eta = np.power(num_updates+2, -alpha)
            algn_list, scores = [], []

            for word1, word2 in word_pairs[index:index+batch_size]:
                score, alg = needleman_wunsch(word1, word2, pmidict)

                if score > margin:
                    algn_list.append(alg)
                    scores.append(score)
                    pruned_word_pairs.append((word1, word2))

            for key, value in calc_pmi(algn_list, scores).items():
                pmidict_val = pmidict[key]
                pmidict[key] = (eta*value) + (1.0-eta) * pmidict_val

            num_updates += 1

        word_pairs = list(pruned_word_pairs)
        print('iteration {!s} (total updates: {!s})'.format(curr_iter, num_updates))

    return pmidict



def run_pmi(dataset, initial_cutoff=0.5, alpha=0.75, margin=1.0, max_iter=15, batch_size=256):
    """
    Run the PMI cognacy identification algorithm on a dataset.Dataset instance.
    Return a {(word, word): distance} dict mapping the dataset's word pairs to
    distance scores, the latter being in the range [0; 1].

    The keyword args comprise the algorithm parameters.
    """
    word_pairs = dataset.get_asjp_pairs(initial_cutoff)
    pmi = train_pmi(word_pairs, alpha, margin, max_iter, batch_size)

    scores = {}

    for concept, words in dataset.get_concepts().items():
        for word1, word2 in itertools.combinations(words, 2):
            score, _ = needleman_wunsch(word1.asjp, word2.asjp, pmi)
            score = 1 - sigmoid(score)

            key = (word1, word2) if word1 < word2 else (word2, word1)
            scores[key] = score

    return scores
