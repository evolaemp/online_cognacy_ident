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



def calc_pmi(alignment_dict, char_list, scores, initialize=False):
    """
    Calculate a pointwise mutual information dictionary from alignments.

    Given a sequence of pairwise alignments and their relative weights,
    calculate the logarithmic pairwise mutual information encoded for the
    character pairs in the alignments.

    This function is sourced from PhyloStar's OnlinePMI repository.
    """
    sound_dict = collections.defaultdict(float)
    relative_align_freq = 0.0
    relative_sound_freq = 0.0
    count_dict = collections.defaultdict(float)

    if initialize == True:
        for c1, c2 in itertools.product(char_list, repeat=2):
            if c1 == "-" or c2 == "-":
                continue
            count_dict[c1,c2] += 0.001
            count_dict[c2,c1] += 0.001
            sound_dict[c1] += 0.001
            sound_dict[c2] += 0.001
            relative_align_freq += 0.001
            relative_sound_freq += 0.002

    for alignment, score in zip(alignment_dict, scores):
        #score = 1.0
        for a1, a2 in alignment:
            if a1 == "-" or a2 == "-":
                continue
            count_dict[a1,a2] += 1.0*score
            count_dict[a2,a1] += 1.0*score
            sound_dict[a1] += 2.0*score
            sound_dict[a2] += 2.0*score
            #relative_align_freq += 2.0
            #relative_sound_freq += 2.0

    relative_align_freq = sum(list(count_dict.values()))
    relative_sound_freq = sum(list(sound_dict.values()))

    for a in count_dict.keys():
        m = count_dict[a]
        if m <=0: print(a, m)
        assert m>0

        num = np.log(m)-np.log(relative_align_freq)
        denom = np.log(sound_dict[a[0]])+np.log(sound_dict[a[1]])-(2.0*np.log(relative_sound_freq))
        val = num - denom
        count_dict[a] = val
        #count_dict[a] = val/(-1.0*num)
    return count_dict



def train_pmi(dataset, initial_cutoff=0.5, alpha=0.75, margin=1.0, max_iter=15, batch_size=256):
    """
    Train a dict mapping pairs of ASJP sounds/chars to their PMI scores on word
    pairs using the EM algorithm with the specified parameters.

    The first arg should be a Dataset or a PairsDataset instance providing the
    word pairs that are potential cognates, i.e. having edit distance above the
    given threshold/cutoff.

    This function is mostly sourced from PhyloStar's OnlinePMI repository.
    """
    word_pairs = dataset.get_asjp_pairs(initial_cutoff)
    alphabet = dataset.get_alphabet()

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
                    scores.append(1.0 - sigmoid(score))
                    pruned_word_pairs.append((word1, word2))

            mb_pmi_dict = calc_pmi(algn_list, alphabet, scores, initialize=True)
            for key, value in mb_pmi_dict.items():
                pmidict_val = pmidict[key]
                pmidict[key] = (eta*value) + (1.0-eta) * pmidict_val

            num_updates += 1

        word_pairs = list(pruned_word_pairs)
        # print('iteration {!s} (total updates: {!s})'.format(curr_iter, num_updates))

    return pmidict



def apply_pmi(dataset, pmi):
    """
    Run the PMI cognacy identification algorithm on a dataset.Dataset instance.
    Return a {(word, word): distance} dict mapping the dataset's word pairs to
    distance scores, the latter being in the range [0; 1].

    The second argument should be a matrix as returned by the train_pmi func.
    """
    scores = {}

    for concept, words in dataset.get_concepts().items():
        for word1, word2 in itertools.combinations(words, 2):
            score, _ = needleman_wunsch(word1.asjp, word2.asjp, pmi)
            score = 1 - sigmoid(score)

            key = (word1, word2) if word1 < word2 else (word2, word1)
            scores[key] = score

    return scores
