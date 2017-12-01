import collections
import itertools
import random

import numpy as np

from online_cognacy_ident.align import needleman_wunsch



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



class OnlinePMITrainer:
    """
    Trains a PMI scorer step-by-step on always improving alignments.

    This class is sourced from PhyloStar's CogDetect library.
    """

    def __init__(self, margin=1.0, alpha=0.75, gop=-2.5, gep=-1.75):
        """
        Create a persistent aligner object.

        margin: scaling factor for scores
        alpha: Decay in update weight (must be between 0.5 and 1)
        gop, gep: Gap opening and extending penalty. gop=None uses character-dependent penalties.
        """
        self.margin = margin
        self.alpha = alpha
        self.n_updates = 0
        self.pmidict = collections.defaultdict(float)
        self.gep = gep
        self.gop = gop

    def align_pairs(self, word_pairs, local=False):
        """
        Align a list of word pairs, removing those that align badly.
        """
        algn_list, scores = [], []
        n_zero = 0
        for w in range(len(word_pairs)-1, -1, -1):
            w1, w2 = word_pairs[w]
            s, alg = needleman_wunsch(
                w1, w2, gop=self.gop, gep=self.gep, lodict=self.pmidict,
                local=local)
            if s <= self.margin:
                n_zero += 1
                word_pairs.pop(w)
                continue
            algn_list.append(alg)
            scores.append(s)
        self.update_pmi_dict(algn_list, scores=scores)
        return algn_list, n_zero

    def update_pmi_dict(self, algn_list, scores=None):
        eta = (self.n_updates + 2) ** (-self.alpha)
        for k, v in calc_pmi(algn_list, scores).items():
            pmidict_val = self.pmidict.get(k, 0.0)
            self.pmidict[k] = (eta * v) + ((1.0 - eta) * pmidict_val)
        self.n_updates += 1



def run_pmi(dataset, initial_cutoff=0.5, alpha=0.75, margin=1.0, max_iter=15, batch_size=256):
    """
    Run the PMI cognacy identification algorithm on a dataset.Dataset instance.
    Return a {(word, word): distance} dict mapping the dataset's word pairs to
    distance scores, the latter being in the range [0; 1].

    The keyword args comprise the algorithm parameters.
    """
    word_pairs = list(dataset.generate_pairs(initial_cutoff))

    online = OnlinePMITrainer(alpha=alpha, margin=margin)

    for n_iter in range(0, max_iter):
        random.shuffle(word_pairs)

        index = 0
        while index < len(word_pairs):
            batch = word_pairs[index:index+batch_size]
            online.align_pairs(batch)
            word_pairs[index:index+batch_size] = batch
            index += len(batch)

        print('iteration {!s} (total updates: {!s})'.format(n_iter, online.n_updates))

    scores = {}

    for concept, words in dataset.get_concepts().items():
        for word1, word2 in itertools.combinations(words, 2):
            score, _ = needleman_wunsch(word1.asjp, word2.asjp, online.pmidict)
            score = 1 - (1/(1 + np.exp(-score)))

            key = (word1, word2) if word1 < word2 else (word2, word1)
            scores[key] = score

    return scores
