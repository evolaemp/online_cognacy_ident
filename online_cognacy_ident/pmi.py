import collections
import itertools
import random

import numpy as np



def normalized_leventsthein(a, b):
    """
    Leventsthein distance normalized
    :param a: word
    :type a: str
    :param b: word
    :type b: str
    :return: distance score
    :rtype: float
    """
    m = [];
    la = len(a) + 1;
    lb = len(b) + 1
    for i in range(0, la):
        m.append([])
        for j in range(0, lb): m[i].append(0)
        m[i][0] = i
    for i in range(0, lb): m[0][i] = i
    for i in range(1, la):
        for j in range(1, lb):
            s = m[i - 1][j - 1]
            if (a[i - 1] != b[j - 1]): s = s + 1
            m[i][j] = min(m[i][j - 1] + 1, m[i - 1][j] + 1, s)
    la = la - 1;
    lb = lb - 1
    return float(m[la][lb])/ float(max(la, lb))



def needleman_wunsch(x, y, lodict={}, gop=-2.5, gep=-1.75, local=False, indel=''):
    """Needleman-Wunsch algorithm with affine gaps penalties.

    This code implements the NW algorithm for pairwise string
    alignment with affine gap penalties.

    'lodict' must be a dictionary with all symbol pairs as keys and
    match scores as values, or a False value (including an empty
    dictionary) to denote (-1, 1) scores. gop and gep are gap
    penalties for opening/extending a gap; alternatively, you can set
    'gop' to None and provide element/gap alignment costs.
    indel takes the character used to denote an indel.

    Returns the alignment score and one optimal alignment.
    """
    n, m = len(x), len(y)
    dp = np.zeros((n + 1, m + 1))
    pointers = np.zeros((n + 1, m + 1), np.int32)
    if not local:
        for i1, c1 in enumerate(x):
            if gop is None:
                dp[i1 + 1, 0] = lodict.get((c1, indel), gep)
            else:
                dp[i1 + 1, 0] = dp[i1, 0]+(gep if i1 + 1 > 1 else gop)
            pointers[i1 + 1, 0] = 1
        for i2, c2 in enumerate(y):
            if gop is None:
                dp[0, i2 + 1] = lodict.get((indel, c2), gep)
            else:
                dp[0, i2 + 1] = dp[0, i2]+(gep if i2 + 1 > 1 else gop)
            pointers[0, i2 + 1] = 2
    for i1, c1 in enumerate(x):
        for i2, c2 in enumerate(y):
            match = dp[i1, i2] + lodict.get(
                (c1, c2),
                1 if c1 == c2 else -1)
            insert = dp[i1, i2 + 1] + (
                lodict.get((c1, indel), gep) if gop is None else
                gep if pointers[i1, i2 + 1] == 1 else gop)
            delet = dp[i1 + 1, i2] + (
                lodict.get((indel, c2), gep) if gop is None else
                gep if pointers[i1 + 1, i2] == 2 else gop)
            pointers[i1 + 1, i2 + 1] = p = np.argmax([match, insert, delet])
            max_score = [match, insert, delet][p]
            if local and max_score < 0:
                max_score = 0
            dp[i1 + 1, i2 + 1] = max_score
    alg = []
    if local:
        i, j = np.unravel_index(dp.argmax(), dp.shape)
    else:
        i, j = n, m
    score = dp[i, j]
    while (i > 0 or j > 0):
        pt = pointers[i, j]
        if pt == 0:
            i -= 1
            j -= 1
            alg = [(x[i], y[j])] + alg
        if pt == 1:
            i -= 1
            alg = [(x[i], indel)] + alg
        if pt == 2:
            j -= 1
            alg = [(indel, y[j])] + alg
        if local and dp[i, j] == 0:
            break
    return score, alg



def calc_pmi(alignments, scores=None):
    """Calculate a pointwise mutual information dictionary from alignments.

    Given a sequence of pairwaise alignments and their relative
    weights, calculate the logarithmic pairwise mutual information
    encoded for the character pairs in the alignments.
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
    """Train a PMI scorer step-by-step on always improving alignments."""

    def __init__(self, margin=1.0, alpha=0.75, gop=-2.5, gep=-1.75):
        """Create a persistent aligner object.

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
        """Align a list of word pairs, removing those that align badly."""
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



def train(dataset, alpha=0.75, margin=1.0, max_iter=15, max_batch=256):
    """
    """
    word_pairs = [pair for pair in dataset.generate_pairs()
        if normalized_leventsthein(pair[0], pair[1]) <= 0.5]

    online = OnlinePMITrainer(alpha=alpha, margin=margin)

    print("Calculating PMIs from very similar words.")
    for n_iter in range(0, max_iter):
        random.shuffle(word_pairs)
        print("Iteration", n_iter)
        idx = 0
        n_zero = 0
        while idx < len(word_pairs):
            #print("Mini Batch", idx)
            wl = word_pairs[idx:idx+max_batch]
            algn_list, z = online.align_pairs(wl)
            n_zero += z
            word_pairs[idx:idx+max_batch] = wl
            idx += len(wl)

        print("Non zero examples went down to {:d} (-{:d}). Updates: {:d}".format(
            len(word_pairs), n_zero, online.n_updates))
        print(collections.Counter(online.pmidict).most_common(8)[::2])
