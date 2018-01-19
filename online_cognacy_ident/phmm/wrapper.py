import collections
import itertools

import numpy as np

from online_cognacy_ident.phmm.model import PairHiddenMarkov
from online_cognacy_ident.pmi import sigmoid



def chunks(some_list, n):
    """
    Yield successive size-sized chunks from some_list.
    :param some_list: list to be chunked
    :type some_list: list
    :param n: size of the chunks
    :type n: int
    :return: Generator returning chunks of size n
    :rtype: list
    """
    for i in range(0, len(some_list), n):
        yield some_list[i:i + n]



def merge(mat1, mat2, run, a):
    """
    Merge two matrices or vectors
    :param mat1: matrix or vector
    :type mat1: numpy.core.ndarray
    :param mat2: matrix or vector
    :type mat2: numpy.core.ndarray
    :param run: number of run
    :type run: int or float
    :param a: scaling parameter
    :type a: float
    :return: matrix or vector
    :rtype: numpy.core.ndarray
    """
    eta = np.power(run + 2, -a)

    return np.multiply(1 - eta, mat1) + np.multiply(eta, mat2)



def model_ll(wordpairs, em, gx, gy, tr):
    """
    calculate model likelihood of phmm using the forward algorithm
    :param wordpairs: list of wordpairs, number coded
    :type wordpairs: list
    :param em: Probabilities of sound correspondence, order as in alphabet
    :type em: np.core.ndarray
    :param gx: Probabilities of gaps in Seq1, order as in alphabet
    :type gx: np.core.ndarray
    :param gy: Probabilities of gaps in seq2, order as in alphabet
    :type gy: np.core.ndarray
    :param tr: Probabilities of state Transitions; order: delta, epsilon, lambda, tauM, tauXY
    :type tr: np.core.ndarray
    :return:
    :rtype:
    """
    sc = 0.0
    ct = 0.0
    model = PairHiddenMarkov(em, gx, gy, tr)
    for seq1, seq2 in wordpairs:
        if len(seq1) > 0 and len(seq2) > 0:
            sc += model.forward(seq1, seq2)[1]
            ct += 1
    return sc/ct



def train_phmm(dataset, initial_cutoff=0.5, alpha=0.75, batch_size=256, rt=0.0001, at=0.001, con_check=False):
    """
    Train a PHMM model using the EM algorithm with the specified parameters.

    The first arg should be a Dataset or a PairsDataset instance providing the
    word pairs that are potential cognates, i.e. having edit distance above the
    given threshold/cutoff.

    :param con_check: Check convergence thorugh change in model likelihood if set to False. Use similarity in parameters
     otherwise. If set to True, convergence tends to be slower.
    :type con_check: bool
    :param rt: relative tolerance
    :type rt: float
    :param at: absolute tolreance
    :type at: float
    :param dataset: dataset containing training data
    :type dataset: online_cognacy_ident.dataset.Dataset
    :param batch_size: chunk size for online EM
    :type batch_size: int
    :param alpha: update strength parameters
    :type alpha: float
    :param initial_cutoff: initial Levenshtein distance cutoff
    :type initial_cutoff: float
    :return: trained parameters, emission matrix, gap x, gap y, Transition
    :rtype: (np.core.ndarray, np.core.ndarray, np.core.ndarray, np.core.ndarray)
    """
    alphabet = dataset.get_alphabet()
    wordpairs = dataset.get_asjp_pairs(initial_cutoff, as_int_tuples=True)

    # create storage for new parameters, include some pseudo counts to facilitate normalization
    em_store = np.zeros((len(alphabet), len(alphabet)))
    em_store[:] = 0.0001

    g_store = np.zeros(len(alphabet))
    g_store[:] = 0.0001

    trans_store = np.array([10.0001, 10.0001, 10.0001, 10.0001, 10.0001, 10.0001, 10.0001])

    # create initial parameters
    em_input = np.ones((len(alphabet), len(alphabet)))
    em_input /= np.sum(em_input)

    gx_input = np.ones(len(alphabet))
    gx_input /= np.sum(gx_input)

    gy_input = np.ones(len(alphabet))
    gy_input /= np.sum(gy_input)

    # delta, epsilon, lambda, taum, tauxy
    trans_input = np.array([0.3, 0.3, 0.3, 0.1, 0.1])

    n_o_batches = 0.0
    converged = False
    run = 0
    ll = 0
    while converged is False:

        np.random.shuffle(wordpairs)
        word_pairs = chunks(wordpairs, batch_size)

        em_check = em_input
        gx_check = gx_input
        gy_check = gy_input
        trans_check = trans_input

        for chunk in word_pairs:

            model = PairHiddenMarkov(em_input, gx_input, gy_input, trans_input)
            new_em, new_gx, new_gy, new_trans = model.baum_welch_train(list_of_seq=chunk,
                                                                        new_em=em_store,
                                                                        new_g_probs=g_store,
                                                                        new_trans=trans_store)

            em_input = merge(em_input, new_em, n_o_batches, alpha)
            gx_input = merge(gx_input, new_gx, n_o_batches, alpha)
            gy_input = merge(gy_input, new_gy, n_o_batches, alpha)
            trans_input = merge(trans_input, new_trans, n_o_batches, alpha)

            n_o_batches += 1

        if con_check:

            results = [np.allclose(em_check, em_input, rtol=rt, atol=at), np.allclose(gx_check, gx_input, rtol=rt, atol=at),
                       np.allclose(gy_check, gy_input, rtol=rt, atol=at), np.allclose(trans_check, trans_input, rtol=rt, atol=at)]
            if False not in results:
                converged = True
        else:

            if run > 0:
                llold = ll
                ll = model_ll(wordpairs, em_input, gx_input, gy_input, trans_input)
                if np.abs(llold-ll) < at:
                    converged = True
            else:
                ll = model_ll(wordpairs, em_input, gx_input, gy_input, trans_input)

        run += 1
    return em_input, gx_input, gy_input, trans_input



def apply_phmm(dataset, em, gx, gy, trans):
    """
    Run the PHMM cognacy identification algorithm on a Dataset instance. Return
    a {(word, word): distance} dict mapping the dataset's synonymous word pairs
    to distance scores, the latter being in the range [0; 1].

    :param dataset: dataset containing training data
    :type dataset: online_cognacy_ident.dataset.Dataset
    :param em: emission probabilities
    :type em: np.core.ndarray
    :param gx: gap probabilities for state x
    :type gx: np.core.ndarray
    :param gy: gap probabilities for state y
    :type gy: np.core.ndarray
    :param trans: transition probabilities
    :type trans: np.core.ndarray
    :param equilibrium: equilibrium probabilities for random model
    :type equilibrium: np.core.ndarray
    :return: dictionary of alignment scores
    :rtype: dict
    """
    alphabet = {char: i for i, char in enumerate(dataset.get_alphabet())}

    score_dict = collections.defaultdict()
    model = PairHiddenMarkov(em, gx, gy, trans)
    equi = dataset.get_equilibrium()
    eq = np.zeros(len(alphabet))
    for k, v in alphabet.items():
        eq[v] = equi[k]
    eq /= sum(eq)

    for concept, words in dataset.get_concepts().items():
        for word1, word2 in itertools.combinations(words, 2):
            s1 = [alphabet[i] for i in word1.asjp]
            s2 = [alphabet[i] for i in word2.asjp]
            v_score = model.viterbi(s1, s2)[1]
            r_score = model.random_model(s1, s2, eq)
            key = (word1, word2) if word1 < word2 else (word2, word1)
            score_dict[key] = sigmoid(v_score / r_score)

    return score_dict
