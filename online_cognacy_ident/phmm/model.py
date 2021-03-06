import itertools

import numpy as np



class PairHiddenMarkov(object):

    def __init__(self, em, gx, gy, trans):
        """
        
        :param em: Probabilities of sound correspondence, order as in alphabet
        :type em: np.core.ndarray
        :param gx: Probabilities of gaps in Seq1, order as in alphabet
        :type gx: np.core.ndarray
        :param gy: Probabilities of gaps in seq2, order as in alphabet
        :type gy: np.core.ndarray
        :param trans: Probabilities of state Transitions; order: delta, epsilon, lambda, tauM, tauXY
        :type trans: np.core.ndarray
        """
        self.em_probs = em
        self.gap_probs_x = gx
        self.gap_probs_y = gy
        self.trans_probs = trans

    def forward(self,
                seq1,
                seq2
                ):
        """
        Forward Algorithm: This Algorithm calculates the joint probability of all alignments, i.e. the probability of
        the sequence pair given the model
        :param seq1: Number coded sequence for alignment, i.e. x = alphabet[i] is represented as i
        :type seq1: list or tuple
        :param seq2: Number coded sequence for alignment, i.e. x = alphabet[i] is represented as i
        :type seq2: list or tuple
        :return: ForwardTrellis (3D numpy array of floats), P
        :rtype: tuple
        """

        __seq1__ = seq1
        __seq2__ = seq2
        e_m = self.em_probs[:].copy()
        g_p_x = self.gap_probs_x[:].copy()
        g_p_y = self.gap_probs_y[:].copy()

        # unpack transition probabilites
        __delta__, __epsilon__, __lambd__, __tauM__, __tauXY__ = self.trans_probs

        m = len(__seq1__)
        n = len(__seq2__)

        forward_trellis = np.zeros((m + 2, n + 2, 3))

        # initialize trellis
        forward_trellis[1][1] = ((1 - 2 * __delta__ - __tauM__), __delta__, __delta__)

        matcharr = np.array([(1 - 2 * __delta__ - __tauM__), (
            1 - __epsilon__ - __tauXY__ - __lambd__), (
                                 1 - __epsilon__ - __tauXY__ - __lambd__)])
        xarr = np.array([__delta__, __epsilon__, __lambd__])
        yarr = np.array([__delta__, __lambd__, __epsilon__])

        inds = itertools.product(list(range(len(__seq1__) + 2))[1:], list(range(len(__seq2__) + 2))[1:])

        for __posi__, __posj__ in inds:

            x = __seq1__[__posi__ - 2]
            y = __seq2__[__posj__ - 2]

            # initialization
            if __posi__ == 1 and __posj__ == 1:
                continue

            if __posi__ > 1 and __posj__ > 1:
                # Matchstate
                forward_trellis[__posi__][__posj__][0] = e_m[(x, y)] * np.sum(
                    np.multiply(forward_trellis[__posi__ - 1][__posj__ - 1], matcharr))
            else:
                forward_trellis[__posi__][__posj__][0] = 0.0

            if __posi__ > 1:
                # state X
                forward_trellis[__posi__][__posj__][1] = g_p_x[x] * np.sum(
                    np.multiply(forward_trellis[__posi__ - 1][__posj__], xarr))
            else:
                forward_trellis[__posi__][__posj__][1] = 0.0

            if __posj__ > 1:
                # State Y
                forward_trellis[__posi__][__posj__][2] = g_p_y[y] * np.sum(
                    np.multiply(forward_trellis[__posi__][__posj__ - 1], yarr))
            else:
                forward_trellis[__posi__][__posj__][2] = 0.0

        p = __tauM__ * forward_trellis[m + 1][n + 1][0] + __tauXY__ * (
            forward_trellis[m + 1][n + 1][1] + forward_trellis[m + 1][n + 1][2])

        return forward_trellis, p

    def backward(self,
                 seq1,
                 seq2
                 ):
        """
        Backward Algorithm: This Algorithm calculates the joint probability of all alignments, i.e. the probability of 
        the sequence pair given the model
        :param seq1: Number coded sequence for alignment, i.e. x = alphabet[i] is represented as i
        :type seq1: tuple or list
        :param seq2: Number coded sequence for alignment, i.e. x = alphabet[i] is represented as i
        :type seq2: tuple or list
        :return: ForwardTrellis (3D numpy array of floats), P
        :rtype: tuple
        """

        __seq1__ = seq1
        __seq2__ = seq2
        e_m = self.em_probs[:].copy()
        g_p_x = self.gap_probs_x[:].copy()
        g_p_y = self.gap_probs_y[:].copy()

        __delta__, __epsilon__, __lambd__, __tauM__, __tauXY__ = self.trans_probs

        m = len(__seq1__)
        n = len(__seq2__)

        backward_trellis = np.zeros((m + 2, n + 2, 4))

        backward_trellis[m][n] = (__tauM__, __tauXY__, __tauXY__, 1.0)  # match state

        inds = itertools.product(reversed(list(range(len(__seq1__) + 1))), reversed(list(range(len(__seq2__) + 1))))
        for __posi__, __posj__ in inds:

            if __posi__ == m and __posj__ == n:

                continue

            elif __posi__ == m:

                y = __seq2__[__posj__]

                __prevY__ = backward_trellis[__posi__][__posj__ + 1][2] * g_p_y[y]

                backward_trellis[__posi__][__posj__] = (
                    __delta__ * __prevY__, __lambd__ * __prevY__, __epsilon__ * __prevY__,
                    0.0)  # assignment to match state

            elif __posj__ == n:

                x = __seq1__[__posi__]

                __prevX__ = backward_trellis[__posi__ + 1][__posj__][1] * g_p_x[x]

                backward_trellis[__posi__][__posj__] = (
                    __delta__ * __prevX__, __epsilon__ * __prevX__, __lambd__ * __prevX__, 0.0)

            else:
                x = __seq1__[__posi__]
                y = __seq2__[__posj__]

                __prevM__ = backward_trellis[__posi__ + 1][__posj__ + 1][0] * e_m[(x, y)]
                __prevX__ = backward_trellis[__posi__ + 1][__posj__][1] * g_p_x[x]
                __prevY__ = backward_trellis[__posi__][__posj__ + 1][2] * g_p_y[y]

                backward_trellis[__posi__][__posj__][0] = \
                    (1 - 2 * __delta__ - __tauM__) * __prevM__ + __delta__ * (__prevX__ + __prevY__)

                backward_trellis[__posi__][__posj__][1] = \
                    (
                        1 - __epsilon__ - __lambd__ - __tauXY__) * __prevM__ + __epsilon__ * __prevX__ + __lambd__ * __prevY__

                backward_trellis[__posi__][__posj__][2] = \
                    (
                        1 - __epsilon__ - __lambd__ - __tauXY__) * __prevM__ + __lambd__ * __prevX__ + __epsilon__ * __prevY__

                backward_trellis[__posi__][__posj__][3] = 0.0

        return backward_trellis

    def viterbi(self,
                seq1,
                seq2
                ):
        """
        Viterbi Algorithm: This Algorithm calculates the score of the most optimal alignment
        This function does no traceback, nor does it provide a traceback matrix.
        :param seq1: Number coded sequence for alignment, i.e. x = alphabet[i] is represented as i
        :type seq1: tuple or list
        :param seq2: Number coded sequence for alignment, i.e. x = alphabet[i] is represented as i
        :type seq2: tuple or list
        :return: ForwardTrellis (3D numpy array of floats), P
        :rtype: tuple
        """

        __seq1__ = seq1
        __seq2__ = seq2

        e_m = self.em_probs[:].copy()
        g_p_x = self.gap_probs_x[:].copy()
        g_p_y = self.gap_probs_y[:].copy()

        __delta__, __epsilon__, __lambd__, __tau_m__, __tau_x_y__ = self.trans_probs

        m = len(__seq1__)
        n = len(__seq2__)

        viterbi_trellis = np.zeros((m + 1, n + 1, 3))

        viterbi_trellis[0][0] = ((1 - 2 * __delta__ - __tau_m__), __delta__, __delta__)

        # Initialization of the first string
        subst = __delta__ * viterbi_trellis[0][0][0]
        delete = __epsilon__ * viterbi_trellis[0][0][1]
        insert = __lambd__ * viterbi_trellis[0][0][2]

        # calculate first value
        sc = g_p_x[__seq1__[0]]
        viterbi_trellis[1][0][1] = sc * max([subst, delete, insert])

        # iterate for the remaining cells

        for ind, itm in itertools.islice(enumerate(__seq1__, start=1), 1, None):
            viterbi_trellis[ind][0][1] = g_p_x[itm] * __epsilon__ * viterbi_trellis[ind - 1][0][1]

        # Initialization of the second string
        delete = __lambd__ * viterbi_trellis[0][0][1]
        insert = __epsilon__ * viterbi_trellis[0][0][2]

        # calculate the first value
        sc = g_p_y[__seq2__[0]]

        viterbi_trellis[1][0][2] = sc * max([subst, delete, insert])

        # iterate for the remaining cells
        for ind, itm in itertools.islice(enumerate(__seq2__, start=1), 1, None):
            viterbi_trellis[0][ind][2] = g_p_y[itm] * __epsilon__ * viterbi_trellis[0][ind - 1][2]

        for __posi__, x in enumerate(__seq1__, start=1):

            for __posj__, y in enumerate(__seq2__, start=1):
                # Matchstate
                __prevM__ = viterbi_trellis[__posi__ - 1][__posj__ - 1][0] * (1 - 2 * __delta__ - __tau_m__)
                __prevX__ = viterbi_trellis[__posi__ - 1][__posj__ - 1][1] * (1 - __epsilon__ - __tau_x_y__ - __lambd__)
                __prevY__ = viterbi_trellis[__posi__ - 1][__posj__ - 1][2] * (1 - __epsilon__ - __tau_x_y__ - __lambd__)

                # State X
                __XprevM__ = viterbi_trellis[__posi__ - 1][__posj__][0] * __delta__
                __XprevX__ = viterbi_trellis[__posi__ - 1][__posj__][1] * __epsilon__
                __XprevY__ = viterbi_trellis[__posi__ - 1][__posj__][2] * __lambd__

                # State Y
                __YprevM__ = viterbi_trellis[__posi__][__posj__ - 1][0] * __delta__
                __YprevX__ = viterbi_trellis[__posi__][__posj__ - 1][1] * __lambd__
                __YprevY__ = viterbi_trellis[__posi__][__posj__ - 1][2] * __epsilon__

                viterbi_trellis[__posi__][__posj__] = [e_m[(x, y)] * max([__prevM__,
                                                                          __prevX__,
                                                                          __prevY__]),
                                                       g_p_x[x] * max([__XprevM__,
                                                                       __XprevX__,
                                                                       __XprevY__]),
                                                       g_p_y[y] * max([__YprevM__,
                                                                       __YprevX__,
                                                                       __YprevY__])]

        p = max(
            [__tau_m__ * viterbi_trellis[m][n][0], __tau_x_y__ * viterbi_trellis[m][n][1],
             __tau_x_y__ * viterbi_trellis[m][n][2]])

        return viterbi_trellis, p

    def random_model(self,
                     seq1, seq2, eq_probs):
        """
        Calculate the similarity of the two strings under the random model
        :param seq1: Number coded sequence for alignment, i.e. x = alphabet[i] is represented as i
        :type seq1: tuple
        :param seq2: Number coded sequence for alignment, i.e. x = alphabet[i] is represented as i
        :type seq2: tuple
        :param eq_probs: equilibrium probabilities of the sounds, order as in alphabet
        :type eq_probs: np.core.ndarray
        :return: probability of relatedness under the random model
        :rtype: float
        """

        lg = np.float(len(seq1) + len(seq2))

        l = lg / 2.0
        eta = 1.0 / (l + 1.0)

        p1 = np.product(eq_probs[seq1])
        p2 = np.product(eq_probs[seq2])

        return np.power(eta, 2) * np.power(1 - eta, lg) * p1 * p2

    def baum_welch_train(self,
                         list_of_seq,
                         new_em,
                         new_g_probs,
                         new_trans,
                         weight=1.0):
        """
        Perform Baum-Welch training for PHMM
        :param list_of_seq: list of pairs of number coded sequences for training
        :type list_of_seq: list of tuple or list of list
        :param new_em: Storage for probabilities of sound correspondence, order as in alphabet
        :type new_em:  np.core.ndarray
        :param new_g_probs: Storage for probabilities of gaps, order as in alphabet
        :type new_g_probs:  np.core.ndarray
        :param new_trans: Storage for state Transitions; order: delta, epsilon, lambda, tauM, tauXY
        :type new_trans:  np.core.ndarray
        :param weight: factor for weighing training iteration
        :type weight: float
        :return: new trained parameters
        :rtype: (np.core.ndarray,np.core.ndarray,np.core.ndarray,np.core.ndarray)
        """

        e_m = self.em_probs[:].copy()
        g_p_x = self.gap_probs_x[:].copy()
        g_p_y = self.gap_probs_y[:].copy()

        __delta__, __epsilon__, __lambd__, __tau_m__, __tau_x_y__ = self.trans_probs

        new_e_m = new_em[:].copy()
        newg_probs = new_g_probs[:].copy()
        #newgy_probs = new_g_probs[:].copy()
        new_trans_probs = new_trans[:].copy()

        new_delta, new_epsilon, new_lambd, new_tau_m, new_tau_x_y, extra_m, extra_x_y = new_trans_probs

        for seq1, seq2 in list_of_seq:

            if len(seq1) > 0 and len(seq2) > 0:

                fwd_trellis, p = self.forward(seq1, seq2)

                bwd_trellis = self.backward(seq1, seq2)

                inv_p = (1.0 * weight) / p

                for i, j in itertools.product(list(range(len(seq1))), list(range(len(seq2)))):
                    x = seq1[i]
                    y = seq2[j]

                    # temporary variables
                    fw_ij = fwd_trellis[i + 2][j + 2]
                    bw_i1 = bwd_trellis[i + 1]
                    bw_i2 = bwd_trellis[i + 2]

                    new_e_m[(x, y)] += inv_p * fw_ij[0] * bw_i1[j + 1][0]
                    new_e_m[(y, x)] += inv_p * fw_ij[0] * bw_i1[j + 1][0]
                    newg_probs[x] += inv_p * fw_ij[1] * bw_i1[j + 1][1]
                    newg_probs[y] += inv_p * fw_ij[2] * bw_i1[j + 1][2]

                    # calculate new transition probabilities
                    if (i != len(seq1) - 1) and (j != len(seq2) - 1):
                        y = seq2[j + 1]
                        x = seq1[i + 1]

                        extra_m += inv_p * fw_ij[0] * (1 - 2 * __delta__ - __tau_m__) * e_m[(x, y)] * bw_i2[j + 2][0]
                        extra_x_y += \
                            inv_p * fw_ij[1] * (1 - __epsilon__ - __lambd__ - __tau_x_y__) * e_m[(x, y)] * bw_i2[j + 2][0]

                    elif j != len(seq2) - 1:
                        y = seq2[j + 1]
                        new_lambd += inv_p * fw_ij[1] * __lambd__ * g_p_y[y] * bw_i1[j + 2][2]

                    elif i != len(seq1) - 1:
                        x = seq1[i + 1]
                        new_epsilon += inv_p * fw_ij[1] * __epsilon__ * g_p_x[x] * bw_i2[j + 1][1]
                        new_delta += inv_p * fw_ij[0] * __delta__ * g_p_x[x] * bw_i2[j + 1][1]

                    new_tau_m += inv_p * fw_ij[0] * __tau_m__ * bw_i1[j + 1][3]
                    new_tau_x_y += inv_p * fw_ij[1] * __tau_x_y__ * bw_i1[j + 1][3]

        trans_count = np.array([new_delta, new_epsilon, new_lambd, new_tau_m, new_tau_x_y, extra_m, extra_x_y])

        # normalize values
        new_e_m /= np.sum(new_e_m)
        newgx_probs = newg_probs/ np.sum(newg_probs)
        newgy_probs = newg_probs/np.sum(newg_probs)

        trans_count = self.normalize_transition(trans_count)

        return new_e_m, newgx_probs, newgy_probs, trans_count

    def normalize_transition(self, trans):
        """
        Normalize the transition values
        :param trans: count of transitions according to Baum-Welch transition
        :type trans: np.core.ndarray
        :return: Normalized, i.e. outgoing transitions sum to 1
        :rtype: np.core.ndarray
        """
        mfac = np.array([2, 0, 0, 1, 0, 1, 0])
        xfac = np.array([0, 1, 1, 0, 1, 0, 1])
        m_norm = np.sum(trans * mfac)
        x_norm = np.sum(trans * xfac)
        return trans[:5]/np.array([m_norm, x_norm, x_norm, m_norm, x_norm])
