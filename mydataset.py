import collections

import itertools

from online_cognacy_ident.align import normalized_levenshtein as ldn
class dataset(object):

    def __init__(self, datei):
        self.datei = datei

    def get_asjp_pairs(self, initial_cutoff, as_int_tuples=True):
        exc = ["%", "~", "*", "$", "\"", " "]

        with open(self.datei, "r") as infile:
            cont = infile.readlines()
        meta = cont.pop(0)
        data = []
        while cont:
            line = cont.pop(0).split("\t")

            data.append([line[i] for i in range(len(line)) if i in [0, 2, 5, 6]])

        data = [[i[0], i[1], tuple([j for j in i[2] if j not in exc]), i[3]] for i in data]
        data = [i for i in data if len(i[2]) != 0]
        alphabet = sorted(list(set([j for i in data for j in i[2]])))
        self.alphabet = alphabet

        retdict = collections.OrderedDict()
        for i in data:
            if i[1] in retdict.keys():
                retdict[i[1]].append([i[0], i[2]])
            else:
                retdict[i[1]] = [[i[0], i[2]]]
        collections.OrderedDict
        wpairs = []
        npairs = 0.0
        for key, value in retdict.items():

            for p1, p2 in itertools.combinations(value, 2):
                score = ldn(p1[1], p2[1])
                npairs += 1
                if score <= 0.5:
                    w1 = [self.alphabet.index(i) for i in p1[1]]
                    w2 = [self.alphabet.index(i) for i in p2[1]]
                    wpairs.append((w1, w2))

        return wpairs

    def get_alphabet(self):

        return self.alphabet
