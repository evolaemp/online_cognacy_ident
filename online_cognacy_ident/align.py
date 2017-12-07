def normalized_levenshtein(a, b):
    """
    Levenshtein distance normalized
    :param a: word
    :type a: str
    :param b: word
    :type b: str
    :return: distance score
    :rtype: float

    This function is sourced from PhyloStar's CogDetect library.
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



def needleman_wunsch(seq_a, seq_b, scores={}, gop=-2.5, gep=-1.75):
    """
    Align two sequences using a flavour of the Needleman-Wunsch algorithm with
    fixed gap opening and gap extension penalties, attributed to Gotoh (1994).

    The scores arg should be a (char_a, char_b): score dict; if a char pair is
    missing, 1/-1 are used as match/mismatch scores.

    Return the best alignment score and one optimal alignment.
    """
    matrix = {}  # (x, y): (score, back)

    for y in range(len(seq_b) + 1):
        for x in range(len(seq_a) + 1):
            cands = []  # [(score, back), ..]

            if x > 0:
                score = matrix[(x-1, y)][0] \
                    + (gep if matrix[(x-1, y)][1] == '←' else gop)
                cands.append((score, '←'))

            if y > 0:
                score = matrix[(x, y-1)][0] \
                    + (gep if matrix[(x, y-1)][1] == '↑' else gop)
                cands.append((score, '↑'))

            if x > 0 and y > 0:
                if (seq_a[x-1], seq_b[y-1]) in scores:
                    score = scores[(seq_a[x-1], seq_b[y-1])]
                else:
                    score = 1 if seq_a[x-1] == seq_b[y-1] else -1
                score += matrix[(x-1, y-1)][0]
                cands.append((score, '.'))
            elif x == 0 and y == 0:
                cands.append((0.0, '.'))

            matrix[(x, y)] = max(cands)

    alignment = []

    while (x, y) != (0, 0):
        if matrix[(x, y)][1] == '←':
            alignment.append((seq_a[x-1], ''))
            x -= 1
        elif matrix[(x, y)][1] == '↑':
            alignment.append(('', seq_b[y-1]))
            y -= 1
        else:
            alignment.append((seq_a[x-1], seq_b[y-1]))
            x, y = x-1, y-1

    return matrix[(len(seq_a), len(seq_b))][0], tuple(reversed(alignment))
