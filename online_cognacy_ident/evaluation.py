import numpy as np



def calc_b_cubed(true_labels, labels):
    """
    Calculate the B-cubed (precision, recall, F-score) of a list of cognate set
    labels against the gold-standard version of the same list.

    This function is a (just slightly) modified version of the b_cubed function
    of PhyloStar's CogDetect library.
    """
    precision = [0.0]*len(true_labels)
    recall = [0.0]*len(true_labels)

    for i, l in enumerate(labels):
        match = 0.0
        prec_denom = 0.0
        recall_denom = 0.0
        for j, m in enumerate(labels):
            if l == m:
                prec_denom += 1.0
                if true_labels[i] == true_labels[j]:
                    match += 1.0
                    recall_denom += 1.0
            elif l != m:
                if true_labels[i] == true_labels[j]:
                    recall_denom += 1.0
        precision[i] = match/prec_denom
        recall[i] = match/recall_denom

    avg_precision = np.average(precision)
    avg_recall = np.average(recall)
    avg_f_score = 2.0*avg_precision*avg_recall/(avg_precision+avg_recall)

    return avg_precision, avg_recall, avg_f_score



def calc_f_score(true_clusters, pred_clusters):
    """
    Calculate the B-cubed F-score of a dataset's cognate sets against their
    gold-standard. This is the metric used to evaluate the performance of the
    cognacy identification algorithms.

    Both args should be dicts mapping concepts to frozen sets of frozen sets of
    Word named tuples. The first comprises the gold-standard clustering and the
    second comprises the inferred one.

    It is assumed that both clusterings comprise the same data and that there
    is at most one word per concept per doculect. An AssertionError is raised
    if these assumptions do not hold true.
    """
    f_scores = []

    for concept in true_clusters.keys():
        assert concept in pred_clusters, str(concept)

        true_labels = {}
        pred_labels = {}

        for index, cog_set in enumerate(true_clusters[concept]):
            label = 'true:{}'.format(index)
            for word in cog_set:
                assert word.doculect not in true_labels, str(word)
                true_labels[word.doculect] = label

        for index, cog_set in enumerate(pred_clusters[concept]):
            label = 'pred:{}'.format(index)
            for word in cog_set:
                assert word.doculect not in pred_labels, str(word)
                pred_labels[word.doculect] = label

        assert set(true_labels.keys()) == set(pred_labels.keys()), str(concept)

        true_labels = [label for _, label in sorted(true_labels.items())]
        pred_labels = [label for _, label in sorted(pred_labels.items())]

        f_scores.append(calc_b_cubed(true_labels, pred_labels)[-1])

    return np.mean(f_scores)
