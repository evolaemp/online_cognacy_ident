import collections

import numpy as np



def compare_cognate_codings(true_cogid_dict, other_cogid_dict):
    """Compare two different cognate codings of the same data by F scores.

    Calculate the F-scores of how well two cognate codings match each other.
    """
    f_scores = []
    # n_clusters = 0
    for concept, forms_by_language in true_cogid_dict.items():
        langs = list(forms_by_language.keys())

        predl, truel = [], []
        for l in langs:
            truel.append(true_cogid_dict[concept][l])
            predl.append(other_cogid_dict[concept][l])
        scores = b_cubed(truel, predl)

        #scores = metrics.f1_score(truel, predl, average="micro")
        print(concept, len(langs), scores, len(set(truel)), "\n")
        f_scores.append(list(scores))
        # n_clusters += len(set(clust.values()))
        #t = utils.dict2binarynexus(predicted_labels, ex_langs, lang_list)
        #print(concept, "\n",t)
        #print("No. of clusters ", n_clusters)
    #print(np.mean(np.array(f_scores), axis=0))
    f_scores = np.mean(np.array(f_scores), axis=0)
    print(f_scores[0], f_scores[1], 2.0*f_scores[0]*f_scores[1]/(f_scores[0]+f_scores[1]))
    return f_scores



def b_cubed(true_labels, labels):
    d = collections.defaultdict()
    precision = [0.0]*len(true_labels)
    recall = [0.0]*len(true_labels)

    for t, l in zip(true_labels, labels):
        d[str(l)] = t

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
    #print precision, recall
    avg_precision = np.average(precision)
    avg_recall = np.average(recall)
    avg_f_score = 2.0*avg_precision*avg_recall/(avg_precision+avg_recall)
    return avg_precision, avg_recall,avg_f_score
