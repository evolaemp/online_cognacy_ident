import itertools

import igraph

import numpy as np

from online_cognacy_ident.pmi import needleman_wunsch



def igraph_clustering(matrix, threshold, method='labelprop'):
    """
    Method computes Infomap clusters from pairwise distance data.
    """

    G = igraph.Graph()
    vertex_weights = []
    for i in range(len(matrix)):
        G.add_vertex(i)
        vertex_weights += [0]

    # variable stores edge weights, if they are not there, the network is
    # already separated by the threshold
    weights = None
    for i,row in enumerate(matrix):
        for j,cell in enumerate(row):
            if i < j:
                if cell <= threshold:
                    G.add_edge(i, j, weight=1-cell, distance=cell)
                    weights = 'weight'

    if method == 'infomap':
        comps = G.community_infomap(edge_weights=weights,
                vertex_weights=None)

    elif method == 'labelprop':
        comps = G.community_label_propagation(weights=weights,
                initial=None, fixed=None)

    elif method == 'ebet':
        dg = G.community_edge_betweenness(weights=weights)
        oc = dg.optimal_count
        comps = False
        while oc <= len(G.vs):
            try:
                comps = dg.as_clustering(dg.optimal_count)
                break
            except:
                oc += 1
        if not comps:
            print('Failed...')
            comps = list(range(len(G.sv)))
            input()
    elif method == 'multilevel':
        comps = G.community_multilevel(return_levels=False)
    elif method == 'spinglass':
        comps = G.community_spinglass()

    D = {}
    for i, comp in enumerate(comps.subgraphs()):
        vertices = [v['name'] for v in comp.vs]
        for vertex in vertices:
            D[vertex] = i

    return D



def cognate_code_infomap2(d, lodict={}, gop=-2.5, gep=-1.75,
                          threshold=0.5, method='labelprop'):
    """Cluster cognates automatically.

    Calculate Needleman-Wunsch distances between forms and cluster
    them into cognate classes using the Infomap algorithm.

    d: A dict-like mapping concepts to a map from languages to forms

    lodict: A similarity matrix, a dict mapping pairs of characters to
    similarity scores

    Returns: A list of sets of (language, concept, form) triples.
    """
    codes = []
    for concept, forms_by_language in d.items():
        # Calculate the Needleman-Wunsch distance for every pair of
        # forms.
        lookup = []
        for language, forms in forms_by_language.items():
            for form in forms:
                lookup.append((concept, language, form))
        if len(lookup) <= 1:
            continue
        #print(lookup)
        distmat = np.zeros((len(lookup), len(lookup)))
        for (i1, (c1, l1, w1)), (i2, (c2, l2, w2)) in itertools.combinations(
                enumerate(lookup), r=2):
            score, align = needleman_wunsch(
                w1, w2, lodict=lodict, gop=gop, gep=gep)
            distmat[i2, i1] = distmat[i1, i2] = 1 - (1/(1 + np.exp(-score)))
            #print(w1, w2, score)

        clust = igraph_clustering(distmat, threshold, method=method)

        similaritygroups = {}
        for entry, group in clust.items():
            similaritygroups.setdefault(group, set()).add(lookup[entry])
        codes += list(similaritygroups.values())
    return codes
