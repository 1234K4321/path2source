#Generate obsever placements ln-Obs

import collections
import networkx as nx
import numpy as np

#ln-Obs
def place_observers_ln(graph, budget): 
    # choose greedly k sensors and returns the instance that maximizes detection
    # proability
    unresolvability = np.inf
    best_s = list(graph.nodes())
    dists = dict(nx.all_pairs_dijkstra_path_length(graph))
    for first in graph.nodes():
        s, classes, lengths = _approx_x(dists, first, budget)
        value = __prob_err(lengths)
        if value < unresolvability or (value == unresolvability and len(s) <
                len(best_s)):
           best_s = s
           unresolvability = value
    return best_s

#subroutine ln-Obs
def _approx_x(d, first, budget):
    s = [first]
    h = np.inf
    classes = [list(d.keys())]
    candidates = list(d.keys())
    while h != 0 and len(s) < budget:
        best_h = np.inf
        candidates.remove(s[-1])
        for new in candidates:
             lengths_new, classes_new = __len_classes(classes, d[first], d[new])
             h_with_new = __prob_err(lengths_new)
             if h_with_new < best_h:
                 best_new = new
                 best_h = h_with_new
                 best_classes = classes_new
                 best_lengths = lengths_new
        h = best_h
        classes = best_classes
        lengths = best_lengths
        s.append(best_new)
    return s, classes, lengths


def __prob_err(len_classes):
    e = 0
    n = float(sum(len_classes))
    for l in len_classes:
        e = e + (l-1)/n
    return e


def __len_classes(classes, d_first, d_new):
    new_classes = []
    for c in classes:
        new_c = collections.defaultdict(list)
        for i in c:
            t = (10**8)*(d_new[i] - d_first[i])
            new_c[t].append(i)
        new_classes.extend(new_c.values()) 
    lengths = [len(x) for x in new_classes]
    return lengths, new_classes


def __factorial(n):
    e = 1
    for i in xrange(1,n+1):
        e = e * float(i)
    return(e)


def __classes(d, b):
    u = b[0]
    vector_to_n = collections.defaultdict(list)
    for n in d.keys():
        vector_to_n[tuple(int((10**8)*(d[n][v] - d[n][u])) for v in b[1:])].append(n)
    classes = vector_to_n.values()
    return classes, [len(c) for c in classes]
