#!/usr/bin/env python
import networkx as nx
import numpy as np
   

def k_median_cost(d, observers):
    """Takes a distance matrix (in dict format) and a set of k nodes and return the k-medians
    value and the clusters assignment
    
    """
    clusterid = {i: i for i in d.keys()}
    sum_dist = 0

    for i in d.keys():
        min_dist = np.inf 
        for j in observers:
            if d[i][j] < min_dist:
                min_dist = d[i][j]
                clusterid[i] = j
        sum_dist = sum_dist + min_dist
    return clusterid, sum_dist/float(len(d))


def greedy_kmedian(graph, k):
    """At every iteration an observer is added
    such that it minimizes the cost at every step   

    """
    d = dict(nx.all_pairs_dijkstra_path_length(graph))
    observers = list()    
    for i in range(k): #for k times
        min_score = np.inf 
        candidate_to_add = None
        for candidate in d.keys():
            if candidate not in observers: #if it has not been chosen yet
                observers_ext = list(observers)
                observers_ext.extend([candidate]) #temporary placement
                tmp, score = k_median_cost(d, observers_ext)
                if score < min_score:
                    min_score = score
                    candidate_to_add = candidate
        assert candidate_to_add != None
        observers.append(candidate_to_add)
    return observers
