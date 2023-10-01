import networkx as nx
import operator

def place_observers_btw(graph,nb_obs):
    btw=sorted(nx.betweenness_centrality(graph).items(), key=operator.itemgetter(1), reverse=True)
    return list(map(lambda x: x[0],btw[:nb_obs]))
