import numpy as np
import networkx as nx

def random_selection(graph,nb_obs):
    nodes=list(graph.nodes())
    #nodes.remove(source)
    return np.random.choice(nodes,nb_obs, replace=False)