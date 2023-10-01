import numpy as np
import networkx as nx
import copy
import pandas as pd

def euclidean_dist_selection(graph, nb_obs,mc=False, distribution=None, exp=False):
    n=graph.number_of_nodes()
    nodes=list(graph.nodes())
    #nodes.remove(source)

    if mc:
        DIFFUSION = 13
        path_lengths = preprocess(nodes, graph, distribution, DIFFUSION)
        dists = compute_mean_shortest_path(path_lengths)
    else:
        dists=dict(nx.all_pairs_dijkstra_path_length(graph))
    tmp_dists=np.zeros((n,n,n))
    for cand in nodes:
        for i in range(n):
            for j in range(i):
                tmp_dists[cand][i][j]+=(dists[cand][i]-dists[cand][j])**2
                tmp_dists[cand][j][i]+=(dists[cand][i]-dists[cand][j])**2
    
    sel=[]
    tmp_s_dists=np.zeros((n,n))
    for o in range(nb_obs):
        max_value=-np.inf
        max_cand=-1
        np.random.shuffle(nodes)
        for cand in nodes:
            tmp_o_dists=tmp_s_dists+tmp_dists[cand] #  shape: (n,n)
            if exp:
                value = -np.sum(np.exp(-tmp_o_dists))
            else:            
                value = np.sum(np.sqrt(tmp_o_dists))

            if value>max_value:
                max_value=value
                max_cand=cand
        
        try:        
            sel.append(max_cand)
            nodes.remove(max_cand)
            tmp_s_dists=tmp_s_dists+tmp_dists[max_cand]
        except:
            print("max_value ",max_value,"max_cand ",max_cand)
    return sel    
    
    
'''
Apply the given distribution to the edge of the graph and then create a dataframe to store
shortest path from every observer to every nodes in the graph
PARAMETERS:
    observer: the observer node
    graph: the nx graph used
    distr: the distribution used
    nb_diffusions: (int) number of time we do the diffusion
Return pandas.DataFrame
'''
def preprocess(observer, graph, distr, nb_diffusions):
    path_lengths = pd.DataFrame()
    rvs = list(distr.rvs(size=nb_diffusions*graph.number_of_edges()))
    for diff in range(nb_diffusions):
        path_lengths_temp = pd.DataFrame()
        for i,(u, v) in enumerate(graph.edges()):
            graph[u][v]['weight'] = rvs[diff*graph.number_of_edges()+i]
        
        for o in observer:
            ### Computation of the shortest paths from every observer to all other nodes
            path_lengths_temp[o] = pd.Series(nx.single_source_dijkstra_path_length(graph, o))
        path_lengths = path_lengths.append(path_lengths_temp)
    return path_lengths



'''
Compute the mean shortest path of every diffusion
PARAMETERS:
    path_lengths:(pandas.DataFrame) containing all shortest path from every diffusion
RETURN: dictionnary of dictionnary: {obs: {node: mean length}}
'''
def compute_mean_shortest_path(path_lengths):
    path_lengths.reset_index(inplace = True)
    path_lengths = path_lengths.rename({'index': 'node'}, axis = 1).set_index('node')
    return path_lengths.groupby(['node']).mean().to_dict()

