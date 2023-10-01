import numpy as np
import networkx as nx
import copy
import pandas as pd
import operator

def mahalanobis_dist_mc_selection(graph, nb_obs, distribution=None, exp=False):
    n=graph.number_of_nodes()
    nodes=list(graph.nodes())
    #nodes.remove(source)
    ref_obs = sorted(nx.betweenness_centrality(graph).items(), key=operator.itemgetter(1), reverse=True)[0][0]    

    DIFFUSION = 13
    path_lengths = preprocess(nodes, graph, distribution, DIFFUSION)
    dists = compute_mean_shortest_path(path_lengths)

    sel=[]
    nodes.remove(ref_obs)
    for o in range(nb_obs-1):
        max_value=0
        max_cand=-1
        np.random.shuffle(nodes)
        for cand in nodes:
            value=0
            for i in range(n):
                for j in range(i):
                    ### computing mu_i[sel] - mu_j[sel]
                    tmp_diff = [(dists[obs][i]-dists[ref_obs][i])-(dists[obs][j]-dists[ref_obs][j]) for obs in sel]
                    tmp_diff += [(dists[cand][i]-dists[ref_obs][i])-(dists[cand][j]-dists[ref_obs][j])]
                    ### Computing the covariance matrix
                    cov_d_s = cov_matrix(path_lengths, sel+[cand], i, ref_obs)
                    if len(tmp_diff)>1:
                        tmp_diff=np.array(tmp_diff).reshape((len(tmp_diff),1))
                        cov_d_s_inv = np.linalg.inv(cov_d_s)
                        if exp:
                            value -= np.exp(-tmp_diff.T @ cov_d_s_inv @ tmp_diff)
                        else:
                            value += np.sqrt(tmp_diff.T @ cov_d_s_inv @ tmp_diff)
                    else:
                        if cov_d_s==0:
                            print(i,j,cov_d_s)
                            print(path_lengths[i][j])
                            
                        if exp:
                            value -= np.exp(-tmp_diff[0]**2/cov_d_s)
                        else:
                            value += tmp_diff[0]/np.sqrt(cov_d_s)
                    
            if value>max_value:
                max_value=value
                max_cand=cand
        
        try:
            sel.append(max_cand)
            nodes.remove(max_cand)
        except:
            print("sel ",sel)
            print("max_cand ",max_cand)
            print("nodes",nodes) 
    return sel+[ref_obs]



'''
Make a certain number of diffusion in order approximate the path length between any node to a observer.
PARAMETERS:
    - observers: list of observers
    - graph: unweighted netwrokx graph
    - distr: scipy.stats object representing the edge delay distribution
    - nb_diffusions: number of  diffusions that have to be made
OUTPUT:
    - path_lengths: Pandas dataframe representing the path length between a node (present in the rows
    of the dataframe) and a observer node (present in the column of the dataframe) for every diffusion.
'''
def preprocess(observers, graph, distr, nb_diffusions):
    graph_copy = graph.copy()
    path_lengths = pd.DataFrame()
    for diff in range(nb_diffusions):
        path_lengths_temp = pd.DataFrame()
        ### edge delay
        edges = graph_copy.edges()
        for (u, v) in edges:
            graph_copy[u][v]['weight'] = abs(distr.rvs())
        for o in observers:
            ### Computation of the shortest paths from every observer to all other nodes
            path_lengths_temp[o] = pd.Series(nx.single_source_dijkstra_path_length(graph_copy, o))
        path_lengths = path_lengths.append(path_lengths_temp, sort = False)
        path_lengths.reset_index(inplace = True)
        path_lengths = path_lengths.rename({'index': 'node'}, axis = 1).set_index('node')
    return path_lengths


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
    path_lengths = path_lengths.reset_index()
    path_lengths = path_lengths.rename({'index': 'node'}, axis = 1).set_index('node')
    return path_lengths.groupby(['node']).mean().to_dict()


'''
Compute the covariance matrix.
PARAMETERS:
    - path_lengths: Pandas dataframe representing the path length of every diffusion
    - selected_obs: observer list without containing the reference observer
    - s: the candidate source
    - ref_obs: the reference observer
OUTPUT:
    - 2D array representing covariance matrix
'''
def cov_matrix(path_lengths, selected_obs, s, ref_obs):
    ref_time = path_lengths[ref_obs].loc[s]
    ref_time = np.tile(ref_time, (len(selected_obs), 1))
    obs_col = [s_obs for s_obs in selected_obs]
    return np.cov(path_lengths[obs_col].transpose().reset_index()[s].to_numpy() - ref_time, ddof = 0)
