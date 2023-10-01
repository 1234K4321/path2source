import math
from itertools import chain, combinations
import networkx as nx
import numpy as np
from scipy.stats import entropy
from statistics import mean
import matplotlib.pyplot as plt
from random import sample

from passive.selection_algorithms.euclidean_dist_selection import euclidean_dist_selection
from passive.selection_algorithms.betweenness_centrality import place_observers_btw
from passive.selection_algorithms.hn_obs import place_observers_hn
from passive.selection_algorithms.kmedian import greedy_kmedian
from passive.selection_algorithms.ln_obs import place_observers_ln
from passive.selection_algorithms.mahalanobis_dist_selection import mahalanobis_dist_mc_selection
from passive.selection_algorithms.random_selection import random_selection

is_tree = False
n = 100
p = 2 * np.log(n) / n
pi = 0.6 if is_tree else 0.1
sub_sample_num = 500

r = dict()  # r[u, v]
w = dict()  # w[i] = [w_i1, w_i2, ..., w_in]  normalized to 1
alpha = dict()
train_subgraphs = []

G = nx.Graph()
all_edges = []

method_name = {
    1: 'count_entropy',
    2: 'mutual_information',
    3: 'betweenness_centrality',
    4: 'euclidean_dist',
    5: 'hn-Obs',
    6: 'k_median',
    7: 'ln-Obs',
    8: 'random_selection'
}
method_num = len(method_name)

final_acc = [[] for _ in range(method_num)]
final_dist = [[] for _ in range(method_num)]
final_rank = [[] for _ in range(method_num)]
t = 0


def main():
    global r, w, alpha, train_subgraphs, G, all_edges, final_acc, final_dist, final_rank, t, pi
    pi = 0.6 if is_tree else 0.1
    k = 1
    for t in range(1, 10):
        if is_tree:
            G = nx.random_tree(n, seed=0)
        else:
            G = nx.erdos_renyi_graph(n, p, seed=0)
            if not nx.is_connected(G):
                return
        all_edges = [e for e in G.edges]
        print(all_edges)
        nx.write_gpickle(G, str(t) + '_Graph.gpickle')
        nx.draw(G)
        plt.show()
        for j in [0, 1]:  # 0: k,  1: pi
            k_values = (2, 3, 4, 5)
            pi_values = (0.1, 0.2, 0.3, 0.4)
            values = [k_values, pi_values][j]
            final_acc = [[] for _ in range(method_num)]
            final_dist = [[] for _ in range(method_num)]
            final_rank = [[] for _ in range(method_num)]
            for it in range(len(values)):
                if j == 0:
                    k = values[it]
                if j == 1:
                    pi = values[it]

                print(k, ' --- ', pi)
                # print(0)
                all_subgraphs = generate_subgraphs(pi, sub_sample_num)
                train_subgraphs, test_subgraphs = train_test_split(all_subgraphs)
                # print(1)
                # if not is_tree:
                # calculate_r()
                # w = calculate_w()
                # alpha = calculate_alpha()
                # print(2)
                sensors = []
                for method in range(1, 1 + method_num):
                    if method <= 2:
                        funct = [g1, g2][method - 1]
                        sensors = greedy(G, k, funct)
                    else:
                        func = [place_observers_btw, euclidean_dist_selection, place_observers_hn, greedy_kmedian,
                                place_observers_ln, random_selection][method - 3]
                        sensors = func(G, k)
                        # print(3)
                    test(G, test_subgraphs, sensors, method)
                    # print(4)
            hist(['k', 'pi'][j], values)


def generate_subgraphs(pi, samp_num):
    for _ in range(samp_num):
        subgraph = nx.Graph()
        subgraph.add_nodes_from(range(n))
        selected_edges = sample(all_edges, math.floor(len(all_edges) * pi))
        # print('=====', len(selected_edges))
        for e in selected_edges:
            subgraph.add_edge(e[0], e[1])
        nx.draw(subgraph)
        plt.show()
        yield subgraph


def test(graph, test_subgraphs, sensors, method):
    global final_acc, final_dist, final_rank
    dist_list = []
    all_pairs = dict(nx.all_pairs_shortest_path_length(graph))
    rank_list = []
    acc_list = []
    sensors_to_source = sensors_to_source_dict_2(sensors)
    for subgraph in test_subgraphs:
        for c in nx.connected_components(subgraph):

            S_value = int(''.join('1' if s in c else '0' for s in sensors), 2)
            sorted_probs = sorted(graph.nodes, key=lambda u: sensors_to_source[S_value][u], reverse=True)

            best_estimated = sorted_probs[0]
            # print(len(c))
            # print(best_estimated)
            for v in c:
                acc_list.append(1 if v == best_estimated else 0)
                dist_list.append(all_pairs[best_estimated][v])
                rank_list.append(sorted_probs.index(v) + 1)

    final_acc[method - 1] = list(final_acc[method - 1]) + [mean(acc_list)]
    final_dist[method - 1] = list(final_dist[method - 1]) + [mean(dist_list)]
    final_rank[method - 1] = list(final_rank[method - 1]) + [mean(rank_list)]


def hist(param_name, values):
    acc_hist(param_name, values)
    dist_hist(param_name, values)
    rank_hist(param_name, values)


def acc_hist(param_name, values):
    for i in range(method_num):
        plt.plot(values, final_acc[i], label='method' + str(i + 1))
    plt.xlabel(param_name)
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('acc_' + param_name + '_' + str(t) + '.png')
    plt.show()


def dist_hist(param_name, values):
    for i in range(method_num):
        plt.plot(values, final_dist[i], label='method' + str(i + 1))
    plt.xlabel(param_name)
    plt.ylabel('distance')
    plt.legend()
    plt.savefig('dist_' + param_name + '_' + str(t) + '.png')
    plt.show()


def rank_hist(param_name, values):
    for i in range(method_num):
        plt.plot(values, final_rank[i], label='method' + str(i + 1))
    plt.xlabel(param_name)
    plt.ylabel('rank')
    plt.legend()
    plt.savefig('rank_' + param_name + '_' + str(t) + '.png')
    plt.show()


def probs_of_test_given_sources(graph, sensors, sensor_values):
    probs = dict()
    for v in graph.nodes:
        probs_of_sensors_given_v = {sensor: r[sensor, v] if sensor_values[sensor] == 1
        else 1 - r[sensor, v] for sensor in sensors}
        prob_of_test_given_v = np.product(list(probs_of_sensors_given_v.values()))
        probs[v] = prob_of_test_given_v
    return probs


def calculate_alpha():
    alpha = dict()
    for u in G.nodes:
        alpha[u] = np.median([r[u, v] for v in G.nodes])
    return alpha


def calculate_w():
    w = dict()  # w[i] = [w_i1, w_i2, ..., w_in]  normalized to 1
    for v in G.nodes:
        w[v] = [r[u, v] for u in G.nodes]
        w[v] = np.array(w[v]) / np.sum(w[v])
    return w


def calculate_r():
    global r
    r = dict()  # r[u, v] = p(u, v)
    for u in G.nodes:
        for v in G.nodes:
            r[u, v] = 1 if u == v else 0
    for subgraph in train_subgraphs:
        components = nx.connected_components(subgraph)
        for c in components:
            for u in c:
                for v in c:
                    if u != v:
                        r[u, v] += 1
    for u in G.nodes:
        for v in G.nodes:
            if u != v:
                r[u, v] /= len(train_subgraphs)


def train_test_split(all_subgraphs):
    train_subgraphs = []
    test_subgraphs = []
    for subgraph in all_subgraphs:
        if np.random.random() < 0.7:
            train_subgraphs.append(subgraph)
        else:
            test_subgraphs.append(subgraph)
    return train_subgraphs, test_subgraphs


def f1(G, S):  # S is a subset of G.nodes
    return n - np.sum([np.product([w[i][j] for i in S]) for j in range(n)])


def f2(G, S):
    k = len(S)
    x = np.array([0 for _ in range(2 ** k)])
    for v in G.nodes:
        binary = ''
        for i in range(k):
            bit = 1 if alpha[v] > r[S[i], v] else 0
            binary += str(bit)
        to_dec = int(binary, 2)
        x[to_dec] += 1
    x = x / len(G.nodes)
    return entropy(x)


def f3(G, S):
    k = len(S)
    S_to_source = {i: {v: 0 for v in G.nodes} for i in range(2 ** k)}
    for subgraph in train_subgraphs:
        for c in nx.connected_components(subgraph):
            S_value = int(''.join('1' if s in c else '0' for s in S), 2)
            for v in c:
                S_to_source[S_value][v] += 1
    cond_entropy = 0
    for S_value in S_to_source.keys():
        counts = np.array(list(S_to_source[S_value].values()))
        if sum(counts) > 0:
            cond_entropy += entropy(counts / sum(counts)) * sum(counts)
    total_sum = sum(list(sum(list(S_to_source[S_value].values())) for S_value in S_to_source.keys()))
    return np.log(n) - (cond_entropy / total_sum)


def sensors_to_source_dict(sensors):
    k = len(sensors)
    S_to_source = {i: {v: 0 for v in G.nodes} for i in range(2 ** k)}
    for subgraph in train_subgraphs:
        for c in nx.connected_components(subgraph):
            S_value = int(''.join('1' if s in c else '0' for s in sensors), 2)
            for v in c:
                S_to_source[S_value][v] += 1
    return S_to_source


def sensors_to_source_dict_2(sensors):  # return dict[S_value(decimal)][v]  (float (tree) or int (else)
    k = len(sensors)
    S_to_source = {i: {v: 0 for v in G.nodes} for i in range(2 ** k)}
    if is_tree:
        my_dict = probs_of_sensors_given_source_in_tree(sensors)
        for i in range(2 ** k):
            for v in G.nodes:
                S_to_source[i][v] = my_dict[v][i]
    else:
        for subgraph in train_subgraphs:
            for c in nx.connected_components(subgraph):
                S_value = int(''.join('1' if s in c else '0' for s in sensors), 2)
                for v in c:
                    S_to_source[S_value][v] += 1
    return S_to_source


def greedy(graph, k, f):
    solution = []
    for _ in range(k):
        best_node = -1
        best_spread = -np.inf
        nodes = [v for v in graph.nodes if v not in solution]
        for node in nodes:
            spread = f(graph, solution + [node])
            if spread > best_spread:
                best_spread = spread
                best_node = node
        solution.append(best_node)
    return solution


def g1(G, S):
    k = len(S)
    x = np.array([0 for _ in range(2 ** k)])
    for v in G.nodes:
        my_dict = probs_of_sensors_given_source_in_tree(S) if is_tree else probs_of_sensors_given_source_by_samplig(S)
        max_prob = max(range(2 ** k), key=lambda y: my_dict[v][y])
        print('====== ', max_prob)
        x[max_prob] += 1
    print('x =========== ', x)
    print(S)
    print(entropy(x))
    return entropy(x)


def g2(G, O):
    k = len(O)
    source_to_O = probs_of_sensors_given_source_in_tree(O) if is_tree else probs_of_sensors_given_source_by_samplig(O)
    O_probs = np.array([0 for _ in range(2 ** k)])
    for i in range(2 ** k):
        for v in G.nodes:
            O_probs[i] += source_to_O[v][i]
    O_entropy = entropy(O_probs)
    O_given_source_entropy = 0
    for v in G.nodes:
        O_given_source_entropy += entropy(list(source_to_O[v].values())) / n
    print(O)
    print(O_entropy - O_given_source_entropy)
    return O_entropy - O_given_source_entropy


def probs_of_sensors_given_source_by_samplig(sensors):  # return dict[v][sensor_values(decimal)]  (int)
    k = len(sensors)
    probs_dict = {v: {i: 0 for i in range(2 ** k)} for v in G.nodes}
    for subgraph in train_subgraphs:
        for c in nx.connected_components(subgraph):
            S_value = int(''.join('1' if s in c else '0' for s in sensors), 2)
            for v in c:
                probs_dict[v][S_value] += 1
    return probs_dict


def probs_of_sensors_given_source_in_tree(sensors):  # return dict[v][sensor_values(decimal)] (only 1s have path to v)
    probs_dict = {v: dict() for v in G.nodes}
    all_paths_from_sensors = dict()
    for u in sensors:
        all_paths_from_sensors[u] = nx.single_source_shortest_path(G, u)  # dict
    for v in G.nodes:
        partial_probs_dict = dict()  # dict[sensor_values_str] (1s have path to v)
        induced_tree = G.subgraph(list(set().union(*[list(all_paths_from_sensors[sensor][v]) for sensor in sensors])))
        for i in range(2 ** len(sensors)):
            bin_str = format(i, '0' + str(len(sensors)) + 'b')
            union_paths = G.subgraph(list(set().union(*[list(all_paths_from_sensors[sensors[j]][v])
                                                        for j in range(len(sensors)) if bin_str[j] == '1'])))
            partial_probs_dict[bin_str] = pow(pi, len(union_paths.edges))

        s = range(len(sensors))
        for zeros_indices in chain.from_iterable(combinations(s, q) for q in range(len(s) + 1)):
            zeros_indices = list(zeros_indices)
            bin_str = ''.join(['1' if k not in zeros_indices else '0' for k in range(len(sensors))])
            to_dec = int(bin_str, 2)
            probs_dict[v][to_dec] = 0
            for zero_subset in chain.from_iterable(
                    combinations(zeros_indices, q) for q in range(len(zeros_indices) + 1)):
                zero_subset = list(zero_subset)
                partial_bin_str = ''.join(['1' if k not in zero_subset else '0' for k in range(len(sensors))])
                probs_dict[v][to_dec] += partial_probs_dict[partial_bin_str] * pow(-1, len(zeros_indices) - len(
                    zero_subset))
    print(probs_dict)
    return probs_dict


if __name__ == '__main__':
    main()
