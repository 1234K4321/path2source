import os
from math import floor, log2
from statistics import mean
import networkx as nx
import matplotlib.pyplot as plt
from model import *
from algorithms import *
from copy import deepcopy
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams.update({'font.size': 5})

all_models = []  # containing pairs (model, model_name)

main_algos = [1, 3, 4, 5, 6, 7, 9]
result = {
    'accuracy': {algo: [] for algo in main_algos},
    'distance': {algo: [] for algo in main_algos},
    '#tests': {algo: [] for algo in main_algos},
    '#contacts': {algo: [] for algo in main_algos}
}

algos = [4, 5, 6, 7, 8, 9, 'theory']
tree_result = {
        'success': {algo: [] for algo in algos},
        '#tests': {algo: [] for algo in algos},
        '#contacts': {algo: [] for algo in algos}
    }

RW_lens = range(1, 11)

pi_default = 0.1
pa_default = 0.1
ph_default = 0.1
pi_values = [i / 10 for i in range(1, 6)]
pa_values = [i / 10 for i in range(1, 6)]
ph_inv_values = [10 * i for i in range(1, 6)]

d_values = [2, 3, 4, 5, 6]


def tree_community_nx_graph(com_num, com_size, d=0):
    # returns a tree on communities ((d+1)-regular if d > 0 else random_tree)
    nx_graph = nx.planted_partition_graph(com_num, com_size, 1, 0, seed=42)
    tree = nx.random_tree(n=com_num, seed=0)
    if d > 0:
        tree = nx.full_rary_tree(r=d, n=com_num)
    for e in tree.edges:
        nx_graph.add_edge(e[0] * com_size, e[1] * com_size)
    return nx_graph


def generate_all_models():
    global all_models
    file_path = os.path.join('datasets', 'facebook_combined.txt')
    G = nx.read_edgelist(file_path, create_using=nx.Graph)
    max_component = max(nx.connected_components(G), key=len)
    G = nx.subgraph(G, max_component)
    model = Model(G, model_type='SI')
    model.test_limit_per_day = 10
    all_models.append((model, 'FACEBOOK'))

    # G = nx.barabasi_albert_graph(500,10)
    # model = Model(G, model_type='SI')
    # model.test_limit_per_day = 5
    # all_models.append((model, 'barabasi_albert_graph(500,10)'))

    # G = tree_community_nx_graph(999,1,d=2)
    # model = Model(G, model_type='SIR')
    # model.test_limit_per_day = 12
    # all_models.append((model, 'SIR_tree_community(1000,1,d=2),lim=12'))

    # G = nx.random_powerlaw_tree(900, seed=0, tries=1000000)
    # model = Model(G, model_type='SI')
    # model.test_limit_per_day = 4
    # all_models.append((model, 'Powerlaw_tree(900)'))

    # G = nx.powerlaw_cluster_graph(550, 15, 0.9, seed=0)
    # model = Model(G, model_type='SI')
    # model.test_limit_per_day = 5
    # all_models.append((model, 'Powerlaw_cluster(550,15,0.9)'))

    # G = nx.ring_of_cliques(65,15)
    # model = Model(G, model_type='SI')
    # model.test_limit_per_day = 5
    # all_models.append((model, 'Ring_of_cliques(65,15)'))

    # G = nx.connected_caveman_graph(25,15)
    # all_models.append((Model(G, model_type='SI'), 'connected_caveman(25,15)'))

    # G = nx.connected_watts_strogatz_graph(500, 10, 0.1)
    # model = Model(G, model_type='SI')
    # model.test_limit_per_day = 5
    # all_models.append((Model(G), 'connected_watts_strogatz(500,10,0.1)'))

    # G = nx.erdos_renyi_graph(200, 0.1)
    # all_models.append((Model(G), 'erdos_renyi_graph(200,0.1)'))
    """canceled:"""
    # G = nx.karate_club_graph()
    # all_models.append((Model(G), 'karate_club_graph'))
    # G = nx.relaxed_caveman_graph(10, 10, 0.05, seed=42)
    # all_models.append((Model(G), 'relaxed_caveman_graph(10,10,0.05)'))
    # G = nx.planted_partition_graph(10, 10, 0.95, 0.05, seed=42)
    # all_models.append((Model(G), 'planted_partition_graph(10,10,0.95,0.05)'))


def draw_diagram(model_name, result_dicts, model_type, test_limit):
    fig, axs = plt.subplots(len(result), 3, constrained_layout=True)
    model_name = model_name + ',' + str(model_type) + ',lim=' + str(test_limit) + \
                 ',pi=' + str(pi_default) + ',pa=' + str(pa_default) + ',ph=' + str(ph_default)
    fig.suptitle(model_name)
    result_keys = list(result.keys())
    for i in range(len(result)):  # result type
        for j in range(3):  # p type (pi, pa, ph)
            x = [pi_values, pa_values, ph_inv_values][j]
            for a in main_algos:  # algo index
                axs[i, j].plot(x, result_dicts[j][result_keys[i]][a], label=alg_name[a], linewidth=1)
            if j == 0:
                axs[i, j].set_ylabel(result_keys[i])
            if i == len(result) - 1:
                axs[i, j].set_xlabel(['pi', 'pa', '1/ph'][j])
            #axs[i, j].set(xlabel=['pi', 'pa', '1/ph'][j], ylabel=result_keys[i])
    # for ax in axs.flat:
    #     ax.label_outer()
    plt.legend(prop={'size': 4})
    plt.tight_layout()
    file_path = os.path.join('diagrams', '%' + model_name)
    plt.savefig(file_path + '.png')
    plt.show()


def experiment_model(model, run_num, algs=main_algos, start_node=None):
    sub_result = {
        # result of each run will be added and the mean will be returned
        'accuracy': {algo: 0 for algo in algs},
        'distance': {algo: 0 for algo in algs},
        '#tests': {algo: 0 for algo in algs},
        '#contacts': {algo: 0 for algo in algs}
    }
    for run_count in range(run_num):
        rand_start_node = random.choice(list(model.nx_graph.nodes)) if start_node is None else start_node
        # if run_count % 10 == 0:
        #     print(run_count)
        model.run_epidemic(start_node=rand_start_node, test_limit_per_day=model.test_limit_per_day, algos_range=algs)
        for algo_index in algs:
            algo = model.all_algos[algo_index]
            sub_result['accuracy'][algo_index] += (1 if algo.best_cand == rand_start_node else 0)
            sub_result['distance'][algo_index] += nx.shortest_path_length(model.nx_graph, source=rand_start_node,
                                                                          target=algo.best_cand)
            sub_result['#tests'][algo_index] += algo.get_test_queries_count()
            sub_result['#contacts'][algo_index] += algo.get_contact_queries_count()
    sub_result = {s: {a: sub_result[s][a] / run_num for a in algs} for s in sub_result.keys()}
    return sub_result


def experiment():
    generate_all_models()
    s0 = [(pi, pa_default, ph_default) for pi in pi_values]
    s1 = [(pi_default, pa, ph_default) for pa in pa_values]
    s2 = [(pi_default, pa_default, 1 / ph_inv) for ph_inv in ph_inv_values]

    for model, model_name in all_models:
        run_num = model.n // 50
        my_result = []
        for k in range(3):  # p type (pi, pa, ph)
            new_result = deepcopy(result)
            for pi, pa, ph in [s0, s1, s2][k]:
                model.pi = pi
                model.pa = pa
                model.ph = ph
                print(str(k), ' ', model_name)
                sub_result = experiment_model(model, run_num=run_num)
                for s in result.keys():
                    for a in main_algos:
                        new_result[s][a].append(sub_result[s][a])
            my_result.append(new_result)
        print('draw')
        draw_diagram(model_name, my_result, model.model_type, model.test_limit_per_day)


def tree_draw_diagram(model_name, result_dict):
    fig, axs = plt.subplots(len(tree_result))
    model_name = model_name + ',pi=' + str(pi_default) + ',pa=' + str(pa_default) + ',ph=' + str(ph_default)
    fig.suptitle(model_name)
    result_keys = list(tree_result.keys())
    for i in range(len(tree_result)):  # result type
        for a in algos:  # algo index
            axs[i].plot(d_values, result_dict[result_keys[i]][a], label=alg_name[a])
        axs[i].set(xlabel='d', ylabel=result_keys[i])
    plt.xticks(d_values)
    for ax in axs.flat:
        ax.label_outer()
    plt.legend(prop={'size': 6})
    plt.tight_layout()
    file_path = os.path.join('diagrams', 'tree', model_name)
    plt.savefig(file_path + '.png')
    plt.show()


def tree_experiment_model(model, d, run_num, start_node):
    p = model.pa / (model.pa + (1 - model.pa) * (1 - model.ph))
    sub_result = {
        # result of each run will be added and the mean will be returned
        'success': {algo: 0 for algo in algos},
        '#tests': {algo: 0 for algo in algos},
        '#contacts': {algo: 0 for algo in algos}
    }
    for run_count in range(run_num):
        first_hosp_node, first_hosp_day = model.run_epidemic(start_node=start_node, test_limit_per_day=model.test_limit_per_day)
        for algo_index in algos:
            if algo_index == 'theory':
                l = nx.shortest_path_length(model.nx_graph, start_node, first_hosp_node)
                sub_result['success'][algo_index] += ((1-p) * pow(1 - p * (1 - 1/d), l-1))
                sub_result['#tests'][algo_index] += (d * l * floor(log2(first_hosp_day)))
                sub_result['#contacts'][algo_index] += (d * l * first_hosp_day)
            else:
                algo = model.all_algos[algo_index]
                sub_result['success'][algo_index] += (1 if algo.best_cand == start_node else 0)
                sub_result['#tests'][algo_index] += algo.get_test_queries_count()
                sub_result['#contacts'][algo_index] += algo.get_contact_queries_count()

    sub_result = {s: {a: sub_result[s][a] / run_num for a in algos} for s in sub_result.keys()}
    return sub_result


def tree_experiment():
    new_result = deepcopy(tree_result)
    n = 5000
    for d in d_values:
        print(d)
        G = nx.full_rary_tree(r=d, n=n)
        model = Model(G, model_type='SI')
        model.test_limit_per_day = d + 1
        run_num = model.n // 50
        sub_result = tree_experiment_model(model, d, run_num, start_node=list(model.nx_graph.nodes)[0])
        for s in new_result.keys():
            for a in algos:
                new_result[s][a].append(sub_result[s][a])
    model_name = 'd_ary_tree(n=' + str(n) + ') SI,lim=d+1'
    tree_draw_diagram(model_name, new_result)


def len_draw_diagram(model_name, result_dict, model_type, test_limit):
    #result_keys = [s for s in result.keys()]
    result_keys = [s for s in result.keys() if s != 'accuracy']
    fig, axs = plt.subplots(len(result_keys), constrained_layout=True)
    model_name = model_name + ',pi=' + str(pi_default) + ',pa=' + str(pa_default) + ',ph=' + str(ph_default)\
                 + ',type=' + str(model_type) + ',lim=' + str(test_limit)
    fig.suptitle('algos with const RW_len in \n' + model_name)
    for i in range(len(result_keys)):  # result type
        axs[i].bar(RW_lens, result_dict[result_keys[i]].values())
        axs[i].set(xlabel='RW len', ylabel=result_keys[i])
    plt.xticks(RW_lens)
    for ax in axs.flat:
        ax.label_outer()
    plt.tight_layout()
    file_path = os.path.join('diagrams', 'RW_len', model_name)
    plt.savefig(file_path + '.png')
    plt.show()


def len_experiment():
    G = nx.barabasi_albert_graph(1000,10)
    model = Model(G, model_type='SI')
    model.test_limit_per_day = 5
    model_name = 'barabasi_albert_graph(1000,10)'
    print('start')

    run_num = model.n // 10
    result_dic = experiment_model(model, run_num, algs=[x + 9 for x in RW_lens])
    print(result_dic)
    len_draw_diagram(model_name, result_dic, model.model_type, model.test_limit_per_day)


if __name__ == '__main__':
    experiment()
    #tree_experiment()
    #len_experiment()
