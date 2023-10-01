import numpy as np
import random
from algorithms import *
from experiment import *

rec_time = 15


class Model:
    def __init__(self, nx_graph, model_type='SI'):
        self.nx_graph = nx_graph
        self.n = nx_graph.number_of_nodes()
        self.nbrs = {v: list(nx_graph.neighbors(v)) for v in list(nx_graph.nodes)}
        self.pi = pi_default  # propagation prob
        self.pa = pa_default  # asymptomatic probability
        self.ph = ph_default  # hospitalization probability
        self.model_type = model_type
        self.test_limit_per_day = 2 * floor(mean(len(nbrs) for nbrs in self.nbrs.values()))

    def init_run(self):
        self.susc = {v: True for v in list(self.nx_graph.nodes)}
        self.infe = {v: False for v in list(self.nx_graph.nodes)}
        self.infe_time = {v: np.inf for v in list(self.nx_graph.nodes)}
        self.infe_nodes = []
        self.iasym = {v: False for v in list(self.nx_graph.nodes)}
        self.rec = {v: False for v in list(self.nx_graph.nodes)}
        self.day_count = 0

    def run_epidemic(self, start_node, test_limit_per_day, algos_range=range(algo_num)):
        self.init_run()
        self.susc[start_node] = False
        self.infe[start_node] = True
        self.infe_nodes = [start_node]
        self.infe_time[start_node] = 0
        self.all_algos = {i: Algorithm(self, version=i, test_limit_per_day=test_limit_per_day) for i in algos_range}
        algo_started = False
        first_hosp_node, first_hosp_day = 0, 0

        while not all(algo.finished for algo in self.all_algos.values()):  # iteration in each day
        #while len([algo for algo in self.all_algos.values() if algo.finished]) < 4:  # iteration in each day
            #print(self.susc)
            #print([algo.finished for algo in self.all_algos.values()])
            #print(algo_started)
            # print('++++')
            # print(start_node)
            # if len(self.infe_nodes) > 0:
            #     v = self.infe_nodes[0]
            #     print(v)
            #     print(self.infe_time[v])
            # print(self.day_count)
            self.day_count += 1
            new_infe_nodes = []
            if ((self.model_type == 'SI' and not any(self.susc.values())) or
                (self.model_type == 'SIR' and len(self.infe_nodes) == 0)) \
                    and not algo_started:
                #print('x')
                algo_started = True
                rand_node = random.choice(list(self.nx_graph.nodes))
                first_hosp_node, first_hosp_day = rand_node, self.day_count
                for algo in self.all_algos.values():
                    algo.init_first_node(rand_node)
            else:
                for v in self.infe_nodes:
                    for w in self.nbrs[v]:
                        if self.susc[w]:
                            if random.random() < self.pi:
                                # v infect w
                                #print('k')
                                self.susc[w] = False
                                self.infe[w] = True
                                new_infe_nodes.append(w)
                                self.infe_time[w] = self.day_count
                                # w symptomatic
                                if random.random() > self.pa:
                                    #print(3)
                                    if random.random() < self.ph:
                                        #print(2)
                                        if not algo_started:
                                            # set algo.best_cand and add it to algo.test_queue
                                            #print('p')
                                            first_hosp_node, first_hosp_day = w, self.day_count
                                            for algo in self.all_algos.values():
                                                algo.init_first_node(w)
                                            #print('q')
                                            algo_started = True
                                # w asymptomatic
                                else:
                                    self.iasym[w] = True

                for v in set(new_infe_nodes):
                    self.infe_nodes.append(v)

            if self.model_type == 'SIR':
                for v in self.infe_nodes:
                    if self.infe_time[v] <= self.day_count - rec_time:
                        #print(self.day_count)
                        self.infe_nodes.remove(v)
                        self.infe[v] = False
                        self.rec[v] = True
                        self.susc[v] = False

            if algo_started:
                #print(0)
                for algo in self.all_algos.values():
                    #print(self.day_count, '-----', algo.version)
                    algo.update()
                #print(1111)
        #print(1)
        return first_hosp_node, first_hosp_day

    def generate_random_walks_for_each_neighbor(self, start_node, length, algo):
        """
        generate random walks of 'length' nodes
        starting from 'start_node' in DFS fashion
        for each neighbor of 'start_node'
        """
        all_walks = []
        algo.contacted[start_node] = True
        for nbr in list(self.nbrs[start_node]):
            all_walks.append(list([start_node] + self.random_walk(nbr, length-1, algo)))
        return all_walks

    def generate_random_walks_for_each_neighbor_daily(self, start_node, length, algo, k=1):
        """
        generate random walks of 'length' nodes
        starting from 'start_node' in DFS fashion
        for each neighbor of 'start_node'
        """
        all_walks = []
        factor = int(np.ceil(algo.test_limit_per_day / len(self.nbrs[start_node])))
        algo.contacted[start_node] = True
        for nbr in list(self.nbrs[start_node]):
            all_walks += [list([start_node] + self.random_walk(nbr, length-1, algo)) for _ in range(k * factor)]
        return all_walks

        # nbr_list = list(self.nbrs[start_node])
        # for i in range(algo.test_limit_per_day):
        #     nbr = nbr_list[i % len(nbr_list)]
        #     all_walks.append(list([start_node] + self.random_walk(nbr, length - 1, algo)))


    # the only correct RW function:
    def generate_DFS_RWs_for_each_neighbor_daily(self, start_node, length, algo, k=1):
        """
        generate random walks of 'length' nodes
        starting from 'start_node' in DFS fashion
        for each neighbor of 'start_node'
        length >= 2
        """
        all_walks = []
        factor = int(np.ceil(algo.test_limit_per_day / len(self.nbrs[start_node])))
        algo.contacted[start_node] = True
        for i in range(factor * len(self.nbrs[start_node])):
            nbr = self.nbrs[start_node][i // factor]
            walk = [start_node, nbr]
            while len(walk) < length:
                cur = walk[-1]
                prev = walk[-2]
                algo.contacted[cur] = True
                cur_nbrs = [x for x in self.nbrs[cur] if x != prev]
                if len(cur_nbrs) > 0:
                    new_nbrs = [nbr for nbr in cur_nbrs if nbr not in self.nbrs[prev]]
                    next = random.choice(new_nbrs) if len(new_nbrs) > 0 else random.choice(cur_nbrs)
                    walk.append(next)
                else:
                    break
            all_walks.append(walk)
        return all_walks


    def generate_random_walks(self, start_node, length, k, algo):
        """generate k random walks of 'length' nodes starting from start_node in DFS fashion"""
        return [self.random_walk(start_node, length, algo) for _ in range(k)]

    def random_walk(self, start_node, length, algo):
        """generate a random walk of 'length' nodes starting from start_node in DFS fashion"""
        walk = [start_node]

        while len(walk) < length:
            cur = walk[-1]
            algo.contacted[cur] = True
            cur_nbrs = self.nbrs[cur]
            if len(cur_nbrs) > 1:
                if len(walk) == 1:
                    walk.append(random.choice(cur_nbrs))
                else:
                    prev = walk[-2]
                    new_nbrs = [nbr for nbr in cur_nbrs if nbr not in self.nbrs[prev]]
                    next = random.choice(new_nbrs) if len(new_nbrs) > 0 else random.choice(cur_nbrs)
                    walk.append(next)
            else:
                break

        return walk
