from math import floor
import numpy as np
from priorityqueue import PriorityQueue

algo_num = 10
alg_name = {
    0: 'LS',
    1: 'LS+',
    2: 'LSv2',
    3: 'LS+v2',
    4: 'RWS',
    5: 'RWS-small',
    6: 'RWS-full',
    7: 'RWS-old',
    8: 'RWS-bin',
    9: 'RWS-bin2',
    'theory': 'theory'
}


class Algorithm:
    def __init__(self, model, version, test_limit_per_day):
        self.version = version
        self.model = model
        self.test_limit_per_day = test_limit_per_day
        self.test_queue = PriorityQueue()  # queue of nodes to be tested
        self.tested = {v: False for v in list(self.model.nx_graph.nodes)}
        self.contacted = {v: False for v in list(self.model.nx_graph.nodes)}

        self.best_cand = None
        self.all_cands = []
        self.initial_RW_factor = 1
        self.RW_factor = self.initial_RW_factor
        self.RWs = []
        self.last_pair = []  # a list of (i1, i2) two final node's indices of binary search on each walk
        # v2 is in test_queue and then swap if needed so that
        # t(v1) <= t(v2)  (if v is asym then t(v) = best_time_on_RW_so_far + 0.5)
        # (same indexing as self.RWs)
        self.node_to_RW = {v: [] for v in list(self.model.nx_graph.nodes)}  # map each testing node to a list of its RW indices
        self.RW_finished = []  # a list of boolean (True: search on the RW is done)

        self.finished = False
        self.contact_queries_count = 0
        self.test_queries_count = 0
        self.day_count = 0

        self.best_cand_local = None  # for RWS version 6 (node with least time before updating candidate)
        self.test_queue_finished = True  # for LS inner loop

    def init_first_node(self, node):
        self.best_cand = node
        self.best_cand_local = node
        self.test_queue.push(node)
        self.model.infe_time[node] = min(self.model.infe_time[node], self.model.day_count)

    def update(self):
        self.day_count += 1
        if self.version >= 4:
            if self.version <= 9:
                self.RWS_upadte()
            else:
                self.RWS_update_const_len(self.version - 9)
        else:
            self.LS_upadte()

    def RWS_update_const_len(self, RW_l):
        best_time = self.model.infe_time[self.best_cand]
        updated = False

        testing_nodes = self.get_testing_nodes()
        for node in testing_nodes:
            if (not self.model.iasym[node]) and self.model.infe_time[node] < best_time:
                updated = True
                self.best_cand = node
                best_time = self.model.infe_time[node]

        if updated or self.day_count < 3:
            self.test_queue = PriorityQueue()
            self.generate_RWs(1 + RW_l, 1)
        elif len(self.test_queue) == 0:
            self.finish_algorithm()
            return

        # if updated:
        #     self.test_queue = PriorityQueue()
        #     self.append_RWs(1 + RW_l)
        # else:
        #     if self.day_count > 5 and len(self.test_queue) == 0:
        #         self.finish_algorithm()
        #         return
        #     else:
        #         self.append_RWs(1 + RW_l)

    def append_RWs(self, RW_len):
        self.RWs = self.model.generate_random_walks_for_each_neighbor(self.best_cand, RW_len, self)
        for walk in self.RWs:
            if (not self.tested[walk[-1]]) and (walk[-1] not in self.test_queue):
                self.test_queue.push(walk[-1])

    def RWS_upadte(self):
        """
        random walk search
        version 4: constant RW factor for each new self.best_cand
        version 5: decreased RW factor at each failure
        version 6: constant RW factor but search among all possible factors to find the best candidate
        version 7: same as 5 but use already generated RWs
        version 8: generate RWs once and full binary search on all walks to find the best candidate
        version 9: generate RWs once and binary search on all walks until a better candidate is found
        """
        # if self.version == 8:
        #     print(self.last_pair)

        best_time = self.model.infe_time[self.best_cand]
        best_time_local = self.model.infe_time[self.best_cand_local]
        updated = False
        updated_RW_indices = set()

        # find self.best_cand
        testing_nodes = self.get_testing_nodes()
        for node in testing_nodes:
            if self.version == 6:
                if (not self.model.iasym[node]) and self.model.infe_time[node] < best_time_local:
                    self.best_cand_local = node
                    best_time_local = self.model.infe_time[node]
            else:
                if self.version != 8 and (not self.model.iasym[node]) and self.model.infe_time[node] < best_time:
                    updated = True
                    self.best_cand = node
                    best_time = self.model.infe_time[node]
                elif self.version in [8, 9]:
                    # update self.last_pair
                    for RW_index in self.node_to_RW[node]:
                        if not self.RW_finished[RW_index]:
                            updated_RW_indices.add(RW_index)
                            i1, i2 = self.last_pair[RW_index]
                            v1, v2 = self.RWs[RW_index][i1], self.RWs[RW_index][i2]
                            if (not self.model.iasym[v2]) and self.model.infe_time[v1] > self.model.infe_time[v2]:
                                self.last_pair[RW_index] = i2, i1
                            if abs(i1 - i2) <= 1:
                                self.RW_finished[RW_index] = True
                    self.node_to_RW[node] = []

        # check end and find self.RW_factor and updated and self.best_cand
        local_end = False
        if self.version in [8, 9]:
            local_end = all(self.RW_finished) and len(self.RW_finished) > 0
        else:
            local_end = (floor(self.RW_factor * best_time) == 0)
        # if self.version == 8:
        #     print(local_end)
        if local_end:
            if self.version in [8, 9]:
                all_top_nodes = [self.RWs[RW_index][self.last_pair[RW_index][0]] for RW_index in range(len(self.RWs))]
                #print(len(self.RWs))
                best_node_found = min(all_top_nodes, key=lambda v: self.model.infe_time[v])
                if best_node_found != self.best_cand:
                    updated = True
                    self.best_cand = best_node_found
                    best_time = self.model.infe_time[best_node_found]
            if self.version == 6 and best_time_local < best_time:
                updated = True
                self.best_cand = self.best_cand_local
                best_time = best_time_local
                self.RW_factor = self.initial_RW_factor
            if not updated:
                self.finish_algorithm()
                return
        elif self.version == 6:
            self.RW_factor *= 0.5  # constant 0.5 could be replaced by other values (0.4, 0.6, 0.7, ...)

        # find self.RW_factor, RW_len
        if self.version != 6:
            if updated:
                if self.version in [4, 7]:
                    self.RW_factor = self.initial_RW_factor
            else:
                self.RW_factor *= 0.5
        if self.version in [8, 9]:
            self.RW_factor = self.initial_RW_factor

        RW_len = 2 + floor(self.RW_factor * best_time)

        # update self.RWs, self.node_to_RW, self.last_pair, self.test_queue
        if (self.version in [4, 5, 6]) or updated or self.day_count == 1:
            self.generate_RWs(RW_len)
        else:
            if self.version == 7:
                self.RWs = [walk[:1 + (len(walk) // 2)] for walk in self.RWs]
            if self.version in [8, 9]:
                for RW_index in updated_RW_indices:
                    RW = self.RWs[RW_index]
                    i1, i2 = self.last_pair[RW_index]
                    new_index = (i1 + i2) // 2
                    self.last_pair[RW_index] = i1, new_index
                    new_node = RW[new_index]
                    self.node_to_RW[new_node].append(RW_index)
                    self.test_queue.push(new_node)

        # update self.test_queue
        if self.version not in [8, 9]:
            self.test_queue = PriorityQueue([walk[-1] for walk in self.RWs], [0 for _ in self.RWs])

    def generate_RWs(self, RW_len, k=1):
        self.RWs = self.model.generate_DFS_RWs_for_each_neighbor_daily(self.best_cand, RW_len, self, k)
        self.test_queue = PriorityQueue([walk[-1] for walk in self.RWs], [0 for _ in self.RWs])
        if self.version in [8, 9]:
            self.RW_finished = [False for _ in self.RWs]
            self.last_pair = [(0, len(walk) - 1) for walk in self.RWs]
            self.node_to_RW = {v: [] for v in list(self.model.nx_graph.nodes)}
            for i in range(len(self.RWs)):
                walk = self.RWs[i]
                self.node_to_RW[walk[-1]].append(i)

    def LS_upadte(self):
        """
        local search
        version 0: LS, 1: LS+, 2: LSv2, 3: LS+v2
        """
        # outer loop
        if self.test_queue_finished:
            last_cand = self.all_cands[-1] if len(self.all_cands) > 0 else None
            if self.best_cand == last_cand:
                self.finish_algorithm()
                return
            self.all_cands.append(self.best_cand)
            self.contact_queries_count += 1
            last_cand = self.all_cands[-1]
            self.contacted[last_cand] = True
            for node in self.model.nbrs[last_cand]:
                if not self.tested[node]:
                    self.test_queue.push(node)
                    self.test_queries_count += 1

            self.test_queue_finished = False
        # inner loop
        else:
            if len(self.test_queue) <= self.test_limit_per_day:
                self.test_queue_finished = True
            # pop last self.test_limit_per_day nodes from self.test_queue
            testing_nodes = self.get_testing_nodes()
            for v in testing_nodes:
                if (not self.model.iasym[v]) and self.model.infe_time[v] < self.model.infe_time[self.best_cand]:
                    self.best_cand = v
                    if self.version in [2, 3]:
                        self.test_queue = PriorityQueue()
                        self.test_queue_finished = True
                        break
                elif self.version in [1, 3] and self.model.iasym[v]:
                    self.contact_queries_count += 1
                    self.contacted[v] = True
                    for node in self.model.nbrs[v]:
                        if not self.tested[node] and node not in self.test_queue:
                            self.test_queue.push(node)
                            self.test_queries_count += 1

    def get_testing_nodes(self):
        testing_nodes = []
        for _ in range(min(self.test_limit_per_day, len(self.test_queue))):
            node = self.test_queue.pop()
            self.tested[node] = True
            testing_nodes.append(node)
        return testing_nodes

    def finish_algorithm(self):
        self.finished = True
        self.contact_queries_count = self.get_contact_queries_count()
        self.test_queries_count = self.get_test_queries_count()

    def get_test_queries_count(self):
        return len([c for c in self.tested.values() if c])

    def get_contact_queries_count(self):
        return len([c for c in self.contacted.values() if c])
