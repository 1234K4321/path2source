


algo_num = 7
pi_default = 0.1


class Model:
    def __init__(self, nx_graph):
        self.nx_graph = nx_graph
        self.n = nx_graph.number_of_nodes()
        self.nbrs = {v: list(nx_graph.neighbors(v)) for v in list(nx_graph.nodes)}
        self.pi = pi_default  # propagation prob

