import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph, spring_layout

class WorkflowGenerator:
    def __init__(self, max_node_num=10):
        self.max_node_num = max_node_num
        self.edge_density = None
    def sample(self, num=1, render=False, edge_density=0.2):
        # zero_adj_matrix = np.zeros((self.max_node_num, self.max_node_num))
        self.edge_density = edge_density
        ret_mt = []
        for i in range(num):
            topological = np.arange(self.max_node_num)
            np.random.shuffle(topological)

            tmp_mt = np.zeros((self.max_node_num, self.max_node_num))
            for i in range(self.max_node_num - 1):
                for j in range(i + 1, self.max_node_num):
                    if np.random.random() < self.edge_density:
                        tmp_mt[topological[i]][topological[j]] = 1
                if not np.any(tmp_mt[topological[i]] == 1):
                    tmp_mt[topological[i]][topological[np.random.randint(i + 1, self.max_node_num)]] = 1
            # print(tmp_mt)
            ret_mt.append(tmp_mt)
            if render:
                G = DiGraph(tmp_mt)
                pos = spring_layout(G)
                nx.draw_networkx(G, pos=pos, arrows=True)
                plt.show()
        return ret_mt


if __name__ == '__main__':
    gen = WorkflowGenerator()
    gs = gen.sample(5)
    print(gs)