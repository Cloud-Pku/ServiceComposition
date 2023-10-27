import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph, spring_layout
from scipy import stats
import pickle
import numpy as np

class ServiceGenerator:
    def __init__(self, df_path = '/home/PJLAB/chenyun/ServiceComposition/simulator/dataset/qws2/qws_df.pkl'):
        with open(df_path, 'rb') as f:
            self.df = pickle.load(f)
        self.attributes = self.df.columns
        self.data = {}
        self.bounds = {}
        self.mean = {}
        self.std = {}
        self.kdes = {}
        for key in self.attributes[:9]:
            self.data[key] = self.df[key].values
            self.bounds[key] = [min(self.data[key]), max(self.data[key])]
            self.mean[key] = np.mean(self.data[key])
            self.std[key] = np.std(self.data[key])
            self.kdes[key] = stats.gaussian_kde(self.data[key])
        self.bounds['Availability'] = [0., 1.]
        self.bounds['Successability'] = [0., 1.]
        self.bounds['Reliability'] = [0., 1.]
        self.mean['Availability'] = 0.
        self.mean['Successability'] = 0.
        self.mean['Reliability'] = 0.
        self.std['Availability'] = 1.
        self.std['Successability'] = 1.
        self.std['Reliability'] = 1.
        print('complete')
        
    def sample(self, nums=1, attributes=['Response Time', 'Availability', 'Throughput', 'Reliability'], norm = True):
        services = [{} for i in range(nums)]
        for att in attributes:
            tmp_atts = self.kdes[att].resample(nums)[0]
            tmp_atts = np.clip(tmp_atts, self.bounds[att][0], self.bounds[att][1])
            for i in range(nums):
                services[i][att] = (tmp_atts[i] - self.mean[att])/self.std[att] if norm else tmp_atts[i]
        return services

class WorkflowGenerator:
    def __init__(self, max_node_num=10):
        self.max_node_num = max_node_num
        self.edge_density = None
    def sample(self, num=1, render=False, edge_density=0.2):
        # zero_adj_matrix = np.zeros((self.max_node_num, self.max_node_num))
        self.edge_density = edge_density
        ret_mt = []
        topologicals = []
        for i in range(num):
            topological = np.arange(self.max_node_num)
            np.random.shuffle(topological)
            topologicals.append(topological)

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
        return ret_mt, topologicals

class ConstraintGenerator:
    def __init__(self, service_gen: ServiceGenerator):
        self.service_gen = service_gen
        
    
    def sample(self, workflows, topologicals, attributes=['Response Time', 'Availability', 'Throughput', 'Reliability'], norm=True, mode='human'):
        
        query_num = len(workflows)
        service_set = []
        for i in range(query_num):
            service_set.append(self.service_gen.sample(len(workflows[i]), attributes=attributes, norm=norm))

        if mode == 'human':
            constraints = []
        else:
            constraints = np.zeros([query_num, len(attributes)])
        for query in range(query_num):
            workflow = workflows[query]
            topological = topologicals[query]
            node_num = len(workflow)
            services = service_set[query]
            constraint = {k : np.array([0.]*node_num) for k in attributes}
            
            
            for node_idx in reversed(topological):
                connect = workflow[node_idx]
                con_idx = np.where(connect==1)[0]
                
                # Response Time
                if len(con_idx) == 0:
                    constraint['Response Time'][node_idx] = services[node_idx]['Response Time']
                else:
                    constraint['Response Time'][node_idx] = services[node_idx]['Response Time'] + max(constraint['Response Time'][con_idx])
                
                # Other
                constraint['Throughput'][node_idx] = services[node_idx]['Throughput']
                constraint['Availability'][node_idx] = services[node_idx]['Availability']
                constraint['Reliability'][node_idx] = services[node_idx]['Reliability']
            
            # Aggregation
            if mode == 'human':
                constraint['Response Time'] = max(constraint['Response Time'])
                constraint['Throughput'] = min(constraint['Throughput'])
                constraint['Availability'] = np.prod(constraint['Availability'])
                constraint['Reliability'] = np.prod(constraint['Reliability'])
                constraints.append(constraint)
            else:
                # avai = np.prod(constraint['Availability']*self.service_gen.std['Availability'] + self.service_gen.mean['Availability'])
                # avai = (avai - self.service_gen.mean['Availability'])/self.service_gen.std['Availability']
                # reli = np.prod(constraint['Reliability']*self.service_gen.std['Reliability'] + self.service_gen.mean['Reliability'])
                # reli = (reli - self.service_gen.mean['Reliability'])/self.service_gen.std['Reliability']
                constraints[query] = np.array([max(constraint['Response Time']), np.prod(constraint['Availability']), min(constraint['Throughput']), np.prod(constraint['Reliability'])])
                # if norm:
                #     constraints[query] = np.array([max(constraint['Response Time']), min(constraint['Throughput']), avai, reli])
                    
        
        return constraints



    


    
        
        

if __name__ == '__main__':
    gen1 = WorkflowGenerator(max_node_num=5)
    wks, tpgs = gen1.sample(1, edge_density=0.8, render=True)
    print(wks)
    
    # gen2 = ServiceGenerator()
    # # print(gen2.sample(9, norm=False))

    # gen3 = ConstraintGenerator(gen2)
    # print(gen3.sample(wks, tpgs, norm=True, mode='tiny'))