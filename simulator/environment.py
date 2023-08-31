import numpy as np
from service_generator import ServiceGenerator
from workflow_generator import WorkflowGenerator

class ServiceComEnv:
    def __init__(self, max_node_num=10, max_ser_set=100, attributes=['Response Time', 'Availability', 'Throughput', 'Reliability']):
        self.max_node_num = max_node_num
        self.max_ser_set = max_ser_set
        self.attributes = attributes
        self.workflow_gen = WorkflowGenerator(max_node_num)
        self.service_gen = ServiceGenerator()
    
    def reset(self, num=1, edge_density=0.2, mode='human', norm=True):
        workflows = self.workflow_gen.sample(num, edge_density=edge_density)
        
        if mode == 'human':
            tasks = [{}] * num
            for i in range(num):
                tasks[i]['adj_matrix'] = workflows[i]
                tasks[i]['ser_set'] = []
                for j in range(self.max_node_num):
                    ser_num = np.random.randint(1, self.max_ser_set)
                    tasks[i]['ser_set'].append(self.service_gen.sample(ser_num))
            return tasks
        else:
            tasks = []
            for i in range(num):
                adj_mt = workflows[i]
                set_i_mt = []
                for j in range(self.max_node_num):
                    ser_num = np.random.randint(1, self.max_ser_set)
                    set_j = self.service_gen.sample(ser_num, self.attributes)
                    
                    set_j_vector = []
                    for n in range(ser_num):
                        for k, v in set_j[n].items():
                            set_j_vector.append((v - self.service_gen.mean[k])/self.service_gen.std[k] if norm else v)
                    set_j_vector += [0]*len(self.attributes)*(self.max_ser_set - ser_num)
                    set_i_mt.append(np.array(set_j_vector))
                set_i_mt = np.vstack(set_i_mt)
                tasks.append(np.concatenate((adj_mt, set_i_mt), 1))
            tasks = np.vstack(tasks).reshape(num, self.max_node_num, -1)
            return tasks
                    

if __name__ == '__main__':
    env = ServiceComEnv()
    tasks = env.reset(2, mode='tiny')
    print(tasks)
        
    