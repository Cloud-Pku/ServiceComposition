import numpy as np
from generator import WorkflowGenerator, ServiceGenerator, ConstraintGenerator

class ServiceComEnv:
    def __init__(self, max_node_num=10, max_ser_set=100, attributes=['Response Time', 'Availability', 'Throughput', 'Reliability']):
        self.max_node_num = max_node_num
        self.max_ser_set = max_ser_set
        self.attributes = attributes
        self.workflow_gen = WorkflowGenerator(max_node_num)
        self.service_gen = ServiceGenerator()
        self.constraint_gen = ConstraintGenerator(self.service_gen)
    
    def reset(self, num=1, edge_density=0.2, mode='human', norm=True):
        workflows, topologicals = self.workflow_gen.sample(num, edge_density=edge_density)
        constrains = self.constraint_gen.sample(workflows, topologicals, self.attributes, norm=norm, mode=mode)
        if mode == 'human':
            tasks = [{}] * num
            for i in range(num):
                tasks[i]['adj_matrix'] = workflows[i]
                tasks[i]['ser_set'] = []
                for j in range(self.max_node_num):
                    ser_num = np.random.randint(1, self.max_ser_set)
                    tasks[i]['ser_set'].append(self.service_gen.sample(ser_num))
            return tasks, None
        else:
            tasks = np.zeros([num, self.max_node_num, self.max_node_num + len(self.attributes) * self.max_ser_set])
            masks = np.ones([num, self.max_node_num, self.max_ser_set])
            for i in range(num):
                tasks[i, :, :self.max_node_num] = workflows[i]
                for j in range(self.max_node_num):
                    
                    ser_num = np.random.randint(1, self.max_ser_set)
                    set_j = self.service_gen.sample(ser_num, self.attributes, norm=norm)
                    order = list(range(self.max_ser_set))
                    np.random.shuffle(order)
                    order = order[:ser_num]
                    masks[i, j, order] = 0

                    for n in range(ser_num):
                        for k_i, k in enumerate(self.attributes):
                            tasks[i, j, self.max_node_num + order[n]*len(self.attributes) + k_i] = set_j[n][k]
            return tasks, masks, constrains
    def step(self, ):
        pass
                    

if __name__ == '__main__':
    env = ServiceComEnv(3, 4)
    tasks, masks, constraints = env.reset(2, mode='tiny')
    print(masks)
    print(tasks)
    print(constraints)        
    