import numpy as np
import random

ss = 1
random.seed(ss)
np.random.seed(ss)
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
        self.workflows, self.topologicals = self.workflow_gen.sample(num, edge_density=edge_density)
        self.constraints = self.constraint_gen.sample(self.workflows, self.topologicals, self.attributes, norm=norm, mode=mode)
        if mode == 'human':
            self.tasks = [{}] * num
            for i in range(num):
                self.tasks[i]['adj_matrix'] = self.workflows[i]
                self.tasks[i]['ser_set'] = []
                for j in range(self.max_node_num):
                    ser_num = np.random.randint(1, self.max_ser_set)
                    self.tasks[i]['ser_set'].append(self.service_gen.sample(ser_num))
            return self.tasks, None
        else:
            self.tasks = np.zeros([num, self.max_node_num, self.max_node_num + len(self.attributes) * self.max_ser_set])
            self.masks = np.ones([num, self.max_node_num, self.max_ser_set])
            if norm:
                self.tasks[:,:, self.max_node_num ::4] = 3. # Response Time
                self.tasks[:,:, self.max_node_num + 1::4] = -3. # Availability
                self.tasks[:,:, self.max_node_num + 2::4] = -3. # Throughput
                self.tasks[:,:, self.max_node_num + 3::4] = -3. # Reliability
            else:
                self.tasks[:,:, self.max_node_num ::4] = np.max(self.service_gen.data['Response Time'])
                # set the other as zeros
            for i in range(num):
                self.tasks[i, :, :self.max_node_num] = self.workflows[i]
                for j in range(self.max_node_num):
                    
                    ser_num = np.random.randint(1, self.max_ser_set)
                    set_j = self.service_gen.sample(ser_num, self.attributes, norm=norm)
                    order = list(range(self.max_ser_set))
                    np.random.shuffle(order)
                    order = order[:ser_num]
                    self.masks[i, j, order] = 0

                    for n in range(ser_num):
                        for k_i, k in enumerate(self.attributes):
                            self.tasks[i, j, self.max_node_num + order[n]*len(self.attributes) + k_i] = set_j[n][k]
            
            return self.workflows, self.tasks, self.masks, self.constraints
    def step(self, solutions):
        query_num = len(self.workflows)
        solution_qos = np.zeros(self.constraints.shape)
        for query in range(query_num):
            workflow = self.workflows[query]
            topological = self.topologicals[query]
            task = self.tasks[query]
            node_num = len(workflow)
            solution = solutions[query]
            aggregation = {k : np.array([0.]*node_num) for k in self.attributes}
            for node_idx in reversed(topological):
                connect = workflow[node_idx]
                con_idx = np.where(connect==1)[0]
                
                # Response Time
                if len(con_idx) == 0:
                    aggregation['Response Time'][node_idx] = task[node_idx, max_node_num + solution[node_idx]*4 ]
                else:
                    aggregation['Response Time'][node_idx] = max(aggregation['Response Time'][con_idx]) + task[node_idx, max_node_num + solution[node_idx]*4 ]
                
                # Other
                aggregation['Availability'][node_idx] = task[node_idx, max_node_num + solution[node_idx]*4 + 1]
                aggregation['Throughput'][node_idx] = task[node_idx, max_node_num + solution[node_idx]*4 + 2]
                aggregation['Reliability'][node_idx] = task[node_idx, max_node_num + solution[node_idx]*4 + 3]
            
            # Aggregation
            solution_qos[query][0] = max(aggregation['Response Time'])
            solution_qos[query][1] = np.prod(aggregation['Availability'])
            solution_qos[query][2] = min(aggregation['Throughput'])
            solution_qos[query][3] = np.prod(aggregation['Reliability'])
        print(solution_qos)
        rewards = np.ones((query_num, 1))
        for query in range(query_num):
            if solution_qos[query][0] < self.constraints[query][0]:
                rewards[query] = -1.
            if np.any(solution_qos[1:] > self.constraints[query]):
                rewards[query] = -1.
        return rewards


                    

if __name__ == '__main__':
    max_node_num = 3
    max_ser_set = 4
    batch = 200
    env = ServiceComEnv(max_node_num, max_ser_set)
    workflows, tasks, masks, constraints = env.reset(batch, mode='tiny', norm=True)
    # print(masks)
    # print(workflows)
    # print(tasks)
    # print(constraints) 
    
    solutions = np.random.randint(0, max_ser_set, [batch, max_node_num])
    rewards = env.step(solutions)
    print(rewards)
    
    
    