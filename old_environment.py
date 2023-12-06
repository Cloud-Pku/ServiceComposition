from typing import Any
import pickle
from solver.model import MatchingDecision
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

class Env():
    def __init__(self, code_service_dict, service_code_dict, service_qos_dict, qos_bound ,tasks_num) -> None:
        with open("/mnt/nfs/chenyun/ServiceComposition/fake_data.pkl",'rb') as f:
            self.data = pickle.load(f)
        
        self.code_service_dict = code_service_dict
        self.service_code_dict = service_code_dict
        self.service_qos_dict = service_qos_dict
        self.qos_bound = qos_bound
        self.tasks_num = tasks_num
        self.qos_constraint = [0, 0]
        
    def reset(self, qos_constraint, weights=[0.5, 0.5]):
        self.weights = weights
        # self.qos_constraint[0] = (qos_constraint[0] - self.tasks_num * self.qos_bound[0][0])/(self.qos_bound[0][1] - self.qos_bound[0][0])
        # self.qos_constraint[1] = (qos_constraint[1] -  self.qos_bound[1][0])/(self.qos_bound[1][1] - self.qos_bound[1][0])
        self.qos_constraint[0] = qos_constraint[0] / 600
        self.qos_constraint[1] = qos_constraint[1] / 40
        
        # tasks, constraints, masks, topologicals
        return self.code_service_dict, self.service_code_dict, self.service_qos_dict, self.qos_constraint
        
    
    def step(self, solutions):
        qos_price = 0.
        qos_time = 0.
        for node_idx, ser_idx in solutions.items():
            qos_price += self.service_qos_dict[ser_idx][0]
            qos_time = max(qos_time, self.service_qos_dict[ser_idx][1])
        if qos_price > self.qos_constraint[0] or qos_time > self.qos_constraint[1]:
            return -1.
        else:
            return self.weights[0] * (self.qos_constraint[0] - qos_price) + self.weights[1] * (self.qos_constraint[1] - qos_time)
        
        

if __name__ == '__main__':
    
    lr = 0.0001
    
    writer = SummaryWriter(log_dir=f'/mnt/nfs/chenyun/ServiceComposition/debug_old/MD', filename_suffix='empty')
    
    with open('fake_data.pkl', 'rb') as f:
        code_service_dict, service_code_dict,  service_qos_dict, qos_bound = pickle.load(f)
    env = Env(code_service_dict, service_code_dict,  service_qos_dict, qos_bound, 10)
    agent = MatchingDecision(10, 1000, 2)
    optim = Adam(agent.parameters(), lr, weight_decay=0.01)
    
    code_service_dict, service_code_dict, service_qos_dict, qos_constraint = env.reset([600,40])
    
    for iter in tqdm(range(10000)):
        
        ser_idx, ser_prob = agent.forward_old(code_service_dict, service_code_dict, service_qos_dict, qos_constraint)
        rewards = env.step(ser_idx)
        rewards = torch.from_numpy(np.array(rewards))
        loss = -torch.mean(torch.sum(torch.log(ser_prob)) * rewards)
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss = loss.mean().cpu().item()
        writer.add_scalar(f'loss', loss, iter)
        writer.add_scalar(f'rewards', rewards, iter)
    
    