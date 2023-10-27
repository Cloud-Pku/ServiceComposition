import torch
import numpy as np
import random
from simulator.environment import ServiceComEnv
from solver.model import NaiveMLPDecision, MHADecision, MatchingDecision
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

ss = 2
random.seed(ss)
np.random.seed(ss)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
max_node_num = 100
max_ser_set = 10
bs = 64
epoch_size = 10000
norm = True
lr = 0.0001
model = 'MHA'

mean_reward_his = []
loss_his = []

if __name__ == '__main__':
    writer = SummaryWriter(log_dir=f'/home/PJLAB/chenyun/ServiceComposition/matching/node{max_node_num}-serset{max_ser_set}/{model}', filename_suffix='empty')
    env = ServiceComEnv(max_node_num, max_ser_set)
    if model == 'MHA':
        agent = MHADecision(max_node_num, max_ser_set).to(device)
    elif model == 'MLP':
        agent = NaiveMLPDecision(max_node_num, max_ser_set).to(device)
    else:
        agent = MatchingDecision(max_node_num, max_ser_set).to(device)
    optim = Adam(agent.parameters(), lr, weight_decay=0.01)
    for epoch in tqdm(range(epoch_size)):
        
        workflows, tasks, masks, constraints, topologicals = env.reset(bs, mode='tiny', norm=norm)
        tasks = torch.from_numpy(tasks).to(torch.float32).to(device)
        constraints = torch.from_numpy(constraints).to(torch.float32).to(device)
        masks = torch.from_numpy(masks).to(device)
        topologicals = torch.from_numpy(topologicals).to(device)
        ser_idx, ser_prob = agent.forward(tasks, constraints, masks, topologicals)
        rewards = env.step(ser_idx.numpy())
        rewards = torch.from_numpy(rewards)
        # mean_reward_his.append(rewards.mean().cpu().item())
        writer.add_scalar(f'reward', rewards.mean().cpu().item(), epoch)
        loss = -torch.mean(torch.sum(torch.log(ser_prob), 1) * rewards.squeeze(-1))
        # loss_his.append(loss.mean().cpu().item())
        writer.add_scalar(f'loss', loss.mean().cpu().item(), epoch)
        optim.zero_grad()
        loss.backward()
        optim.step()
    
    # plt.plot(range(epoch_size), mean_reward_his, label='reward')
    # plt.savefig('/home/PJLAB/chenyun/ServiceComposition/reward.png')
    # plt.close()
    
    # plt.plot(range(epoch_size), loss_his, label='loss')
    # plt.savefig('/home/PJLAB/chenyun/ServiceComposition/loss.png')
    # plt.close()
    writer.close()
        
        