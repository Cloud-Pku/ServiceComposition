import torch
import numpy as np
import random
from simulator.environment import ServiceComEnv
from solver.model import NaiveMLPDecision, MHADecision, MatchingDecision, DQNDecision
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

ss = 2
random.seed(ss)
np.random.seed(ss)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
max_node_num = 10
max_ser_set = 10
bs = 64
epoch_size = 10000
norm = True
lr = 0.0001
model = 'DQN'

mean_reward_his = []
loss_his = []

if __name__ == '__main__':
    writer = SummaryWriter(log_dir=f'/home/PJLAB/chenyun/ServiceComposition/dqn/node{max_node_num}-serset{max_ser_set}/{model}', filename_suffix='empty')
    env = ServiceComEnv(max_node_num, max_ser_set)
    if model == 'MHA':
        agent = MHADecision(max_node_num, max_ser_set).to(device)
    elif model == 'MLP':
        agent = NaiveMLPDecision(max_node_num, max_ser_set).to(device)
    elif model == 'MD':
        agent = MatchingDecision(max_node_num, max_ser_set).to(device)
    else:
        agent = DQNDecision(max_node_num, max_ser_set).to(device)
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
        if model == 'DQN':
            agent.reward_list.append(rewards)
            agent.post_process()
            loss = agent.learn(optim, writer, epoch)
            writer.add_scalar(f'total_loss', loss, epoch)
        else:
            loss = -torch.mean(torch.sum(torch.log(ser_prob), 1) * rewards.squeeze(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss = loss.mean().cpu().item()
            writer.add_scalar(f'loss', loss, epoch)
        # mean_reward_his.append(rewards.mean().cpu().item())
        writer.add_scalar(f'reward', rewards.mean().cpu().item(), epoch)

    
    # plt.plot(range(epoch_size), mean_reward_his, label='reward')
    # plt.savefig('/home/PJLAB/chenyun/ServiceComposition/reward.png')
    # plt.close()
    
    # plt.plot(range(epoch_size), loss_his, label='loss')
    # plt.savefig('/home/PJLAB/chenyun/ServiceComposition/loss.png')
    # plt.close()
    writer.close()
        
        