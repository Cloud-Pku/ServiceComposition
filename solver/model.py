import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset
import random

class NaiveMLPDecision(nn.Module):
    def __init__(self, max_node_num, max_ser_set, att_num=4, hidden_list=[128, 128, 64]):
        super().__init__()
        self.max_node_num = max_node_num
        self.max_ser_set = max_ser_set
        self.hidden_list = hidden_list
        
        self.layers = []
        curr_dim = self.max_node_num * (self.max_node_num + self.max_ser_set * att_num) + att_num
        for dim in hidden_list:
            fc = nn.Linear(curr_dim, dim)
            actv = nn.SiLU()
            self.layers += [fc, actv]
            curr_dim = dim
            
        self.layers += [nn.Linear(self.hidden_list[-1], max_node_num * max_ser_set)]
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, tasks, constraints, masks, topologicals):
        # logits = self.model(torch.cat((tasks.reshape(tasks.shape[0], -1), constraints), -1))
        # ser_prob = F.softmax(logits.reshape((-1, self.max_ser_set))) * masks.reshape((-1, self.max_ser_set))
        # ser_idx = torch.distributions.Categorical(ser_prob).sample()
        # select_prob = ser_prob[torch.arange(0, ser_prob.shape[0]),ser_idx]
        # return ser_idx.reshape((tasks.shape[0], self.max_node_num)), select_prob.reshape((tasks.shape[0], self.max_node_num))
        
        logits = self.model(torch.cat((tasks.reshape(tasks.shape[0], -1), constraints), -1)).reshape((-1, self.max_node_num, self.max_ser_set))
        logits = logits.masked_fill(masks == 0, -1e9)
        ser_prob = F.softmax(logits, -1)
        ser_idx = torch.distributions.Categorical(ser_prob).sample()
        
        tmp_ser_prob = ser_prob.reshape((-1, ser_prob.shape[-1]))
        tmp_ser_idx = ser_idx.reshape((-1,))
        select_prob = tmp_ser_prob[torch.arange(0, tmp_ser_prob.shape[0]), tmp_ser_idx].reshape(ser_idx.shape)
        return ser_idx, select_prob
    
class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.

        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )

class MHADecision(nn.Module):
    def __init__(self, max_node_num, max_ser_set, att_num=4, emb_dim = 128, bias = True):
        super().__init__()
        self.max_node_num = max_node_num
        self.max_ser_set = max_ser_set
        self.emb_dim = emb_dim
        node_dim = self.max_node_num + self.max_ser_set * att_num
        self.init_emb_layer = nn.Sequential(
            nn.Linear(node_dim + att_num, 128),
            nn.SiLU(),
            nn.Linear(128, emb_dim),
        )
        
        self.MHA_layer = MultiHeadAttention(emb_dim, 8, bias)
        
        self.ser_prob_head = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.SiLU(),
            nn.Linear(128, max_ser_set),
        )

    def forward(self, tasks, constraints, masks, topologicals):
        
        constraints = constraints.unsqueeze(1).repeat([1, self.max_node_num, 1]) # just cat
        
        node_emb = self.init_emb_layer(torch.cat((tasks, constraints), -1))
        
        out = self.MHA_layer(node_emb, node_emb, node_emb) + node_emb # residual
        
        logits = self.ser_prob_head(out)
        logits = logits.masked_fill(masks == 0, -1e9)
        ser_prob = F.softmax(logits, -1)
        ser_idx = torch.distributions.Categorical(ser_prob).sample()
        
        tmp_ser_prob = ser_prob.reshape((-1, ser_prob.shape[-1]))
        tmp_ser_idx = ser_idx.reshape((-1,))
        select_prob = tmp_ser_prob[torch.arange(0, tmp_ser_prob.shape[0]), tmp_ser_idx].reshape(ser_idx.shape)

        return ser_idx, select_prob


class MatchingDecision(nn.Module):
    def __init__(self, max_node_num, max_ser_set, att_num=4, emb_dim = 128, bias = True):
        super().__init__()
        self.max_node_num = max_node_num
        self.max_ser_set = max_ser_set
        self.emb_dim = emb_dim
        self.att_num = att_num
        node_dim = self.max_node_num
        self.init_emb_layer = nn.Sequential(
            nn.Linear(node_dim + att_num, 128),
            nn.SiLU(),
            nn.Linear(128, emb_dim),
        )
        
        self.MHA_layer = MultiHeadAttention(emb_dim, 8, bias)
        
        self.node_emb_layer = nn.Sequential(
            nn.Linear(emb_dim + att_num, 128),
            nn.SiLU(),
            nn.Linear(128, emb_dim),
        )
        
        self.service_emb_layer = nn.Sequential(
            nn.Linear(att_num, 128),
            nn.SiLU(),
            nn.Linear(128, emb_dim),
        )
    def make_onehot(self, topological):
        ones_matrix = torch.zeros((topological.shape[0], self.max_node_num))
        ones_matrix[torch.arange(topological.shape[0]), topological] = 1
        return ones_matrix

    def forward(self, tasks, constraints, masks, topologicals):
        constraints = constraints.unsqueeze(1).repeat([1, self.max_node_num, 1]) # just cat
        workflows = tasks[:,:, :self.max_node_num]
        node_emb = self.init_emb_layer(torch.cat((workflows, constraints), -1))
        
        nodes_emb = self.MHA_layer(node_emb, node_emb, node_emb) + node_emb # residual
        
        query_num = tasks.shape[0]
        qos_aggregation = torch.zeros((query_num, self.max_node_num, self.att_num))
        qos_aggregation[:, :, 0] = -3.
        qos_aggregation[:, :, 1] = 1.
        qos_aggregation[:, :, 2] = 3.
        qos_aggregation[:, :, 3] = 1.
        
        availability = torch.ones(query_num)
        throughput = torch.zeros(query_num) + 3
        reliability = torch.ones(query_num)
        
        return_idx = torch.zeros((query_num, self.max_node_num))
        return_prob = torch.zeros((query_num, self.max_node_num))
        
        for i, node_idx in enumerate(reversed(range(self.max_node_num))):
            topological = topologicals[:, node_idx]
            topological_idx = topological.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, self.max_node_num + self.max_ser_set * 4])
            task = torch.gather(tasks, 1, topological_idx).squeeze(1)
            workflow = task[:, :self.max_node_num]
            response_time = workflow * qos_aggregation[:,:,0]
            response_time = torch.max(response_time, 1)[0]
            if i == (self.max_node_num - 1):
                response_time = -3.
            
            qos_agg_feature = torch.cat((response_time.unsqueeze(-1), availability.unsqueeze(-1), throughput.unsqueeze(-1), reliability.unsqueeze(-1)), -1)
            topological_idx_tmp = topological.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, self.emb_dim])
            inp = torch.cat((torch.gather(nodes_emb, 1, topological_idx_tmp).squeeze(1), qos_agg_feature), -1) # maybe FiLM
            task_node_emb = self.node_emb_layer(inp)
            
            # service embedding
            services = torch.gather(tasks, 1, topological_idx).squeeze(1)[:, self.max_node_num:].reshape((query_num, -1, self.att_num))
            services_emb = self.service_emb_layer(services)
            
            # matching
            logits = torch.sum(services_emb * task_node_emb.unsqueeze(1), -1)
            topological_idx_tmp = topological.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, self.max_ser_set])
            mask = torch.gather(masks, 1, topological_idx_tmp).squeeze(1)
            logits = logits.masked_fill(mask == 0, -1e9)
            ser_prob = F.softmax(logits, -1)
            ser_idx = torch.distributions.Categorical(ser_prob).sample()
            # print(ser_idx)
            
            # add to return
            topo_onehot = self.make_onehot(topological)
            idx_resi_mtx = topo_onehot * ser_idx.unsqueeze(-1)
            prob_resi_mtx = topo_onehot * ser_prob[torch.arange(query_num), ser_idx].unsqueeze(-1)
            return_idx += idx_resi_mtx
            return_prob += prob_resi_mtx
            
            # calculate aggretation
            rt_idx = ser_idx * 4 + self.max_node_num
            ava_idx = ser_idx * 4 + self.max_node_num + 1
            tp_idx = ser_idx * 4 + self.max_node_num + 2
            rel_idx = ser_idx * 4 + self.max_node_num + 3
            
            ser_rt = task.gather(1, rt_idx.unsqueeze(-1)).squeeze(-1)
            ser_ava = task.gather(1, ava_idx.unsqueeze(-1)).squeeze(-1)
            ser_tp = task.gather(1, tp_idx.unsqueeze(-1)).squeeze(-1)
            ser_rel = task.gather(1, rel_idx.unsqueeze(-1)).squeeze(-1)
            qos_aggregation[torch.arange(query_num), topological, 0] = ser_rt + response_time
            qos_aggregation[torch.arange(query_num), topological, 1] = ser_ava * availability
            qos_aggregation[torch.arange(query_num), topological, 2] = torch.min(torch.cat((ser_tp.unsqueeze(-1), throughput.unsqueeze(-1)), -1), 1)[0]
            qos_aggregation[torch.arange(query_num), topological, 3] = ser_rel * reliability
            availability = ser_ava * availability
            throughput = torch.min(torch.cat((ser_tp.unsqueeze(-1), throughput.unsqueeze(-1)), -1), -1)[0]
            reliability = ser_rel * reliability
            
        
        return return_idx.to(torch.int16), return_prob

class MemoryReplay(object):

    def __init__(self,
                 max_size=10000,
                 bs=64,
                 obs_size=84,
                 num_heads = 10,
                 p = 0.5):

        self.obs = np.zeros((max_size, obs_size), dtype=np.float32)
        self.reward = np.zeros(max_size, dtype=np.float32)
        self.action = np.zeros(max_size, dtype=np.int32)
        self.next_obs = np.zeros((max_size, obs_size), dtype=np.float32)
        self.weight = np.zeros(max_size, dtype=np.float32)
        #self.ss = np.zeros_like(self.s)
        self.done = np.array([True]*max_size)
        self.head_mask = np.zeros((max_size, num_heads), dtype=np.int32)
        
        self.num_heads = num_heads
        self.p = p
        self.max_size = max_size
        self.bs = bs
        self._cursor = 0
#        self.total_idx = list(range(self.max_size))
        self.occupied = False


    def put(self, s, a, r, s_, d, w):

        if self._cursor == (self.max_size-1):
            self._cursor = 0
            self.occupied = True
        else:
            self._cursor += 1

        self.obs[self._cursor] = np.array(s)
        self.action[self._cursor] = np.array(a)
        self.reward[self._cursor] = np.array(r)
        self.next_obs[self._cursor] = np.array(s_)
        self.done[self._cursor] = np.array(d)
        self.weight[self._cursor] = np.array(w)
        self.head_mask[self._cursor] = np.random.binomial(1, self.p, self.num_heads)


    def batch(self, k):
        if self.occupied:
            idx =  np.array(range(self.max_size))
            idx_k = self.head_mask[:,k].astype(bool)
        else:
            idx =  np.array(range(self._cursor))
            idx_k = self.head_mask[:self._cursor,k].astype(bool)
        sample_idx = random.sample(list(idx[idx_k]), self.bs)
        s = self.obs[sample_idx]
        a = self.action[sample_idx]
        r = self.reward[sample_idx]
        s_ = self.next_obs[sample_idx]
        d = self.done[sample_idx]

        return torch.from_numpy(s), torch.from_numpy(a), torch.from_numpy(r), torch.from_numpy(s_), torch.from_numpy(d)

# class Buffer(Dataset):
#     def __init__(self, obs, action, next_obs, reward, done, mask, weight, transition_size) -> None:
#         super().__init__()
#         # transition is passed as list and updated outside
#         self.obs = obs
#         self.action = action
#         self.next_obs = next_obs
#         self.reward = reward
#         self.done = done
#         self.mask = mask
#         self.weight = weight
#         self.transition_size = transition_size
#     def __len__(self):
#         return max(1, len(self.obs))
#     def __getitem__(self, idx):
#         idx = idx % len(self.obs)
#         obs = self.obs[idx]
#         action = self.action[idx]
#         next_obs = self.next_obs[idx]
#         done = self.done[idx]
#         reward = torch.Tensor([self.reward[idx]])
#         mask = self.mask[idx]
#         weight = self.weight[idx]
#         return (obs, action, next_obs, reward, done, mask, weight)

class DQNDecision(nn.Module):
    def __init__(self, max_node_num, max_ser_set, att_num=4, emb_dim = 128, q_heads_num = 4, iter_num = 10, gamma = 0.9, bias = True):
        super().__init__()
        
        self.iter_step = 0
        self.target_update_freq = 100
        self.max_node_num = max_node_num
        self.max_ser_set = max_ser_set
        self.emb_dim = emb_dim
        self.att_num = att_num
        self.iter_num = iter_num
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.state_dim = self.max_node_num + self.max_ser_set * att_num + 2 * att_num
        self.state_emb_layer = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.SiLU(),
            nn.Linear(128, emb_dim),
        )
        self.state_emb_layer_target = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.SiLU(),
            nn.Linear(128, emb_dim),
        )
        self.q_heads_num = q_heads_num
        self.q_heads = nn.ModuleList(
            [nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, 128),
                nn.SiLU(),
                nn.Linear(128, self.max_ser_set),
            ) for _ in range(q_heads_num)]
        )
        self.q_heads_target = nn.ModuleList(
            [nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, 128),
                nn.SiLU(),
                nn.Linear(128, self.max_ser_set),
            ) for _ in range(q_heads_num)]
        )
        self.obs_list = []
        self.action_list = []
        self.next_obs_list = []
        self.reward_list = []
        self.done_list = []
        self.mask_list = []
        self.weights_list = []
        self.buffer = MemoryReplay(obs_size=self.state_dim, num_heads=q_heads_num)
        
    def make_onehot(self, topological):
        ones_matrix = torch.zeros((topological.shape[0], self.max_node_num))
        ones_matrix[torch.arange(topological.shape[0]), topological] = 1
        return ones_matrix

    def forward(self, tasks, constraints, masks, topologicals):
        # constraints = constraints.unsqueeze(1).repeat([1, self.max_node_num, 1]) # just cat
        
        k = np.random.randint(0, self.q_heads_num)
        workflows = tasks[:,:, :self.max_node_num]
        
        query_num = tasks.shape[0]
        qos_aggregation = torch.zeros((query_num, self.max_node_num, self.att_num))
        qos_aggregation[:, :, 0] = -3.
        qos_aggregation[:, :, 1] = 1.
        qos_aggregation[:, :, 2] = 3.
        qos_aggregation[:, :, 3] = 1.
        
        availability = torch.ones(query_num)
        throughput = torch.zeros(query_num) + 3
        reliability = torch.ones(query_num)
        
        return_idx = torch.zeros((query_num, self.max_node_num))
        # return_prob = torch.zeros((query_num, self.max_node_num))
        
        for i, node_idx in enumerate(reversed(range(self.max_node_num))):
            topological = topologicals[:, node_idx]
            topological_idx = topological.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, self.max_node_num + self.max_ser_set * 4])
            task = torch.gather(tasks, 1, topological_idx).squeeze(1)
            workflow = task[:, :self.max_node_num]
            response_time = workflow * qos_aggregation[:,:,0]
            response_time = torch.max(response_time, 1)[0]
            if i == 0:
                response_time -= 3.
            qos_agg_feature = torch.cat((response_time.unsqueeze(-1), availability.unsqueeze(-1), throughput.unsqueeze(-1), reliability.unsqueeze(-1)), -1)
            # topological_idx_tmp = topological.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, task])
            state = torch.cat((task, constraints, qos_agg_feature), -1) # maybe FiLM
            self.obs_list.append(state)
            if i != 0:
                self.next_obs_list.append(state)
                self.reward_list.append(torch.zeros(query_num))
                self.done_list.append(torch.zeros(query_num))
            self.mask_list.append(torch.randint(0, 2, (query_num, self.q_heads_num)))
            state_emb = self.state_emb_layer(state)

            q_value = self.q_heads[k](state_emb)
            
            topological_idx_tmp = topological.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, self.max_ser_set])
            mask = torch.gather(masks, 1, topological_idx_tmp).squeeze(1)
            q_value = q_value.masked_fill(mask == 0, -1e9)
            ser_idx = torch.argmax(q_value, -1)
            self.action_list.append(ser_idx)
            
            q = q_value.gather(1, ser_idx.unsqueeze(-1))
            q_ = self.q_heads_target[k](self.state_emb_layer_target(state)).gather(1, ser_idx.unsqueeze(-1))
            self.weights_list.append(torch.abs(q - q_).detach())
            
            topo_onehot = self.make_onehot(topological)
            idx_resi_mtx = topo_onehot * ser_idx.unsqueeze(-1)
            return_idx += idx_resi_mtx
            
            
            # calculate aggretation
            rt_idx = ser_idx * 4 + self.max_node_num
            ava_idx = ser_idx * 4 + self.max_node_num + 1
            tp_idx = ser_idx * 4 + self.max_node_num + 2
            rel_idx = ser_idx * 4 + self.max_node_num + 3
            
            ser_rt = task.gather(1, rt_idx.unsqueeze(-1)).squeeze(-1)
            ser_ava = task.gather(1, ava_idx.unsqueeze(-1)).squeeze(-1)
            ser_tp = task.gather(1, tp_idx.unsqueeze(-1)).squeeze(-1)
            ser_rel = task.gather(1, rel_idx.unsqueeze(-1)).squeeze(-1)
            qos_aggregation[torch.arange(query_num), topological, 0] = ser_rt + response_time
            qos_aggregation[torch.arange(query_num), topological, 1] = ser_ava * availability
            qos_aggregation[torch.arange(query_num), topological, 2] = torch.min(torch.cat((ser_tp.unsqueeze(-1), throughput.unsqueeze(-1)), -1), 1)[0]
            qos_aggregation[torch.arange(query_num), topological, 3] = ser_rel * reliability
            availability = ser_ava * availability
            throughput = torch.min(torch.cat((ser_tp.unsqueeze(-1), throughput.unsqueeze(-1)), -1), -1)[0]
            reliability = ser_rel * reliability
        
        self.next_obs_list.append(state)
        self.done_list.append(torch.ones(query_num))

        return return_idx.to(torch.int16), None
    
    def post_process(self):
        task_num = len(self.obs_list)
        query_num = self.obs_list[0].shape[0]
        for task in range(task_num):
            for query in range(query_num):
                s = self.obs_list[task][query]
                a = self.action_list[task][query]
                r = self.reward_list[task][query]
                s_ = self.next_obs_list[task][query]
                d = self.done_list[task][query]
                w = self.weights_list[task][query]
                self.buffer.put(s, a, r, s_, d, w)
        self.obs_list = []
        self.action_list = []
        self.reward_list = []
        self.next_obs_list = []
        self.done_list = []
        self.weights_list = []

    def learn(self, optim, writer, epoch):
        total_loss_his = []
        for iter in range(self.iter_num):
            loss_his = []
            for k in range(self.q_heads_num):
                k = np.random.randint(0, self.q_heads_num)
                s, a, r, s_, d = self.buffer.batch(k)
                q = self.q_heads[k](self.state_emb_layer(s)).gather(1, a.unsqueeze(-1).to(torch.int64)).squeeze(-1)
                y = r + self.gamma * (~d) * torch.max(self.q_heads_target[k](self.state_emb_layer_target(s_)), -1)[0]
                q_loss = self.mse(q, y.detach())
                optim.zero_grad()
                q_loss.backward()
                optim.step()
                loss_his.append(q_loss.detach().cpu().item())
            writer.add_scalars(f'Loss/q_{k}_loss', np.mean(loss_his), epoch)
            total_loss_his.append(np.mean(loss_his))
        return np.mean(total_loss_his)

        
if __name__ == '__main__':
    max_node_num = 3
    max_ser_set = 4
    # agent = MatchingDecision(max_node_num, max_ser_set)
    # tasks = torch.randint(0, 2, (64, max_node_num, max_node_num + max_ser_set * 4))
    # constraints = torch.rand((64, 4))
    # masks = torch.randint(0, 2, (64, max_node_num, max_ser_set))
    # topologicals = torch.arange(0, max_node_num).unsqueeze(0).repeat([64, 1])
    
    # res = agent(tasks, constraints, masks, topologicals)
    
    agent = DQNDecision(max_node_num, max_ser_set)
    inp = torch.rand((64, max_node_num + 2 * 4))
    agent(inp)
        
        
        