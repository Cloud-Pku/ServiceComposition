import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        qos_aggregation[:, :, 0] = 3.
        qos_aggregation[:, :, 3] = 1.
        
        availability = torch.ones(query_num)
        throughput = torch.zeros(query_num) + 3
        reliability = torch.ones(query_num)
        
        return_idx = torch.zeros((query_num, self.max_node_num))
        return_prob = torch.zeros((query_num, self.max_node_num))
        
        for node_idx in reversed(range(self.max_node_num)):
            topological = topologicals[:, node_idx]
            topological_idx = topological.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, self.max_node_num + self.max_ser_set * 4])
            task = torch.gather(tasks, 1, topological_idx).squeeze(1)
            workflow = task[:, :self.max_node_num]
            response_time = workflow * qos_aggregation[:,:,0]
            response_time = torch.max(response_time, 1)[0]
            
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

if __name__ == '__main__':
    max_node_num = 3
    max_ser_set = 4
    agent = MatchingDecision(max_node_num, max_ser_set)
    tasks = torch.randint(0, 2, (64, max_node_num, max_node_num + max_ser_set * 4))
    constraints = torch.rand((64, 4))
    masks = torch.randint(0, 2, (64, max_node_num, max_ser_set))
    topologicals = torch.arange(0, max_node_num).unsqueeze(0).repeat([64, 1])
    
    
    res = agent(tasks, constraints, masks, topologicals)
        
        
        