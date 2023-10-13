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
        
    def forward(self, tasks, constraints, masks):
        logits = self.model(torch.cat((tasks.reshape(tasks.shape[0], -1), constraints), -1))
        ser_prob = F.softmax(logits.reshape((-1, self.max_ser_set))) * masks.reshape((-1, self.max_ser_set))
        ser_idx = torch.distributions.Categorical(ser_prob).sample()
        select_prob = ser_prob[torch.arange(0, ser_prob.shape[0]),ser_idx]
        return ser_idx.reshape((tasks.shape[0], self.max_node_num)), select_prob.reshape((tasks.shape[0], self.max_node_num))
        
        # logits = self.model(torch.cat((tasks.reshape(tasks.shape[0], -1), constraints), -1)).reshape((-1, self.max_node_num, self.max_ser_set))
        # logits = logits.masked_fill(masks == 0, -1e9)
        # ser_prob = F.softmax(logits, -1)
        # ser_idx = torch.distributions.Categorical(ser_prob).sample()
        
        # tmp_ser_prob = ser_prob.reshape((-1, ser_prob.shape[-1]))
        # tmp_ser_idx = ser_idx.reshape((-1,))
        # select_prob = tmp_ser_prob[torch.arange(0, tmp_ser_prob.shape[0]), tmp_ser_idx].reshape(ser_idx.shape)
        # return ser_idx, select_prob
    
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

    def forward(self, tasks, constraints, masks):
        
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
        

if __name__ == '__main__':
    max_node_num = 3
    max_ser_set = 4
    agent = MHADecision(max_node_num, max_ser_set)
    tasks = torch.rand((64, max_node_num, max_node_num + max_ser_set * 4))
    res = agent(tasks, None, None)
        
        
        