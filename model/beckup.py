import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from typing import Optional, Union
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence, pad_sequence, PackedSequence
import ot
import numpy as np

class NTN(nn.Module):
    def __init__(self, input_dim, out_dim, args):
        super(NTN, self).__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.args = args

        self.W = nn.Parameter(torch.Tensor(input_dim, input_dim, out_dim))
        self.V = nn.Parameter(torch.Tensor(out_dim, 2 * input_dim))
        self.B = nn.Parameter(torch.Tensor(out_dim))
        self.U = nn.Parameter(torch.Tensor(out_dim, 1))

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.U)
        nn.init.zeros_(self.B)

    def forward(self, features1, features2):
        features1 = features1
        features2 = features2
        tensor_product = torch.einsum('ai,ijk,bj->abk', [features1, self.W, features2])
        features1_expanded = features1.unsqueeze(1).expand(features1.size(0),features2.size(0),features1.size(1))  # [a, 1, 200]
        features2_expanded = features2.unsqueeze(0).expand(features1.size(0),features2.size(0),features1.size(1))  # [a, 1, 200]
        concat_features = torch.cat((features1_expanded, features2_expanded),dim=2)
        Linear_term = torch.matmul(concat_features, self.V.t())
        x = F.relu(tensor_product + Linear_term + self.B).squeeze()
        #x = torch.matmul(x, self.U).squeeze()
        return x

class SimpleGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_out_features):
        super(SimpleGNN, self).__init__()
        # 定义第一个图卷积层，输入特征维度为num_node_features，输出特征维度为100
        self.conv1 = GCNConv(num_node_features, 100)
        # 定义第二个图卷积层，输入特征维度为100，输出特征维度为num_classes
        self.conv2 = GCNConv(100, 200)
        self.fc = torch.nn.Linear(200, num_out_features)

    def forward(self, edge_features, edge_indices):
        x = self.conv1(edge_features, edge_indices)
        x = F.relu(x)
        
        x = self.conv2(x, edge_indices)
        x = F.relu(x)

        x = self.fc(x)
        return x

class TripleConv(MessagePassing):
    """
    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn, eps: float = 0., 
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        
        self.DIRECTIONAL = True

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, 'in_features'):
                in_channels = nn.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            in_channels = 101
            self.lin = Linear(3 * edge_dim, edge_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()
    
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)
    
    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:

        # Switch i and j based on direction of triple:
        '''
        reverse = edge_attr[:, -1] == -1

        if self.DIRECTIONAL:
            x_i[reverse], x_j[reverse] = x_j[reverse], x_i[reverse]

        #x_i[edge_attr[:, -1] == -1] = x_j[edge_attr[:, -1] == -1]
        edge_attr = edge_attr[:, :-1]
        '''
       
        return self.lin(torch.cat((x_i, edge_attr, x_j), 1)).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

class RuleEmbedding(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RuleEmbedding, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 256
        self.num_layers = 2
        self.rnn = nn.RNN(self.input_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, ori_lengths):
        '''
        input: x: (batch_size, max_length, input_dim) --> the input sequence
               ori_lengths: (batch_size) --> the length of each sequence in the batch
        output: (batch_size, output_dim) --> the last hidden output of the RNN
        '''
        packed_x = pack_padded_sequence(x, ori_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_x)
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)

        hidden = hidden[-1].squeeze()
        output = self.linear(hidden)

        return output
    
    def padding_data(self, x):
        '''
        input: x: [(length, input_dim)] --> the list of input sequences
        output: (batch_size, max_length, input_dim) --> the padded input sequence
        '''
        max_length = max(tensor.shape[1] for tensor in x)
        ori_lengths = [tensor.shape[1] for tensor in x]
        padded_tensor = [torch.cat([tensor, torch.zeros(max_length - tensor.shape[1], tensor.shape[2]).to(tensor.device).unsqueeze(0)], dim=1) for tensor in x]
        padded_tensor = torch.cat(padded_tensor, dim=0)
        
        return padded_tensor, ori_lengths

class RuleRanker(torch.nn.Module):
    def __init__(self, input_dim):
        """
        input: input_dim(int)
        """
        self.hidden_dim1 = 128
        self.hidden_dim2 = 64
        super(RuleRanker, self).__init__()
        self.fc1 = nn.Linear(input_dim, self.hidden_dim1)
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, 1)

    def forward(self, x):
        """
        input: x(batch_size, dim)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        scores = self.fc3(x)
        scores = torch.sigmoid(scores)
        return scores.squeeze()


class GraphNet(torch.nn.Module):
    def __init__(self, args, rule_nums):
        super(GraphNet, self).__init__()

        self.args = args
        self.has_wasserstein = args.distance
        self.rule_nums = rule_nums
        self.setup_layers()
    
    def setup_layers(self):
        self.alpha = nn.Parameter(torch.tensor(0.9, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
        )
        '''
        self.mlp3 = torch.nn.Sequential(
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 200),
        )
        '''
        self.conv1 = TripleConv(self.mlp, edge_dim=100)
        self.conv2 = TripleConv(self.mlp2, edge_dim=100)
        #self.conv3 = TripleConv(self.mlp3, edge_dim=100)
        self.ntn = NTN(200, 200, self.args)
        self.fully_connected1 = nn.Linear(200 + self.rule_nums * 100, 500)
        self.fully_connected2 = nn.Linear(500, 100)
        self.fully_connected3 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(p=0.2)

        self.fully_connected_wasserstein = nn.Linear(2, 1)
        self.normalize_distance = NormalizeDistance()

        self.rule_embeder = RuleEmbedding(100, 100)
        self.rule_ranker = RuleRanker(100)
        #linear graph
        '''
        self.conv3 = SimpleGNN(100, 200)
        self.ntn2 = NTN(200, 200, self.args)
        '''
    def convolutional_pass(self, features, edge_index, edge_attr):
        features = self.conv1(features, edge_index, edge_attr)
        features = F.relu(features)

        features = self.conv2(features, edge_index, edge_attr)
        features = F.relu(features)

        #features = self.conv3(features, edge_index, edge_attr)
        #features = F.relu(features)
        
        return features
    '''
    def edge_convolutional_pass(self, features, edge_index):
        features = self.conv3(features, edge_index)
        features = F.relu(features)
        
        return features
    '''
    def forward(self, data):
        features_1 = data['node_features_0'].squeeze()
        features_2 = data['node_features_1'].squeeze()
        edge_index_1 = data['edge_indices_0'].squeeze()
        edge_index_2 = data['edge_indices_1'].squeeze()
        edge_attr_1 = data['edge_features_0'].squeeze()
        edge_attr_2 = data['edge_features_1'].squeeze()
        #line_indices_1 = data['line_indices_0'].squeeze()
        #line_indices_2 = data['line_indices_1'].squeeze()
        sampled_rules_1 = data['sampled_rules_0']
        sampled_rules_2 = data['sampled_rules_1']
        sampled_rules = sampled_rules_1 + sampled_rules_2

        #rule process
        padded_rules, ori_lengths = self.rule_embeder.padding_data(sampled_rules)
        rules_embedding = self.rule_embeder(padded_rules, ori_lengths)
        rules_score = self.rule_ranker(rules_embedding)
        top_scores, top_indices = torch.topk(rules_score, self.rule_nums)
        top_rules_embedding = rules_embedding[top_indices]

        #graph vector process
        features_1 = self.convolutional_pass(features_1, edge_index_1, edge_attr_1)
        features_2 = self.convolutional_pass(features_2, edge_index_2, edge_attr_2)
        g1 = global_add_pool(features_1, torch.zeros(features_1.size(0), dtype=torch.long).to(features_1.device))
        g2 = global_add_pool(features_2, torch.zeros(features_2.size(0), dtype=torch.long).to(features_1.device))

        #linear graph
        '''
        features_3 = self.edge_convolutional_pass(edge_attr_1, line_indices_1)
        features_4 = self.edge_convolutional_pass(edge_attr_2, line_indices_2)
        g3 = global_add_pool(features_3, torch.zeros(features_3.size(0), dtype=torch.long).to(features_1.device))
        g4 = global_add_pool(features_4, torch.zeros(features_4.size(0), dtype=torch.long).to(features_1.device))
        '''
        # wasserstein distance
        wasserstein_distance = self.wasserstein_distance(g1, g2)
        wasser_tensor = torch.tensor([wasserstein_distance], dtype=torch.float32).to(features_1.device)
        
        x = self.ntn(g1, g2)
        #score2 = self.ntn2(g3, g4)
        #score = torch.cat((score1, score2))
        y = top_rules_embedding.view(-1)
        x = torch.cat((x, y))
        x = self.fully_connected1(x)
        #score = self.dropout(score)
        x = F.relu(x)
        x = self.fully_connected2(x)
        x = F.relu(x)
        x = self.fully_connected3(x)
        x = torch.abs(x)
        #x = torch.cat((wasser_tensor, x))
        #x = self.fully_connected_wasserstein(x)
        if self.has_wasserstein:
            wasser_tensor = torch.relu(wasser_tensor)#self.normalize_distance(wasser_tensor)
            x = torch.cat((wasser_tensor, x))
            x = self.fully_connected_wasserstein(x)
            return torch.abs(x)
            #return (1-self.beta) * torch.abs(x) + self.beta * torch.abs(wasser_tensor) / data['norm']
        else:
            return torch.abs(x)
    
    def wasserstein_distance(self, x1, x2):
        x1 = x1.view(-1, 1).cpu().detach().numpy()
        x2 = x2.view(-1, 1).cpu().detach().numpy()
        
        a = np.ones((x1.shape[0],)) / x1.shape[0]
        b = np.ones((x2.shape[0],)) / x2.shape[0]
        
        # 计算距离矩阵
        M = ot.dist(x1, x2, metric='euclidean')
        
        # 计算 Wasserstein 距离
        distance = ot.emd2(a, b, M)
    
        return distance
    
class NormalizeDistance(nn.Module):
    def __init__(self):
        super(NormalizeDistance, self).__init__()
        self.fc = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x