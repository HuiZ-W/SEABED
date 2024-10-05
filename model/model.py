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
from .layers.ntn import NTN
from .layers.ruleEmbedding import RuleEmbedding, RuleRanker
from .layers.attention import SelfAttention
from .layers.updateGate import LinearGate
import time

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

        return self.lin(torch.cat((x_i, edge_attr, x_j), 1)).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class GraphNet(torch.nn.Module):
    def __init__(self, args):
        super(GraphNet, self).__init__()
        self.dim = args.rule_graph_dim
        self.args = args
        self.init_emb_size = 100
        self.rule_dims = self.dim
        self.graph_vector_dims = self.dim
        self.hidden_dim1 = 100
        self.hidden_dim2 = 64
        self.hidden_dim3 = self.dim
        self.rule_nums = args.sample_rule_nums

        self.info_type = args.info_type
        self.combine_type = args.combine_type
        self.setup_layers()
    
    def setup_layers(self):
        #graph convolution process
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim1, self.hidden_dim1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim1, self.hidden_dim1),
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim1, self.hidden_dim2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim2, self.hidden_dim3),
        )
        self.conv1 = TripleConv(self.mlp, edge_dim=self.init_emb_size)
        self.conv2 = TripleConv(self.mlp2, edge_dim=self.init_emb_size)
        #graph vector process
        self.ntn = NTN(self.graph_vector_dims, self.graph_vector_dims, self.args)
        #rule embedding process
        self.rule_embeder = RuleEmbedding(self.init_emb_size, self.rule_dims)
        #vector combination process
        if self.combine_type == 'concat':
            self.rule_attention = SelfAttention(self.rule_dims)
            self.norm = torch.nn.Linear(self.rule_dims + self.graph_vector_dims, self.hidden_dim3)
            self.fully_connected1 = nn.Linear(self.hidden_dim3, 32)
            self.fully_connected2 = nn.Linear(32, 16)
            self.fully_connected3 = nn.Linear(16, 1)

        elif self.combine_type == 'fusion':
            if self.info_type == 'attention':
                self.rule_attention = SelfAttention(self.rule_dims)
            elif self.info_type == 'concat':
                self.rule_process = torch.nn.Sequential(
                                        torch.nn.Linear(self.rule_nums * self.rule_dims * 2, self.hidden_dim3),
                                        torch.nn.ReLU()
                                        )
            self.fusion_gate = LinearGate(self.graph_vector_dims, self.rule_dims, self.hidden_dim3)
            self.fully_connected1 = nn.Linear(self.hidden_dim3, 32)
            self.fully_connected2 = nn.Linear(32, 16)
            self.fully_connected3 = nn.Linear(16, 1)

    def convolutional_pass(self, features, edge_index, edge_attr):
        features = self.conv1(features, edge_index, edge_attr)
        features = F.relu(features)

        features = self.conv2(features, edge_index, edge_attr)
        features = F.relu(features)
        return features
    
    def forward(self, data):
        features_1 = data['node_features_0'].squeeze()
        features_2 = data['node_features_1'].squeeze()
        edge_index_1 = data['edge_indices_0'].squeeze()
        edge_index_2 = data['edge_indices_1'].squeeze()
        edge_attr_1 = data['edge_features_0'].squeeze()
        edge_attr_2 = data['edge_features_1'].squeeze()
        sampled_rules = data['rules'].squeeze(0)
        ori_lengths = data['ori_lengths']

        t_start = time.time()
        #graph vector process
        features_1 = self.convolutional_pass(features_1, edge_index_1, edge_attr_1)
        features_2 = self.convolutional_pass(features_2, edge_index_2, edge_attr_2)
        g1 = global_add_pool(features_1, torch.zeros(features_1.size(0), dtype=torch.long).to(features_1.device))
        g2 = global_add_pool(features_2, torch.zeros(features_2.size(0), dtype=torch.long).to(features_1.device))
        graph_vector, x1 = self.ntn(g1, g2)
        #rule embedding process
        rules_embedding = self.rule_embeder(sampled_rules, ori_lengths)
        #vector combination process
        if rules_embedding.dim() < 2:
            rules_embedding = rules_embedding.unsqueeze(0)

        t_end = time.time()
        GR_time = t_end - t_start
        
        t_start = time.time()
        if self.combine_type == 'fusion':
            if self.info_type == 'attention':
                rules_fusion, attention_weight = self.rule_attention(rules_embedding, graph_vector)
                final_vector = self.fusion_gate(graph_vector, rules_fusion)
            elif self.info_type == 'concat':
                attention_weight = []
                rules_embedding = torch.flatten(rules_embedding, start_dim=0)
                rules_fusion = self.rule_process(rules_embedding)
                final_vector = self.fusion_gate(graph_vector, rules_fusion)
        elif self.combine_type == 'concat':
            rules_embedding, attention_weight = self.rule_attention(rules_embedding, graph_vector)
            final_vector = torch.cat((graph_vector, rules_embedding), dim=0)
            final_vector = self.norm(final_vector)
        x = self.fully_connected1(final_vector)
        x = F.relu(x)
        x = self.fully_connected2(x)
        x = F.relu(x)
        x = self.fully_connected3(x)
        t_end = time.time()
        F_time = t_end - t_start
        return torch.abs(x), attention_weight#, [GR_time, F_time]