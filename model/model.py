import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch_geometric.nn import global_add_pool, global_mean_pool
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
from scipy.stats import wasserstein_distance
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
                 train_eps: bool = False, node_dim: Optional[int] = None, edge_dim: Optional[int] = None,
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
            self.lin = Linear(2* node_dim + edge_dim, node_dim)
            self.lin2 = Linear(2 *node_dim + edge_dim, node_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()
            self.lin2.reset_parameters()
    
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
        #x_i和x_j是根据edge_index对node_features进行重排(x_i是源节点，x_j是目标节点)
        #最终结果会聚合到x_j的排列上
        #x_i x_j
        # 1   2
        # 2   1
        #edge_feature在外面模型进行了反向边添加，也就是edge_index的一半是正向边，一半是反向边
        half_length = x_i.size(0) // 2
        x_i_1, x_i_2 = x_i[:half_length], x_i[half_length:]
        x_j_1, x_j_2 = x_j[:half_length], x_j[half_length:]
        edge_attr_1, edge_attr_2 = edge_attr[:half_length], edge_attr[half_length:]
        # 将1 edge 2聚合到目标节点
        res1 = self.lin(torch.cat((x_i_1, edge_attr_1, x_j_1), 1)).relu()
        # 将1 edge 2聚合到源节点
        res2 = self.lin2(torch.cat((x_j_2, edge_attr_2, x_i_2), 1)).relu()
        res = torch.cat((res1, res2), 0)
        return res
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
        trans_edge_index_1 = torch.stack([edge_index_1[1], edge_index_1[0]], dim=0)
        combined_edge_index1 = torch.cat([edge_index_1, trans_edge_index_1], dim=1)
        edge_index_2 = data['edge_indices_1'].squeeze()
        trans_edge_index_2 = torch.stack([edge_index_2[1], edge_index_2[0]], dim=0)
        combined_edge_index2 = torch.cat([edge_index_2, trans_edge_index_2], dim=1)
        edge_attr_1 = data['edge_features_0'].squeeze()
        combined_edge_attr1 = torch.cat([edge_attr_1, edge_attr_1], dim=0)
        edge_attr_2 = data['edge_features_1'].squeeze()
        combined_edge_attr2 = torch.cat([edge_attr_2, edge_attr_2], dim=0)
        sampled_rules = data['rules'].squeeze(0)
        ori_lengths = data['ori_lengths']

        t1 = time.time()
        #graph vector process
        features_1 = self.convolutional_pass(features_1, combined_edge_index1,combined_edge_attr1)
        features_2 = self.convolutional_pass(features_2, combined_edge_index2,combined_edge_attr2)
        g1 = global_add_pool(features_1, torch.zeros(features_1.size(0), dtype=torch.long).to(features_1.device))
        g2 = global_add_pool(features_2, torch.zeros(features_2.size(0), dtype=torch.long).to(features_1.device))
        graph_vector, x1 = self.ntn(g1, g2)
        t2 = time.time()
        #rule embedding process
        rules_embedding = self.rule_embeder(sampled_rules, ori_lengths)
        #vector combination process
        if rules_embedding.dim() < 2:
            rules_embedding = rules_embedding.unsqueeze(0)

        t3 = time.time()
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
        t4 = time.time()
        F_time = t4 - t3
        G_time = t2 - t1
        R_time = t3 - t2
        return torch.abs(x), attention_weight, [G_time,R_time,F_time]
    

class SubGraphNet(torch.nn.Module):
    def __init__(self, args):
        super(SubGraphNet, self).__init__()
        self.dim = args.rule_graph_dim
        self.n_patch = args.n_patch
        self.args = args
        self.init_emb_size = 100
        if args.use_local_count:
            self.init_count_size = 107
        else:
            self.init_count_size = 100
        self.graph_vector_dims = self.dim
        self.hidden_dim1 = 100
        self.hidden_dim2 = 64
        self.hidden_dim3 = self.dim
        self.position_encoding = self.create_position_encoding(self.hidden_dim3, self.n_patch * 2 + 1).to(args.device)
        self.use_patch = args.use_patch
        self.use_extra_loss = args.use_extra_loss
        self.use_local_count = args.use_local_count
        self.setup_layers()
        

    def create_position_encoding(self, dim, num):
        # 生成从1到9的sin位置编码
        position = torch.arange(1, num+1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(num, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def setup_layers(self):
        #graph convolution process(whole graph)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.init_count_size, self.hidden_dim1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim1, self.hidden_dim1),
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim1, self.hidden_dim2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim2, self.hidden_dim3),
        )
        self.conv1 = TripleConv(self.mlp, node_dim=self.init_count_size, edge_dim=self.init_emb_size)
        self.conv2 = TripleConv(self.mlp2, node_dim=self.init_emb_size, edge_dim=self.init_emb_size)
        self.ntn = NTN(self.graph_vector_dims, self.graph_vector_dims, self.args)
        #graph convolution process(subgraph)
        self.mlp3 = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim1, self.hidden_dim1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim1, self.hidden_dim1),
        )
        self.mlp4 = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim1, self.hidden_dim2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim2, self.hidden_dim3),
        )
        self.conv3 = TripleConv(self.mlp3, node_dim=self.init_emb_size, edge_dim=self.init_emb_size)
        self.conv4 = TripleConv(self.mlp4, node_dim=self.init_emb_size, edge_dim=self.init_emb_size)
        #subgraph vector process
        self.position_encoder = nn.Linear(8, self.hidden_dim3)

        self.multhead_attention = nn.MultiheadAttention(self.hidden_dim3, 4)
        if self.use_patch:
            self.mlp5 = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim3 * 2, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
            )
        else:
            self.mlp5 = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim3, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
            )

    def convolutional_pass(self, features, edge_index, edge_attr):
        features = self.conv1(features, edge_index, edge_attr)
        features = F.relu(features)

        features = self.conv2(features, edge_index, edge_attr)
        #features = F.relu(features)
        return features
    
    def convolutional_pass_subgraph(self, features, edge_index, edge_attr):
        features = self.conv3(features, edge_index, edge_attr)
        features = F.relu(features)

        features = self.conv4(features, edge_index, edge_attr)
        #features = F.relu(features)
        return features

    def emd_loss(self, u_value, v_value):
        # 假设 x 和 y 是形状为 (batch_size, dim) 的张量
    
        u_weights = torch.ones_like(u_value) / u_value.size(0)
        u_weights = u_weights / u_weights.sum()
        v_weights = torch.ones_like(v_value) / v_value.size(0)
        v_weights = v_weights / v_weights.sum()

        u_sorter = torch.argsort(u_value)
        v_sorter = torch.argsort(v_value)
        all_values = torch.cat((u_value, v_value))
        all_values, _ = torch.sort(all_values)

        deltas = torch.diff(all_values)

        u_cdf_indices = torch.searchsorted(u_value[u_sorter], all_values[:-1], right=True)
        v_cdf_indices = torch.searchsorted(v_value[v_sorter], all_values[:-1], right=True)

        u_cdf = torch.zeros_like(all_values)
        v_cdf = torch.zeros_like(all_values)

        u_sorted_cumweights = torch.cat((torch.tensor([0], dtype=torch.float32).to(u_value.device),
                                            torch.cumsum(u_weights[u_sorter], dim=0)))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

        v_sorted_cumweights = torch.cat((torch.tensor([0], dtype=torch.float32).to(v_value.device),
                                            torch.cumsum(v_weights[v_sorter], dim=0)))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

        return torch.sum(torch.abs(u_cdf - v_cdf) * deltas)

    def forward(self, data):
        start_time = time.time()
        #Process whole-graph features
        graph1 = data['whole_graph_0']
        graph2 = data['whole_graph_1']
        features_1 = graph1['node_features'].squeeze()
        features_2 = graph2['node_features'].squeeze()
        edge_index_1 = graph1['edge_indices'].squeeze()
        edge_index_2 = graph2['edge_indices'].squeeze()
        edge_features_1 = graph1['edge_features'].squeeze()
        edge_features_2 = graph2['edge_features'].squeeze()
        features_1 = self.convolutional_pass(features_1, edge_index_1, edge_features_1)
        features_2 = self.convolutional_pass(features_2, edge_index_2, edge_features_2)
        g1 = global_add_pool(features_1, torch.zeros(features_1.size(0), dtype=torch.long).to(features_1.device))
        g2 = global_add_pool(features_2, torch.zeros(features_2.size(0), dtype=torch.long).to(features_1.device))
        graph_vector, x1 = self.ntn(g1, g2)
        if self.training and self.use_extra_loss:
            emb_loss = self.emd_loss(g1.squeeze(), g2.squeeze())
        else:
            emb_loss = 0
        global_time = time.time() - start_time
        start_time = time.time()
        #Process subgraph features
        if self.use_patch:
            subgraph1 = data['subgraph_features_0']
            subgraph2 = data['subgraph_features_1']
            subgraph1_pe = data['patch_pe_0'].squeeze()
            subgraph2_pe = data['patch_pe_1'].squeeze()
            subgraph1_index = data['batch_index_0'].squeeze()
            subgraph2_index = data['batch_index_1'].squeeze()
            #subgraph1
            subgraph_features = subgraph1['node_features'].squeeze()
            edge_index = subgraph1['edge_indices'].squeeze(0)
            edge_features = subgraph1['edge_features'].squeeze()
            features = self.convolutional_pass_subgraph(subgraph_features, edge_index, edge_features)
            sub1 = global_add_pool(features, subgraph1_index).to(features.device)
            #subgraph2
            subgraph_features = subgraph2['node_features'].squeeze()
            edge_index = subgraph2['edge_indices'].squeeze(0)
            edge_features = subgraph2['edge_features'].squeeze()
            features = self.convolutional_pass_subgraph(subgraph_features, edge_index, edge_features)
            sub2 = global_add_pool(features, subgraph2_index).to(features.device)

            subgraph1_global = sub1 + self.position_encoder(subgraph1_pe)
            subgraph2_global = sub2 + self.position_encoder(subgraph2_pe)

            combined_global = torch.cat([subgraph1_global, subgraph2_global], dim=0)
            cls_token = torch.zeros(1, combined_global.size(1)).to(combined_global.device)
            combined_global = torch.cat([cls_token, combined_global], dim=0)
            combined_global = combined_global + self.position_encoding
            attn_output, attn_output_weights = self.multhead_attention(combined_global, combined_global, combined_global) 
            attn_output = attn_output[0]
            local_time = time.time() - start_time
            start_time = time.time()
            x = torch.cat([attn_output, graph_vector], dim=0)
            x = self.mlp5(x)
            F_time = time.time() - start_time
            return torch.abs(x), emb_loss, global_time, local_time, F_time
        else:  
            x = self.mlp5(graph_vector)
            return torch.abs(x), emb_loss



class RuleGraphNet(torch.nn.Module):
    def __init__(self, args):
        super(RuleGraphNet, self).__init__()
        self.dim = args.rule_graph_dim
        self.rule_nums = args.sample_rule_nums
        self.args = args
        self.init_emb_size = 100
        if args.use_local_count:
            self.init_count_size = 107
        else:
            self.init_count_size = 100
        self.graph_vector_dims = self.dim
        self.hidden_dim1 = 100
        self.hidden_dim2 = 64
        self.hidden_dim3 = self.dim
        self.use_rules = args.use_rules
        self.use_extra_loss = args.use_extra_loss
        self.use_local_count = args.use_local_count
        self.setup_layers()
    
    def setup_layers(self):
        #graph convolution process(whole graph)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.init_count_size, self.hidden_dim1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim1, self.hidden_dim1),
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim1, self.hidden_dim2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim2, self.hidden_dim3),
        )
        self.conv1 = TripleConv(self.mlp, node_dim=self.init_count_size, edge_dim=self.init_emb_size)
        self.conv2 = TripleConv(self.mlp2, node_dim=self.init_emb_size, edge_dim=self.init_emb_size)
        self.ntn = NTN(self.graph_vector_dims, self.graph_vector_dims, self.args)
        #graph convolution process(subgraph)
        self.mlp3 = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim1, self.hidden_dim1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim1, self.hidden_dim1),
        )
        self.mlp4 = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim1, self.hidden_dim2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim2, self.hidden_dim3),
        )
        self.conv3 = TripleConv(self.mlp3, node_dim=self.init_emb_size, edge_dim=self.init_emb_size)
        self.conv4 = TripleConv(self.mlp4, node_dim=self.init_emb_size, edge_dim=self.init_emb_size)
        #subgraph vector process
        self.position_encoder = nn.Linear(8, self.hidden_dim3)
        self.multhead_attention = nn.MultiheadAttention(self.hidden_dim3, 4)
        if self.use_rules:
            self.mlp5 = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim3 * 2, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
            )
        else:
            self.mlp5 = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim3, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
            )
    def convolutional_pass(self, features, edge_index, edge_attr):
        features = self.conv1(features, edge_index, edge_attr)
        features = F.relu(features)

        features = self.conv2(features, edge_index, edge_attr)
        #features = F.relu(features)
        return features
    
    def convolutional_pass_subgraph(self, features, edge_index, edge_attr):
        features = self.conv3(features, edge_index, edge_attr)
        features = F.relu(features)

        features = self.conv4(features, edge_index, edge_attr)
        #features = F.relu(features)
        return features

    def emd_loss(self, u_value, v_value):
        # 假设 x 和 y 是形状为 (batch_size, dim) 的张量
    
        u_weights = torch.ones_like(u_value) / u_value.size(0)
        u_weights = u_weights / u_weights.sum()
        v_weights = torch.ones_like(v_value) / v_value.size(0)
        v_weights = v_weights / v_weights.sum()

        u_sorter = torch.argsort(u_value)
        v_sorter = torch.argsort(v_value)
        all_values = torch.cat((u_value, v_value))
        all_values, _ = torch.sort(all_values)

        deltas = torch.diff(all_values)

        u_cdf_indices = torch.searchsorted(u_value[u_sorter], all_values[:-1], right=True)
        v_cdf_indices = torch.searchsorted(v_value[v_sorter], all_values[:-1], right=True)

        u_cdf = torch.zeros_like(all_values)
        v_cdf = torch.zeros_like(all_values)

        u_sorted_cumweights = torch.cat((torch.tensor([0], dtype=torch.float32).to(u_value.device),
                                            torch.cumsum(u_weights[u_sorter], dim=0)))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

        v_sorted_cumweights = torch.cat((torch.tensor([0], dtype=torch.float32).to(v_value.device),
                                            torch.cumsum(v_weights[v_sorter], dim=0)))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

        return torch.sum(torch.abs(u_cdf - v_cdf) * deltas)

    def forward(self, data):
        #Process whole-graph features
        graph1 = data['whole_graph_0']
        graph2 = data['whole_graph_1']
        features_1 = graph1['node_features'].squeeze()
        features_2 = graph2['node_features'].squeeze()
        edge_index_1 = graph1['edge_indices'].squeeze()
        edge_index_2 = graph2['edge_indices'].squeeze()
        edge_features_1 = graph1['edge_features'].squeeze()
        edge_features_2 = graph2['edge_features'].squeeze()
        features_1 = self.convolutional_pass(features_1, edge_index_1, edge_features_1)
        features_2 = self.convolutional_pass(features_2, edge_index_2, edge_features_2)
        g1 = global_add_pool(features_1, torch.zeros(features_1.size(0), dtype=torch.long).to(features_1.device))
        g2 = global_add_pool(features_2, torch.zeros(features_2.size(0), dtype=torch.long).to(features_1.device))
        graph_vector, x1 = self.ntn(g1, g2)
        if self.training and self.use_extra_loss:
            emb_loss = self.emd_loss(g1.squeeze(), g2.squeeze())
        else:
            emb_loss = 0
        #Process rules features
        if self.use_rules == True:
            graph1_rules = data['sampled_rules_0']
            rules1_batch_index = data['rules_0_info']
            graph2_rules = data['sampled_rules_1']
            rules2_batch_index = data['rules_1_info']
            graph1_rule_global = []
            graph2_rule_global = []
            if len(rules1_batch_index) != 0:
                features_0 = graph1_rules['node_features'].squeeze()
                edge_index_0 = graph1_rules['edge_indices'].squeeze(0)
                edge_attr_0 = graph1_rules['edge_features'].squeeze()
                P_rules_0 = self.convolutional_pass_subgraph(features_0, edge_index_0, edge_attr_0)
                g1 = global_add_pool(P_rules_0, rules1_batch_index.squeeze())
            else:
                g1 = torch.zeros(1, self.dim).to(graph_vector.device)
            if len(rules2_batch_index) != 0:
                features_1 = graph2_rules['node_features'].squeeze()
                edge_index_1 = graph2_rules['edge_indices'].squeeze(0)
                edge_attr_1 = graph2_rules['edge_features'].squeeze()
                P_rules_1 = self.convolutional_pass_subgraph(features_1, edge_index_1, edge_attr_1)
                g2 = global_add_pool(P_rules_1, rules2_batch_index.squeeze())
            else:
                g2 = torch.zeros(1, self.dim).to(graph_vector.device)
            combined_global = torch.cat([g1, g2], dim=0)
            cls_token = torch.zeros(1, combined_global.size(1)).to(combined_global.device)
            combined_global = torch.cat([cls_token, combined_global], dim=0)
            attn_output, attn_output_weights = self.multhead_attention(combined_global, combined_global, combined_global) 
            attn_output = attn_output[0]
            x = torch.cat([attn_output, graph_vector], dim=0)
            x = self.mlp5(x)
            return torch.abs(x), emb_loss
        else:
            x = self.mlp5(graph_vector)
            return torch.abs(x), emb_loss
                
            
