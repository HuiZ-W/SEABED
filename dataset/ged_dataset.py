from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import glob
import json
import torch
import random
import itertools
import time
import torch.nn.functional as F
from multiprocessing import Pool
class GEDDataSet:
    def __init__(self, data_dir, PreLoad=False, device=None, args=None):
        if device == None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        if args != None:
            self.sample_num = args.sample_rule_nums
            self.rule_length = args.rule_length
        else:
            self.sample_num = 32
            self.rule_length = 10

        self.data_dir = data_dir
        self.data_files = glob.glob(os.path.join(data_dir, "*.json"))
        self.preload = PreLoad
        self.data = []
        if PreLoad:
            for file in self.data_files:
                data = self.load_file(file)
                self.data.append(data)

    def __len__(self):
        return len(self.data_files)

    def load_file(self, file):
        with open(file, "r") as f:
            data = {}
            metadata = json.load(f)
            i = 0
            for j, Graph in metadata.items():
                if j == 'GED':
                    #data['gt_ged'] = torch.tensor(Graph, dtype=torch.float32).to(self.device)
                    data['gt_ged'] = torch.tensor(Graph/(0.25*(len(data["node_features_1"])+len(data["node_features_0"])+\
                                                               len(data["edge_features_1"])+len(data["edge_features_0"]))),dtype=torch.float32).to(self.device)
                    data['real_ged'] = torch.tensor(Graph, dtype=torch.float32).to(self.device)
                    continue
                node_features = Graph["node_features"]
                data["node_features_" + str(i)] = torch.tensor([node["embedding"] for node in node_features], dtype=torch.float32).to(self.device).squeeze()
                edge_indices = Graph["edge_indices"]
                edge_indices = torch.tensor(edge_indices)
                data["edge_indices_" + str(i)] = torch.tensor(edge_indices).to(self.device).squeeze().transpose(0, 1)
                edge_features = Graph["edge_features"]
                data["edge_features_" + str(i)] = torch.tensor([edge["embedding"] for edge in edge_features], dtype=torch.float32).to(self.device).squeeze()
                #data["line_indices_" + str(i)] = torch.tensor(self.build_linear_graph(edge_indices)).to(self.device).squeeze().transpose(0, 1)
                data["sampled_rules_" + str(i)] = self.rule_sample(self.sample_num, self.rule_length, len(node_features), edge_features, edge_indices)
                i += 1
            data['norm'] = torch.tensor(0.25*(len(data["node_features_1"])+len(data["node_features_0"])+\
                                                 len(data["edge_features_1"])+len(data["edge_features_0"]))).to(self.device)
        return data

    def __getitem__(self, index):
        if self.preload:
            return self.data[index]
        else:
            filename = self.data_files[index]
            return self.load_file(filename)
        
    def build_linear_graph(self, edge_indices):
        line_graph_indices = []
        for i, edge1 in enumerate(edge_indices):
            for j, edge2 in enumerate(edge_indices):
                if edge1[0] == edge2[1] or edge1[0] == edge2[0] or edge1[1] == edge2[1] or edge1[1] == edge2[0]:
                    line_graph_indices.append([i,j])
    
        return line_graph_indices
    
    def rule_sample(self, num_samples, num_steps, nodes, edges, edge_indices):
        """
        Randomly sample rules from the graph
        :param num_samples(int): number of samples
        :param num_steps(list[min_step, max_step]): number of steps in the random walk
        :param nodes(list[node1,....]): nodes in the graph
        :param edges(list[edge1,....]): edges in the graph
        :param edge_indices(list[[edge_index1],....]): edge indices in the graph
        """
        sample_rules = []
        while len(sample_rules) < num_samples:
            start_node = random.choice(list(range(nodes)))
            edge_indices = edge_indices
            sampled_edges = self.random_walk_sampling(start_node, num_steps, edge_indices, edges)
            if sampled_edges == None:
                continue
            sample_rules.extend(sampled_edges)
        return sample_rules

    def random_walk_sampling(self, start_node, num_steps, edge_indices, edges):
        """
        Random walk sampling
        :param start_node(int): start node
        :param numsteps(int): number of steps in the random walk
        :param edge_indices(list[[edge_index1],....]): edge indices in the graph
        :param edges(list[edge1,....]): edges in the graph
        """
        sampled_edges = []
        current_node = start_node
        for _ in range(num_steps):
            neighbors = []
            for i, edge in enumerate(edge_indices):
                if edge[0] == current_node:
                    neighbors.append(i)
            if not neighbors:
                break
            chosen_edge_index = random.choice(neighbors)
            sampled_edges.append(edges[chosen_edge_index]["embedding"])
            current_node = edge_indices[chosen_edge_index][1]
        if len(sampled_edges) == 0:
            return None
            sampled_edge_embeddings_tensor = torch.zeros_like(torch.tensor(edges[0]["embedding"])).to(self.device).unsqueeze(0)
        else:
            sampled_edge_embeddings_tensor = torch.tensor(sampled_edges, dtype=torch.float32).to(self.device)
        if sampled_edge_embeddings_tensor.dim() == 2:
            sampled_edge_embeddings_tensor = sampled_edge_embeddings_tensor.to(self.device).unsqueeze(0)
        else:
            sampled_edge_embeddings_tensor.permute(1, 0, 2).to(self.device)
        return sampled_edge_embeddings_tensor 

class NEWGEDDataSet:
    def __init__(self, data_dir, pairs_file_path, PreLoad=False, device=None, args=None):
        if device == None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        if args != None:
            self.sample_num = args.sample_rule_nums
            self.rule_length = args.rule_length
            self.deduplicate = args.deduplicate
        else:
            self.sample_num = 32
            self.rule_length = 10
            self.deduplicate = False

        self.data_dir = data_dir
        self.data_pairs = self.load_pairs(pairs_file_path) 
        self.preload = PreLoad
        self.data = []
        if PreLoad:
            for pair in self.data_pairs:
                data = self.load_file(pair)
                self.data.append(data)

    def __len__(self):
        return len(self.data_pairs)

    def load_pairs(self, file):
        with open(file, "r") as f:
            data = json.load(f)
        pairs = data['pairs_info']
        return pairs

    def load_file(self, pair):
        time_start = time.time()
        data = {}
        data['graph1'] = pair[0]
        data['graph2'] = pair[1]
        file1, file2, ged = pair[0], pair[1], pair[2]
        file1 = os.path.join(self.data_dir, file1)
        file2 = os.path.join(self.data_dir, file2)
        t1 = time.time()
        with open(file1, "r") as f:
            metadata = json.load(f)
            Graph = metadata['0']  
            node_features = Graph["node_features"]
            data["node_features_0"] = torch.tensor([node["embedding"] for node in node_features], dtype=torch.float32).to(self.device).squeeze()
            edge_indices = Graph["edge_indices"]
            edge_indices = torch.tensor(edge_indices)
            data["edge_indices_0"] = torch.tensor(edge_indices).to(self.device).squeeze().transpose(0, 1)
            edge_features = Graph["edge_features"]
            data["edge_features_0"] = torch.tensor([edge["embedding"] for edge in edge_features], dtype=torch.float32).to(self.device).squeeze()
            t2 = time.time()
            data["sampled_rules_0"], index1 = self.rule_sample(self.sample_num, self.rule_length, len(node_features), edge_features, edge_indices)
            t3 = time.time()
            data['n1'] = len(node_features)
        with open(file2, "r") as f:
            metadata = json.load(f)
            Graph = metadata['0']  
            node_features = Graph["node_features"]
            data["node_features_1"] = torch.tensor([node["embedding"] for node in node_features], dtype=torch.float32).to(self.device).squeeze()
            edge_indices = Graph["edge_indices"]
            edge_indices = torch.tensor(edge_indices)
            data["edge_indices_1"] = torch.tensor(edge_indices).to(self.device).squeeze().transpose(0, 1)
            edge_features = Graph["edge_features"]
            data["edge_features_1"] = torch.tensor([edge["embedding"] for edge in edge_features], dtype=torch.float32).to(self.device).squeeze()
            t5 = time.time()
            data["sampled_rules_1"], index2 = self.rule_sample(self.sample_num, self.rule_length, len(node_features), edge_features, edge_indices)
            t6 = time.time()
            data['n2'] = len(node_features)
        t7 = time.time()
        sampled_rules = data["sampled_rules_0"] + data["sampled_rules_1"]
        sampled_rules, ori_length = self.rules_padding(sampled_rules)
        if self.deduplicate:
            sampled_rules, ori_length = self.deduplicate_rules(sampled_rules, ori_length)
            if sampled_rules.dim() < 2:
                sampled_rules = sampled_rules.unsqueeze(0) 
        data['rules'] = sampled_rules
        data['ori_lengths'] = ori_length
        data['norm'] = torch.tensor(0.25*(len(data["node_features_1"])+len(data["node_features_0"])+\
                                                 len(data["edge_features_1"])+len(data["edge_features_0"]))).to(self.device)
        data['gt_ged'] = torch.tensor(ged/(0.25*(len(data["node_features_1"])+len(data["node_features_0"])+\
                                                               len(data["edge_features_1"])+len(data["edge_features_0"]))),dtype=torch.float32).to(self.device)
        data['real_ged'] = torch.tensor(ged, dtype=torch.float32).to(self.device)
        data['rule_index'] = index1+index2
        time_end = time.time()
        data['Graph_Time'] = t7 - t1 - (t6 - t5 + t3 - t2)
        data['Rule_Time'] = t6 - t5 + t3 - t2
        return data

    def __getitem__(self, index):
        if self.preload:
            return self.data[index]
        else:
            pair = self.data_pairs[index]
            return self.load_file(pair)
        
    def build_linear_graph(self, edge_indices):
        line_graph_indices = []
        for i, edge1 in enumerate(edge_indices):
            for j, edge2 in enumerate(edge_indices):
                if edge1[0] == edge2[1] or edge1[0] == edge2[0] or edge1[1] == edge2[1] or edge1[1] == edge2[0]:
                    line_graph_indices.append([i,j])
    
        return line_graph_indices
    
    def rules_padding(self, x):
        max_length = max([tensor.shape[0] for tensor in x])
        ori_lengths = [tensor.shape[0] for tensor in x]
        padded_tensor = [torch.cat([tensor.unsqueeze(0), torch.zeros(max_length - tensor.shape[0], tensor.shape[1]).to(tensor.device).unsqueeze(0)], dim=1) for tensor in x]
        padded_tensor = torch.cat(padded_tensor, dim=0)
        
        return padded_tensor, ori_lengths
    
    def deduplicate_rules(self, x, ori_lengths):
        padded_rules, inverse_index= torch.unique(x, dim=0, return_inverse=True)
        new_lengths = []
        for i in range(padded_rules.size(0)):
            first_index = torch.nonzero(inverse_index == i, as_tuple=False)[0].item()
            new_lengths.append(ori_lengths[first_index])
        return padded_rules, new_lengths

    def rule_sample(self, num_samples, num_steps, nodes, edges, edge_indices):
        """
        Randomly sample rules from the graph
        :param num_samples(int): number of samples
        :param num_steps(list[min_step, max_step]): number of steps in the random walk
        :param nodes(list[node1,....]): nodes in the graph
        :param edges(list[edge1,....]): edges in the graph
        :param edge_indices(list[[edge_index1],....]): edge indices in the graph
        """
        sample_rules = []
        rules_index = []
        while len(sample_rules) < num_samples:
            start_node = random.choice(list(range(nodes)))
            edge_indices = edge_indices
            sampled_edges, sample_index = self.random_walk_sampling(start_node, num_steps, edge_indices, edges)
            if sampled_edges == None:
                continue
            sample_rules.extend(sampled_edges)
            rules_index.extend(sample_index)
        return sample_rules, rules_index

    def random_walk_sampling(self, start_node, num_steps, edge_indices, edges):
        """
        Random walk sampling
        :param start_node(int): start node
        :param numsteps(int): number of steps in the random walk
        :param edge_indices(list[[edge_index1],....]): edge indices in the graph
        :param edges(list[edge1,....]): edges in the graph
        """
        sampled_edges = []
        sampled_index = []
        current_node = start_node
        for _ in range(num_steps):
            neighbors = []
            for i, edge in enumerate(edge_indices):
                if edge[0] == current_node:
                    neighbors.append(i)
            if not neighbors:
                break
            chosen_edge_index = random.choice(neighbors)
            sampled_index.append(chosen_edge_index)
            sampled_edges.append(edges[chosen_edge_index]["embedding"])
            current_node = edge_indices[chosen_edge_index][1]
        if len(sampled_edges) == 0:
            return None, None
            sampled_edge_embeddings_tensor = torch.zeros_like(torch.tensor(edges[0]["embedding"])).to(self.device).unsqueeze(0)
        else:
            sampled_edge_embeddings_tensor = torch.tensor(sampled_edges, dtype=torch.float32).to(self.device)
        if sampled_edge_embeddings_tensor.dim() == 2:
            sampled_edge_embeddings_tensor = sampled_edge_embeddings_tensor.to(self.device).unsqueeze(0)
        else:
            sampled_edge_embeddings_tensor = sampled_edge_embeddings_tensor.permute(1, 0, 2).to(self.device)
        return sampled_edge_embeddings_tensor, [sampled_index]

import pymetis
import networkx as nx
class MetisGEDDataSet:
    def __init__(self, data_dir, pairs_file_path, PreLoad=False, device=None, args=None):
        if device == None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        if args != None:
            self.n_patch = args.n_patch
        else:
            self.n_patch = 4
        self.use_local_count = args.use_local_count
        self.data_dir = data_dir
        self.data_pairs = self.load_pairs(pairs_file_path) 
        self.preload = PreLoad
        self.num_worker = args.num_worker
        self.data = []
        if PreLoad:
            if self.num_worker > 1:
                self.load_data_multiprocessing()
            else:
                for pair in self.data_pairs:
                    data = self.load_file(pair)
                    self.data.append(data)

    def __len__(self):
        return len(self.data_pairs)

    def load_pairs(self, file):
        with open(file, "r") as f:
            data = json.load(f)
        pairs = data['pairs_info']
        return pairs
    
    def load_data_multiprocessing(self):
        with Pool(processes=self.num_worker) as pool:
            self.data = pool.map(self.load_file, self.data_pairs)

    def load_file(self, pair):
        data = {}
        data['graph1'] = pair[0]
        data['graph2'] = pair[1]
        file1, file2, ged = pair[0], pair[1], pair[2]
        file1 = os.path.join(self.data_dir, file1)
        file2 = os.path.join(self.data_dir, file2)

        with open(file1, "r") as f:
            metadata = json.load(f)
            Graph = metadata['0']  
            node_features = Graph["node_features"]
            edge_indices = Graph["edge_indices"]
            edge_features = Graph["edge_features"]
            G1 = nx.Graph()
            for i in range(len(node_features)):
                G1.add_node(i)
            for edge in edge_indices:
                G1.add_edge(edge[0], edge[1])
            # 生成子图特征
            subgraphs = self.get_subgraph(G1, edge_indices, node_features)
            subgraphs_nodes_mask = torch.zeros((self.n_patch, G1.number_of_nodes()), dtype=torch.bool)
            for subgraph_id, subgraph in enumerate(subgraphs):
                subgraphs_nodes_mask[subgraph_id, subgraph['node']] = True
            coarsen_adj = self.cal_coarsen_adj(subgraphs_nodes_mask)
            patch_pe = self.random_walk(coarsen_adj, n_iter=8).to(self.device)
            subgraph_features, batch_index = self.batch_subgraph(subgraphs, node_features, edge_features)
            data['subgraph_features_0'] = subgraph_features
            data['patch_pe_0'] = patch_pe
            data['batch_index_0'] = batch_index
            
            # 保留完整图特征
            whole_graph = {'node_features': [], 'edge_indices': [], 'edge_features': []}
            node_features = torch.tensor([node["embedding"] for node in node_features], dtype=torch.float32).to(self.device).squeeze()
            if self.use_local_count:
                # 获取local count
                local_count_matrix = self.local_count(G1).to(self.device)
                node_features = torch.cat([node_features, local_count_matrix], dim=-1)
            whole_graph['node_features'] = node_features
            edge_indices = torch.tensor(edge_indices).to(self.device).squeeze().transpose(0, 1)
            trans_edge_index = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
            combined_edge_index = torch.cat([edge_indices, trans_edge_index], dim=1)
            whole_graph['edge_indices'] = combined_edge_index.to(self.device).squeeze()
            edge_features = torch.tensor([edge["embedding"] for edge in edge_features], dtype=torch.float32).to(self.device).squeeze()
            total_edge_features = torch.cat([edge_features, edge_features], dim=0)
            whole_graph['edge_features'] = total_edge_features
            data['whole_graph_0'] = whole_graph
            #图1节点数
            data['n1'] = len(node_features)
            
        with open(file2, "r") as f:
            metadata = json.load(f)
            Graph = metadata['0']
            node_features = Graph["node_features"]
            edge_indices = Graph["edge_indices"]
            edge_features = Graph["edge_features"]  
            G2 = nx.Graph()
            for i in range(len(node_features)):
                G2.add_node(i)
            for edge in edge_indices:
                G2.add_edge(edge[0], edge[1])
            # 生成子图特征
            subgraphs = self.get_subgraph(G2, edge_indices, node_features)
            subgraphs_nodes_mask = torch.zeros((self.n_patch, G2.number_of_nodes()), dtype=torch.bool)
            for subgraph_id, subgraph in enumerate(subgraphs):
                subgraphs_nodes_mask[subgraph_id, subgraph['node']] = True
            coarsen_adj = self.cal_coarsen_adj(subgraphs_nodes_mask)
            patch_pe = self.random_walk(coarsen_adj, n_iter=8).to(self.device)
            subgraph_features, batch_index = self.batch_subgraph(subgraphs, node_features, edge_features)
            data['subgraph_features_1'] = subgraph_features
            data['patch_pe_1'] = patch_pe
            data['batch_index_1'] = batch_index
            # 保留完整图特征
            whole_graph = {'node_features': [], 'edge_indices': [], 'edge_features': []}
            node_features = torch.tensor([node["embedding"] for node in node_features], dtype=torch.float32).to(self.device).squeeze()
            if self.use_local_count:
                # 获取local count
                local_count_matrix = self.local_count(G2).to(self.device)
                node_features = torch.cat([node_features, local_count_matrix], dim=-1)
            whole_graph['node_features'] = node_features
            edge_indices = torch.tensor(edge_indices).to(self.device).squeeze().transpose(0, 1)
            trans_edge_index = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
            combined_edge_index = torch.cat([edge_indices, trans_edge_index], dim=1)
            whole_graph['edge_indices'] = combined_edge_index.to(self.device).squeeze()
            edge_features = torch.tensor([edge["embedding"] for edge in edge_features], dtype=torch.float32).to(self.device).squeeze()
            total_edge_features = torch.cat([edge_features, edge_features], dim=0)
            whole_graph['edge_features'] = total_edge_features
            data['whole_graph_1'] = whole_graph
            #图2节点数
            data['n2'] = len(node_features)

        #计算规范化常数
        data['norm'] = torch.tensor(0.25*(len(data['whole_graph_0']["node_features"])+len(data['whole_graph_1']["node_features"])+\
                                                 len(data['whole_graph_0']["edge_features"])+len(data['whole_graph_1']["edge_features"]))).to(self.device)
        #计算GED, 用于训练
        data['gt_ged'] = torch.tensor(ged/data['norm'],dtype=torch.float32).to(self.device)
        #真实GED, 用于评估
        data['real_ged'] = torch.tensor(ged, dtype=torch.float32).to(self.device)
        return data
    
    def cal_coarsen_adj(self, subgraphs_nodes_mask):
        mask = subgraphs_nodes_mask.to(torch.float)
        coarsen_adj = torch.matmul(mask, mask.t())
        return coarsen_adj

    def random_walk(self, A, n_iter):
        Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)  # D^-1
        RW = A * Dinv
        M = RW
        M_power = M
        PE = [torch.diagonal(M)]
        for _ in range(n_iter-1):
            M_power = torch.matmul(M_power, M)
            PE.append(torch.diagonal(M_power))
        PE = torch.stack(PE, dim=-1)
        return PE
    
    def get_subgraph(self, G, edge_indices, node_features):
        if G.number_of_nodes() < self.n_patch:
            membership = []
        else:
            node_features = torch.tensor([node["embedding"] for node in node_features], dtype=torch.float32).to(self.device).squeeze()
            xadj = [0]
            adjncy = []
            eweights = []
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                adjncy.extend(neighbors)
                xadj.append(len(adjncy))
                for neighbor in neighbors:
                    cos_sim = F.cosine_similarity(node_features[node].unsqueeze(0), node_features[neighbor].unsqueeze(0)).item()
                    weight = 1 - cos_sim
                    int_weight = int(weight * 1000)
                    eweights.append(int_weight)
            _, membership = pymetis.part_graph(self.n_patch, xadj=xadj, adjncy=adjncy, eweights=eweights)
            sub = self.k_hop_neighbor(G, membership, edge_indices)
        return sub
    
    def k_hop_neighbor(self, G, membership, edge_indices):
        subgraphs = []
        for subgraph_id in range(self.n_patch):
            subgraph = {'node': [], 'edge': []}
            subgraph_nodes = [i for i, m in enumerate(membership) if m == subgraph_id]
            neighbors = set(subgraph_nodes)
            edges = set()
            # 找到 subgraph_nodes 之间的边
            for i, node in enumerate(subgraph_nodes):
                for j in range(i + 1, len(subgraph_nodes)):
                    neighbor = subgraph_nodes[j]
                    if [node, neighbor] in edge_indices:
                        indexs = [i for i, edge in enumerate(edge_indices) if edge == [node, neighbor]]
                        for index in indexs:
                            edges.add((node, neighbor, index))
            # 找到 1-hop邻居及边
            for node in subgraph_nodes:
                neighbors.update(G.neighbors(node))
                for neighbor in G.neighbors(node):
                    neighbors.add(neighbor)
                    if [node, neighbor] in edge_indices:
                        indexs = [i for i, edge in enumerate(edge_indices) if edge == [node, neighbor]]
                        for index in indexs:
                            edges.add((node, neighbor, index))
                    if [neighbor, node] in edge_indices:
                        indexs = [i for i, edge in enumerate(edge_indices) if edge == [neighbor, node]]
                        for index in indexs:
                            edges.add((neighbor, node, index))
            subgraph['node'] = list(neighbors)
            subgraph['edge'] = list(edges)
            subgraphs.append(subgraph)
        return subgraphs
    
    def batch_subgraph(self, subgraphs, node_features, edge_features):
        start = 0
        index = []
        combined_subgraph = {
            'node_features': [],
            'edge_indices': [],
            'edge_features': []
        }
        final_node_features = []
        final_edge_indices = []
        final_edge_features = []
        for subgraph in subgraphs:
            node_map = {node: idx for idx, node in enumerate(subgraph["node"])}
            for node in subgraph['node']:
                final_node_features.append(node_features[node]['embedding']) 
            for edge in subgraph['edge']:
                final_edge_indices.append([node_map[edge[0]] + start, node_map[edge[1]] + start])
                final_edge_indices.append([node_map[edge[1]] + start, node_map[edge[0]] + start])
                final_edge_features.append(edge_features[edge[2]]['embedding'])
                final_edge_features.append(edge_features[edge[2]]['embedding'])
            index.append([start, start + len(subgraph['node'])])
            start += len(subgraph['node'])
        combined_subgraph['node_features'] = torch.tensor(final_node_features, dtype=torch.float32).to(self.device).squeeze()
        combined_subgraph['edge_indices'] = torch.tensor(final_edge_indices, dtype=torch.long).to(self.device).squeeze().transpose(0, 1)
        combined_subgraph['edge_features'] = torch.tensor(final_edge_features, dtype=torch.float32).to(self.device).squeeze()
        batch_index = torch.zeros(combined_subgraph['node_features'].size(0), dtype=torch.long).to(self.device)
        for i, (start, end) in enumerate(index):
            batch_index[start:end+1] = i
        return combined_subgraph, batch_index
    
    def local_count(self, graph):
        from collections import defaultdict
        #'chain_3_middle': 0, 'chain_3_boundary': 0, 'chain_5_middle': 0, 'chain_5_bridge': 0, 'chain_5_boundary':0,'triangle': 0, 'quadrilateral': 0
        node_roles = defaultdict(lambda: {'chain_3_middle': 0, 'chain_3_boundary': 0, 'chain_5_middle': 0, 'chain_5_bridge': 0, 'chain_5_boundary':0,'triangle': 0, 'quadrilateral': 0})
        # triangle
        for node in graph.nodes():
            neighbors = set(graph.neighbors(node))
            for neighbor in neighbors:
                common_neighbors = neighbors & set(graph.neighbors(neighbor))
                for common_neighbor in common_neighbors:
                    node_roles[node]['triangle'] += 1
                    node_roles[neighbor]['triangle'] += 1
                    node_roles[common_neighbor]['triangle'] += 1
        # chain_3
        for node in graph.nodes():
            for leaf_node in graph.nodes():
                paths = nx.all_simple_paths(graph, source=node, target=leaf_node, cutoff=5)
                for path in paths:
                    if len(path) == 3:
                        node_roles[path[0]]['chain_3_boundary'] += 1
                        node_roles[path[1]]['chain_3_middle'] += 1
                        node_roles[path[2]]['chain_3_boundary'] += 1
                    if len(path) == 5:
                        node_roles[path[0]]['chain_5_boundary'] += 1
                        node_roles[path[1]]['chain_5_bridge'] += 1
                        node_roles[path[2]]['chain_5_middle'] += 1
                        node_roles[path[3]]['chain_5_bridge'] += 1
                        node_roles[path[4]]['chain_5_boundary'] += 1 
        # quadrilateral'
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    common_neighbors = set(graph.neighbors(neighbors[i])) & set(graph.neighbors(neighbors[j]))
                    for common_neighbor in common_neighbors:
                        if common_neighbor != node and not graph.has_edge(node, common_neighbor):
                            node_roles[node]['quadrilateral'] += 1
                            node_roles[neighbors[i]]['quadrilateral'] += 1
                            node_roles[neighbors[j]]['quadrilateral'] += 1
                            node_roles[common_neighbor]['quadrilateral'] += 1
                        
        node_num = graph.number_of_nodes()
        local_count_matrix = torch.zeros((node_num, 7), dtype=torch.float32)
        for node, roles in node_roles.items():
            local_count_matrix[node, 0] = roles['chain_3_middle'] / 2
            local_count_matrix[node, 1] = roles['chain_3_boundary'] / 2
            local_count_matrix[node, 2] = roles['chain_5_middle'] / 3
            local_count_matrix[node, 3] = roles['chain_5_boundary'] / 2
            local_count_matrix[node, 4] = roles['chain_5_bridge'] / 2
            local_count_matrix[node, 5] = roles['triangle'] / 3
            local_count_matrix[node, 6] = roles['quadrilateral'] / 4
        local_count_matrix = F.normalize(local_count_matrix, p=1, dim=-1)
        return local_count_matrix

    def __getitem__(self, index):
        if self.preload:
            return self.data[index]
        else:
            pair = self.data_pairs[index]
            return self.load_file(pair)

class StruturalGEDDataSet:
    def __init__(self, data_dir, pairs_file_path, PreLoad=False, device=None, args=None):
        if device == None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        if args != None:
            self.sample_num = args.sample_rule_nums
        else:
            self.sample_num = 8
        self.use_local_count = args.use_local_count
        self.data_dir = data_dir
        self.data_pairs = self.load_pairs(pairs_file_path) 
        self.preload = PreLoad
        self.data = []
        if PreLoad:
            for pair in self.data_pairs:
                data = self.load_file(pair)
                self.data.append(data)

    def __len__(self):
        return len(self.data_pairs)

    def load_pairs(self, file):
        with open(file, "r") as f:
            data = json.load(f)
        pairs = data['pairs_info']
        return pairs
    
 
    def load_file(self, pair):
        data = {}
        data['graph1'] = pair[0]
        data['graph2'] = pair[1]
        file1, file2, ged = pair[0], pair[1], pair[2]
        file1 = os.path.join(self.data_dir, file1)
        file2 = os.path.join(self.data_dir, file2)

        with open(file1, "r") as f:
            metadata = json.load(f)
            Graph = metadata['0']  
            node_features = Graph["node_features"]
            edge_indices = Graph["edge_indices"]
            edge_features = Graph["edge_features"]
            G1 = nx.DiGraph()
            for i in range(len(node_features)):
                G1.add_node(i)
            for edge in edge_indices:
                G1.add_edge(edge[0], edge[1])
            # 提取规则
            sample_rules = self.sample_strutral_rules(self.sample_num, node_features, edge_features, edge_indices)
            start = 0
            index = []
            combined_rules = {
                'node_features': [],
                'edge_indices': [],
                'edge_features': []
            }
            for rule in sample_rules:
                combined_rules['node_features'].append(torch.tensor(rule['node_features'], dtype=torch.float32).to(self.device).squeeze())
                combined_rules['edge_features'].append(torch.tensor(rule['edge_features'], dtype=torch.float32).to(self.device).squeeze())
                combined_rules['edge_features'].append(torch.tensor(rule['edge_features'], dtype=torch.float32).to(self.device).squeeze())
                # 添加反向边
                dir_edge_indices = [[edge[0] + start, edge[1] + start] for edge in rule['edge_indices']]
                trans_edge_indices = [[edge[1] + start, edge[0] + start] for edge in rule['edge_indices']]
                combined_rules['edge_indices'].append(torch.tensor(dir_edge_indices, dtype=torch.long).to(self.device))
                combined_rules['edge_indices'].append(torch.tensor(trans_edge_indices, dtype=torch.long).to(self.device))
                index.append([start, start + len(rule['node_features'])])
                start += len(rule['node_features'])
             # 将列表中的张量按顺序堆叠在一起
            if len(combined_rules['node_features']) != 0:
                combined_rules['node_features'] = torch.cat(combined_rules['node_features'], dim=0)
                combined_rules['edge_indices'] = torch.cat(combined_rules['edge_indices'], dim=0).transpose(0, 1)
                combined_rules['edge_features'] = torch.cat(combined_rules['edge_features'], dim=0)
                batch_index = torch.zeros(combined_rules['node_features'].size(0), dtype=torch.long).to(self.device)
                for i, (start, end) in enumerate(index):
                    batch_index[start:end+1] = i
            else:
                batch_index = []
            data['sampled_rules_0'] = combined_rules
            data['rules_0_info'] = batch_index
            # 获取local count
            local_count_matrix = self.local_count(G1).to(self.device)
            # 保留完整图特征
            whole_graph = {'node_features': [], 'edge_indices': [], 'edge_features': []}
            node_features = torch.tensor([node["embedding"] for node in node_features], dtype=torch.float32).to(self.device).squeeze()
            if self.use_local_count:
                node_features = torch.cat([node_features, local_count_matrix], dim=-1)
            whole_graph['node_features'] = node_features
            whole_graph['edge_indices'] = torch.tensor(edge_indices).to(self.device).squeeze().transpose(0, 1)
            whole_graph['edge_features'] = torch.tensor([edge["embedding"] for edge in edge_features], dtype=torch.float32).to(self.device).squeeze()
            data['whole_graph_0'] = whole_graph
            #图1节点数
            data['n1'] = len(node_features)
        with open(file2, "r") as f:
            metadata = json.load(f)
            Graph = metadata['0']
            node_features = Graph["node_features"]
            edge_indices = Graph["edge_indices"]
            edge_features = Graph["edge_features"]
            G2 = nx.DiGraph()
            for i in range(len(node_features)):
                G2.add_node(i)
            for edge in edge_indices:
                G2.add_edge(edge[0], edge[1]) 
            # 提取规则
            start = 0
            index = []
            combined_rules = {
                'node_features': [],
                'edge_indices': [],
                'edge_features': []
            }
            sample_rules = self.sample_strutral_rules(self.sample_num, node_features, edge_features, edge_indices)
            for rule in sample_rules:
                combined_rules['node_features'].append(torch.tensor(rule['node_features'], dtype=torch.float32).to(self.device).squeeze())
                combined_rules['edge_features'].append(torch.tensor(rule['edge_features'], dtype=torch.float32).to(self.device).squeeze())
                combined_rules['edge_features'].append(torch.tensor(rule['edge_features'], dtype=torch.float32).to(self.device).squeeze())
                # 添加反向边
                dir_edge_indices = [[edge[0] + start, edge[1] + start] for edge in rule['edge_indices']]
                trans_edge_indices = [[edge[1] + start, edge[0] + start] for edge in rule['edge_indices']]
                combined_rules['edge_indices'].append(torch.tensor(dir_edge_indices, dtype=torch.long).to(self.device))
                combined_rules['edge_indices'].append(torch.tensor(trans_edge_indices, dtype=torch.long).to(self.device))
                index.append([start, start + len(rule['node_features'])])
                start += len(rule['node_features'])
             # 将列表中的张量按顺序堆叠在一起
            if len(combined_rules['node_features']) != 0:
                combined_rules['node_features'] = torch.cat(combined_rules['node_features'], dim=0)
                combined_rules['edge_indices'] = torch.cat(combined_rules['edge_indices'], dim=0).transpose(0, 1)
                combined_rules['edge_features'] = torch.cat(combined_rules['edge_features'], dim=0)
                batch_index = torch.zeros(combined_rules['node_features'].size(0), dtype=torch.long).to(self.device)
                for i, (start, end) in enumerate(index):
                    batch_index[start:end+1] = i
            else:
                batch_index = []
            data['sampled_rules_1'] = combined_rules
            data['rules_1_info'] = batch_index
            # 获取local count
            local_count_matrix = self.local_count(G2).to(self.device)
            # 保留完整图特征
            whole_graph = {'node_features': [], 'edge_indices': [], 'edge_features': []}
            node_features = torch.tensor([node["embedding"] for node in node_features], dtype=torch.float32).to(self.device).squeeze()
            if self.use_local_count:
                node_features = torch.cat([node_features, local_count_matrix], dim=-1)
            whole_graph['node_features'] = node_features
            edge_indices = torch.tensor(edge_indices).to(self.device).squeeze().transpose(0, 1)
            trans_edge_index = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
            combined_edge_index = torch.cat([edge_indices, trans_edge_index], dim=1)
            whole_graph['edge_indices'] = combined_edge_index.to(self.device).squeeze()
            edge_features = torch.tensor([edge["embedding"] for edge in edge_features], dtype=torch.float32).to(self.device).squeeze()
            total_edge_features = torch.cat([edge_features, edge_features], dim=0)
            whole_graph['edge_features'] = total_edge_features
            data['whole_graph_1'] = whole_graph
            #图2节点数
            data['n2'] = len(node_features)
            
        #计算规范化常数
        data['norm'] = torch.tensor(0.25*(len(data['whole_graph_0']["node_features"])+len(data['whole_graph_1']["node_features"])+\
                                                 len(data['whole_graph_0']["edge_features"])+len(data['whole_graph_1']["edge_features"]))).to(self.device)
        #计算GED, 用于训练
        data['gt_ged'] = torch.tensor(ged/data['norm'],dtype=torch.float32).to(self.device)
        #真实GED, 用于评估
        data['real_ged'] = torch.tensor(ged, dtype=torch.float32).to(self.device)
        return data

    def sample_strutral_rules(self, num_samples, node_features, edge_features, edge_indices):
        Final_rules = []
        G = nx.DiGraph()
        for i in range(len(node_features)):
            G.add_node(i)
        for edge in edge_indices:
            G.add_edge(edge[0], edge[1])
        #  Pm2 rules
        Pm2_Final = []
        pm2_rules = set()
        for path in itertools.permutations(G.nodes(), 3):
            if G.has_edge(path[0], path[1]) and G.has_edge(path[1], path[2]):
                pm2_rules.add((path, (edge_indices.index([path[0], path[1]]), edge_indices.index([path[1], path[2]]))))
        for path in pm2_rules:
            rule = {}
            nodes = path[0]
            rule['node_features'] = [node_features[node]['embedding'] for node in nodes]
            rule['edge_indices'] = [[0, 1], [1, 2]]
            index = path[1]
            rule['edge_features'] = [edge_features[index[0]]['embedding'], edge_features[index[1]]['embedding']]
            Pm2_Final.append(rule)
        # Pm3 rules
        Pm3_Final = []
        pm3_rules = set()
        for path in itertools.permutations(G.nodes(), 4):
            if G.has_edge(path[0], path[1]) and G.has_edge(path[1], path[2]) and G.has_edge(path[2], path[3]):
                pm3_rules.add((path, (edge_indices.index([path[0], path[1]]), edge_indices.index([path[1], path[2]]), edge_indices.index([path[2], path[3]]))))
        for path in pm3_rules:
            rule = {}
            nodes = path[0]
            rule['node_features'] = [node_features[node]['embedding'] for node in nodes]
            rule['edge_indices'] = [[0, 1], [1, 2], [2, 3]]
            index = path[1]
            rule['edge_features'] = [edge_features[index[0]]['embedding'], edge_features[index[1]]['embedding'], edge_features[index[2]]['embedding']]
            Pm3_Final.append(rule)
        # trangle rules
        triangle_Final = []
        trangle_rules = set()
        for path in itertools.permutations(G.nodes(), 3):
            if G.has_edge(path[0], path[1]) and G.has_edge(path[1], path[2]) and G.has_edge(path[2], path[0]):
                trangle_rules.add((path, (edge_indices.index([path[0], path[1]]), edge_indices.index([path[1], path[2]]), edge_indices.index([path[2], path[0]]))))
        for path in trangle_rules:
            rule = {}
            nodes = path[0]
            rule['node_features'] = [node_features[node]['embedding'] for node in nodes]
            rule['edge_indices'] = [[0, 1], [1, 2], [2, 0]]
            index = path[1]
            rule['edge_features'] = [edge_features[index[0]]['embedding'], edge_features[index[1]]['embedding'], edge_features[index[2]]['embedding']]
            triangle_Final.append(rule)
        # il2 rules
        il2_Final = []
        il2_rules = set()
        for node in G.nodes():
            neighbors = list(G.predecessors(node))
            for pair in itertools.combinations(neighbors, 2):
                if G.has_edge(pair[0], node) and G.has_edge(pair[1], node):
                    il2_rules.add(((pair[0],pair[1],node), (edge_indices.index([pair[0], node]), edge_indices.index([pair[1], node]))))
        for path in il2_rules:
            rule = {}
            nodes = path[0]
            rule['node_features'] = [node_features[node]['embedding'] for node in nodes]
            rule['edge_indices'] = [[0, 2], [1, 2]]
            index = path[1]
            rule['edge_features'] = [edge_features[index[0]]['embedding'], edge_features[index[1]]['embedding']]
            il2_Final.append(rule)
        # il3 rules
        il3_Final = []
        il3_rules = set()
        for node in G.nodes():
            neighbors = list(G.predecessors(node))
            for triplet in itertools.combinations(neighbors, 3                                                                                                                                                                                                                                                                                                                                                                                                     ):
                if G.has_edge(triplet[0], node) and G.has_edge(triplet[1], node) and G.has_edge(triplet[2], node):
                    il3_rules.add(((triplet[0],triplet[1],triplet[2],node), (edge_indices.index([triplet[0], node]), edge_indices.index([triplet[1], node]), edge_indices.index([triplet[2], node]))))
        for path in il3_rules:
            rule = {}
            nodes = path[0]
            rule['node_features'] = [node_features[node]['embedding'] for node in nodes]
            rule['edge_indices'] = [[0, 3], [1, 3], [2, 3]]
            index = path[1]
            rule['edge_features'] = [edge_features[index[0]]['embedding'], edge_features[index[1]]['embedding'], edge_features[index[2]]['embedding']]
            il3_Final.append(rule)
        # il2 extend
        il2_extend_Final = []
        il2_extend_rules = set()
        for node in G.nodes():
            neighbors = list(G.predecessors(node))
            for pair in itertools.combinations(neighbors, 2):
                if G.has_edge(pair[0], node) and G.has_edge(pair[1], node):
                    for neighbor in G.neighbors(node):
                        if neighbor != pair[0] and neighbor != pair[1]:
                            il2_extend_rules.add(((pair[0],pair[1], node, neighbor), (edge_indices.index([pair[0], node]), edge_indices.index([pair[1], node]), edge_indices.index([node, neighbor]))))
        for path in il2_extend_rules:
            rule = {}
            nodes = path[0]
            rule['node_features'] = [node_features[node]['embedding'] for node in nodes]
            rule['edge_indices'] = [[0, 2], [1, 2], [2, 3]]
            index = path[1]
            rule['edge_features'] = [edge_features[index[0]]['embedding'], edge_features[index[1]]['embedding'], edge_features[index[2]]['embedding']]
            il2_extend_Final.append(rule)
        rules_num = {
            'Pm2': len(Pm2_Final),
            'Pm3': len(Pm3_Final),
            'triangle': len(triangle_Final),
            'il2': len(il2_Final),
            'il3': len(il3_Final),
            'il2_extend': len(il2_extend_Final)
        }
        total_num = sum(rules_num.values())
        if total_num <= num_samples:
            total_rules = Pm2_Final + Pm3_Final + triangle_Final + il2_Final + il3_Final + il2_extend_Final
            random.shuffle(total_rules)
            Final_rules = total_rules
        else:
            ratio = [num / total_num for num in rules_num.values()]
            sample_num = [int(num_samples * r) for r in ratio]
            Final_rules.extend(random.sample(Pm2_Final, sample_num[0]))
            Final_rules.extend(random.sample(Pm3_Final, sample_num[1]))
            Final_rules.extend(random.sample(triangle_Final, sample_num[2]))
            Final_rules.extend(random.sample(il2_Final, sample_num[3]))
            Final_rules.extend(random.sample(il3_Final, sample_num[4]))
            Final_rules.extend(random.sample(il2_extend_Final, sample_num[5]))
        return Final_rules[:num_samples]
    
    def local_count(self, graph):
        from collections import defaultdict
        #'chain_3_middle': 0, 'chain_3_boundary': 0, 'chain_5_middle': 0, 'chain_5_bridge': 0, 'chain_5_boundary':0,'triangle': 0, 'quadrilateral': 0
        node_roles = defaultdict(lambda: {'chain_3_middle': 0, 'chain_3_boundary': 0, 'chain_5_middle': 0, 'chain_5_bridge': 0, 'chain_5_boundary':0,'triangle': 0, 'quadrilateral': 0})
        # triangle
        for node in graph.nodes():
            neighbors = set(graph.neighbors(node))
            for neighbor in neighbors:
                common_neighbors = neighbors & set(graph.neighbors(neighbor))
                for common_neighbor in common_neighbors:
                    node_roles[node]['triangle'] += 1
                    node_roles[neighbor]['triangle'] += 1
                    node_roles[common_neighbor]['triangle'] += 1
        # chain_3
        for node in graph.nodes():
            for leaf_node in graph.nodes():
                paths = nx.all_simple_paths(graph, source=node, target=leaf_node, cutoff=5)
                for path in paths:
                    if len(path) == 3:
                        node_roles[path[0]]['chain_3_boundary'] += 1
                        node_roles[path[1]]['chain_3_middle'] += 1
                        node_roles[path[2]]['chain_3_boundary'] += 1
                    if len(path) == 5:
                        node_roles[path[0]]['chain_5_boundary'] += 1
                        node_roles[path[1]]['chain_5_bridge'] += 1
                        node_roles[path[2]]['chain_5_middle'] += 1
                        node_roles[path[3]]['chain_5_bridge'] += 1
                        node_roles[path[4]]['chain_5_boundary'] += 1 
        # quadrilateral'
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    common_neighbors = set(graph.neighbors(neighbors[i])) & set(graph.neighbors(neighbors[j]))
                    for common_neighbor in common_neighbors:
                        if common_neighbor != node and not graph.has_edge(node, common_neighbor):
                            node_roles[node]['quadrilateral'] += 1
                            node_roles[neighbors[i]]['quadrilateral'] += 1
                            node_roles[neighbors[j]]['quadrilateral'] += 1
                            node_roles[common_neighbor]['quadrilateral'] += 1
                        
        node_num = graph.number_of_nodes()
        local_count_matrix = torch.zeros((node_num, 7), dtype=torch.float32)
        for node, roles in node_roles.items():
            local_count_matrix[node, 0] = roles['chain_3_middle'] / 2
            local_count_matrix[node, 1] = roles['chain_3_boundary'] / 2
            local_count_matrix[node, 2] = roles['chain_5_middle'] / 3
            local_count_matrix[node, 3] = roles['chain_5_boundary'] / 2
            local_count_matrix[node, 4] = roles['chain_5_bridge'] / 2
            local_count_matrix[node, 5] = roles['triangle'] / 3
            local_count_matrix[node, 6] = roles['quadrilateral'] / 4
        local_count_matrix = F.normalize(local_count_matrix, p=1, dim=-1)
        return local_count_matrix

    def __getitem__(self, index):
        if self.preload:
            return self.data[index]
        else:
            pair = self.data_pairs[index]
            return self.load_file(pair)

if __name__ == "__main__":
    dir = "/home/huizhong/GED_Process/NeuralGED/data/newdata/swdf_unlabel/processed_data/train"
    pairs_path = "/home/huizhong/GED_Process/NeuralGED/data/newdata/swdf_unlabel/processed_data/train_GEDINFO.json"
    dataset = MetisGEDDataSet(data_dir=dir, pairs_file_path=pairs_path, PreLoad=False, device="cuda")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for data in data_loader:
        print(data)
        break