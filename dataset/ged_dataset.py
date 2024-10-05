from torch.utils.data import DataLoader
import os
import glob
import json
import torch
import random
import time
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
            data["sampled_rules_0"], index1 = self.rule_sample(self.sample_num, self.rule_length, len(node_features), edge_features, edge_indices)
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
            data["sampled_rules_1"], index2 = self.rule_sample(self.sample_num, self.rule_length, len(node_features), edge_features, edge_indices)
            data['n2'] = len(node_features)
            
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
        data['time_use'] = time_end - time_start
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
    
if __name__ == "__main__":
    path = "/home/huizhong/GED_Process/NeuralGED/data/swdf/processed_data/train"
    dataset = GEDDataSet(path, PreLoad=False, device="cuda")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for data in data_loader:
        print(data)
        break