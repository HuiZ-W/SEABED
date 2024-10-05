import os
import glob
import json

data_info = {"avg_node_num": 0, "avg_edge_num": 0, "max_node": 0, "max_edge": 0, "avg_ged": 0, 'total_graph_num': 0}
data_dir = "/home/huizhong/GED_Process/NeuralGED/data/wikidata_small/processed_data"
sub_dirs = ['train','val','test']
files = []

files += glob.glob(os.path.join(data_dir, sub_dirs[0], "*.json"))
files += glob.glob(os.path.join(data_dir, sub_dirs[1], "*.json"))
files += glob.glob(os.path.join(data_dir, sub_dirs[2], "*.json"))

#files = glob.glob(os.path.join(data_dir, "*.json"))
for file in files:
    with open(file, "r") as f:
        data = json.load(f)
    graphs = [data['0'], data['1']]
    ged = data['GED']
    
    for graph in graphs:
        nodes = len(graph['node_features'])
        edges = len(graph['edge_features'])
        data_info['avg_node_num'] += nodes
        data_info["avg_edge_num"] += edges
        if nodes > data_info["max_node"]:
            data_info['max_node'] = nodes
        if edges > data_info['max_edge']:
            data_info['max_edge'] = edges
    '''
    nodes = len(graphs[0]['node_features'])
    edges = len(graphs[0]['edge_features'])
    data_info['avg_node_num'] += nodes
    data_info["avg_edge_num"] += edges
    if nodes > data_info["max_node"]:
        data_info['max_node'] = nodes
    if edges > data_info['max_edge']:
        data_info['max_edge'] = edges
    '''
    data_info['avg_ged'] += ged
data_info["avg_edge_num"] /= (len(files) *2)
data_info["avg_node_num"] /= (len(files) *2)
data_info["avg_ged"] /= len(files)
data_info['total_graph_num'] = len(files)
with open(os.path.join(os.path.dirname(data_dir), "dataset_info.json"), "w") as f:
    json.dump(data_info, f)
