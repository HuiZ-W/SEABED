import os
import json
import random
import networkx as nx
import glob
from tqdm import tqdm
from multiprocessing import Pool
def read_all_json_files(folder_path):
    json_data = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为 JSON 文件
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # 读取 JSON 文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                json_data.append(data)
    return json_data

def node_match(n1, n2):
    return n1['label'] == n2['label']

def edge_match(e1, e2):
    return e1['label'] == e2['label']

def cal_ged(file1, file2):
    with open(file1, "r") as f:
        data = json.load(f)
        data1 = data['0']
    with open(file2, "r") as f:
        data = json.load(f)
        data2 = data['0']
    G1 = nx.Graph()
    G2 = nx.Graph()
    for i, node in enumerate(data1['node_features']):
        G1.add_node(i)
    for i, edge_index in enumerate(data1['edge_indices']):
        edge = data1['edge_features'][i]
        G1.add_edge(edge_index[0], edge_index[1], label=edge['id'])
    for i, node in enumerate(data2['node_features']):
        G2.add_node(i)
    for i, edge_index in enumerate(data2['edge_indices']):
        edge = data2['edge_features'][i]
        G2.add_edge(edge_index[0], edge_index[1], label=edge['id'])
    ged = nx.graph_edit_distance(G1, G2, edge_match=edge_match)
    noedge_label_ged = nx.graph_edit_distance(G1, G2)
    file1_name = os.path.basename(file1)
    file2_name = os.path.basename(file2)
    return [file1_name, file2_name, ged, noedge_label_ged]

def mul_cal_ged(index, all_json_data):
    ori_file = all_json_data[index]
    result = {'pairs_info':[],"total Ged":0}
    other_files = random.sample(all_json_data[:index] + all_json_data[index:], dereive_num)
    for file in tqdm(other_files):
        pairs = cal_ged(ori_file, file)
        result['pairs_info'].append(pairs)
        result['total Ged'] += pairs[2]
    return result

if __name__ == "__main__":
    folder_path = '/home/huizhong/GED_Process/NeuralGED/data/newdata/wikidata_unlabel/processed_data'
    sub_dirs = ['train','val','test']
    dereive_num = 50
    for sub_dir in sub_dirs:
        GED_INFO = {'pairs_info':[],"Avg Ged":0}
        Avg_Ged = 0
        all_json_data = glob.glob(os.path.join(folder_path, sub_dir, "*.json"))
        with Pool(20) as pool:
            results = list(tqdm(pool.starmap(mul_cal_ged, [(i, all_json_data) for i in range(len(all_json_data))]), total=len(all_json_data)))
        for result in results:
            GED_INFO['pairs_info'].extend(result['pairs_info'])
            Avg_Ged += result['total Ged']
        Avg_Ged /= (len(all_json_data) * dereive_num)
        GED_INFO["Avg Ged"] = Avg_Ged
        with open(os.path.join(folder_path, f"{sub_dir}_GEDINFO.json"), "w") as f:
            json.dump(GED_INFO, f)
'''
if __name__ == "__main__":
    folder_path = '/home/huizhong/GED_Process/NeuralGED/data/newdata/swdf_samelabel/processed_data'
    pair_path = '/home/huizhong/GED_Process/NeuralGED/data/newdata/swdf_samelabel/processed_data'
    sub_dirs = ['train', 'val', 'test']
    for sub_dir in sub_dirs:
        GED_INFO = {'pairs_info':[],"Avg Ged":0}
        pair_file = os.path.join(pair_path,f"{sub_dir}_GEDINFO.json")
        with open(pair_file, 'r') as f:
            data = json.load(f)
            data = data['pairs_info']
        for pair in data:
            file1, file2 = pair[0], pair[1]
            file1s = os.path.join(folder_path,sub_dir,file1)
            file2s = os.path.join(folder_path,sub_dir,file2)
            with open(file1s, 'r') as f:
                graph1 = json.load(f)
                graph1 = graph1['0']
            with open(file2s, 'r') as f:
                graph2 = json.load(f)
                graph2 = graph2['0']
            label1 = graph1['node_features']
            label2 = graph2['node_features']
            edge1 = graph1['edge_features']
            edge2 = graph2['edge_features']
            ged = len(label2) - len(label1) + len(edge2) - len(edge1)
            pairs = [file1, file2, ged]
            GED_INFO['pairs_info'].append(pairs)
            GED_INFO['Avg Ged'] += ged
        GED_INFO['Avg Ged'] /= len(data)
'''


# 