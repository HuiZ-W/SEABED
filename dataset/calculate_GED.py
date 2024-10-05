import json
import os
import subprocess
import random
import time
import glob
import re
from tqdm import tqdm

def create_tmpdata(file_name, other_file):
    with open(file_name, "r") as f:
        data = json.load(f)
        graph1 = data['0']
    with open(other_file, "r") as f:
        data = json.load(f)
        graph2 = data['0']
    new_file_name = []
    nodes_label = []
    edges_label = []
    node_features1 = graph1['node_features']
    node_features2 = graph2['node_features']
    edge_indices1 = graph1['edge_indices']
    edge_indices2 = graph2['edge_indices']
    edge_features1 = graph1['edge_features']
    edge_features2 = graph2['edge_features']
    for edge in edge_features1:
            if edge["id"] not in edges_label:
                edges_label.append(edge["id"])
    for edge in edge_features2:
            if edge["id"] not in edges_label:
                edges_label.append(edge["id"])
    tmp_name1 = file_name.replace(".json" , "_1") + ".txt"
    tmp_name2 = other_file.replace(".json" , "_2") + ".txt"
    with open(tmp_name1, 'w') as f:
        f.write("t # 1\n")
        for j, node in enumerate(node_features1):
            id = 1
            f.write("v " + str(j) + " " + str(id)+"\n")
        for j, edge in enumerate(edge_indices1):
            id = edges_label.index(edge_features1[j]["id"])
            f.write("e " + str(edge[0]) + " " + str(edge[1]) + " " + str(id) + "\n")
        new_file_name.append(tmp_name1)
    with open(tmp_name2, 'w') as f:
        f.write("t # 1\n")
        for j, node in enumerate(node_features2):
            id = 1
            f.write("v " + str(j) + " " + str(id)+"\n")
        for j, edge in enumerate(edge_indices2):
            id = edges_label.index(edge_features2[j]["id"])
            f.write("e " + str(edge[0]) + " " + str(edge[1]) + " " + str(id) + "\n")
        new_file_name.append(tmp_name2)
    return new_file_name


def delete_tmpdata(file_names):
    for file in file_names:
        os.remove(file)

if __name__ == "__main__":
    data_dir = "/home/GED_Process/NeuralGED/data/newdata/wikidata_unlabel"
    data_path = os.path.join(data_dir, "processed_data")
    sub_dirs = ["train", "val", "test"]
    dereive_num = 50
    for sub_dir in sub_dirs:
        GED_INFO = {'pairs_info':[],"Avg GED":0}
        all_json_data = glob.glob(os.path.join(data_path, sub_dir, "*.json"))
        for index, file1 in tqdm(enumerate(all_json_data)):
            other_files = random.sample(all_json_data[:index] + all_json_data[index:], dereive_num)
            for file in other_files:
                file_name = os.path.join(data_dir, sub_dir, file1)
                file1_name = os.path.join(data_dir, sub_dir, file)
                tmp_files = create_tmpdata(file_name, file1_name) 
                tstart = time.time()
                command = " /home/huizhong/Graph_Edit_Distance/ged -d {file1} -q {file2} -m pair -p astar -l LSa -g".format(file1=tmp_files[0], file2=tmp_files[1])
                result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
                tend = time.time()
                process_result = result.stdout.decode("utf-8")
                ged_result = re.search(r"\*\*\* GEDs \*\*\*\n(\d+)", process_result)
                if ged_result:
                    ged_result = ged_result.group(1)
                else:
                    ged_result = "Error"
                delete_tmpdata(tmp_files)
                if ged_result:
                    GED_INFO['pairs_info'].append([file, file1, ged_result])
                    GED_INFO['Avg GED'] += ged_result
        GED_INFO_FILE = os.path.join(data_dir,f"{sub_dir}_GEDINFO.json")
        with open(GED_INFO_FILE, "w") as f:
            json.dump(GED_INFO, f)
