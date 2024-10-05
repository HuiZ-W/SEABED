import os
import random
import json
from tqdm import tqdm
import rdflib
import glob
import argparse
import shutil
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
import requests
import copy
NEW_TRIPLE = -1
NEW_SUBJECT = 0
NEW_OBJECT = 1
NEW_PREDICTS = 2
params = {}

def add_triple_to_graph_data(triple, nodes, graph_data, mode=NEW_TRIPLE):
    graph_data["KG"].append([str(triple[0]), str(triple[1]), str(triple[2])])
    sIdx = nodes.index(str(triple[0]))
    oIdx = nodes.index(str(triple[2]))
    if mode == NEW_TRIPLE:
        graph_data["node_features"].extend([
            {"id": str(triple[0]), "embedding": "null"},
            {"id": str(triple[2]), "embedding": "null"}
        ])
    if mode == NEW_SUBJECT:
        graph_data["node_features"].append({"id": str(triple[0]), "embedding": "null"})
    elif mode == NEW_OBJECT:
        graph_data["node_features"].append({"id": str(triple[2]), "embedding": "null"})
    graph_data["edge_indices"].append([sIdx, oIdx])
    graph_data["edge_features"].append({"id": str(triple[1]), "embedding": "null"})
    return

def generate_subgraphs(KG_path, des_dir, args):
    KG = rdflib.Graph()
    KG.parse(KG_path, format="nt")
    all_triple = list(KG)
    data_info = {"avg_node_num": 0, "avg_edge_num": 0, "max_node": 0, "max_edge": 0}
    shuffle = True
    only_add_edges = 0.5
    #start generating subgraphs
    for i in tqdm(range(args.data_num)):
        json_data = {}
        #chosen a triple as the first triple
        first_triple = random.choice(all_triple)
        graph_data = {"KG": [], "node_features": [], "edge_indices": [], "edge_features": []}
        num_nodes = random.randint(args.min_nodes, args.max_nodes)
        chosen_nodes = []
        chosen_edges = []
        chosen_nodes.append(str(first_triple[0]))
        chosen_nodes.append(str(first_triple[2]))
        chosen_edges.append(str(first_triple[1]))
        add_triple_to_graph_data(first_triple, chosen_nodes, graph_data)
        #add more triples to the subgraph
        while len(chosen_nodes) < num_nodes:
            if len(chosen_nodes) >= num_nodes//2 and random.random() < only_add_edges:
                next_n = random.choice(chosen_nodes)
                next_o = random.choice(chosen_nodes)
                query = """
                        SELECT ?s ?p ?o WHERE {{
                            {{ <{n}> ?p <{o}> . BIND(<{n}> AS ?s) BIND(<{o}> AS ?o) }} 
                            UNION 
                            {{ <{o}> ?p <{n}> . BIND(<{n}> AS ?o) BIND(<{o}> AS ?s) }}
                        }}
                        """.format(n=next_n, o=next_o)
            else:
                next_n = random.choice(chosen_nodes)
                query = """
                        SELECT ?s ?p ?o WHERE {{
                            {{ <{n}> ?p ?o . BIND(<{n}> AS ?s) }} UNION {{ ?s ?p <{n}> . BIND(<{n}> AS ?o) }}
                        }}
                        """.format(n=next_n)
            possible_triple = KG.query(query)
            possible_triple = list(possible_triple)
            if shuffle:
                random.shuffle(possible_triple)
            for row in possible_triple:
                s = str(row[0])
                p = str(row[1])
                o = str(row[2])
                if graph_data["KG"].count([s, p, o]) > 0:
                    continue
                else:
                    chosen_edges.append(p)
                    if s not in chosen_nodes and o in chosen_nodes:
                        chosen_nodes.append(s)
                        add_triple_to_graph_data(row, chosen_nodes, graph_data, NEW_SUBJECT)
                    elif o not in chosen_nodes and s in chosen_nodes:
                        chosen_nodes.append(o)
                        add_triple_to_graph_data(row, chosen_nodes, graph_data, NEW_OBJECT)
                    else:
                        add_triple_to_graph_data(row, chosen_nodes, graph_data, NEW_PREDICTS)
                    break
        json_data[0] = graph_data
        #store the subgraph information
        data_info["avg_node_num"] += len(chosen_nodes)
        data_info["avg_edge_num"] += len(chosen_edges)
        if len(chosen_nodes) > data_info["max_node"]:
            data_info["max_node"] = len(chosen_nodes)
        if len(chosen_edges) > data_info["max_edge"]:
            data_info["max_edge"] = len(chosen_edges)
        file_name = os.path.join(des_dir, "graphPair_{}.json".format(i))
        with open(file_name, "w") as f:
            json.dump(json_data, f) 
    data_info["avg_node_num"] /= args.data_num
    data_info["avg_edge_num"] /= args.data_num
    file_name = os.path.join(os.path.dirname(des_dir), "dataset_info.json")
    with open(file_name, "w") as f:
        json.dump(data_info, f)
    return

def modify_subgraphs1(KG_path, des_dir, source_dir, args):
    add_nodes_num = [2, 3, 4, 5]
    sub_dirs = ['train', 'val', 'test']
    KG = rdflib.Graph()
    KG.parse(KG_path, format="nt")
    for sub_dir in sub_dirs:
        source = os.path.join(source_dir, sub_dir)
        GED_INFO = {'pari_info':[], 'Avg Ged': 0}
        avg_ged = 0
        files = glob.glob(os.path.join(source,'*.json'))
        for file in files:
            with open(file, "r") as f:
                data = json.load(f)
            data = data['0']
            file_name = os.path.basename(file)
            filebase_name = os.path.splitext(os.path.basename(file))[0]
            for i in range(args.derive_data_num):
                graph_data = {"KG": copy.deepcopy(data['KG']), 
                            "node_features": copy.deepcopy(data['node_features']), 
                            "edge_indices": copy.deepcopy(data['edge_indices']), 
                            "edge_features": copy.deepcopy(data["edge_features"])}
                json_data = {}
                chosen_nodes = [node["id"] for node in graph_data["node_features"]]
                chosen_edges = [edge["id"] for edge in graph_data["edge_features"]]
                only_add_edges = 0.3
                shuffle = True
                #edit subgraph via add nodes and attacted edges
                add_num = random.choice(add_nodes_num)
                ori_len = len(chosen_nodes) + len(chosen_edges)
                final_node_num = len(chosen_nodes) + add_num
                while len(chosen_nodes) < final_node_num:
                    if random.random() < only_add_edges:
                            next_n = random.choice(chosen_nodes)
                            next_o = random.choice(chosen_nodes)
                            query = """
                                    SELECT ?s ?p ?o WHERE {{
                                        {{ <{n}> ?p <{o}> . BIND(<{n}> AS ?s) BIND(<{o}> AS ?o) }} 
                                        UNION 
                                        {{ <{o}> ?p <{n}> . BIND(<{n}> AS ?o) BIND(<{o}> AS ?s) }}
                                    }}
                                    """.format(n=next_n, o=next_o)
                    else:
                        next_n = random.choice(chosen_nodes)
                        query = """
                                SELECT ?s ?p ?o WHERE {{
                                    {{ <{n}> ?p ?o . BIND(<{n}> AS ?s) }} UNION {{ ?s ?p <{n}> . BIND(<{n}> AS ?o) }}
                                }}
                                """.format(n=next_n)
                    possible_triple = KG.query(query)
                    possible_triple = list(possible_triple)
                    if shuffle:
                        random.shuffle(possible_triple)
                    for row in possible_triple:
                        s = str(row[0])
                        p = str(row[1])
                        o = str(row[2])
                        if graph_data["KG"].count([s, p, o]) > 0:
                            continue
                        else:
                            chosen_edges.append(p)
                            if s not in chosen_nodes and o in chosen_nodes:
                                chosen_nodes.append(s)
                                add_triple_to_graph_data(row, chosen_nodes, graph_data, NEW_SUBJECT)
                            elif o not in chosen_nodes and s in chosen_nodes:
                                chosen_nodes.append(o)
                                add_triple_to_graph_data(row, chosen_nodes, graph_data, NEW_OBJECT)
                            else:
                                add_triple_to_graph_data(row, chosen_nodes, graph_data, NEW_PREDICTS)
                            break
                json_data[0] = graph_data
                newfilename = f"{filebase_name}_{i}.json"
                ged = len(chosen_nodes) + len(chosen_edges) - ori_len
                avg_ged += ged
                GED_INFO['pari_info'].append([file_name, newfilename, ged])
                with open(os.path.join(source, newfilename), "w") as f:
                    json.dump(json_data, f)
        GED_INFO["Avg Ged"] = avg_ged / (len(files) * args.derive_data_num)
        GED_FILE_PATH = os.path.join(source_dir,f'{sub_dir}_GEDINFO.json')
        with open(GED_FILE_PATH, "w") as f:
            json.dump(GED_INFO, f)
    return

def modify_subgraphs(KG_path, source_dir, args):
    edit_nums = [3, 4, 5, 6, 7, 8, 9, 10]
    #sub_dirs = ['train', 'val', 'test']
    sub_dirs = ['mytest']
    KG = rdflib.Graph()
    KG.parse(KG_path, format="nt")
    for sub_dir in tqdm(sub_dirs):
        source = os.path.join(source_dir, sub_dir)
        GED_INFO = {'pairs_info':[], 'Avg Ged': 0}
        avg_ged = 0
        files = glob.glob(os.path.join(source,'*.json'))
        for file in tqdm(files):
            with open(file, "r") as f:
                data = json.load(f)
            data = data['0']
            file_name = os.path.basename(file)
            filebase_name = os.path.splitext(os.path.basename(file))[0]
            for i in tqdm(range(args.derive_data_num)):
                graph_data = {"KG": copy.deepcopy(data['KG']), 
                            "node_features": copy.deepcopy(data['node_features']), 
                            "edge_indices": copy.deepcopy(data['edge_indices']), 
                            "edge_features": copy.deepcopy(data["edge_features"])}
                json_data = {}
                chosen_nodes = [node["id"] for node in graph_data["node_features"]]
                chosen_edges = [edge["id"] for edge in graph_data["edge_features"]]
                only_add_edges = 0.3
                shuffle = True
                #edit subgraph via add nodes and attacted edges
                edit_num = random.choice(edit_nums)
                edit = 0
                while edit < edit_num:
                    if random.random() < only_add_edges:
                            next_n = random.choice(chosen_nodes)
                            next_o = random.choice(chosen_nodes)
                            query = """
                                    SELECT ?s ?p ?o WHERE {{
                                        {{ <{n}> ?p <{o}> . BIND(<{n}> AS ?s) BIND(<{o}> AS ?o) }} 
                                        UNION 
                                        {{ <{o}> ?p <{n}> . BIND(<{n}> AS ?o) BIND(<{o}> AS ?s) }}
                                    }}
                                    """.format(n=next_n, o=next_o)
                    else:
                        next_n = random.choice(chosen_nodes)
                        query = """
                                SELECT ?s ?p ?o WHERE {{
                                    {{ <{n}> ?p ?o . BIND(<{n}> AS ?s) }} UNION {{ ?s ?p <{n}> . BIND(<{n}> AS ?o) }}
                                }}
                                """.format(n=next_n)
                    possible_triple = KG.query(query)
                    possible_triple = list(possible_triple)
                    if shuffle:
                        random.shuffle(possible_triple)
                    for row in possible_triple:
                        s = str(row[0])
                        p = str(row[1])
                        o = str(row[2])
                        if graph_data["KG"].count([s, p, o]) > 0:
                            continue
                        else:
                            chosen_edges.append(p)
                            if s not in chosen_nodes and o in chosen_nodes:
                                chosen_nodes.append(s)
                                edit += 2
                                add_triple_to_graph_data(row, chosen_nodes, graph_data, NEW_SUBJECT)
                            elif o not in chosen_nodes and s in chosen_nodes:
                                chosen_nodes.append(o)
                                edit += 2
                                add_triple_to_graph_data(row, chosen_nodes, graph_data, NEW_OBJECT)
                            else:
                                edit += 1
                                add_triple_to_graph_data(row, chosen_nodes, graph_data, NEW_PREDICTS)
                            break
                json_data[0] = graph_data
                newfilename = f"{filebase_name}_{i}.json"
                ged = edit_num
                avg_ged += ged
                GED_INFO['pairs_info'].append([file_name, newfilename, ged])
                with open(os.path.join(source, newfilename), "w") as f:
                    json.dump(json_data, f)
        GED_INFO["Avg Ged"] = avg_ged / (len(files) * args.derive_data_num)
        GED_FILE_PATH = os.path.join(source_dir,f'{sub_dir}_GEDINFO.json')
        with open(GED_FILE_PATH, "w") as f:
            json.dump(GED_INFO, f)
    return

def process_data(statistic_file, raw_dir):
    with open(statistic_file, "r") as f:
        statistics = json.load(f)
    sub_dirs = ['train', 'val', 'test']
    for sub_dir in sub_dirs:
        raw_files = glob.glob(os.path.join(raw_dir, sub_dir, "*.json"))
        for raw_file in tqdm(raw_files):
            with open(raw_file, "r") as f:
                data = json.load(f)
            if len(data) != 1:
                continue
            Graph = data['0']
            node_features = Graph["node_features"]
            edge_features = Graph["edge_features"]
            for node in node_features:
                id = node["id"]
                node["embedding"] = statistics[id]["embedding"]
            for edge in edge_features:
                id = edge["id"]
                edge["embedding"] = statistics[id]["embedding"]
            with open(raw_file, "w") as f:
                json.dump(data, f)
    return

def generate_subgraph(param):
    index = param[0]
    args = param[1]
    des_dir = param[2]
    total_triple_num = param[3]
    sparql_endpoint = param[4]
    url = f"{sparql_endpoint}"
    shuffle = True
    only_add_edges = 0.5
    data_info = {'node':0, "edge":0,"ged":0}
    #start generating subgraphs
    json_data = {}
    #chosen a triple as the first triple
    first_triple_index = random.randint(0, total_triple_num)
    query = f"""
            SELECT ?s ?p ?o WHERE {{
                ?s ?p ?o
            }} 
            OFFSET {first_triple_index}
            LIMIT {first_triple_index+1}
            """
    params["sparql"] = query
    with requests.get(url, params=params) as res:
        res = res.json()
        result = res['results']['bindings'][0]
    first_triple = [result['s']['value'], result['p']['value'], result['o']['value']]
    graph_data = {"KG": [], "node_features": [], "edge_indices": [], "edge_features": []}
    num_nodes = random.randint(args.min_nodes, args.max_nodes)
    chosen_nodes = []
    chosen_edges = []
    chosen_nodes.append(str(first_triple[0]))
    chosen_nodes.append(str(first_triple[2]))
    chosen_edges.append(str(first_triple[1]))
    add_triple_to_graph_data(first_triple, chosen_nodes, graph_data)
    #add more triples to the subgraph
    while len(chosen_nodes) < num_nodes:
        if len(chosen_nodes) >= num_nodes//2 and random.random() < only_add_edges:
            next_n = random.choice(chosen_nodes)
            next_o = random.choice(chosen_nodes)
            query = """
                    SELECT ?s ?p ?o WHERE {{
                        {{ <{n}> ?p <{o}> . BIND(<{n}> AS ?s) BIND(<{o}> AS ?o) }} 
                        UNION 
                        {{ <{o}> ?p <{n}> . BIND(<{n}> AS ?o) BIND(<{o}> AS ?s) }}
                    }}
                    LIMIT 1000
                    """.format(n=next_n, o=next_o)
        else:
            next_n = random.choice(chosen_nodes)
            query = """
                    SELECT ?s ?p ?o WHERE {{
                        {{ <{n}> ?p ?o . BIND(<{n}> AS ?s) }} UNION {{ ?s ?p <{n}> . BIND(<{n}> AS ?o) }}
                    }}
                    LIMIT 1000
                    """.format(n=next_n)
        params['sparql'] = query
        with requests.get(url, params=params) as res:
            res = res.json()
        possible_triple = res['results']['bindings']
        possible_triple = list(possible_triple)
        if shuffle:
            random.shuffle(possible_triple)
        for row in possible_triple:
            s = str(row['s']['value'])
            p = str(row['p']['value'])
            o = str(row['o']['value'])
            triple = [s, p, o]
            if graph_data["KG"].count([s, p, o]) > 0:
                continue
            else:
                chosen_edges.append(p)
                if s not in chosen_nodes and o in chosen_nodes:
                    chosen_nodes.append(s)
                    add_triple_to_graph_data(triple, chosen_nodes, graph_data, NEW_SUBJECT)
                elif o not in chosen_nodes and s in chosen_nodes:
                    chosen_nodes.append(o)
                    add_triple_to_graph_data(triple, chosen_nodes, graph_data, NEW_OBJECT)
                else:
                    add_triple_to_graph_data(triple, chosen_nodes, graph_data, NEW_PREDICTS)
                break
    json_data[0] = graph_data
    #store the subgraph information
    data_info["node"] = len(chosen_nodes)
    data_info["edge"] = len(chosen_edges)
    file_name = os.path.join(des_dir, "graphPair_{}.json".format(index))
    with open(file_name, "w") as f:
        json.dump(json_data, f)
    return data_info

def modify_subgraph(param):
    file_name = param[0]
    args = param[1]
    raw_dir = param[2]
    sub_dir = param[3]
    sparql_endpoint = param[4]
    url = f"{sparql_endpoint}"
    shuffle = True
    only_add_edges = 0.3
    edit_nums = [5,6,7,8,9,10,11,12,13,14,15]
    data_info = []

    with open(file_name, "r") as f:
        data = json.load(f)
    data = data['0']
    file_name = os.path.basename(file_name)
    filebase_name = os.path.splitext(os.path.basename(file_name))[0]
    for i in tqdm(range(args.derive_data_num)):
        graph_data = {"KG": copy.deepcopy(data['KG']), 
                    "node_features": copy.deepcopy(data['node_features']), 
                    "edge_indices": copy.deepcopy(data['edge_indices']), 
                    "edge_features": copy.deepcopy(data["edge_features"])}
        json_data = {}
        chosen_nodes = [node["id"] for node in graph_data["node_features"]]
        chosen_edges = [edge["id"] for edge in graph_data["edge_features"]]
        #edit subgraph via add nodes and attacted edges
        edit_num = random.choice(edit_nums)
        edit = 0
        while edit < edit_num:
            if random.random() < only_add_edges:
                next_n = random.choice(chosen_nodes)
                next_o = random.choice(chosen_nodes)
                query = """
                        SELECT ?s ?p ?o WHERE {{
                            {{ <{n}> ?p <{o}> . BIND(<{n}> AS ?s) BIND(<{o}> AS ?o) }} 
                            UNION 
                            {{ <{o}> ?p <{n}> . BIND(<{n}> AS ?o) BIND(<{o}> AS ?s) }}
                        }}
                        LIMIT 1000
                        """.format(n=next_n, o=next_o)
            else:
                next_n = random.choice(chosen_nodes)
                query = """
                        SELECT ?s ?p ?o WHERE {{
                            {{ <{n}> ?p ?o . BIND(<{n}> AS ?s) }} UNION {{ ?s ?p <{n}> . BIND(<{n}> AS ?o) }}
                        }}
                        LIMIT 1000
                        """.format(n=next_n)
            params['sparql'] = query
            with requests.get(url, params=params) as res:
                res = res.json()
            possible_triple = res['results']['bindings']
            possible_triple = list(possible_triple)
            if shuffle:
                random.shuffle(possible_triple)
            for row in possible_triple:
                s = str(row['s']['value'])
                p = str(row['p']['value'])
                o = str(row['o']['value'])
                triple = [s, p, o]
                if graph_data["KG"].count([s, p, o]) > 0:
                    continue
                else:
                    chosen_edges.append(p)
                    if s not in chosen_nodes and o in chosen_nodes:
                        chosen_nodes.append(s)
                        add_triple_to_graph_data(triple, chosen_nodes, graph_data, NEW_SUBJECT)
                        edit +=2
                    elif o not in chosen_nodes and s in chosen_nodes:
                        chosen_nodes.append(o)
                        add_triple_to_graph_data(triple, chosen_nodes, graph_data, NEW_OBJECT)
                        edit +=2
                    else:
                        add_triple_to_graph_data(triple, chosen_nodes, graph_data, NEW_PREDICTS)
                        edit +=1
                    break
        json_data[0] = graph_data
        newfilename = f"{filebase_name}_{i}.json"
        ged = edit
        with open(os.path.join(raw_dir, sub_dir, newfilename), "w") as f:
            json.dump(json_data, f)
        data_info.append([filebase_name, newfilename, ged])
    return data_info


if __name__ == "__main__":
    
    dataset = 'lubm'
    dataset_name = f'{dataset}_unlabel'
    parser = argparse.ArgumentParser(description="Generate subgraphs from a Knowledge Graph")
    parser.add_argument("--max_nodes", type=int, default=25, help="Maximum number of nodes in a subgraph")
    parser.add_argument("--min_nodes", type=int, default=15, help="Minimum number of nodes in a subgraph")
    parser.add_argument("--data_num", type=int, default=1000, help="Number of base graphs to generate")
    parser.add_argument("--derive_data_num", type=int, default=30, help="Number of dereive graphs of per graph")
    args = parser.parse_args()

    remote = False
    Trans_embedding = True
    GenDeriveGraph = False
    GenBaseGraph = False
    params = {
    "username": "root",
    "password": "123456",
    "operation": "query",
    "db_name": dataset,
    "format": "json"
    }
    KG_path = f"/home/GED_Process/NeuralGED/data/newdata/{dataset}/{dataset}.nt"
    embedding_path = f"/home/GED_Process/NeuralGED/data/newdata/{dataset}/statistic.json"
    raw_dir = f"/home/GED_Process/NeuralGED/data/newdata/{dataset_name}/processed_data"
    #raw_dir = f"/home/GED_Process/NeuralGED/data/previous_data/swdf_small/raw_data"
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if remote == False:
        if GenBaseGraph:
            print(f"Generating base subgraph from KG {dataset}")
            generate_subgraphs(KG_path, raw_dir, args)
            print("Finished generating subgraph")
        if GenDeriveGraph:
            print(f"Generating derive subgraph from KG {dataset}")
            modify_subgraphs(KG_path, raw_dir, args)
            print("Finished generating subgraph")
        if Trans_embedding:
            print(f"Processing data for {dataset}")
            process_data(embedding_path, raw_dir=raw_dir)
            print("Finished processing data")
    else:
        if GenBaseGraph:
            print(f"Generating Base subgraph from KG {dataset}")
            data_info = {"avg_node_num": 0, "avg_edge_num": 0, "max_node": 0, "max_edge": 0}
            results = []
            url = "http://127.0.0.1:9002/grpc/api/"
            query = """
                SELECT (COUNT(*) as ?count) WHERE {
                    ?s ?p ?o
                }
                """
            params["sparql"] = query
            with requests.get(url, params=params) as res:
                res = res.json()
            total_triple_num = int(res['results']['bindings'][0]['count']['value'])
            with Pool(processes=10) as pool:
                for result in tqdm(pool.imap_unordered(
                    generate_subgraph, 
                    [[i, args, raw_dir, total_triple_num, "http://127.0.0.1:9002/grpc/api/"] for i in range(args.data_num)],
            ), total = args.data_num):
                    results.append(result)
            for res in results:
                data_info["avg_node_num"] += res['node']
                data_info['avg_edge_num'] += res['edge']
                if res['node'] > data_info["max_node"]:
                    data_info["max_node"] = res["node"]
                if res['edge'] > data_info["max_edge"]:
                    data_info["max_edge"] = res["edge"]
            data_info["avg_node_num"] /= args.data_num
            data_info["avg_edge_num"] /= args.data_num
            data_info['total_graph_num'] = args.data_num
            file_name = os.path.join(os.path.dirname(raw_dir), "dataset_info.json")
            with open(file_name, "w") as f:
                json.dump(data_info, f)
            print("Finished generating subgraph")
        
        if GenDeriveGraph:
            print(f"Generating derive subgraph from KG {dataset}")
            sub_dirs = ['val', 'test']
            for sub_dir in sub_dirs:
                files = glob.glob(os.path.join(raw_dir, sub_dir, "*.json"))
                results = {'pairs_info':[], 'Avg Ged': 0}
                avg_ged = 0
                with Pool(processes=10) as pool:
                    for result in tqdm(pool.imap_unordered(
                        modify_subgraph, 
                        [[file, args, raw_dir, sub_dir, "http://127.0.0.1:9002/grpc/api/"] for file in files],
                ), total = len(files)):
                        for pair in result:
                            results['pairs_info'].append(pair)
                            avg_ged += int(pair[2])
                results["Avg Ged"] = avg_ged / (len(files) * args.derive_data_num)
                GED_FILE_PATH = os.path.join(raw_dir,f'{sub_dir}_GEDINFO.json')
                with open(GED_FILE_PATH, "w") as f:
                    json.dump(results, f)
            print("Finished generating subgraph")
        if Trans_embedding:
            print(f"Processing data for {dataset}")
            process_data(embedding_path, raw_dir=raw_dir)
            print("Finished processing data")
