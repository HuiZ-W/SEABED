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
    data_info = {"avg_node_num": 0, "avg_edge_num": 0, "max_node": 0, "max_edge": 0, "avg_ged": 0}
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
        #edit subgraph via del nodes and attacted edges
        del_percent = random.uniform(args.edit_percent_min, args.edit_percent_max)
        del_nodes_num = int(len(chosen_nodes)*del_percent)
        del_nodes = random.sample(chosen_nodes, del_nodes_num)
        new_graph_data = {"KG": [], "node_features": [], "edge_indices": [], "edge_features": []}
        new_chosen_nodes = []
        new_chosen_edges = []
        chosen_triples = graph_data["KG"].copy() 
        for triple in chosen_triples:
            if triple[0] in del_nodes or triple[2] in del_nodes:
                continue
            else:
                new_chosen_edges.append(triple[1])
                if triple[0] in new_chosen_nodes and triple[2] in new_chosen_nodes:
                    add_triple_to_graph_data(triple, new_chosen_nodes, new_graph_data, NEW_PREDICTS)
                elif triple[0] not in new_chosen_nodes and triple[2] not in new_chosen_nodes:
                    new_chosen_nodes.append(triple[0])
                    new_chosen_nodes.append(triple[2])
                    add_triple_to_graph_data(triple, new_chosen_nodes, new_graph_data, NEW_TRIPLE)
                elif triple[2] not in new_chosen_nodes:
                    new_chosen_nodes.append(triple[2])
                    add_triple_to_graph_data(triple, new_chosen_nodes, new_graph_data, NEW_OBJECT)
                elif triple[0] not in new_chosen_nodes:
                    new_chosen_nodes.append(triple[0])
                    add_triple_to_graph_data(triple, new_chosen_nodes, new_graph_data, NEW_SUBJECT)    
        json_data[1] = new_graph_data
        #calculate the Ged_distance
        GED = len(chosen_nodes) + len(chosen_edges) - len(new_chosen_nodes) - len(new_chosen_edges)
        json_data["GED"] = GED
        data_info["avg_ged"] += GED
        file_name = os.path.join(des_dir, "graphPair_10_20_{}.json".format(i))
        with open(file_name, "w") as f:
            json.dump(json_data, f)
        
    data_info["avg_node_num"] /= args.data_num
    data_info["avg_edge_num"] /= args.data_num
    data_info["avg_ged"] /= args.data_num
    file_name = os.path.join(os.path.dirname(des_dir), "dataset_info.json")
    with open(file_name, "w") as f:
        json.dump(data_info, f)
    return


def process_data(statistic_file, raw_dir, des_dir):
    with open(statistic_file, "r") as f:
        statistics = json.load(f)
    raw_files = glob.glob(os.path.join(raw_dir, "*.json"))
    for raw_file in tqdm(raw_files):
        with open(raw_file, "r") as f:
            data = json.load(f)
        for i, Graph in data.items():
            if isinstance(Graph, int) or isinstance(Graph, str):
                continue
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
        destination_path = os.path.join(des_dir, os.path.basename(raw_file))
        if os.path.exists(destination_path):
            os.remove(destination_path)
        shutil.move(raw_file, des_dir)
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
    #edit subgraph via del nodes and attacted edges
    del_percent = random.uniform(args.edit_percent_min, args.edit_percent_max)
    del_nodes_num = int(len(chosen_nodes)*del_percent)
    del_nodes = random.sample(chosen_nodes, del_nodes_num)
    new_graph_data = {"KG": [], "node_features": [], "edge_indices": [], "edge_features": []}
    new_chosen_nodes = []
    new_chosen_edges = []
    chosen_triples = graph_data["KG"].copy() 
    for triple in chosen_triples:
        if triple[0] in del_nodes or triple[2] in del_nodes:
            continue
        else:
            new_chosen_edges.append(triple[1])
            if triple[0] in new_chosen_nodes and triple[2] in new_chosen_nodes:
                add_triple_to_graph_data(triple, new_chosen_nodes, new_graph_data, NEW_PREDICTS)
            elif triple[0] not in new_chosen_nodes and triple[2] not in new_chosen_nodes:
                new_chosen_nodes.append(triple[0])
                new_chosen_nodes.append(triple[2])
                add_triple_to_graph_data(triple, new_chosen_nodes, new_graph_data, NEW_TRIPLE)
            elif triple[2] not in new_chosen_nodes:
                new_chosen_nodes.append(triple[2])
                add_triple_to_graph_data(triple, new_chosen_nodes, new_graph_data, NEW_OBJECT)
            elif triple[0] not in new_chosen_nodes:
                new_chosen_nodes.append(triple[0])
                add_triple_to_graph_data(triple, new_chosen_nodes, new_graph_data, NEW_SUBJECT)    
    json_data[1] = new_graph_data
    #calculate the Ged_distance
    GED = len(chosen_nodes) + len(chosen_edges) - len(new_chosen_nodes) - len(new_chosen_edges)
    json_data["GED"] = GED
    data_info["ged"] = GED
    file_name = os.path.join(des_dir, "graphPair_10_20_{}.json".format(index))
    with open(file_name, "w") as f:
        json.dump(json_data, f)
    return data_info




if __name__ == "__main__":
    
    dataset = 'lubm'
    dataset_name = 'lubm'
    parser = argparse.ArgumentParser(description="Generate subgraphs from a Knowledge Graph")
    parser.add_argument("--max_nodes", type=int, default=20, help="Maximum number of nodes in a subgraph")
    parser.add_argument("--min_nodes", type=int, default=10, help="Minimum number of nodes in a subgraph")
    parser.add_argument("--data_num", type=int, default=5000, help="Number of subgraphs to generate")
    parser.add_argument("--edit_percent_max", type=float, default=0.20, help="Percentage of edges to edit")
    parser.add_argument("--edit_percent_min", type=float, default=0.10, help="Percentage of edges to edit")
    args = parser.parse_args()

    remote = False
    Trans_embedding = True
    Gen_graph = True
    params = {
    "username": "root",
    "password": "123456",
    "operation": "query",
    "db_name": dataset,
    "format": "json"
    }
    KG_path = f"/home/GED_Process/NeuralGED/data/nwedata/{dataset}/lubm.nt"
    embedding_path = f"/home/GED_Process/NeuralGED/data/newdata/{dataset}/statistic.json"
    raw_dir = f"/home/GED_Process/NeuralGED/data/newdata/{dataset_name}l/raw_data"
    des_dir = f"/home/GED_Process/NeuralGED/data/newdata/{dataset_name}/processed_data"
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if remote == False:
        if Gen_graph:
            print(f"Generating subgraph from KG {dataset}")
            generate_subgraphs(KG_path, des_dir, args)
            print("Finished generating subgraph")
        if Trans_embedding:
            print(f"Processing data for {dataset}")
            process_data(embedding_path, raw_dir=raw_dir, des_dir=des_dir)
            print("Finished processing data")
    else:
        if Gen_graph:
            print(f"Generating subgraph from KG {dataset}")
            data_info = {"avg_node_num": 0, "avg_edge_num": 0, "max_node": 0, "max_edge": 0, "avg_ged": 0}
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
            with Pool(processes=8) as pool:
                for result in tqdm(pool.imap_unordered(
                    generate_subgraph, 
                    [[i, args, raw_dir, total_triple_num, "http://127.0.0.1:9002/grpc/api/"] for i in range(args.data_num)],
            ), total = args.data_num):
                    results.append(result)
            for res in results:
                data_info["avg_node_num"] += res['node']
                data_info['avg_edge_num'] += res['edge']
                data_info["avg_ged"] += res['ged']
                if res['node'] > data_info["max_node"]:
                    data_info["max_node"] = res["node"]
                if res['edge'] > data_info["max_edge"]:
                    data_info["max_edge"] = res["edge"]
            data_info["avg_node_num"] /= args.data_num
            data_info["avg_edge_num"] /= args.data_num
            data_info['total_graph_num'] = args.data_num
            data_info["avg_ged"] /= args.data_num
            file_name = os.path.join(os.path.dirname(raw_dir), "dataset_info.json")
            with open(file_name, "w") as f:
                json.dump(data_info, f)
            print("Finished generating subgraph")
        if Trans_embedding:
            print(f"Processing data for {dataset}")
            process_data(embedding_path, raw_dir=raw_dir, des_dir=des_dir)
            print("Finished processing data")
