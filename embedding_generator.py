from pyrdf2vec.graphs import KG
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker
from cachetools import MRUCache
import json
from tqdm import tqdm
import os
import glob
import re
import shutil

def uri_to_id(uri):
    return uri.split('/')[-1]


def get_embeddings(dataset_name, kg_file, entities=None, remote=True, sparql_endpoint="http://127.0.0.1:9002/grpc/api/"):
    if remote:
        GRAPH = KG(sparql_endpoint, skip_verify=True, mul_req=True)
    else:
        GRAPH = KG(kg_file, skip_verify=True)
    if remote:
        entities = entities
    else:
        if entities is not None:
            entities = entities
        else:
            train_entities = [entity.name for entity in list(GRAPH._entities)]
            test_entities = [entity.name for entity in list(GRAPH._vertices)]
            entities = set(train_entities + test_entities)
            entities = list(entities)
    # create RDF2vec model
    transformer = RDF2VecTransformer(
        Word2Vec(epochs=10, vector_size=100),
        walkers=[RandomWalker(4, max_walks=5, with_reverse=True, n_jobs=32, md5_bytes=None)],
        verbose=2
    )
    '''
    #Define the path of the walk folder
    folder_path = "/home/huizhong/GED_Process/NeuralGED/data/walk"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print("walk folder removed")
    os.makedirs(folder_path)
    print("walk folder recreated")
    '''

    # Generate the embeddings
    print("Starting to fit model")
    embeddings, literals =transformer.fit_transform(GRAPH, entities)
    print("Finished fitting model")
    # Generating embedding for all entities
    test_entities_cleaned = {}
    print("Calculating Embeddings")
    for i, entity in enumerate(tqdm(entities)):
        try:
            embedding = embeddings[i]
            cov_embedding = [item.tolist() for item in embedding]
            test_entities_cleaned[entity] = {
                "embedding": cov_embedding,
            }
        except:
            print(entity)
            raise

    print("Saving statistics")
    with open(os.path.join(os.path.dirname(kg_file), "statistic.json"),"w") as f:
        json.dump(test_entities_cleaned, f)
    print("Finished saving statistics")  

    return

if __name__ == "__main__":

    data_dir = "/home/huizhong/GED_Process/NeuralGED/data/newdata/yago"
    subdirs = ["train", "val", 'test']
    rawData_path = os.path.join(data_dir, "raw_data")
    data_path = os.path.join(data_dir, "processed_data")
    kg_path = os.path.join(data_dir, "wikidata.nt")
    statistic_file = os.path.join(data_dir, "statistics.json")

    if os.path.exists(statistic_file):
        print("print exist embedding file: ", statistic_file)
    else:
        entities = []
        files = []
        for subdir in subdirs:
            files += glob.glob(os.path.join(rawData_path, subdir, "*.json"))
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                KGs = [data["0"]['KG']]
            for kg in KGs:
                for triple in kg:
                    entities.append(triple[0])
                    entities.append(triple[2])
                    entities.append(triple[1])
        '''
        with open(kg_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.match(r'<(.+?)> <(.+?)> <(.+?)> \.', line)
                if match:
                    sub, pre, obj = match.groups()
                    entities.append(sub)
                    entities.append(obj)
                    entities.append(pre)
        '''
        entities = list(set(entities))
        print("Generating Embeddings Via KG:", kg_path, "for dataset: yago")
        get_embeddings("yago", kg_path, entities=entities, remote=True)
        print("Finished generating embeddings")