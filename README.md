# GINIE

This repository contains the implementation of GINIE, a method for predicting knowledge graph edit distance through embedding and graph neural networks. The method is developed using PyTorch.

---

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
---

## Installation
### Requirements
You will need to install Python 3. This code has been tested with Python 3.8.10.
### Setup
You can install the required packages using the following command:
```bash
pip install -r requirements.txt
```
## Usage
### Data
The used datasets, graph pairs can be found under the following link:[GENIE_Data](https://cuhko365-my.sharepoint.com/:f:/g/personal/224045005_link_cuhk_edu_cn/Eq1RC0HTdnhHka5r6Bd2LpMB9PSlJx2XEEc3TvtSRfGFCQ?e=24EaDJ)

We expect you to have a directory containing the data you want to predict in the following format:
```
|---dataset
|    |---graph_pair_0.json
|    |---graph_pair_1.json
|---graph_pair_info.json
```
the graph_pair_0.json contains the info of your graph, the format is following:
```
{
    "0":{
        "KG":[
            [S1,P1,O1],
            ...,
        ]
        "node_features":[
            {
                "id":"S1",
                "embedding":[...],
            },
            ...,
        ]
        "edge_indices":[
            [0,1],
            ...,
        ]
        "edge_features":[
            {
                "id":"P1",
                "embedding":[...],
            },
            ...,
        ]"
    }
}
```
Here, `KG` refers to the entire graph corresponding to the original RDF format;

`node_features` represents each entity, containing the RDF identifier of the entity in the KG as well as its embedding;

`edge_indices` represents the connections between nodes;

`edge_features` represents each predicate, containing the RDF identifier of the predicate in the KG as well as its embedding. 

In this case, the i-th connection in edge_indices corresponds to the i-th edge in edge_features.

The graph_pair_info.json contains the graph pair info that you want to calculate,the format is following:
```
{
    "pairs_info":[["graphPair_0.json", "graphPair_1.json", gt_ged], ...]
}
```
Here, `gt_ged` (ground truth Graph Edit Distance) is used to compare the performance of models during training and testing phases.

### Training

Next, you can train the GNN model to predict the GED. 
For that, the file `train.py` is used.

The best model will be saved under ```model_xxx_best.pth```. Here xxx is the time when you start the taining.
