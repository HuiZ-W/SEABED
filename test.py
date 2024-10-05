import torch
import torch.nn as nn 
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.ged_dataset import NEWGEDDataSet
from model.model import GraphNet
import logging
import argparse
import os
import datetime
import time
import json
from scipy.stats import spearmanr, kendalltau
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(args):

    result_dir = os.path.join(os.path.dirname(args.test_dir), "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger_path = os.path.join(result_dir, f"test_{current_time}.log")
    logging.basicConfig(filename=logger_path, level=logging.INFO)
    
    device = torch.device(args.device)

    test_dataset = NEWGEDDataSet(args.test_dir, PreLoad=args.preload, device=device, args=args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = GraphNet(args, rule_nums=args.model_rule_num).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0
    start_time = time.time()
    with torch.no_grad(): 
        for data in tqdm(test_loader):
            gt_ged = data["gt_ged"]
            output = model(data)
            loss = criterion(output, gt_ged)
            total_loss += loss.item()
    end_time = time.time()
    total_loss /= len(test_loader)
    logging.info(f'Test Loss: {total_loss}, Test_time: {(end_time - start_time)*100/len(test_loader)}')


def score(args):
    nums = 0
    time_usage = []
    mse = []  # score mse
    mae = []  # ged mae
    num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
    num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
    rho = []
    tau = []
    pk1 = []
    pk5 = []
    pk10 = []
    pk15 = []
    pk20 = []
    ndcg_1 = []
    ndcg_5 = []
    ndcg_10 = []
    ndcg_15 = []
    ndcg_20 = []
    time_percent={"fir_time":0,"snd_time":0,"trd_time":0}
    result_dir = os.path.join(os.path.dirname(args.test_dir), "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger_path = os.path.join(result_dir, f"test_{current_time}.log")
    logging.basicConfig(filename=logger_path, level=logging.INFO)
    
    device = torch.device(args.device)

    test_dataset = NEWGEDDataSet(args.test_dir, args.test_pairs, PreLoad=args.preload, device=device, args=args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = GraphNet(args).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    criterion = nn.MSELoss()
    
    result = {}
    gt = []
    pre = []
    start_time = time.time()
    with torch.no_grad(): 
        for data in tqdm(test_loader):
            nums += 1
            output, attention_weight, Time_Use = model(data)
            '''
            time_percent["fir_time"] += data['time_use'].item()
            time_percent["snd_time"] += Time_Use[0]
            time_percent["trd_time"] += Time_Use[1]
            '''
            graph_id = data["graph1"][0]
            gt_ged = data["gt_ged"]
            real_ged = data["real_ged"]
            pre_ged = output*data['norm']
            round_pre_ged = torch.round(pre_ged)
            mse.append(criterion(output, gt_ged).item())
            mae.append(abs(pre_ged - real_ged).item())
            pre.append(pre_ged.item())
            gt.append(real_ged.item())
            if round_pre_ged == real_ged:
                num_acc += 1
                num_fea += 1
            elif round_pre_ged > real_ged:
                num_fea += 1 
            if graph_id not in result:
                result[graph_id] = {"pair_name": [],"gt": [], "pre": []} 
            result[graph_id]["pair_name"].append(data["graph1"][0])
            result[graph_id]["gt"].append(real_ged.item())
            result[graph_id]["pre"].append(pre_ged.item())
    end_time = time.time()
    time_usage.append((end_time - start_time)*100/len(test_loader))
    acc = num_acc/nums
    fea = num_fea/nums
    for key in result:
        pre = result[key]["pre"]
        gt = result[key]["gt"]
        pair_info = result[key]["pair_name"]
        pk1.append(cal_pk(1, pre, gt))
        pk5.append(cal_pk(5, pre, gt))
        pk10.append(cal_pk(10, pre, gt))
        pk15.append(cal_pk(15, pre, gt))
        pk20.append(cal_pk(20, pre, gt))
        ndcg_1.append(ndcg_k(1, pre, gt))
        ndcg_5.append(ndcg_k(5, pre, gt))
        ndcg_10.append(ndcg_k(10, pre, gt))
        ndcg_15.append(ndcg_k(15, pre, gt))
        ndcg_20.append(ndcg_k(20, pre, gt))
        rho.append(spearmanr(pre, gt)[0])
        tau.append(kendalltau(pre, gt)[0])
    pk1 = sum(pk1)/len(pk1)
    pk5 = sum(pk5)/len(pk5)
    pk10 = sum(pk10)/len(pk10)
    pk15 = sum(pk15)/len(pk15)
    pk20 = sum(pk20)/len(pk20)
    ndcg_1 = sum(ndcg_1)/len(ndcg_1)
    ndcg_5 = sum(ndcg_5)/len(ndcg_5)
    ndcg_10 = sum(ndcg_10)/len(ndcg_10)
    ndcg_15 = sum(ndcg_15)/len(ndcg_15)
    ndcg_20 = sum(ndcg_20)/len(ndcg_20)
    rho = sum(rho)/len(rho)
    tau = sum(tau)/len(tau)
    mse = sum(mse)/len(mse)
    mae = sum(mae)/len(mae)
    time_percent["fir_time"] = time_percent["fir_time"]/len(test_loader)
    time_percent["snd_time"] = time_percent["snd_time"]/len(test_loader)
    time_percent["trd_time"] = time_percent["trd_time"]/len(test_loader)
    result = {
        'mse': mse,
        'mae': mae,
        'acc': acc,
        'fea': fea,
        'rho': rho,
        'tau': tau,
        'pk1': pk1,
        'pk5': pk5,
        'pk10': pk10,
        'pk15': pk15,
        'pk20': pk20,
        'ndcg_1': ndcg_1,
        'ndcg_5': ndcg_5,
        'ndcg_10': ndcg_10,
        'ndcg_15': ndcg_15,
        'ndcg_20': ndcg_20,
        'time': time_usage,
        'time_percent': time_percent
    }
    print(result)
    logging.info("result: %s", json.dumps(result, indent=4))

def cal_pk(num, pre, gt):
    tmp = list(zip(gt, pre))
    tmp.sort(key=lambda x: x[0])
    beta = []
    for i, p in enumerate(tmp):
        beta.append((p[1], p[0], i))
    beta.sort()
    ans = 0
    for i in range(num):
        if beta[i][2] < num:
            ans += 1
    return ans/num
import numpy as np


def ndcg_k(k, pre, gt):
    tmp = list(zip(gt, pre))
    tmp.sort(key=lambda x: x[0])
    beta = []
    for i, p in enumerate(tmp):
        beta.append((p[1], p[0], i))
    beta.sort()
    revlevance_scores = []
    best_scores = []
    for i in range(len(beta)):
        if beta[i][2] == i:
            revlevance_scores.append(1)
        else:
            revlevance_scores.append(0)
        best_scores.append(1)
    revlevance_scores = np.asfarray(revlevance_scores)[:k]
    best_scores = np.asfarray(best_scores)[:k]
    dcg = np.sum((2**revlevance_scores - 1) / np.log2(np.arange(2, revlevance_scores.size + 2)))
    idcg = np.sum((2**best_scores - 1) / np.log2(np.arange(2, best_scores.size + 2)))
    if idcg == 0:
        return 0
    return dcg/idcg

if __name__ == "__main__":
    set_seed(42)
    dataset = "swdf_unlabel"
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default=f'/home/GED_Process/NeuralGED/data/newdata/{dataset}/processed_data/test')
    parser.add_argument('--test_pairs', type=str, default=f'/home/GED_Process/NeuralGED/data/newdata/{dataset}/processed_data/test_GEDINFO.json')
    parser.add_argument('--model_path', type=str, default='/home/GED_Process/NeuralGED/data/newdata/swdf_unlabel/processed_data/tree_mlp_all_rules/model_2024-09-11_20-39-08_best.pth')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--preload', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_rule_nums', type=int, default=8)
    parser.add_argument('--sample_rule_nums', type=int, default=16)
    parser.add_argument('--rule_graph_dim', type=int, default=32)
    parser.add_argument('--rule_length', type=int, default=4)
    parser.add_argument('--deduplicate', type=bool, default=False)
    parser.add_argument('--combine_type', type=str, default="fusion")
    parser.add_argument('--info_type', type=str, default='attention')
    args = parser.parse_args()
    #test(args)
    score(args)
