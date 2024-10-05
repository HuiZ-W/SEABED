import torch
import torch.nn as nn 
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.ged_dataset import GEDDataSet
from model.model import GraphNet
import logging
import argparse
import os
import datetime
import time
import json

def test(args):

    result_dir = os.path.join(os.path.dirname(args.test_dir), "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger_path = os.path.join(result_dir, f"test_{current_time}.log")
    logging.basicConfig(filename=logger_path, level=logging.INFO)
    
    device = torch.device(args.device)

    test_dataset = GEDDataSet(args.test_dir, PreLoad=args.preload, device=device, args=args)
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
    pk1 = []
    pk5 = []
    pk10 = []
    pk15 = []
    pk20 = []
    pk30 = []
    pk40 = []
    pk50 = []
    result_dir = os.path.join(os.path.dirname(args.test_dir), "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger_path = os.path.join(result_dir, f"test_{current_time}.log")
    logging.basicConfig(filename=logger_path, level=logging.INFO)
    
    device = torch.device(args.device)

    test_dataset = GEDDataSet(args.test_dir, PreLoad=args.preload, device=device, args=args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = GraphNet(args, rule_nums=args.model_rule_num).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    criterion = nn.MSELoss()
    
    pre = []
    gt = []
    start_time = time.time()
    with torch.no_grad(): 
        for data in tqdm(test_loader):
            nums += 1
            output = model(data)
            gt_ged = data["gt_ged"]
            real_ged = data["real_ged"]
            pre_ged = output*data['norm']
            round_pre_ged = torch.round(pre_ged)
            mse.append(criterion(output, gt_ged))
            mae.append(abs(pre_ged - real_ged))
            pre.append(pre_ged.item())
            gt.append(real_ged.item())
            if round_pre_ged == real_ged:
                num_acc += 1
                num_fea += 1
            elif round_pre_ged > real_ged:
                num_fea += 1  
    end_time = time.time()
    time_usage.append((end_time - start_time)*100/len(test_loader))
    acc = num_acc/nums
    fea = num_fea/nums
    pk1.append(cal_pk(1, pre, gt))
    pk5.append(cal_pk(5, pre, gt))
    pk10.append(cal_pk(10, pre, gt))
    pk15.append(cal_pk(15, pre, gt))
    pk20.append(cal_pk(20, pre, gt))
    pk30.append(cal_pk(30, pre, gt))
    pk40.append(cal_pk(40, pre, gt))
    pk50.append(cal_pk(50, pre, gt))
    mse = sum(mse)/len(mse)
    mae = sum(mae)/len(mae)
    result = {
        'mse': mse,
        'mae': mae,
        'acc': acc,
        'fea': fea,
        'pk1': pk1,
        'pk5': pk5,
        'pk10': pk10,
        'pk15': pk15,
        'pk20': pk20,
        'pk30': pk30,
        'pk40': pk40,
        'pk50': pk50,
        'time': time_usage
    }
    logging.info("result: %s", json.dumps(result, indent=4))

def cal_pk(num, pre, gt):
    tmp = list(zip(pre, gt))
    tmp.sort()
    beta = []
    for i, p in enumerate(tmp):
        beta.append((p[1], p[0], i))
    beta.sort()
    ans = 0
    for i in range(num):
        if beta[i][2] < num:
            ans += 1
    return ans/num

if __name__ == "__main__":
    dataset = "wikidata"
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default=f'/home/huizhong/GED_Process/NeuralGED/data/{dataset}/processed_data/test')
    parser.add_argument('--model_path', type=str, default='/home/huizhong/GED_Process/NeuralGED/data/wikidata/processed_data/new_model/model_2024-08-13_20-11-10_best.pth')
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--preload', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_rule_num', type=int, default=32)
    parser.add_argument('--sample_rule_num', type=int, default=32)
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--rule_length', type=int, default=10)
    parser.add_argument('--combine_type', type=str, default="fusion")
    args = parser.parse_args()
    #test(args)
    score(args)