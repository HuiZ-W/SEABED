import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import DataLoader
from dataset.ged_dataset import GEDDataSet
from model.model import GraphNet
import numpy as np
from tqdm import tqdm
import os
import datetime
import argparse
import random
import json

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(args):
    set_seed(args.seed)
    model_dir = os.path.join(os.path.dirname(args.train_dir), "bir_RNN")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    device = torch.device(args.device)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    best_model_path = os.path.join(model_dir, f"model_{current_time}_best.pth")
    last_model_path = os.path.join(model_dir, f"model_{current_time}_last.pth")
    logger_path = os.path.join(model_dir, f"log_{current_time}.log")
    logging.basicConfig(filename=logger_path, level=logging.INFO)
    params = {
        'seed': args.seed,
        'dataset': args.dataset,
        'combine_type': args.combine_type,
        'weight_decay': args.weight_decays,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'avg_loss': args.avg_loss,
        'rule&graph_dim': args.rule_graph_dim,
        'combine_type': args.combine_type,
        'model_rule_nums': args.model_rule_nums,
        'sample_rule_nums': args.sample_rule_nums,
        'rule_length': args.rule_length
    }
    logging.info("Training parameters: %s", json.dumps(params, indent=4))
    # load data
    train_dataset = GEDDataSet(args.train_dir, PreLoad=args.preload, device=device, args=args)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    if args.val_dir != "none":
        val_dataset = GEDDataSet(args.val_dir, PreLoad=args.preload, device=device, args=args)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    if args.test_dir != "none":
        test_dataset = GEDDataSet(args.test_dir, PreLoad=args.preload, device=device, args=args)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # define model
    model = GraphNet(args).to(device)
    if args.model_path != "none":
        model.load_state_dict(torch.load(args.model_path))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decays)
    # train model
    losses = 0
    best_val_loss = float("inf")
    best_val_epoch = 0
    for epoch in tqdm(range(args.epochs)):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for i, data in enumerate(train_loader):
            gt_ged = data["gt_ged"]
            output = model(data)
            losses += criterion(output, gt_ged)
            if (i + 1) % args.batch_size == 0 or (i + 1) == len(train_loader):
                total_loss += losses.item()
                if args.avg_loss:
                    losses /= args.batch_size
                losses.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
                losses = 0
        total_loss /= len(train_loader)
        logging.info(f'Epoch {epoch}, Loss: {total_loss}')
        # validate
        if args.val_dir != "none":
            model.eval()
            val_loss = 0
            for data in val_loader:
                gt_ged = data["gt_ged"]
                output = model(data)
                loss = criterion(output, gt_ged)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            logging.info(f'Epoch {epoch}, Validation Loss: {val_loss}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
    logging.info(f'Best val epoch:{best_val_epoch}')
    logging.info(f'Best val loss:{best_val_loss}')
    # test
    if args.test_dir != "none":
        test_loss = 0
        for data in test_loader:
            gt_ged = data["gt_ged"]
            output = model(data)
            loss = criterion(output, gt_ged)
            test_loss += loss.item()
        test_loss /= len(test_loader)
        logging.info(f'Test Loss: {test_loss}')
        # save model
        torch.save(model.state_dict(), last_model_path)
        # test again
        model1 = GraphNet(args).to(device)
        state_dict = torch.load(best_model_path)
        model1.load_state_dict(state_dict)
        model1.eval()
        test_loss = 0
        for data in test_loader:
            gt_ged = data["gt_ged"]
            output = model1(data)
            loss = criterion(output, gt_ged)
            test_loss += loss.item()
        test_loss /= len(test_loader)
        logging.info(f'Best Model Test Loss: {test_loss}')
    else:
        torch.save(model.state_dict(), last_model_path)

if __name__ == "__main__":
    dataset = 'swdf_unlabel'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--train_dir', type=str, default=f'/home/huizhong/GED_Process/NeuralGED/data/{dataset}/processed_data/train')
    parser.add_argument('--val_dir', type=str, default=f'/home/huizhong/GED_Process/NeuralGED/data/{dataset}/processed_data/val')
    parser.add_argument('--test_dir', type=str, default=f'/home/huizhong/GED_Process/NeuralGED/data/{dataset}/processed_data/test')
    parser.add_argument('--model_path', type=str, default="none")
    parser.add_argument('--preload', type=bool, default=True)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--avg_loss', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model_rule_nums', type=int, default=8)
    parser.add_argument('--sample_rule_nums', type=int, default=16)
    parser.add_argument('--rule_length', type=int, default=4)
    parser.add_argument('--rule_graph_dim', type=int, default=32)
    parser.add_argument('--combine_type', type=str, default="fusion")
    parser.add_argument('--weight_decays', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    train(args)
