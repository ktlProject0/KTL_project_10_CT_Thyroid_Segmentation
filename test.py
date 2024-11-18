from __future__ import print_function
import os
import argparse
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Net
from dataset import CustomDataset
from loss import DiceChannelLoss
from sklearn.metrics import precision_score, recall_score

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Segmentation')
    parser.add_argument('--data_direc', type=str, default='./data', help="data directory")
    parser.add_argument('--n_classes', type=int, default=1, help="num of classes")
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=123')
    parser.add_argument('--testBatchSize', type=int, default=4, help='test batch size')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints', help='Path for save best model')
    opt = parser.parse_args()
    
    if not os.path.isdir(opt.model_save_path):
        raise Exception("checkpoints not found, please run train.py first")

    os.makedirs("test_results", exist_ok=True)
    
    print(opt)
    
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
    torch.manual_seed(opt.seed)
    
    device = 'cuda'
    
    print('===> Loading datasets')
    
    test_set = CustomDataset(f"{opt.data_direc}/test", mode='eval')
    test_dataloader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
    
    print('===> Building model')
    model = Net(in_channels=1, out_channels=opt.n_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(opt.model_save_path, 'model_statedict.pth'), map_location=device))
    model.eval()

    with open(os.path.join(opt.model_save_path, 'metric_logger.json'), 'r') as f:
        metric_logger = json.load(f)
    
    criterion = nn.BCELoss()
    criterion_dice = DiceChannelLoss()
    
    total_test_num = len(test_dataloader.sampler)
    test_dice, test_ce = torch.zeros(opt.n_classes), 0
    
    # For precision and recall calculation
    all_pred = []
    all_target = []
    all_precision = []
    all_recall = []
    all_dice_loss = []

    with torch.no_grad():
        for data in tqdm(test_dataloader, total=len(test_dataloader), position=0, desc='Test', colour='green'):
            batch_num = len(data['input'])
        
            image = data['input'].to(device)
            target = data['target'].to(device)

            pred_logit = model(image.float())
            pred = (pred_logit > 0.5).float()
            ce_loss = criterion(pred, target.float())
            dice_channel_loss, dice_loss = criterion_dice(pred, target)
            
            dice_score = 1 - dice_channel_loss
            test_dice += dice_score.cpu() * batch_num
            test_ce += ce_loss.item() * batch_num
            
            # Save predictions and targets for metrics
            all_pred.append(pred.cpu().numpy())
            all_target.append(target.cpu().numpy())
            all_dice_loss.append(dice_loss.item())

            # Calculate precision and recall for the current batch
            precision = precision_score(target.cpu().numpy().flatten(), pred.cpu().numpy().flatten(), average='binary', zero_division=1)
            recall = recall_score(target.cpu().numpy().flatten(), pred.cpu().numpy().flatten(), average='binary', zero_division=1)
            all_precision.append(precision)
            all_recall.append(recall)

    test_dice /= total_test_num
    test_ce /= total_test_num
    dice_std = np.std(all_dice_loss)
    # Flatten the lists of predictions and targets
    all_pred = np.concatenate(all_pred)
    all_target = np.concatenate(all_target)

    # Calculate overall precision and recall
    overall_precision = precision_score(all_target.flatten(), all_pred.flatten(), average='binary', zero_division=1)
    overall_recall = recall_score(all_target.flatten(), all_pred.flatten(), average='binary', zero_division=1)

    # Calculate standard deviation
    precision_std = np.std(all_precision)
    recall_std = np.std(all_recall)

    eval_df = pd.DataFrame({
        "Test Dice Coefficient Score": [test_dice.numpy()],
        "Test Dice Std": [dice_std],  
        "Test Precision": [overall_precision],
        "Test Precision Std": [precision_std],
        "Test Recall": [overall_recall],
        "Test Recall Std": [recall_std]
    })

    eval_df.to_csv(f"test_results/metric_df.csv", index=None)

    plt.figure()
    for k in ['train_loss', 'val_loss']:
        plt.plot(np.arange(len(metric_logger[k])), metric_logger[k], label=k)
    plt.title("Dice Coefficient Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"test_results/learning_graph_dice_coefficient.png", dpi=200)
