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


def visualize_sample(image, target, prediction, output_dir, idx):
    """
    Save 3D visualization of input, target, prediction, input+target, and input+prediction slices.
    """
    os.makedirs(output_dir, exist_ok=True)
    for slice_idx in range(image.shape[0]):
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))

        axes[0].imshow(image[slice_idx], cmap='gray')
        axes[0].set_title(f"Input - Slice {slice_idx}")
        axes[0].axis('off')

        axes[1].imshow(target[slice_idx], cmap='gray', alpha=0.7)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')

        axes[2].imshow(prediction[slice_idx], cmap='gray', alpha=0.7)
        axes[2].set_title("Prediction")
        axes[2].axis('off')

        axes[3].imshow(image[slice_idx], cmap='gray')
        axes[3].imshow(target[slice_idx], cmap='gray', alpha=0.5)
        axes[3].set_title("Input + GT")
        axes[3].axis('off')

        axes[4].imshow(image[slice_idx], cmap='gray')
        axes[4].imshow(prediction[slice_idx], cmap='gray', alpha=0.5)
        axes[4].set_title("Input + Pred")
        axes[4].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_{idx}_slice_{slice_idx}.png"))
        plt.close()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Segmentation')
    parser.add_argument('--data_direc', type=str, default='./data', help="data directory")
    parser.add_argument('--n_classes', type=int, default=1, help="num of classes")
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=123')
    parser.add_argument('--testBatchSize', type=int, default=1, help='test batch size (3D volume)')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints', help='Path for save best model')
    opt = parser.parse_args()

    if not os.path.isdir(opt.model_save_path):
        raise Exception("checkpoints not found, please run train.py first")

    os.makedirs("test_results", exist_ok=True)
    os.makedirs("test_results/visualizations", exist_ok=True)

    print(opt)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    device = 'cuda' if opt.cuda and torch.cuda.is_available() else 'cpu'

    print('===> Loading datasets')
    test_set = CustomDataset(f"{opt.data_direc}/test", mode='eval')
    test_dataloader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    print('===> Building model')
    model = Net(in_channels=1, out_channels=opt.n_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(opt.model_save_path, 'model_statedict.pth'), map_location=device))
    model.eval()

    criterion = nn.BCELoss()
    criterion_dice = DiceChannelLoss()

    total_test_num = len(test_dataloader.sampler)
    test_dice, test_ce = torch.zeros(opt.n_classes), 0

    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_dataloader, total=len(test_dataloader), desc='Test', colour='green')):
            image = data['input'].to(device)  # Shape: (1, 1, D, H, W)
            target = data['target'].to(device)  # Shape: (1, 1, D, H, W)

            pred_logit = model(image.float())  # Output of shape (1, 1, D, H, W)
            pred = (pred_logit > 0.5).float()

            ce_loss = criterion(pred, target.float())
            dice_channel_loss, dice_loss = criterion_dice(pred, target)

            dice_score = 1 - dice_channel_loss
            test_dice += dice_score.cpu()
            test_ce += ce_loss.item()

            # Save visualization
            visualize_sample(
                image.squeeze().cpu().numpy(),
                target.squeeze().cpu().numpy(),
                pred.squeeze().cpu().numpy(),
                output_dir="test_results/visualizations",
                idx=idx
            )

    print("[INFO] Visualizations saved in 'test_results/visualizations'")
