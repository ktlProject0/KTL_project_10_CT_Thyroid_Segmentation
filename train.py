from __future__ import print_function
import os
import argparse
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from dataset import CustomDataset
from util import EarlyStopping
from loss import BCELoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Thyroid Segmentation')
    parser.add_argument('--data_direc', type=str, default='./data', help="Data directory")
    parser.add_argument('--n_classes', type=int, default=1, help="Number of classes")
    parser.add_argument('--batchSize', type=int, default=1, help='Training batch size')
    parser.add_argument('--total_epoch', type=int, default=200, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_schedule_patience', type=int, default=10, help='Learning rate schedule patience')
    parser.add_argument('--earlystop_patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads for data loader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints', help='Path to save the best model')
    opt = parser.parse_args()

    os.makedirs(opt.model_save_path, exist_ok=True)
    print(opt)

    # Set device
    torch.manual_seed(opt.seed)
    device = 'cuda:0'

    # Load datasets
    print('===> Loading datasets')
    train_set = CustomDataset(f"{opt.data_direc}/train", mode='train')
    val_set = CustomDataset(f"{opt.data_direc}/valid", mode='eval')
    train_dataloader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    val_dataloader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

    # Initialize model, loss function, optimizer, and scheduler
    print('===> Building model')
    model = Net(in_channels=1, out_channels=opt.n_classes).to(device)
    criterion = BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.lr_schedule_patience)
    monitor = EarlyStopping(patience=opt.earlystop_patience, verbose=True, path=os.path.join(opt.model_save_path, 'model_statedict.pth'))

    # Initialize metrics
    metric_logger = {k: [] for k in ['train_ce', 'val_ce', 'train_loss', 'val_loss', 'lr']}
    total_train_num = len(train_dataloader.sampler)
    total_val_num = len(val_dataloader.sampler)

    for epoch in range(opt.total_epoch):
        for param in optimizer.param_groups:
            lr_status = param['lr']
        metric_logger['lr'].append(lr_status)

        epoch_loss = {k: 0 for k in metric_logger if k != 'lr'}

        print(f"Epoch {epoch + 1:03d}/{opt.total_epoch:03d}\tLR: {lr_status:.0e}")

        # Training phase
        model.train()
        for data in tqdm(train_dataloader, total=len(train_dataloader), desc='Train', colour='blue'):
            image, target = data['input'].to(device), data['target'].to(device)
            pred = model(image.float())
            ce_loss = criterion(pred, target.float())

            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()

            batch_num = image.size(0)
            epoch_loss['train_ce'] += ce_loss.item() * batch_num
            epoch_loss['train_loss'] += ce_loss.item() * batch_num

        # Validation phase
        model.eval()
        with torch.no_grad():
            for data in tqdm(val_dataloader, total=len(val_dataloader), desc='Val', colour='green'):
                image, target = data['input'].to(device), data['target'].to(device)
                pred = model(image.float())
                ce_loss = criterion(pred, target.float())

                batch_num = image.size(0)
                epoch_loss['val_ce'] += ce_loss.item() * batch_num
                epoch_loss['val_loss'] += ce_loss.item() * batch_num

        # Normalize metrics by dataset size
        epoch_loss = {k: v / (total_train_num if 'train' in k else total_val_num) for k, v in epoch_loss.items()}
        for k, v in epoch_loss.items():
            metric_logger[k].append(v)

        # Early stopping
        monitor(epoch_loss['val_loss'], model)
        if monitor.early_stop:
            print(f"Training early stopped. Minimum validation loss: {monitor.val_loss_min}")
            break

        # Adjust learning rate
        scheduler.step(epoch_loss['val_loss'])

        print(f"Train loss: {epoch_loss['train_loss']:.7f}\tTrain CE: {epoch_loss['train_ce']:.7f}\n"
              f"Val loss: {epoch_loss['val_loss']:.7f}\tVal CE: {epoch_loss['val_ce']:.7f}")

    # Save metrics to JSON
    with open(os.path.join(opt.model_save_path, 'metric_logger.json'), 'w') as f:
        json.dump(metric_logger, f)
