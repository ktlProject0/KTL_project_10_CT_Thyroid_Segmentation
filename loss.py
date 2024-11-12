import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        target = target.float()
        bce_loss = - (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        return bce_loss.mean()

class DiceChannelLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceChannelLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        
        num_channels = pred.shape[1]
        dice_loss_per_channel = torch.zeros(num_channels, device=pred.device)
        
        for i in range(num_channels):
            pred_channel = pred[:, i, ...]
            target_channel = target[:, i, ...]

            intersection = (pred_channel * target_channel).sum()
            union = pred_channel.sum() + target_channel.sum()
            
            dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss_per_channel[i] = 1 - dice_coeff

        avg_dice_loss = dice_loss_per_channel.mean()
        
        return dice_loss_per_channel, avg_dice_loss