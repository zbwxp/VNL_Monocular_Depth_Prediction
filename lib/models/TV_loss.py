import torch
import torch.nn as nn
import math

class TV_Loss(nn.Module):
    """
    Virtual Normal Loss Function.
    """

    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self, gt_depth, pred_depth):
        gt_tv_h = gt_depth[:, :, 1:, :] - gt_depth[:, :, :-1, :]
        gt_tv_w = gt_depth[:, :, :, 1:] - gt_depth[:, :, :, :-1]

        pred_tv_h = pred_depth[:, :, 1:, :] - pred_depth[:, :, :-1, :]
        pred_tv_w = pred_depth[:, :, :, 1:] - pred_depth[:, :, :, :-1]

        loss = (torch.exp(-torch.abs(gt_tv_h)) * torch.abs(pred_tv_h)).sum() + (torch.exp(-torch.abs(gt_tv_w)) * torch.abs(pred_tv_w)).sum()
        return loss
