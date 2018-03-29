from torch import nn
import torch.nn.functional as F

def dice_score(logits, targets, thresh):
    hard_preds = (F.sigmoid(logits) > thresh).float()
    m1 = hard_preds.view(-1)  # Flatten
    m2 = targets.view(-1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection) / (m1.sum() + m2.sum() + 1e-6)

def dice_coeff(pred, target, smooth=0):
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth + 1e-10)