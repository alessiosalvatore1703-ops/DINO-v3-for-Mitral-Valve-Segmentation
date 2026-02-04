import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (N, 1, H, W) or (N, C, H, W)
        # targets: same shape, values 0 or 1
        probs = torch.sigmoid(logits)
        probs = probs.view(logits.size(0), -1)
        targets = targets.view(logits.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class PowerJaccardLoss(nn.Module):
    '''
        From https://www.scitepress.org/Papers/2021/103040/103040.pdf
        Based from https://github.com/luke-ck/ultrasound-segmentation/blob/master/src/metrics.py
    '''
    def __init__(self, p=2, smooth=1e-6):
        super().__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (N, 1, H, W) or (N, C, H, W)
        # targets: same shape, values 0 or 1
        probs = torch.sigmoid(logits)
        probs = probs.view(logits.size(0), -1)
        targets = targets.view(logits.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        total = (torch.pow(probs, self.p) + torch.pow(targets, self.p)).sum(dim=1)
        union = total - intersection

        IoU = (intersection + self.smooth) / (union + self.smooth)

        return (1 - IoU).mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss.
    BCE penalizes false positives/negatives per pixel.
    Dice focuses on overlap.
    This combination helps prevent mode collapse (all 0s or all 1s).
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        # BCE loss (works on logits)
        bce_loss = self.bce(logits, targets)
        
        # Dice loss
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(logits.size(0), -1)
        targets_flat = targets.view(logits.size(0), -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    Down-weights easy examples and focuses on hard ones.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Compute focal weights
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting for class imbalance
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        focal_loss = alpha_weight * focal_weight * bce
        return focal_loss.mean()