import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def bce_dice_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice
if __name__ == "__main__":
    pred = torch.randn(2, 1, 256, 256)  # Float tensor (default)
    target = torch.randint(0, 2, (2, 256, 256)).unsqueeze(1).float()  # 🔽 Changed to float
    
    loss = bce_dice_loss(pred, target)
    print(f"Loss value: {loss.item()}")


