import torch

def compute_iou(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred) > threshold
    target = target.bool()
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    if union == 0:
        return torch.tensor(1.0)
    return intersection / union

# Test the function
if __name__ == "__main__":
    pred = torch.tensor([[0.2, 0.7], [0.6, 0.9]])
    target = torch.tensor([[0, 1], [1, 1]])
    iou = compute_iou(pred, target)
    print(f"IoU: {iou}")

