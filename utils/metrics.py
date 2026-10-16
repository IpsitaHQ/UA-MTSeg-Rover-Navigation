import torch
import numpy as np
import torch.nn.functional as F

def calculate_iou(preds, labels, num_classes=8):
    preds = torch.argmax(preds, dim=1)
    iou_per_class = []
    
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            iou_per_class.append(float('nan'))  # Ignore class if it is absent in both
        else:
            iou_per_class.append((intersection / union).item())
            
    # filter out nans
    valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    return miou, iou_per_class

def expected_calibration_error(confidences, accuracies, num_bins=10):
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        in_bin = np.where((confidences > bin_lower) & (confidences <= bin_upper))[0]
        if len(in_bin) > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (len(in_bin) / total_samples) * np.abs(bin_acc - bin_conf)
            
    return ece

def mc_dropout_predict(model, rgb, depth, thermal, passes=15):
    """
    Runs MC dropout inference.
    rgb, depth, thermal should be tensors of shape (B, C, H, W).
    """
    model.train() # Keep dropout active
    predictions = []
    
    with torch.no_grad():
        for _ in range(passes):
            logits = model(rgb, depth, thermal)
            probs = F.softmax(logits, dim=1)
            predictions.append(probs)
            
    stacked_probs = torch.stack(predictions) # (passes, B, C, H, W)
    mean_probs = stacked_probs.mean(dim=0)    # (B, C, H, W)
    
    # Entropy: H = -sum(p * log(p))
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
    
    # Normalize by log(C) where C is number of classes
    num_classes = mean_probs.shape[1]
    norm_entropy = entropy / np.log(num_classes)
    
    return mean_probs, norm_entropy
