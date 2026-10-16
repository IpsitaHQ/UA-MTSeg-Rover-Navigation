import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from .metrics import calculate_iou

def train_epoch(model, dataloader, criterion, optimizer, device, is_multimodal=False):
    model.train()
    epoch_loss = 0.0
    all_preds, all_labels = [], []
    
    for batch_x, batch_y in tqdm(dataloader, desc="Training", leave=False):
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        if is_multimodal:
            # batch_x splits: rgb(3), depth(1), thermal(1)
            # Assuming channel order is RGB(3), D(1), T(1)
            rgb = batch_x[:, 0:3, :, :].to(device)
            depth = batch_x[:, 3:4, :, :].to(device)
            thermal = batch_x[:, 4:5, :, :].to(device)
            logits = model(rgb, depth, thermal)
        else:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        all_preds.append(logits.detach())
        all_labels.append(batch_y.detach())
        
    avg_loss = epoch_loss / len(dataloader)
    
    cat_preds = torch.cat(all_preds, dim=0)
    cat_labels = torch.cat(all_labels, dim=0)
    miou, _ = calculate_iou(cat_preds, cat_labels)
    
    return avg_loss, miou

def evaluate(model, dataloader, criterion, device, is_multimodal=False):
    model.eval()
    epoch_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Evaluating", leave=False):
            batch_y = batch_y.to(device)
            
            if is_multimodal:
                rgb = batch_x[:, 0:3, :, :].to(device)
                depth = batch_x[:, 3:4, :, :].to(device)
                thermal = batch_x[:, 4:5, :, :].to(device)
                logits = model(rgb, depth, thermal)
            else:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                
            loss = criterion(logits, batch_y)
            epoch_loss += loss.item()
            
            all_preds.append(logits)
            all_labels.append(batch_y)
            
    avg_loss = epoch_loss / len(dataloader)
    
    cat_preds = torch.cat(all_preds, dim=0)
    cat_labels = torch.cat(all_labels, dim=0)
    miou, ious = calculate_iou(cat_preds, cat_labels)
    
    return avg_loss, miou, ious
