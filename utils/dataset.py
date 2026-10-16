import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
import json

# Pre-defined transforms
def get_transforms(is_train=True):
    geometric_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    ], additional_targets={'depth': 'image', 'thermal': 'image'}) if is_train else A.Compose([], additional_targets={'depth': 'image', 'thermal': 'image'})

    rgb_transform = A.Compose([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    ]) if is_train else A.Compose([])

    depth_transform = A.Compose([
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.2),
    ]) if is_train else A.Compose([])

    thermal_transform = A.Compose([
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
    ]) if is_train else A.Compose([])
    
    return geometric_transform, rgb_transform, depth_transform, thermal_transform


class MultimodalTerrainDataset(Dataset):
    def __init__(self, df, modalities=['rgb', 'depth', 'thermal'], is_train=True):
        self.df = df.reset_index(drop=True)
        self.modalities = modalities
        self.is_train = is_train
        self.geo_tfm, self.rgb_tfm, self.depth_tfm, self.therm_tfm = get_transforms(is_train)
        
    def __len__(self):
        return len(self.df)
        
    def preprocess_depth(self, depth):
        depth = np.nan_to_num(depth, nan=0.0)
        depth = np.clip(depth, 0, 255.0) # Using 255 if raw 8-bit, adjust if metric
        depth = depth / 255.0
        return depth
        
    def preprocess_thermal(self, thermal):
        t_min, t_max = np.min(thermal), np.max(thermal)
        if t_max > t_min:
            thermal = (thermal - t_min) / (t_max - t_min)
        else:
            thermal = np.zeros_like(thermal)
        return thermal

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        rgb = cv2.imread(row['rgb_path'])
        if rgb is None: rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        else: rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None: mask = np.zeros((720, 1280), dtype=np.uint8)
            
        depth = cv2.imread(row['depth_path'], cv2.IMREAD_GRAYSCALE)
        if depth is None: depth = np.zeros((720, 1280), dtype=np.uint8)
            
        thermal = cv2.imread(row['thermal_path'], cv2.IMREAD_GRAYSCALE)
        if thermal is None: thermal = np.zeros((720, 1280), dtype=np.uint8)
        
        # 0. Spatial Registration (Resize to RGB shape)
        target_shape = (rgb.shape[1], rgb.shape[0]) # (W, H)
        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, target_shape, interpolation=cv2.INTER_NEAREST)
        if thermal.shape[:2] != rgb.shape[:2]:
            thermal = cv2.resize(thermal, target_shape, interpolation=cv2.INTER_NEAREST)
        if mask.shape[:2] != rgb.shape[:2]:
            mask = cv2.resize(mask, target_shape, interpolation=cv2.INTER_NEAREST)
            
        # 1. Geo transforms
        geo_aug = self.geo_tfm(image=rgb, depth=depth, thermal=thermal, mask=mask)
        rgb = geo_aug['image']
        depth = geo_aug['depth']
        thermal = geo_aug['thermal']
        mask = geo_aug['mask']
        
        # 2. Photometric transforms
        if self.is_train:
            rgb = self.rgb_tfm(image=rgb)['image']
            depth = self.depth_tfm(image=depth)['image']
            thermal = self.therm_tfm(image=thermal)['image']
            
        # 3. Normalization
        rgb = rgb.astype(np.float32) / 255.0
        depth = self.preprocess_depth(depth.astype(np.float32))
        thermal = self.preprocess_thermal(thermal.astype(np.float32))
        
        # Build tensor
        channels = []
        if 'rgb' in self.modalities:
            channels.append(rgb.transpose(2, 0, 1))
        if 'depth' in self.modalities:
            channels.append(np.expand_dims(depth, axis=0))
        if 'thermal' in self.modalities:
            channels.append(np.expand_dims(thermal, axis=0))
            
        x = np.concatenate(channels, axis=0) if channels else np.zeros((1, rgb.shape[0], rgb.shape[1]))
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)
