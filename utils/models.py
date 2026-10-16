import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def get_unet(in_channels, num_classes=8):
    # If using timm-efficientnet or resnet, weights can only be loaded if in_channels == 3
    # For robust multimodal baseline, we can use a randomly initialized encoder or load weights and patch the first conv layer.
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if in_channels == 3 else None,
        in_channels=in_channels,
        classes=num_classes,
    )
    return model

class ModalityEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.conv(x)

class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Simple SE-like attention or channel attention
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels * 3, 1),
            nn.Sigmoid()
        )
    def forward(self, rgb, depth, thermal):
        concat = torch.cat([rgb, depth, thermal], dim=1)
        weights = self.attention(concat)
        fused = concat * weights
        return fused

class UAMTSeg(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.rgb_enc = ModalityEncoder(3, 64)
        self.depth_enc = ModalityEncoder(1, 64)
        self.thermal_enc = ModalityEncoder(1, 64)
        
        self.fusion = AttentionFusion(64)
        
        # MC Dropout for uncertainty (always active if configured for MC inference)
        self.dropout = nn.Dropout2d(p=0.5)
        
        # Simple Decoder (In practice, a full UNet decoder with skip connections would be used)
        # We simulate a lightweight one here to keep parameters low
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64*3, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
    def forward(self, rgb, depth, thermal):
        feat_rgb = self.rgb_enc(rgb)
        feat_depth = self.depth_enc(depth)
        feat_thermal = self.thermal_enc(thermal)
        
        fused = self.fusion(feat_rgb, feat_depth, feat_thermal)
        fused = self.dropout(fused) 
        
        out = self.decoder(fused)
        
        # Upsample if needed to match input resolution (720, 1280)
        out = nn.functional.interpolate(out, size=(rgb.shape[2], rgb.shape[3]), mode='bilinear', align_corners=False)
        return out
