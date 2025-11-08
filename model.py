"""
MobileNetV3-Large Model for ChestMNIST Binary Classification
Lightweight and fast architecture optimized for speed and accuracy
"""

import torch
import torch.nn as nn
from torchvision import models


class MobileNetV3ChestMNIST(nn.Module):
    """
    MobileNetV3-Large dengan custom classifier untuk binary classification
    Optimized untuk ChestMNIST dataset
    """
    
    def __init__(self, in_channels=1, num_classes=1, pretrained=True, dropout=0.4):
        super(MobileNetV3ChestMNIST, self).__init__()
        
        # Load pre-trained MobileNetV3-Large
        self.mobilenet = models.mobilenet_v3_large(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Modify first conv layer untuk grayscale input (1 channel)
        if in_channels == 1:
            original_conv = self.mobilenet.features[0][0]
            self.mobilenet.features[0][0] = nn.Conv2d(
                in_channels, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # Transfer weights dari 3 channels ke 1 channel (average across channels)
            if pretrained:
                with torch.no_grad():
                    self.mobilenet.features[0][0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        
        # Get feature dimension dari classifier
        in_features = self.mobilenet.classifier[0].in_features
        
        # Custom classifier untuk binary classification dengan regularization
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(),  # MobileNetV3 uses Hardswish
            nn.Dropout(p=dropout),
            nn.Linear(1280, 640),
            nn.Hardswish(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(640, num_classes)
        )
    
    def forward(self, x):
        return self.mobilenet(x)


def get_mobilenet_model(in_channels=1, num_classes=1, pretrained=True, dropout=0.4):
    """
    Factory function untuk create MobileNetV3 model
    
    Args:
        in_channels (int): Number of input channels (1 for grayscale)
        num_classes (int): Number of output classes (1 for binary)
        pretrained (bool): Use ImageNet pre-trained weights
        dropout (float): Dropout rate
    
    Returns:
        model: MobileNetV3ChestMNIST model
    """
    model = MobileNetV3ChestMNIST(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    return model


if __name__ == "__main__":
    # Test model
    print("Testing MobileNetV3-Large model...")
    model = get_mobilenet_model(in_channels=1, num_classes=1, pretrained=True, dropout=0.4)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 128, 128)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test successful!")
