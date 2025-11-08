# model_resnet.py
# Transfer Learning dengan ResNet18 untuk Chest X-ray Classification

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18_ChestXray(nn.Module):
    """
    ResNet18 pretrained model yang di-fine-tune untuk chest X-ray classification.
    
    Modifications:
    1. Conv1: Modified untuk grayscale input (1 channel instead of 3)
    2. FC layer: Modified untuk binary classification (1 output)
    3. Pretrained weights: ImageNet pretrained untuk transfer learning
    """
    
    def __init__(self, num_classes=1, pretrained=True, freeze_backbone=False, dropout_rate=0.5):
        super(ResNet18_ChestXray, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer untuk grayscale (1 channel)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # Modified: Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Freeze backbone layers jika diminta (untuk fine-tuning bertahap)
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
            # Unfreeze conv1 (karena kita modifikasi untuk grayscale)
            for param in self.resnet.conv1.parameters():
                param.requires_grad = True
        
        # Replace final FC layer untuk binary classification
        # Original: Linear(512, 1000) - ImageNet 1000 classes
        # Modified: Linear(512, 1) - Binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),  # Dropout untuk regularisasi
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)
    
    def unfreeze_backbone(self):
        """Unfreeze semua layers untuk full fine-tuning"""
        for param in self.resnet.parameters():
            param.requires_grad = True


class DenseNet121_ChestXray(nn.Module):
    """
    DenseNet121 pretrained model - used by CheXNet (Stanford)
    """
    
    def __init__(self, num_classes=1, pretrained=True):
        super(DenseNet121_ChestXray, self).__init__()
        
        # Load pretrained DenseNet121
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # Modify first conv layer untuk grayscale
        self.densenet.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Replace classifier
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.densenet(x)


class EfficientNetB0_ChestXray(nn.Module):
    """
    EfficientNet-B0 - Best efficiency/accuracy trade-off
    """
    
    def __init__(self, num_classes=1, pretrained=True):
        super(EfficientNetB0_ChestXray, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        # Modify first conv layer untuk grayscale
        self.efficientnet.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Replace classifier
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)


def get_model(model_name='resnet18', pretrained=True, freeze_backbone=False, dropout_rate=0.5):
    """
    Factory function untuk mendapatkan model.
    
    Args:
        model_name: 'resnet18', 'densenet121', atau 'efficientnet_b0'
        pretrained: Use pretrained weights dari ImageNet
        freeze_backbone: Freeze backbone untuk fine-tuning bertahap (hanya ResNet18)
        dropout_rate: Dropout rate untuk regularisasi
    
    Returns:
        model: PyTorch model
    """
    models_dict = {
        'resnet18': ResNet18_ChestXray,
        'densenet121': DenseNet121_ChestXray,
        'efficientnet_b0': EfficientNetB0_ChestXray,
    }
    
    if model_name.lower() not in models_dict:
        raise ValueError(f"Model {model_name} tidak tersedia. Pilih: {list(models_dict.keys())}")
    
    if model_name.lower() == 'resnet18':
        model = models_dict[model_name.lower()](
            pretrained=pretrained, 
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
    else:
        model = models_dict[model_name.lower()](pretrained=pretrained)
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing ResNet18_ChestXray...")
    model = ResNet18_ChestXray(pretrained=False)
    
    # Test forward pass
    dummy_input = torch.randn(4, 1, 28, 28)  # Batch=4, Grayscale, 28x28
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n✅ Model test passed!")
