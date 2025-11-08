import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN_HighRes(nn.Module):
    """
    SimpleCNN Enhanced untuk High Resolution (128x128)
    
    Arsitektur yang lebih dalam untuk memanfaatkan resolusi tinggi:
    - 4 Conv layers (vs 3 di SimpleCNN original)
    - Lebih banyak filters untuk capture detail
    - Spatial Pyramid Pooling untuk multi-scale features
    """
    
    def __init__(self, in_channels=1, num_classes=1, dropout_rate=0.4, input_size=128):
        super(SimpleCNN_HighRes, self).__init__()
        
        self.input_size = input_size
        
        # --- Convolutional Layers ---
        # Conv1: 128x128 -> 64x64
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.25)
        
        # Conv2: 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=0.25)
        
        # Conv3: 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=0.3)
        
        # Conv4: 16x16 -> 8x8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Fully Connected Layers ---
        # After 4 pooling layers: 128 -> 64 -> 32 -> 16 -> 8
        # Feature map: 256 channels * 8 * 8 = 16,384
        fc_input_size = 256 * 8 * 8
        
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout_fc2 = nn.Dropout(p=dropout_rate)
        
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC Block 1
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)
        
        # FC Block 2
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout_fc2(x)
        
        # Output
        x = self.fc3(x)
        
        return x


class AttentionCNN_HighRes(nn.Module):
    """
    CNN dengan Channel Attention Mechanism untuk High Resolution
    
    Menggunakan Squeeze-and-Excitation (SE) blocks untuk:
    - Fokus pada channel yang paling informatif
    - Meningkatkan representational power
    """
    
    def __init__(self, in_channels=1, num_classes=1, dropout_rate=0.4):
        super(AttentionCNN_HighRes, self).__init__()
        
        # Conv blocks dengan SE attention
        self.conv1 = self._make_conv_block(in_channels, 32, se_ratio=16)
        self.conv2 = self._make_conv_block(32, 64, se_ratio=16)
        self.conv3 = self._make_conv_block(64, 128, se_ratio=16)
        self.conv4 = self._make_conv_block(128, 256, se_ratio=16)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # FC layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        
    def _make_conv_block(self, in_ch, out_ch, se_ratio=16):
        """Create conv block with SE attention"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SEBlock(out_ch, se_ratio),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Squeeze
        y = self.squeeze(x).view(b, c)
        
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        
        # Scale
        return x * y.expand_as(x)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("HIGH RESOLUTION MODEL TEST")
    print("=" * 60)
    
    # Test SimpleCNN_HighRes
    model1 = SimpleCNN_HighRes(in_channels=1, num_classes=1, dropout_rate=0.4, input_size=128)
    print(f"\n1️⃣ SimpleCNN_HighRes")
    print(f"   Total parameters: {count_parameters(model1):,}")
    
    # Test forward pass
    x = torch.randn(4, 1, 128, 128)  # Batch of 4 images
    out1 = model1(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out1.shape}")
    print(f"   ✅ Forward pass successful!")
    
    # Test AttentionCNN_HighRes
    print(f"\n2️⃣ AttentionCNN_HighRes (with SE blocks)")
    model2 = AttentionCNN_HighRes(in_channels=1, num_classes=1, dropout_rate=0.4)
    print(f"   Total parameters: {count_parameters(model2):,}")
    
    out2 = model2(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out2.shape}")
    print(f"   ✅ Forward pass successful!")
    
    print("\n" + "=" * 60)
    print("COMPARISON:")
    print("=" * 60)
    print(f"SimpleCNN_HighRes:    {count_parameters(model1):>10,} params")
    print(f"AttentionCNN_HighRes: {count_parameters(model2):>10,} params")
    print("=" * 60)
