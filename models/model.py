# model.py

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, dropout_rate=0.3, input_size=128):
        super().__init__()
        # Convolutional layers dengan Batch Normalization
        # Input: 128x128 untuk better feature extraction
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2)  # 128x128 → 128x128
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)           # 64x64 → 64x64
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)           # 32x32 → 32x32
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)                                               # Pooling 2x2
        self.dropout_conv = nn.Dropout2d(p=0.25)                                     # Dropout untuk conv layers
        
        # Calculate FC input size based on input_size
        # After 3 pooling layers: input_size / (2^3) = input_size / 8
        fc_input_size = 64 * (input_size // 8) * (input_size // 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        self.fc3 = nn.Linear(128, 1 if num_classes == 2 else num_classes)

    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)              # (N, 16, 14, 14)
        x = self.dropout_conv(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)              # (N, 32, 7, 7)
        x = self.dropout_conv(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool(x)              # (N, 64, 3, 3)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # FC Block 1
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        # FC Block 2
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        # Output
        x = self.fc3(x)
        return x

# --- Bagian untuk pengujian ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("--- Menguji Model 'SimpleCNN' ---")
    
    model = SimpleCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print("Arsitektur Model:")
    print(model)
    
    dummy_input = torch.randn(64, IN_CHANNELS, 28, 28)
    output = model(dummy_input)
    
    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}")
    print("Pengujian model 'SimpleCNN' berhasil.")