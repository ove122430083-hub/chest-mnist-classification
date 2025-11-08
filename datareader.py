import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from medmnist import ChestMNIST
from PIL import Image, ImageFilter

# --- Konfigurasi Kelas Biner ---
CLASS_A_IDX = 1  # 'Cardiomegaly'
CLASS_B_IDX = 7  # 'Pneumothorax'

# --- UPGRADE: High Resolution (128x128) untuk Better Feature Learning ---
TARGET_RESOLUTION = 128  # Naik dari 28x28 ke 128x128

# --- Medical-Safe Augmentation (Conservative) ---
# Hanya gunakan transformasi yang aman untuk chest X-ray
TRAIN_TRANSFORM = transforms.Compose([
    # 1. Resize ke resolusi tinggi
    transforms.Resize((TARGET_RESOLUTION, TARGET_RESOLUTION), 
                     interpolation=transforms.InterpolationMode.BICUBIC),
    
    # 2. Geometric augmentations (subtle untuk medical images)
    transforms.RandomRotation(degrees=5),  # Sangat kecil untuk menghindari distorsi anatomi
    transforms.RandomAffine(
        degrees=0, 
        translate=(0.05, 0.05),  # Subtle shift
        scale=None,  # No scaling untuk preserve size
    ),
    
    # 3. Intensity augmentations (safe untuk chest X-ray)
    transforms.RandomApply([
        transforms.Lambda(lambda x: x.filter(ImageFilter.SHARPEN))
    ], p=0.3),  # Sharpen untuk enhance edges
    
    transforms.RandomAutocontrast(p=0.3),  # Automatic contrast adjustment
    
    # 4. Convert to tensor and normalize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --- Transform untuk Validation (tanpa augmentation) ---
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((TARGET_RESOLUTION, TARGET_RESOLUTION),
                     interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

NEW_CLASS_NAMES = {0: 'Cardiomegaly', 1: 'Pneumothorax'}
ALL_CLASS_NAMES = [
    'Atelectasis',        # 0
    'Cardiomegaly',       # 1
    'Effusion',           # 2
    'Infiltration',       # 3
    'Mass',               # 4
    'Nodule',             # 5
    'Pneumonia',          # 6
    'Pneumothorax',       # 7
    'Consolidation',      # 8
    'Edema',              # 9
    'Emphysema',          # 10
    'Fibrosis',           # 11
    'Pleural_Thickening', # 12
    'Hernia',             # 13
]

class FilteredBinaryDataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        
        # Muat dataset lengkap
        full_dataset = ChestMNIST(split=split, transform=None, download=True)
        original_labels = full_dataset.labels

        # Cari indeks untuk gambar yang HANYA memiliki satu label yang kita inginkan
        # ChestMNIST uses multi-hot encoding: shape (N, 14) where each column is a class
        indices_a = np.where((original_labels[:, CLASS_A_IDX] == 1) & (original_labels.sum(axis=1) == 1))[0]
        indices_b = np.where((original_labels[:, CLASS_B_IDX] == 1) & (original_labels.sum(axis=1) == 1))[0]

        # Simpan data yang sudah difilter
        self.imgs = []
        self.labels = []

        # Tambahkan data untuk kelas Cardiomegaly (dipetakan ke label 0)
        for idx in indices_a:
            self.imgs.append(full_dataset.imgs[idx])
            self.labels.append(0)

        # Tambahkan data untuk kelas Pneumothorax (dipetakan ke label 1)
        for idx in indices_b:
            self.imgs.append(full_dataset.imgs[idx])
            self.labels.append(1)
        
        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)
        
        print(f"[{split.upper()}] Total samples: {len(self.imgs)}")
        print(f"  - Cardiomegaly: {np.sum(self.labels == 0)}")
        print(f"  - Pneumothorax: {np.sum(self.labels == 1)}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL Image
        img = Image.fromarray(img.squeeze(), mode='L')  # 'L' for grayscale
        
        # Apply transform
        if self.transform:
            img = self.transform(img)
        
        # Return label as tensor with correct shape
        return img, torch.tensor(label, dtype=torch.float32)

def get_data_loaders(batch_size=32, use_augmentation=True):
    """
    Membuat DataLoader untuk training dan validation.
    
    Args:
        batch_size: Ukuran batch
        use_augmentation: Jika True, gunakan augmentasi untuk training
    
    Returns:
        train_loader, val_loader, num_classes (2), in_channels (1)
    """
    
    # Pilih transform berdasarkan use_augmentation
    if use_augmentation:
        train_transform = TRAIN_TRANSFORM
        print("✅ Menggunakan Medical-Safe Augmentation + High Resolution")
    else:
        train_transform = VAL_TRANSFORM
        print("⚠️ Tidak menggunakan augmentasi (only high resolution)")
    
    # Buat dataset
    train_dataset = FilteredBinaryDataset(split='train', transform=train_transform)
    val_dataset = FilteredBinaryDataset(split='val', transform=VAL_TRANSFORM)
    
    # Buat DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 untuk Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    num_classes = 1  # Binary classification (output sigmoid)
    in_channels = 1  # Grayscale images
    
    print(f"\n📊 Dataset Summary:")
    print(f"  - Input Resolution: {TARGET_RESOLUTION}x{TARGET_RESOLUTION}")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Num classes: {num_classes} (binary)")
    print(f"  - In channels: {in_channels} (grayscale)\n")
    
    return train_loader, val_loader, num_classes, in_channels

def calculate_class_weights(train_loader):
    """Menghitung class weights untuk weighted loss function."""
    
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    
    all_labels = np.array(all_labels)
    
    # Hitung jumlah sampel per kelas
    class_0_count = np.sum(all_labels == 0)  # Cardiomegaly
    class_1_count = np.sum(all_labels == 1)  # Pneumothorax
    
    total = len(all_labels)
    
    # Hitung weight (inverse frequency)
    weight_class_0 = total / (2 * class_0_count)
    weight_class_1 = total / (2 * class_1_count)
    
    print(f"Class Weights:")
    print(f"  - Cardiomegaly (0): {weight_class_0:.4f}")
    print(f"  - Pneumothorax (1): {weight_class_1:.4f}")
    
    return torch.tensor([weight_class_0, weight_class_1], dtype=torch.float32)

def visualize_batch(data_loader, num_images=8):
    """Visualisasi batch dari data loader."""
    
    images, labels = next(iter(data_loader))
    
    # Denormalize
    mean = 0.5
    std = 0.5
    images = images * std + mean
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        img = images[i].squeeze().cpu().numpy()
        label = labels[i].item()
        class_name = NEW_CLASS_NAMES[int(label)]
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'{class_name}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_batch_highres.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Batch visualization saved to 'sample_batch_highres.png'")

if __name__ == "__main__":
    print("=" * 60)
    print("HIGH RESOLUTION DATAREADER TEST")
    print("=" * 60)
    
    # Test dengan augmentation
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(
        batch_size=32,
        use_augmentation=True
    )
    
    # Visualize
    print("\n🖼️ Creating visualization...")
    visualize_batch(train_loader, num_images=8)
    
    # Test satu batch
    images, labels = next(iter(train_loader))
    print(f"\n✅ Test successful!")
    print(f"   Image batch shape: {images.shape}")  # Should be [32, 1, 128, 128]
    print(f"   Label batch shape: {labels.shape}")  # Should be [32]
    print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
