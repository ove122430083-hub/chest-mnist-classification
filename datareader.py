
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from medmnist import ChestMNIST

# --- Konfigurasi Kelas Biner ---
CLASS_A_IDX = 1  # 'Cardiomegaly'
CLASS_B_IDX = 7 # 'Pneumothorax'

# --- Data Augmentation untuk Training (Medical-Safe) ---
# NOTE: Augmentation MENURUNKAN performa untuk dataset ini!
# Best practice: TIDAK menggunakan augmentation
TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalisasi
])

# --- Transform untuk Validation (tanpa augmentation) ---
VAL_TRANSFORM = transforms.Compose([
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
        indices_a = np.where((original_labels[:, CLASS_A_IDX] == 1) & (original_labels.sum(axis=1) == 1))[0]
        indices_b = np.where((original_labels[:, CLASS_B_IDX] == 1) & (original_labels.sum(axis=1) == 1))[0]

        # Simpan gambar dan label yang sudah dipetakan ulang
        self.images = []
        self.labels = []

        # Tambahkan data untuk kelas Cardiomegaly (dipetakan ke label 0)
        for idx in indices_a:
            self.images.append(full_dataset[idx][0])
            self.labels.append(0)

        # Tambahkan data untuk kelas Pneumothorax (dipetakan ke label 1)
        for idx in indices_b:
            self.images.append(full_dataset[idx][0])
            self.labels.append(1)
        
        print(f"Split: {split}")
        print(f"Jumlah Cardiomegaly (label 0): {len(indices_a)}")
        print(f"Jumlah Pneumothorax (label 1): {len(indices_b)}")
        print()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor([label])

def get_data_loaders(batch_size, use_augmentation=True):
    """
    Get data loaders with optional data augmentation for training.
    
    Args:
        batch_size: Batch size for dataloaders
        use_augmentation: If True, apply data augmentation to training data
    
    Returns:
        train_loader, val_loader, num_classes, in_channels
    """
    # Use augmentation for training if enabled
    if use_augmentation:
        train_transform = TRAIN_TRANSFORM
        print("✅ Data Augmentation AKTIF untuk training data")
    else:
        train_transform = VAL_TRANSFORM
        print("⚠️  Data Augmentation TIDAK AKTIF")
    
    # Validation always uses basic transform (no augmentation)
    val_transform = VAL_TRANSFORM

    train_dataset = FilteredBinaryDataset('train', train_transform)
    val_dataset = FilteredBinaryDataset('test', val_transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    n_classes = 2
    n_channels = 1
    
    print("Dataset ChestMNIST berhasil difilter untuk klasifikasi biner!")
    print(f"Kelas yang digunakan: {NEW_CLASS_NAMES[0]} (Label 0) dan {NEW_CLASS_NAMES[1]} (Label 1)")
    print(f"Jumlah data training: {len(train_dataset)}")
    print(f"Jumlah data validasi: {len(val_dataset)}")
    
    return train_loader, val_loader, n_classes, n_channels

def show_samples(dataset):
    cardiomegaly_imgs = []
    pneumothorax_imgs = []
    
    for img, label in dataset:
        if label.item() == 0 and len(cardiomegaly_imgs) < 5:
            cardiomegaly_imgs.append(img)
        elif label.item() == 1 and len(pneumothorax_imgs) < 5:
            pneumothorax_imgs.append(img)
        
        if len(cardiomegaly_imgs) == 5 and len(pneumothorax_imgs) == 5:
            break
            
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Perbandingan Gambar: Cardiomegaly (atas) vs Pneumothorax (bawah)", fontsize=16)
    
    for i, img in enumerate(cardiomegaly_imgs):
        ax = axes[0, i]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f"Cardiomegaly #{i+1}")
        ax.axis('off')
        
    for i, img in enumerate(pneumothorax_imgs):
        ax = axes[1, i]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f"Pneumothorax #{i+1}")
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def show_class_distribution(split='train'):
    full_dataset = ChestMNIST(split=split, transform=None, download=True)
    original_labels = full_dataset.labels
    
    class_counts = {}
    for idx, class_name in enumerate(ALL_CLASS_NAMES):
        count = np.where((original_labels[:, idx] == 1) & (original_labels.sum(axis=1) == 1))[0].shape[0]
        class_counts[class_name] = count
    
    # Urutkan berdasarkan jumlah (descending)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"Distribusi Kelas di ChestMNIST ({split.upper()} set)")
    print(f"(Hanya single-label samples)")
    print(f"{'='*60}")
    print(f"{'No':<4} {'Kelas':<25} {'Jumlah Sampel':<15}")
    print(f"{'-'*60}")
    
    total_samples = 0
    for i, (class_name, count) in enumerate(sorted_classes, 1):
        print(f"{i:<4} {class_name:<25} {count:<15}")
        total_samples += count
    
    print(f"{'-'*60}")
    print(f"{'TOTAL':<29} {total_samples:<15}")
    print(f"{'='*60}\n")
    
    return sorted_classes

if __name__ == '__main__':
    print("Memuat dataset untuk plotting...")
    
    # Tampilkan distribusi kelas
    print("\n--- Distribusi Kelas Training Set ---")
    show_class_distribution('train')
    
    print("\n--- Distribusi Kelas Test Set ---")
    show_class_distribution('test')
    
    plot_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = FilteredBinaryDataset('train', transform=plot_transform)
    
    if len(train_dataset) > 0:
        print("\n--- Menampilkan 5 Contoh Gambar per Kelas ---")
        show_samples(train_dataset)
    else:
        print("Dataset tidak berisi sampel untuk kelas yang dipilih.")
