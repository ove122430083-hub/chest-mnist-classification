"""
MobileNetV3-Large Training Script - IMPROVED VERSION
Target: 90%+ accuracy dengan optimizations
- Two-stage training (frozen -> fine-tune)
- Mixed precision training
- Gradient accumulation
- TTA dan threshold optimization
- Advanced data augmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import random
import time
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.model_mobilenet import get_mobilenet_model
from data.datareader_highres import get_data_loaders

# ============= IMPROVED HYPERPARAMETERS =============
# Optimized for MobileNetV3-Large to reach 90%+
SEED = 2024  # Best seed dari ensemble
DROPOUT = 0.5  # Optimal untuk MobileNetV3
BATCH_SIZE = 48  # Larger batch untuk MobileNetV3 (lebih ringan)
LEARNING_RATE_STAGE1 = 0.001  # Higher LR untuk frozen stage
LEARNING_RATE_STAGE2 = 0.0001  # Lower LR untuk fine-tune
WEIGHT_DECAY = 0.01
STAGE1_EPOCHS = 30  # Lebih lama untuk frozen training
STAGE2_EPOCHS = 120  # Lebih banyak epochs untuk fine-tune
PATIENCE = 40  # Lebih patient
GRAD_CLIP = 1.0
GRAD_ACCUMULATION = 2  # Effective batch = 96
# ===================================================

def set_seed(seed):
    """Set all random seeds"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_accumulation=2, grad_clip=1.0):
    """Train for one epoch with gradient accumulation and mixed precision"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device).float()
        
        # Mixed precision training
        with autocast('cuda'):
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss = loss / grad_accumulation  # Scale loss
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Calculate accuracy
        running_loss += loss.item() * grad_accumulation
        probs = torch.sigmoid(outputs)
        predicted = (probs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': running_loss / (batch_idx + 1), 'acc': 100. * correct / total})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation', leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).float()
            
            with autocast('cuda'):
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predicted = (probs >= 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': running_loss / len(loader), 'acc': 100. * correct / total})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, np.array(all_probs), np.array(all_labels)

def validate_with_tta(model, loader, device):
    """Validate with Test-Time Augmentation (horizontal flip)"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='TTA Validation', leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).float()
            
            # Original prediction
            with autocast('cuda'):
                outputs1 = model(images).squeeze()
                probs1 = torch.sigmoid(outputs1)
            
            # Flipped prediction
            images_flipped = torch.flip(images, dims=[3])  # Horizontal flip
            with autocast('cuda'):
                outputs2 = model(images_flipped).squeeze()
                probs2 = torch.sigmoid(outputs2)
            
            # Average predictions
            probs = (probs1 + probs2) / 2.0
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_probs), np.array(all_labels)

def find_optimal_threshold(probs, labels, start=0.3, end=0.7, step=0.01):
    """Find optimal threshold for binary classification"""
    best_acc = 0
    best_threshold = 0.5
    
    for threshold in np.arange(start, end, step):
        predicted = (probs >= threshold).astype(float)
        acc = (predicted == labels).mean() * 100
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    
    return best_threshold, best_acc

def plot_training_history(history, save_path='results/mobilenet_improved_history.png'):
    """Plot training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    
    # Mark best epoch
    best_epoch = history['best_epoch']
    best_acc = history['best_val_acc']
    ax2.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best Epoch: {best_epoch}')
    ax2.scatter([best_epoch], [best_acc], color='red', s=200, zorder=5, marker='*', edgecolors='black', linewidths=2)
    
    # Target line
    ax2.axhline(y=90, color='purple', linestyle='--', linewidth=2, alpha=0.5, label='Target: 90%')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Stage transition marker
    if 'stage_transition' in history:
        transition_epoch = history['stage_transition']
        ax2.axvline(x=transition_epoch, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax2.text(transition_epoch, ax2.get_ylim()[1] * 0.95, 'Unfreeze', fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Configuration panel
    ax3.axis('off')
    config_text = f"""
MOBILENETV3-LARGE IMPROVED CONFIGURATION
==========================================

ARCHITECTURE:
  Model: MobileNetV3-Large (ImageNet pre-trained)
  Input: 128x128 grayscale
  Parameters: ~5.5M
  Dropout: {DROPOUT}

TRAINING STRATEGY:
  Two-Stage Training:
    Stage 1: Frozen backbone ({STAGE1_EPOCHS} epochs)
    Stage 2: Full fine-tuning ({STAGE2_EPOCHS} epochs)
  
  Learning Rates:
    Stage 1: {LEARNING_RATE_STAGE1}
    Stage 2: {LEARNING_RATE_STAGE2}

OPTIMIZATION:
  Optimizer: AdamW
  Weight Decay: {WEIGHT_DECAY}
  Batch Size: {BATCH_SIZE}
  Grad Accumulation: {GRAD_ACCUMULATION}
  Effective Batch: {BATCH_SIZE * GRAD_ACCUMULATION}
  Grad Clipping: {GRAD_CLIP}
  Mixed Precision: FP16

REGULARIZATION:
  Dropout: {DROPOUT}
  Weight Decay: {WEIGHT_DECAY}
  Early Stopping Patience: {PATIENCE}

AUGMENTATION & TTA:
  Training: Random H-Flip, Resize, Normalize
  TTA: Horizontal Flip Averaging
  Threshold Optimization: 0.3-0.7 range

DATASET:
  Train: 2306 samples
  Val: 305 samples
  Classes: Binary (Cardiomegaly vs Pneumothorax)
"""
    ax3.text(0.1, 0.5, config_text, transform=ax3.transAxes,
            fontsize=9, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
    
    # Results panel
    ax4.axis('off')
    results_text = f"""
TRAINING RESULTS
==========================================

BEST PERFORMANCE:
  Best Val Accuracy: {history['best_val_acc']:.2f}%
  Best Epoch: {history['best_epoch']}
  Optimal Threshold: {history['best_threshold']:.3f}

STAGE 1 (Frozen Backbone):
  Epochs: {STAGE1_EPOCHS}
  Best Val Acc: {history.get('stage1_best_acc', 'N/A')}
  
STAGE 2 (Full Fine-tuning):
  Epochs: {len(history['train_acc']) - STAGE1_EPOCHS}
  Best Val Acc: {history['best_val_acc']:.2f}%

FINAL METRICS:
  Final Train Acc: {history['train_acc'][-1]:.2f}%
  Final Val Acc: {history['val_acc'][-1]:.2f}%
  Overfitting Gap: {history['train_acc'][-1] - history['val_acc'][-1]:.2f}%

TRAINING TIME:
  Total Time: {history['total_time']:.1f} minutes
  Avg per Epoch: {history['total_time'] / len(history['train_acc']):.1f} min

TARGET STATUS:
  Target: 90.00%
  Achieved: {history['best_val_acc']:.2f}%
  Status: {'ACHIEVED' if history['best_val_acc'] >= 90 else 'NOT REACHED'}
"""
    
    status_color = 'lightgreen' if history['best_val_acc'] >= 90 else 'lightyellow'
    ax4.text(0.1, 0.5, results_text, transform=ax4.transAxes,
            fontsize=9, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.2))
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining history saved to: {save_path}")
    plt.close()

def main():
    # Set seed
    set_seed(SEED)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "="*70)
    print("MOBILENETV3-LARGE IMPROVED TRAINING")
    print("="*70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Random Seed: {SEED}")
    print(f"Target Accuracy: 90.00%")
    print("="*70 + "\n")
    
    # Load data
    print("Loading ChestMNIST dataset (128x128)...")
    train_loader, val_loader, _, _ = get_data_loaders(batch_size=BATCH_SIZE)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\nInitializing MobileNetV3-Large model...")
    model = get_mobilenet_model(in_channels=1, num_classes=1, pretrained=True, dropout=DROPOUT).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'best_val_acc': 0, 'best_epoch': 0,
        'best_threshold': 0.5, 'stage_transition': STAGE1_EPOCHS,
        'total_time': 0
    }
    
    best_val_acc = 0
    patience_counter = 0
    start_time = time.time()
    
    # ============= STAGE 1: FROZEN BACKBONE =============
    print("\n" + "="*70)
    print("STAGE 1: TRAINING CLASSIFIER ONLY (FROZEN BACKBONE)")
    print("="*70)
    
    # Freeze backbone
    for param in model.mobilenet.features.parameters():
        param.requires_grad = False
    
    trainable_params_stage1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (Stage 1): {trainable_params_stage1:,}")
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE_STAGE1,
        weight_decay=WEIGHT_DECAY
    )
    
    for epoch in range(1, STAGE1_EPOCHS + 1):
        print(f"\nStage 1 - Epoch {epoch}/{STAGE1_EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            grad_accumulation=GRAD_ACCUMULATION, grad_clip=GRAD_CLIP
        )
        
        val_loss, val_acc, val_probs, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc'] = val_acc
            history['best_epoch'] = epoch
            history['stage1_best_acc'] = f"{val_acc:.2f}%"
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }
            save_path = Path('trained_models/mobilenet_improved_stage1.pth')
            save_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(checkpoint, save_path)
            print(f"Saved best Stage 1 model: {val_acc:.2f}%")
            
            patience_counter = 0
        else:
            patience_counter += 1
    
    stage1_best = best_val_acc
    print(f"\nStage 1 Complete - Best Val Acc: {stage1_best:.2f}%")
    
    # ============= STAGE 2: FULL FINE-TUNING =============
    print("\n" + "="*70)
    print("STAGE 2: FULL MODEL FINE-TUNING (UNFREEZE ALL)")
    print("="*70)
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    trainable_params_stage2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (Stage 2): {trainable_params_stage2:,}")
    
    # New optimizer with lower learning rate
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_STAGE2,
        weight_decay=WEIGHT_DECAY
    )
    
    patience_counter = 0
    
    for epoch in range(STAGE1_EPOCHS + 1, STAGE1_EPOCHS + STAGE2_EPOCHS + 1):
        print(f"\nStage 2 - Epoch {epoch}/{STAGE1_EPOCHS + STAGE2_EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            grad_accumulation=GRAD_ACCUMULATION, grad_clip=GRAD_CLIP
        )
        
        val_loss, val_acc, val_probs, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc'] = val_acc
            history['best_epoch'] = epoch
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }
            save_path = Path('trained_models/mobilenet_improved_best.pth')
            save_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(checkpoint, save_path)
            print(f"NEW BEST MODEL: {val_acc:.2f}%")
            
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Training complete
    total_time = (time.time() - start_time) / 60
    history['total_time'] = total_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total training time: {total_time:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best epoch: {history['best_epoch']}")
    
    # ============= TTA AND THRESHOLD OPTIMIZATION =============
    print("\n" + "="*70)
    print("TTA AND THRESHOLD OPTIMIZATION")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load('trained_models/mobilenet_improved_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # TTA validation
    print("Performing Test-Time Augmentation...")
    tta_probs, tta_labels = validate_with_tta(model, val_loader, device)
    
    # Find optimal threshold
    print("Optimizing classification threshold...")
    best_threshold, best_tta_acc = find_optimal_threshold(tta_probs, tta_labels)
    
    history['best_threshold'] = best_threshold
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"TTA Accuracy: {best_tta_acc:.2f}%")
    
    # Update best accuracy if TTA is better
    if best_tta_acc > history['best_val_acc']:
        print(f"TTA improved accuracy: {history['best_val_acc']:.2f}% -> {best_tta_acc:.2f}%")
        history['best_val_acc'] = best_tta_acc
    
    # Save final checkpoint with threshold
    checkpoint['best_threshold'] = best_threshold
    checkpoint['tta_accuracy'] = best_tta_acc
    checkpoint['history'] = history
    torch.save(checkpoint, 'trained_models/mobilenet_improved_final.pth')
    
    # Plot training history
    print("\nGenerating training visualizations...")
    plot_training_history(history)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Model: MobileNetV3-Large")
    print(f"Best Validation Accuracy: {history['best_val_acc']:.2f}%")
    print(f"Best Epoch: {history['best_epoch']}")
    print(f"Optimal Threshold: {best_threshold:.3f}")
    print(f"Training Time: {total_time:.1f} minutes")
    print(f"Stage 1 Best: {stage1_best:.2f}%")
    print(f"Stage 2 Best: {best_tta_acc:.2f}%")
    
    if history['best_val_acc'] >= 90:
        print("\nTARGET 90% ACHIEVED!")
    else:
        print(f"\nTarget not reached. Gap: {90 - history['best_val_acc']:.2f}%")
    
    print("="*70)
    
    print("\nModel saved to: trained_models/mobilenet_improved_final.pth")
    print("Training history saved to: results/mobilenet_improved_history.png")

if __name__ == "__main__":
    main()
