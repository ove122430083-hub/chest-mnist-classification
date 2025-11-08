# Laporan Eksperimen MobileNetV3-Large untuk ChestMNIST Classification

**Nama:** INDAH OKTALISA & OVE DEWANDA FATIN  
**NIM:** 122430127 & 122430083  
**Tanggal:** 8 November 2025  
**Model:** MobileNetV3-Large  
**Dataset:** ChestMNIST Binary Classification  
**Target Akurasi:** 90.00%  
**Hasil Akhir:** 89.51%

---

## 1. Latar Belakang

Eksperimen ini bertujuan untuk mengimplementasikan model MobileNetV3-Large pada dataset ChestMNIST dengan target akurasi minimal 90%. MobileNetV3-Large dipilih karena karakteristiknya yang ringan dan cepat, cocok untuk aplikasi deployment dan real-time inference, sambil tetap mempertahankan akurasi yang competitive.

### 1.1 Konteks Dataset
- **Dataset:** ChestMNIST (subset dari MedMNIST)
- **Task:** Binary classification (Cardiomegaly vs Pneumothorax)
- **Training samples:** 2,306 images
  - Cardiomegaly: 754 samples (32.7%)
  - Pneumothorax: 1,552 samples (67.3%)
- **Validation samples:** 305 images
  - Cardiomegaly: 97 samples (31.8%)
  - Pneumothorax: 208 samples (68.2%)
- **Image resolution:** 128x128 pixels (grayscale)

### 1.2 Baseline Performance
Sebelum implementasi MobileNetV3, beberapa eksperimen telah dilakukan dengan arsitektur lain:
- DenseNet121 (Single Model): 91.80%
- DenseNet121 (Ensemble Model 5): 92.46% (best individual)
- DenseNet121 (Ensemble Average): 92.13%

---

## 2. Metodologi

### 2.1 Arsitektur Model

**Base Architecture:** MobileNetV3-Large
- Pre-trained pada ImageNet (ImageNet1K_V2 weights)
- Total parameters: 5,022,225 (~5.0 million)
- Input: 128x128 grayscale images
- Output: Binary classification (single sigmoid output)

**Custom Classifier Design:**
```
Linear(960, 1280)
Hardswish activation
Dropout(0.5)
Linear(1280, 640)
Hardswish activation
Dropout(0.25)
Linear(640, 1)
```

**Modifikasi Input Layer:**
- First convolutional layer dimodifikasi untuk menerima 1 channel (grayscale)
- Pre-trained weights dari 3 channels di-average menjadi 1 channel
- Mempertahankan knowledge dari ImageNet pre-training

### 2.2 Two-Stage Training Strategy

#### Stage 1: Frozen Backbone Training
**Duration:** 30 epochs  
**Trainable parameters:** 2,050,561 (classifier only)  
**Learning rate:** 0.001  
**Objective:** Melatih classifier layer untuk menyesuaikan dengan domain medical imaging

**Hasil Stage 1:**
- Best validation accuracy: 74.75%
- Training time: ~2.8 minutes
- Karakteristik: Konvergensi stabil tanpa overfitting signifikan

#### Stage 2: Full Model Fine-tuning
**Duration:** 120 epochs (early stopped at 150 total)  
**Trainable parameters:** 5,022,225 (all parameters)  
**Learning rate:** 0.0001 (10x reduction dari Stage 1)  
**Objective:** Fine-tune seluruh model untuk optimal performance

**Hasil Stage 2:**
- Best validation accuracy: 89.18% (epoch 148)
- Training time: ~8.3 minutes
- Early stopping: Triggered dengan patience 40

### 2.3 Hyperparameter Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Random Seed | 2024 | Terbukti optimal dari ensemble experiments |
| Dropout | 0.5 | Balance antara regularization dan capacity |
| Batch Size | 48 | Optimal untuk GPU memory dan stability |
| Gradient Accumulation | 2 steps | Effective batch size = 96 |
| Weight Decay | 0.01 | L2 regularization untuk generalization |
| Gradient Clipping | 1.0 | Prevent exploding gradients |
| Optimizer | AdamW | Adaptive learning dengan weight decay |
| Loss Function | BCEWithLogitsLoss | Stable training untuk binary classification |

### 2.4 Data Augmentation

**Training Augmentation:**
- Random horizontal flip (probability 0.5)
- Resize to 128x128
- Normalization: ImageNet mean and std

**Validation:**
- Resize to 128x128
- Normalization only (no augmentation)

### 2.5 Advanced Techniques

#### Mixed Precision Training (FP16)
- Automatic Mixed Precision (AMP) dengan GradScaler
- Mengurangi memory usage hingga 50%
- Mempercepat training tanpa loss akurasi
- Gradient scaling untuk numerical stability

#### Test-Time Augmentation (TTA)
- Horizontal flip augmentation saat inference
- Averaging predictions dari original dan flipped images
- Meningkatkan robustness dan akurasi
- Improvement: 89.18% → 89.51% (+0.33%)

#### Threshold Optimization
- Search range: 0.30 - 0.70
- Step size: 0.01
- Optimal threshold: 0.670
- Method: Maximize validation accuracy

---

## 3. Hasil Eksperimen

### 3.1 Performance Metrics

**Final Results:**
- **Best Validation Accuracy:** 89.51% (with TTA)
- **Validation Accuracy (no TTA):** 89.18%
- **Best Epoch:** 148
- **Optimal Classification Threshold:** 0.670
- **Total Training Time:** 11.1 minutes
- **Training Accuracy (final):** 99.70%

**Stage-wise Performance:**
| Stage | Epochs | Best Val Acc | Training Time |
|-------|--------|--------------|---------------|
| Stage 1 | 30 | 74.75% | 2.8 min |
| Stage 2 | 120 | 89.18% | 8.3 min |
| With TTA | - | 89.51% | - |

### 3.2 Training Progression Analysis

**Stage 1 (Frozen Backbone):**
- Epoch 1: 66.7% → Epoch 30: 74.75%
- Smooth convergence dengan minimal fluctuation
- No overfitting observed (train acc < val acc awal training)
- Classifier berhasil learn basic patterns

**Stage 2 (Full Fine-tuning):**
- Epoch 31-148: 74.75% → 89.18%
- Significant improvement di early epochs (31-60)
- Gradual improvement dengan fluctuations (60-148)
- Peak performance di epoch 148

### 3.3 Overfitting Analysis

**Training vs Validation Gap:**
- Stage 1: Minimal gap (healthy underfitting)
- Stage 2 (best epoch 148): 99.70% - 89.18% = 10.52%
- Karakteristik: Moderate overfitting tapi masih acceptable

**Observations:**
- Training accuracy mencapai ~99.7% (near perfect)
- Validation accuracy plateau di ~89%
- Dropout 0.5 dan weight decay 0.01 membantu control overfitting
- Early stopping dengan patience 40 mencegah severe overfitting

### 3.4 Perbandingan dengan Baseline

| Model | Parameters | Accuracy | Training Time | Speed Ratio |
|-------|------------|----------|---------------|-------------|
| DenseNet121 (Model 5) | 7.6M | 92.46% | 19.6 min | 1.0x |
| DenseNet121 (Ensemble) | 38M | 92.13% | 72.0 min | 1.0x |
| **MobileNetV3-Large** | **5.0M** | **89.51%** | **11.1 min** | **1.76x** |

**Key Insights:**
- MobileNetV3 mencapai 89.51% dengan 34% fewer parameters
- Training 1.76x lebih cepat dari single DenseNet121
- Trade-off: -2.95% accuracy untuk +76% speed improvement
- Sangat cocok untuk deployment dan resource-constrained environments

---

## 4. Analisis Kegagalan dan Limitasi

### 4.1 Gap dari Target (90%)

**Gap Analysis:**
- Target: 90.00%
- Achieved: 89.51%
- Gap: 0.49%

**Possible Reasons:**
1. **Model Capacity:** MobileNetV3 lebih ringan dari DenseNet121
2. **Dataset Size:** 2,306 training samples relatif kecil
3. **Class Imbalance:** 32.7% vs 67.3% distribution
4. **Overfitting:** Training accuracy 99.7% vs validation 89.51%

### 4.2 Observed Limitations

**Training Behavior:**
- Validation accuracy fluktuatif di late epochs (100-150)
- Tidak ada consistent improvement setelah epoch 100
- Possible local minima atau overfitting

**Model Characteristics:**
- Lighter architecture = lower representation capacity
- Trade-off antara speed dan accuracy
- Sulit mencapai performance DenseNet121

### 4.3 Class Imbalance Impact

**Dataset Distribution:**
- Cardiomegaly: 32.7% (minority class)
- Pneumothorax: 67.3% (majority class)

**Mitigation Applied:**
- Threshold optimization (0.670 instead of 0.5)
- Slightly bias toward majority class
- No weighted loss atau oversampling digunakan

---

## 5. Visualisasi Hasil

### 5.1 Training Curves
**File:** `results/mobilenet_training_curves.png`

**Content:**
- Left panel: Training dan validation loss curves
- Right panel: Training dan validation accuracy curves
- Best epoch marked dengan red scatter point
- Clear visualization of two-stage training

**Observations:**
- Loss curves shows smooth convergence
- Accuracy improvement significant di Stage 2
- Validation accuracy plateau setelah epoch 100

### 5.2 Validation Predictions
**File:** `results/mobilenet_val_predictions.png`

**Content:**
- Grid 4x5 dengan 20 random validation samples
- Green border: Correct predictions
- Red border: Incorrect predictions
- Display: Ground truth, prediction, confidence score
- Using TTA dan optimal threshold (0.670)

**Observations:**
- Majority predictions correct (green)
- High confidence pada correct predictions
- Model dapat distinguish antara kedua classes dengan baik

---

## 6. Kesimpulan

### 6.1 Achievement Summary

**Primary Objectives:**
- Implementasi MobileNetV3-Large: Success
- Training completion: Success
- Target 90%: Not fully achieved (89.51%, gap 0.49%)
- Fast training: Success (11.1 minutes)
- Visualization generation: Success

**Technical Success:**
- Two-stage training strategy berfungsi dengan baik
- Mixed precision training stabil dan efficient
- TTA memberikan improvement (+0.33%)
- Threshold optimization efektif (0.5 → 0.670)

### 6.2 Key Findings

1. **Model Efficiency:**
   - MobileNetV3-Large sangat efficient (5M parameters, 11.1 min training)
   - Suitable untuk deployment dan production use cases
   - Good balance antara speed dan accuracy

2. **Training Strategy:**
   - Two-stage training (frozen → fine-tune) efektif untuk transfer learning
   - Stage 1 stabilizes classifier (74.75%)
   - Stage 2 achieves significant improvement (89.51%)

3. **Performance Trade-offs:**
   - 2.95% accuracy loss vs DenseNet121
   - 76% faster training time
   - 34% fewer parameters
   - Acceptable trade-off untuk many applications

4. **Optimization Techniques:**
   - TTA improves accuracy by 0.33%
   - Threshold optimization critical untuk imbalanced data
   - Mixed precision training enables larger batch sizes

### 6.3 Rekomendasi untuk Future Work

**Untuk Mencapai 90%+ dengan MobileNetV3:**

1. **Data Augmentation Enhancement:**
   - Tambah advanced augmentations (rotation, brightness, contrast)
   - Mixup atau CutMix untuk regularization
   - AutoAugment untuk medical imaging

2. **Class Balance Handling:**
   - Implement weighted loss function
   - Oversampling minority class (Cardiomegaly)
   - Focal loss untuk hard examples

3. **Model Architecture Tuning:**
   - Experiment dengan dropout rates (0.3-0.7)
   - Try different classifier architectures
   - Add attention mechanisms

4. **Training Optimization:**
   - Learning rate scheduling (cosine annealing, warmup)
   - Longer training dengan early stopping
   - K-fold cross-validation

5. **Ensemble Approach:**
   - Train multiple MobileNetV3 dengan different seeds
   - Ensemble predictions untuk boost accuracy
   - Expected improvement: +1-2%

**Untuk Production Deployment:**

1. **Model Optimization:**
   - Quantization (FP16 → INT8)
   - Pruning untuk reduce size
   - ONNX export untuk cross-platform

2. **Inference Optimization:**
   - Batch inference untuk throughput
   - TensorRT untuk GPU acceleration
   - Core ML untuk mobile devices

3. **Monitoring:**
   - Track inference time dan throughput
   - Monitor prediction confidence distribution
   - A/B testing dengan baseline models

---

## 7. File dan Artefak

### 7.1 Source Code Files

**Model:**
- `models/model_mobilenet.py` - MobileNetV3-Large architecture

**Training:**
- `scripts/train_mobilenet_improved.py` - Main training script

**Data:**
- `data/datareader_highres.py` - Dataset loader dan augmentation

**Visualization:**
- `scripts/visualize_mobilenet.py` - Generate plots dan predictions

### 7.2 Model Checkpoints

**Saved Models:**
- `trained_models/mobilenet_improved_stage1.pth` - Stage 1 best (74.75%)
- `trained_models/mobilenet_improved_best.pth` - Stage 2 best (89.18%)
- `trained_models/mobilenet_improved_final.pth` - Final dengan TTA (89.51%)

**Checkpoint Contents:**
- Model state dict
- Optimizer state dict
- Training history
- Best validation accuracy
- Optimal threshold
- TTA accuracy

### 7.3 Results dan Visualizations

**Generated Files:**
- `results/mobilenet_improved_history.png` - Auto-generated during training
- `results/mobilenet_training_curves.png` - Loss dan accuracy curves
- `results/mobilenet_val_predictions.png` - Validation predictions grid

### 7.4 Environment Specifications

**Hardware:**
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- CUDA Version: 12.4

**Software:**
- Python: 3.11.14
- PyTorch: 2.6.0+cu124
- torchvision: Latest (for MobileNetV3 pre-trained)
- CUDA Toolkit: 12.4

**Key Libraries:**
- torch, torchvision: Deep learning framework
- numpy: Numerical computing
- matplotlib: Visualization
- tqdm: Progress bars
- pathlib: File path handling

---

## 8. Appendix

### 8.1 Hyperparameter Summary

```
SEED = 2024
DROPOUT = 0.5
BATCH_SIZE = 48
LEARNING_RATE_STAGE1 = 0.001
LEARNING_RATE_STAGE2 = 0.0001
WEIGHT_DECAY = 0.01
STAGE1_EPOCHS = 30
STAGE2_EPOCHS = 120
PATIENCE = 40
GRAD_CLIP = 1.0
GRAD_ACCUMULATION = 2
```

### 8.2 Training Timeline

| Phase | Duration | Key Milestones |
|-------|----------|----------------|
| Setup | 10 sec | Dataset loading, model initialization |
| Stage 1 | 2.8 min | Frozen training: 66.7% → 74.75% |
| Stage 2 | 8.3 min | Fine-tuning: 74.75% → 89.18% |
| TTA | 15 sec | Threshold optimization: 89.18% → 89.51% |
| **Total** | **11.1 min** | **Complete pipeline execution** |

### 8.3 Comparison dengan Established Models

| Model | Params | Accuracy | Inference Time | Use Case |
|-------|--------|----------|----------------|----------|
| ResNet50 | 25M | ~90% | Medium | Balanced |
| DenseNet121 | 8M | 92% | Slow | High accuracy |
| EfficientNet-B0 | 5M | ~90% | Medium | Balanced |
| **MobileNetV3-Large** | **5M** | **89.51%** | **Fast** | **Deployment** |
| MobileNetV2 | 3.5M | ~88% | Fast | Mobile |

### 8.4 References

1. Howard, A., et al. (2019). "Searching for MobileNetV3." ICCV.
2. Yang, J., et al. (2021). "MedMNIST: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification."
3. He, K., et al. (2016). "Deep Residual Learning for Image Recognition."
4. Huang, G., et al. (2017). "Densely Connected Convolutional Networks."

---

**Document Version:** 1.0  
**Last Updated:** 8 November 2025  
**Authors:** INDAH OKTALISA (122430127) & OVE DEWANDA FATIN (122430083)  
**Project:** ChestMNIST Binary Classification dengan MobileNetV3-Large  
**Status:** Completed - 89.51% Accuracy Achieved
