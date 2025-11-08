"""
Ensemble Model untuk ChestMNIST Binary Classification
Menggabungkan SimpleCNN Enhanced + ResNet18 untuk performa maksimal
"""

import torch
import torch.nn as nn
from model import SimpleCNN
from model_resnet import get_model

class EnsembleModel(nn.Module):
    def __init__(self, input_size=28, device='cuda'):
        super(EnsembleModel, self).__init__()
        self.input_size = input_size
        self.device = device
        
        # Model 1: SimpleCNN Enhanced (binary classification)
        self.model_simple = SimpleCNN(
            in_channels=1, 
            num_classes=1,  # Binary classification
            dropout_rate=0.3, 
            input_size=input_size
        )
        
        # Model 2: ResNet18
        self.model_resnet = get_model(
            model_name='resnet18',
            pretrained=True,
            freeze_backbone=False,
            dropout_rate=0.5
        )
        
        # Ensemble weights (learnable)
        self.weight_simple = nn.Parameter(torch.tensor(0.6))  # SimpleCNN lebih baik
        self.weight_resnet = nn.Parameter(torch.tensor(0.4))  # ResNet kontribusi lebih kecil
        
    def forward(self, x):
        """
        Forward pass dengan weighted averaging
        """
        # Pastikan input sesuai ukuran
        if x.size(-1) != self.input_size:
            x_resized = torch.nn.functional.interpolate(
                x, size=(self.input_size, self.input_size), 
                mode='bilinear', align_corners=False
            )
        else:
            x_resized = x
        
        # Prediksi dari kedua model
        pred_simple = self.model_simple(x_resized)
        pred_resnet = self.model_resnet(x_resized)
        
        # Weighted averaging dengan softmax weights
        weights = torch.softmax(torch.stack([self.weight_simple, self.weight_resnet]), dim=0)
        ensemble_pred = weights[0] * pred_simple + weights[1] * pred_resnet
        
        return ensemble_pred
    
    def load_pretrained_models(self, simple_path, resnet_path):
        """
        Load model yang sudah di-train sebelumnya
        """
        # Load SimpleCNN
        if simple_path and os.path.exists(simple_path):
            checkpoint = torch.load(simple_path, map_location=self.device)
            # Handle checkpoint yang menyimpan dictionary dengan 'model_state_dict'
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model_simple.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model_simple.load_state_dict(checkpoint)
            print(f"✓ SimpleCNN loaded from {simple_path}")
        else:
            print(f"⚠️ SimpleCNN path not found: {simple_path}, using random init")
        
        # Load ResNet18
        if resnet_path and os.path.exists(resnet_path):
            checkpoint = torch.load(resnet_path, map_location=self.device)
            # Handle checkpoint yang menyimpan dictionary dengan 'model_state_dict'
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model_resnet.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model_resnet.load_state_dict(checkpoint)
            print(f"✓ ResNet18 loaded from {resnet_path}")
        else:
            print(f"⚠️ ResNet18 path not found: {resnet_path}, using random init")
    
    def freeze_base_models(self):
        """
        Freeze kedua model untuk hanya train ensemble weights
        """
        for param in self.model_simple.parameters():
            param.requires_grad = False
        for param in self.model_resnet.parameters():
            param.requires_grad = False
        print("✓ Base models frozen, only ensemble weights trainable")
    
    def unfreeze_all(self):
        """
        Unfreeze semua untuk fine-tuning
        """
        for param in self.model_simple.parameters():
            param.requires_grad = True
        for param in self.model_resnet.parameters():
            param.requires_grad = True
        print("✓ All parameters unfrozen for fine-tuning")


class VotingEnsemble:
    """
    Hard voting ensemble untuk prediksi biner
    """
    def __init__(self, models, device='cuda'):
        self.models = models
        self.device = device
        
    def predict(self, x):
        """
        Hard voting: mayoritas menang
        """
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append((pred > 0.5).float())
        
        # Voting
        votes = torch.stack(predictions).sum(dim=0)
        final_pred = (votes > len(self.models) / 2).float()
        
        return final_pred


class AveragingEnsemble:
    """
    Soft voting ensemble dengan probability averaging
    """
    def __init__(self, models, weights=None, device='cuda'):
        self.models = models
        self.device = device
        
        # Default equal weights
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
    
    def predict(self, x):
        """
        Weighted averaging dari probabilities
        """
        predictions = []
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                pred = torch.sigmoid(model(x))
                predictions.append(pred * weight)
        
        # Weighted average
        final_pred = torch.stack(predictions).sum(dim=0)
        
        return final_pred


def create_ensemble_from_checkpoints(simple_path, resnet_path, input_size=28, device='cuda'):
    """
    Helper function untuk create ensemble dari checkpoint files
    """
    ensemble = EnsembleModel(input_size=input_size, device=device)
    ensemble.load_pretrained_models(simple_path, resnet_path)
    ensemble.to(device)
    
    return ensemble


import os
