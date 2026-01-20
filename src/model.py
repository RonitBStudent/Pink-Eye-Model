"""
Pink eye detection model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from utils import load_config


class PinkEyeClassifier(nn.Module):
    def __init__(self, config_path="config.yaml"):
        super(PinkEyeClassifier, self).__init__()
        
        config = load_config(config_path)
        self.num_classes = config['model']['num_classes']
        self.architecture = config['model']['architecture']
        self.dropout_rate = config['model']['dropout_rate']
        
        # Load pre-trained backbone
        self.backbone = timm.create_model(
            self.architecture,
            pretrained=config['model']['pretrained'],
            num_classes=0,  # Remove classifier head
            global_pool='avg'
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 4),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.feature_dim // 2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim // 2, self.num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Classification
        logits = self.classifier(attended_features)
        
        return logits, attention_weights
    
    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ModelEnsemble(nn.Module):
    """Ensemble of multiple models for improved performance"""
    
    def __init__(self, models):
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        outputs = []
        attention_weights = []
        
        for model in self.models:
            logits, attn = model(x)
            outputs.append(logits)
            attention_weights.append(attn)
        
        # Average predictions
        ensemble_output = torch.stack(outputs).mean(dim=0)
        ensemble_attention = torch.stack(attention_weights).mean(dim=0)
        
        return ensemble_output, ensemble_attention


def create_model(config_path="config.yaml"):
    """Factory function to create model"""
    model = PinkEyeClassifier(config_path)
    return model


def get_model_summary(model, input_size=(3, 224, 224)):
    """Print model summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Architecture: {model.architecture}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input size: {input_size}")
    print(f"Number of classes: {model.num_classes}")
    print("=" * 50)
    
    # Test forward pass
    dummy_input = torch.randn(1, *input_size)
    with torch.no_grad():
        output, attention = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Attention shape: {attention.shape}")


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    get_model_summary(model)
