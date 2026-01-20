"""
Advanced model architectures for pink eye detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models import efficientnet_b0, efficientnet_b3, efficientnet_b7
from utils import load_config


class MultiScaleEfficientNet(nn.Module):
    """Multi-scale feature extraction with EfficientNet"""
    
    def __init__(self, config_path="config.yaml"):
        super(MultiScaleEfficientNet, self).__init__()
        
        config = load_config(config_path)
        self.num_classes = config['model']['num_classes']
        
        # Multiple EfficientNet scales
        self.backbone_b0 = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.backbone_b3 = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        
        # Feature dimensions
        self.feat_dim_b0 = self.backbone_b0.num_features
        self.feat_dim_b3 = self.backbone_b3.num_features
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_dim_b0 + self.feat_dim_b3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classifier
        self.classifier = nn.Linear(256, self.num_classes)
    
    def forward(self, x):
        # Multi-scale feature extraction
        feat_b0 = self.backbone_b0(x)
        feat_b3 = self.backbone_b3(x)
        
        # Concatenate features
        combined_feat = torch.cat([feat_b0, feat_b3], dim=1)
        
        # Fusion and classification
        fused_feat = self.fusion(combined_feat)
        logits = self.classifier(fused_feat)
        
        return logits


class AttentionEfficientNet(nn.Module):
    """EfficientNet with advanced attention mechanisms"""
    
    def __init__(self, config_path="config.yaml"):
        super(AttentionEfficientNet, self).__init__()
        
        config = load_config(config_path)
        self.num_classes = config['model']['num_classes']
        
        # Backbone
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.feature_dim = self.backbone.num_features
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim, 
            num_heads=8, 
            dropout=0.1
        )
        
        # Spatial attention
        self.spatial_attention = SpatialAttention()
        
        # Channel attention
        self.channel_attention = ChannelAttention(self.feature_dim)
        
        # Classifier with residual connection
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.feature_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim // 2, self.num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply channel attention
        features = self.channel_attention(features.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        
        # Multi-head self-attention
        features_attn = features.unsqueeze(1)  # Add sequence dimension
        attn_output, _ = self.multihead_attn(features_attn, features_attn, features_attn)
        features = features + attn_output.squeeze(1)  # Residual connection
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention module"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pooling
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply attention
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention.expand_as(x)


class EnsembleModel(nn.Module):
    """Ensemble of different architectures"""
    
    def __init__(self, config_path="config.yaml"):
        super(EnsembleModel, self).__init__()
        
        config = load_config(config_path)
        self.num_classes = config['model']['num_classes']
        
        # Different architectures
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=self.num_classes)
        self.resnet = timm.create_model('resnet50', pretrained=True, num_classes=self.num_classes)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=self.num_classes)
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x):
        # Get predictions from each model
        pred_eff = self.efficientnet(x)
        pred_res = self.resnet(x)
        pred_vit = self.vit(x)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_pred = (weights[0] * pred_eff + 
                        weights[1] * pred_res + 
                        weights[2] * pred_vit)
        
        return ensemble_pred


class VisionTransformerCustom(nn.Module):
    """Custom Vision Transformer for medical imaging"""
    
    def __init__(self, config_path="config.yaml"):
        super(VisionTransformerCustom, self).__init__()
        
        config = load_config(config_path)
        self.num_classes = config['model']['num_classes']
        
        # Vision Transformer backbone
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        # Medical-specific attention layers
        self.medical_attention = nn.MultiheadAttention(
            embed_dim=768,  # ViT base embedding dimension
            num_heads=12,
            dropout=0.1
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.3),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(384, self.num_classes)
        )
    
    def forward(self, x):
        # Extract ViT features
        features = self.vit(x)
        
        # Apply medical attention
        features_attn = features.unsqueeze(1)
        attn_output, attn_weights = self.medical_attention(features_attn, features_attn, features_attn)
        features = features + attn_output.squeeze(1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits, attn_weights


def create_advanced_model(model_type="attention", config_path="config.yaml"):
    """Factory function for advanced models"""
    
    if model_type == "multiscale":
        return MultiScaleEfficientNet(config_path)
    elif model_type == "attention":
        return AttentionEfficientNet(config_path)
    elif model_type == "ensemble":
        return EnsembleModel(config_path)
    elif model_type == "vit":
        return VisionTransformerCustom(config_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Test different architectures
if __name__ == "__main__":
    # Test model creation
    models = {
        "multiscale": create_advanced_model("multiscale"),
        "attention": create_advanced_model("attention"),
        "vit": create_advanced_model("vit")
    }
    
    for name, model in models.items():
        print(f"{name} model parameters: {sum(p.numel() for p in model.parameters()):,}")
