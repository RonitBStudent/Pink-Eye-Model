"""
Advanced training strategies for pink eye detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import numpy as np
from utils import load_config


class AdvancedTrainer:
    def __init__(self, model, config_path="config.yaml"):
        self.model = model
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def setup_advanced_optimizer(self):
        """Advanced optimizer configurations"""
        
        # Differential learning rates for different parts
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name or 'features' in name:
                backbone_params.append(param)
            else:
                classifier_params.append(param)
        
        # Different learning rates for backbone and classifier
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config['training']['learning_rate'] * 0.1},
            {'params': classifier_params, 'lr': self.config['training']['learning_rate']}
        ], weight_decay=self.config['training']['weight_decay'])
        
        return optimizer
    
    def setup_advanced_scheduler(self, optimizer, train_loader):
        """Advanced learning rate scheduling"""
        
        # One Cycle Learning Rate
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config['training']['learning_rate'],
            epochs=self.config['training']['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # 30% of training for warmup
            anneal_strategy='cos'
        )
        
        return scheduler
    
    def setup_cosine_restarts(self, optimizer):
        """Cosine annealing with warm restarts"""
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Initial restart period
            T_mult=2,  # Multiply restart period by this factor
            eta_min=1e-6
        )


class LabelSmoothingLoss(nn.Module):
    """Label smoothing for better generalization"""
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class FocalLossAdvanced(nn.Module):
    """Advanced Focal Loss with class balancing"""
    
    def __init__(self, alpha=None, gamma=2, num_classes=2, size_average=True):
        super(FocalLossAdvanced, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.size_average = size_average
        
        if isinstance(alpha, (float, int)):
            self.alpha = torch.ones(num_classes)
            self.alpha = self.alpha * alpha
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt = -ce_loss
            focal_loss = at * (1 - pt) ** self.gamma * logpt
        else:
            focal_loss = (1 - pt) ** self.gamma * (-ce_loss)
        
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class MixUpTrainer:
    """MixUp training implementation"""
    
    @staticmethod
    def mixup_data(x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    @staticmethod
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class GradualUnfreezing:
    """Gradual unfreezing strategy for transfer learning"""
    
    def __init__(self, model, total_epochs):
        self.model = model
        self.total_epochs = total_epochs
        self.layers = list(model.children())
        
    def unfreeze_layers(self, epoch):
        """Gradually unfreeze layers during training"""
        unfreeze_at = self.total_epochs // 3  # Start unfreezing at 1/3 of training
        
        if epoch < unfreeze_at:
            # Keep backbone frozen
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        else:
            # Gradually unfreeze from top layers
            layers_to_unfreeze = min(len(self.layers), 
                                   int((epoch - unfreeze_at) / (self.total_epochs - unfreeze_at) * len(self.layers)))
            
            for i, layer in enumerate(reversed(self.layers)):
                if i < layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True


class AdvancedLossFunction(nn.Module):
    """Combined loss function for better training"""
    
    def __init__(self, num_classes=2, focal_alpha=0.25, focal_gamma=2, 
                 label_smoothing=0.1, loss_weights=None):
        super(AdvancedLossFunction, self).__init__()
        
        self.focal_loss = FocalLossAdvanced(alpha=focal_alpha, gamma=focal_gamma, num_classes=num_classes)
        self.label_smooth_loss = LabelSmoothingLoss(num_classes, smoothing=label_smoothing)
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Default weights for combining losses
        if loss_weights is None:
            self.loss_weights = {'focal': 0.5, 'smooth': 0.3, 'ce': 0.2}
        else:
            self.loss_weights = loss_weights
    
    def forward(self, predictions, targets):
        focal = self.focal_loss(predictions, targets)
        smooth = self.label_smooth_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)
        
        total_loss = (self.loss_weights['focal'] * focal + 
                     self.loss_weights['smooth'] * smooth + 
                     self.loss_weights['ce'] * ce)
        
        return total_loss


class ModelEMA:
    """Exponential Moving Average of model weights"""
    
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# Training configuration improvements
ADVANCED_CONFIG = {
    "training_strategies": {
        "use_mixup": True,
        "mixup_alpha": 0.2,
        "use_cutmix": True,
        "cutmix_alpha": 1.0,
        "use_label_smoothing": True,
        "label_smoothing_factor": 0.1,
        "use_model_ema": True,
        "ema_decay": 0.9999,
        "gradual_unfreezing": True,
        "differential_lr": True,
        "backbone_lr_factor": 0.1
    },
    
    "advanced_augmentation": {
        "use_advanced_aug": True,
        "aug_probability": 0.8,
        "medical_specific_aug": True
    },
    
    "loss_function": {
        "use_combined_loss": True,
        "focal_weight": 0.5,
        "smooth_weight": 0.3,
        "ce_weight": 0.2
    }
}
