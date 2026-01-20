"""
Training script for pink eye detection model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from model import create_model, FocalLoss
from utils import (load_config, save_model_checkpoint, plot_training_history, 
                  calculate_class_weights, create_directories)


class EyeDataset(Dataset):
    """Custom dataset for eye images"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load class mapping
        config = load_config()
        self.class_to_idx = {v: k for k, v in config['classes'].items()}
        
        # Collect all image paths and labels
        for class_name in self.class_to_idx.keys():
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_file))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class Trainer:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        create_directories(self.config)
        
        # Initialize model
        self.model = create_model(config_path).to(self.device)
        
        # Setup data transforms
        self.setup_transforms()
        
        # Setup data loaders
        self.setup_data_loaders()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Setup loss function
        self.setup_loss_function()
        
        # Setup logging
        self.writer = SummaryWriter(self.config['paths']['logs_dir'])
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_auc': [], 'val_auc': []
        }
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def setup_transforms(self):
        """Setup data transforms"""
        # Training transforms (with augmentation)
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transforms (no augmentation)
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def setup_data_loaders(self):
        """Setup data loaders"""
        # Dataset paths
        train_dir = os.path.join(self.config['data']['splits_path'], 'train')
        val_dir = os.path.join(self.config['data']['splits_path'], 'val')
        
        # Create datasets
        self.train_dataset = EyeDataset(train_dir, self.train_transform)
        self.val_dataset = EyeDataset(val_dir, self.val_transform)
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        if self.config['training']['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif self.config['training']['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['training']['weight_decay']
            )
        
        # Setup scheduler
        if self.config['training']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config['training']['epochs']
            )
        elif self.config['training']['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
    
    def setup_loss_function(self):
        """Setup loss function with class weights"""
        # Calculate class weights for imbalanced data
        class_weights = calculate_class_weights(self.train_dataset)
        weights = torch.tensor([class_weights.get(i, 1.0) for i in range(self.config['model']['num_classes'])])
        weights = weights.to(self.device)
        
        # Use Focal Loss for better handling of class imbalance
        self.criterion = FocalLoss(alpha=1, gamma=2)
        
        print(f"Class weights: {weights}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, attention = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().detach().numpy())
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        # Calculate AUC
        try:
            all_probs = np.array(all_probs)
            if self.config['model']['num_classes'] == 2:
                # Binary classification - use probabilities for positive class
                epoch_auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                # Multi-class classification - use one-vs-rest
                epoch_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        except Exception as e:
            print(f"Warning: Could not calculate training AUC: {e}")
            epoch_auc = 0.0
        
        return epoch_loss, epoch_acc, epoch_auc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs, attention = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        # Calculate AUC
        try:
            all_probs = np.array(all_probs)
            if self.config['model']['num_classes'] == 2:
                # Binary classification - use probabilities for positive class
                epoch_auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                # Multi-class classification - use one-vs-rest
                epoch_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        except Exception as e:
            print(f"Warning: Could not calculate validation AUC: {e}")
            epoch_auc = 0.0
        
        return epoch_loss, epoch_acc, epoch_auc, all_preds, all_labels
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Training for {self.config['training']['epochs']} epochs")
        
        for epoch in range(self.config['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            print("-" * 30)
            
            # Train
            train_loss, train_acc, train_auc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_auc, val_preds, val_labels = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('AUC/Train', train_auc, epoch)
            self.writer.add_scalar('AUC/Validation', val_auc, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = os.path.join(
                    self.config['paths']['model_save_dir'], 
                    'best_model.pth'
                )
                save_model_checkpoint(
                    self.model, self.optimizer, epoch, 
                    val_loss, val_acc, checkpoint_path
                )
                print(f"New best model saved! Validation accuracy: {val_acc:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Close tensorboard writer
        self.writer.close()
        
        # Plot training history
        plot_path = os.path.join(self.config['paths']['results_dir'], 'training_history.png')
        plot_training_history(self.history, plot_path)
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")


def main():
    """Main training function"""
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()
