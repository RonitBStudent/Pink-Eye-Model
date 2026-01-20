"""
Enhanced Confusion Matrix Generator for Pink Eye Detection
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                           accuracy_score, precision_recall_fscore_support)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms

from model import create_model
from train import EyeDataset
from utils import load_config


class ConfusionMatrixGenerator:
    def __init__(self, config_path="config.yaml", model_path=None):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = list(self.config['classes'].values())
        
        # Load model
        self.model = create_model(config_path).to(self.device)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: No model loaded. Using untrained model.")
        
        self.model.eval()
    
    def setup_data_loader(self, split='test'):
        """Setup data loader for specified split"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        data_dir = os.path.join(self.config['data']['splits_path'], split)
        dataset = EyeDataset(data_dir, transform)
        
        loader = DataLoader(
            dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
        
        return loader, dataset
    
    def get_predictions(self, data_loader):
        """Get model predictions"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs, attention = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def plot_enhanced_confusion_matrix(self, y_true, y_pred, save_path=None, 
                                     normalize=False, title="Confusion Matrix"):
        """Plot enhanced confusion matrix with additional metrics"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f"Normalized {title}"
        else:
            fmt = 'd'
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax1, cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        ax1.set_title(title)
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Calculate and plot metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)
        
        # Create metrics table
        metrics_data = []
        for i, class_name in enumerate(self.class_names):
            metrics_data.append([
                class_name,
                f"{precision[i]:.3f}",
                f"{recall[i]:.3f}", 
                f"{f1[i]:.3f}",
                f"{support[i]}"
            ])
        
        # Add overall metrics
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics_data.append([
            'Overall',
            f"{overall_precision:.3f}",
            f"{overall_recall:.3f}",
            f"{overall_f1:.3f}",
            f"{len(y_true)}"
        ])
        
        # Plot metrics table
        ax2.axis('tight')
        ax2.axis('off')
        table = ax2.table(cellText=metrics_data,
                         colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color code the table
        for i in range(len(metrics_data)):
            for j in range(len(metrics_data[0])):
                if i == len(metrics_data) - 1:  # Overall row
                    table[(i+1, j)].set_facecolor('#E8F4FD')
                elif i < len(self.class_names):  # Class rows
                    table[(i+1, j)].set_facecolor('#F0F8FF')
        
        ax2.set_title('Classification Metrics', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
        return cm
    
    def plot_class_distribution(self, y_true, y_pred, save_path=None):
        """Plot class distribution comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # True distribution
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        class_names_true = [self.class_names[i] for i in unique_true]
        
        ax1.pie(counts_true, labels=class_names_true, autopct='%1.1f%%', startangle=90)
        ax1.set_title('True Class Distribution')
        
        # Predicted distribution
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        class_names_pred = [self.class_names[i] for i in unique_pred]
        
        ax2.pie(counts_pred, labels=class_names_pred, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Predicted Class Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution saved to {save_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self, split='test', save_dir=None):
        """Generate comprehensive confusion matrix report"""
        if save_dir is None:
            save_dir = self.config['paths'].get('results_dir', 'results')
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Get data loader and predictions
        loader, dataset = self.setup_data_loader(split)
        y_true, y_pred, y_probs = self.get_predictions(loader)
        
        print(f"\n{'='*60}")
        print(f"CONFUSION MATRIX REPORT - {split.upper()} SET")
        print(f"{'='*60}")
        print(f"Total samples: {len(y_true)}")
        
        # Print classification report
        print("\nClassification Report:")
        print("-" * 40)
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names,
                                     digits=4)
        print(report)
        
        # Generate plots
        print("\nGenerating visualizations...")
        
        # Regular confusion matrix
        cm_path = os.path.join(save_dir, f'confusion_matrix_{split}.png')
        self.plot_enhanced_confusion_matrix(y_true, y_pred, cm_path, 
                                           normalize=False, 
                                           title=f"Confusion Matrix - {split.title()} Set")
        
        # Normalized confusion matrix
        cm_norm_path = os.path.join(save_dir, f'confusion_matrix_{split}_normalized.png')
        self.plot_enhanced_confusion_matrix(y_true, y_pred, cm_norm_path, 
                                           normalize=True,
                                           title=f"Normalized Confusion Matrix - {split.title()} Set")
        
        # Class distribution
        dist_path = os.path.join(save_dir, f'class_distribution_{split}.png')
        self.plot_class_distribution(y_true, y_pred, dist_path)
        
        # Save raw data
        np.save(os.path.join(save_dir, f'y_true_{split}.npy'), y_true)
        np.save(os.path.join(save_dir, f'y_pred_{split}.npy'), y_pred)
        np.save(os.path.join(save_dir, f'y_probs_{split}.npy'), y_probs)
        
        print(f"\nAll results saved to: {save_dir}")
        
        return y_true, y_pred, y_probs


def main():
    """Main function to generate confusion matrix"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Confusion Matrix for Pink Eye Detection')
    parser.add_argument('--model_path', type=str, 
                       default='models/saved/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], 
                       default='test', help='Dataset split to evaluate')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create generator
    generator = ConfusionMatrixGenerator(args.config, args.model_path)
    
    # Generate comprehensive report
    generator.generate_comprehensive_report(args.split, args.save_dir)


if __name__ == "__main__":
    main()
