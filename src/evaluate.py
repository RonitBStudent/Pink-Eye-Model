"""
Evaluation script for pink eye detection model
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                           confusion_matrix, classification_report, roc_auc_score)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
try:
    from grad_cam import GradCAM
    from grad_cam.utils.model_targets import ClassifierOutputTarget
    from grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
from PIL import Image
import cv2

from model import create_model
from train import EyeDataset
from utils import (load_config, plot_confusion_matrix, plot_roc_curves, 
                  print_classification_report, load_model_checkpoint)


class ModelEvaluator:
    def __init__(self, config_path="config.yaml", model_path=None):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = create_model(config_path).to(self.device)
        
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        
        self.model.eval()
        
        # Setup test data loader
        self.setup_test_loader()
        
        # Class names
        self.class_names = list(self.config['classes'].values())
    
    def setup_test_loader(self):
        """Setup test data loader"""
        import torchvision.transforms as transforms
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        test_dir = os.path.join(self.config['data']['splits_path'], 'test')
        self.test_dataset = EyeDataset(test_dir, test_transform)
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
        
        print(f"Test samples: {len(self.test_dataset)}")
    
    def evaluate(self):
        """Evaluate model on test set"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs, attention = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Calculate per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )
        
        # Calculate AUC for multi-class
        try:
            y_true_bin = label_binarize(all_labels, classes=range(len(self.class_names)))
            auc_scores = []
            for i in range(len(self.class_names)):
                auc = roc_auc_score(y_true_bin[:, i], all_probs[:, i])
                auc_scores.append(auc)
            mean_auc = np.mean(auc_scores)
        except:
            auc_scores = [0] * len(self.class_names)
            mean_auc = 0
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall: {recall:.4f}")
        print(f"Overall F1-Score: {f1:.4f}")
        print(f"Mean AUC: {mean_auc:.4f}")
        
        print("\nPer-Class Metrics:")
        print("-" * 30)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name}:")
            print(f"  Precision: {precision_per_class[i]:.4f}")
            print(f"  Recall: {recall_per_class[i]:.4f}")
            print(f"  F1-Score: {f1_per_class[i]:.4f}")
            print(f"  AUC: {auc_scores[i]:.4f}")
        
        # Detailed classification report
        print_classification_report(all_labels, all_preds, self.class_names)
        
        # Plot confusion matrix
        results_dir = self.config['paths']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        cm_path = os.path.join(results_dir, 'confusion_matrix.png')
        plot_confusion_matrix(all_labels, all_preds, self.class_names, cm_path)
        
        # Plot ROC curves
        if mean_auc > 0:
            roc_path = os.path.join(results_dir, 'roc_curves.png')
            plot_roc_curves(all_labels, all_probs, self.class_names, roc_path)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': mean_auc,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
    
    def generate_gradcam_visualizations(self, num_samples=10):
        """Generate Grad-CAM visualizations for model interpretability"""
        if not GRADCAM_AVAILABLE:
            print("Grad-CAM not available. Install with: pip install grad-cam")
            return
            
        # Setup Grad-CAM
        target_layers = [self.model.backbone.features[-1]]  # Last conv layer
        cam = GradCAM(model=self.model, target_layers=target_layers)
        
        results_dir = self.config['paths']['results_dir']
        gradcam_dir = os.path.join(results_dir, 'gradcam')
        os.makedirs(gradcam_dir, exist_ok=True)
        
        # Get some test samples
        sample_indices = np.random.choice(len(self.test_dataset), num_samples, replace=False)
        
        for idx, sample_idx in enumerate(sample_indices):
            image, label = self.test_dataset[sample_idx]
            image_tensor = image.unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output, _ = self.model(image_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
            
            # Generate Grad-CAM
            targets = [ClassifierOutputTarget(predicted_class)]
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Convert tensor to numpy for visualization
            image_np = image.permute(1, 2, 0).cpu().numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = std * image_np + mean
            image_np = np.clip(image_np, 0, 1)
            
            # Create visualization
            visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
            
            # Save visualization
            true_class = self.class_names[label]
            pred_class = self.class_names[predicted_class]
            filename = f'gradcam_{idx}_true_{true_class}_pred_{pred_class}.png'
            save_path = os.path.join(gradcam_dir, filename)
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.imshow(image_np)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(grayscale_cam, cmap='hot')
            plt.title('Grad-CAM Heatmap')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(visualization)
            plt.title(f'Overlay\nTrue: {true_class}\nPred: {pred_class}')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Generated {num_samples} Grad-CAM visualizations in {gradcam_dir}")
    
    def predict_single_image(self, image_path):
        """Predict on a single image"""
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output, attention = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
        
        # Get results
        predicted_label = self.class_names[predicted_class]
        confidence = probabilities[0][predicted_class].item()
        
        print(f"Prediction for {image_path}:")
        print(f"Predicted class: {predicted_label}")
        print(f"Confidence: {confidence:.4f}")
        
        print("\nAll class probabilities:")
        for i, class_name in enumerate(self.class_names):
            prob = probabilities[0][i].item()
            print(f"  {class_name}: {prob:.4f}")
        
        return predicted_label, confidence, probabilities[0].cpu().numpy()


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Pink Eye Detection Model')
    parser.add_argument('--model_path', type=str, 
                       default='models/saved/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to single image for prediction')
    parser.add_argument('--gradcam', action='store_true',
                       help='Generate Grad-CAM visualizations')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(model_path=args.model_path)
    
    if args.image_path:
        # Single image prediction
        evaluator.predict_single_image(args.image_path)
    else:
        # Full evaluation
        results = evaluator.evaluate()
        
        if args.gradcam:
            evaluator.generate_gradcam_visualizations()


if __name__ == "__main__":
    main()
