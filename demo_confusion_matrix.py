"""
Simple Confusion Matrix Demo for Pink Eye Detection
Creates a confusion matrix with sample data for demonstration
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                           accuracy_score, precision_recall_fscore_support)

# Class names for pink eye detection
CLASS_NAMES = ['healthy_eye', 'infected_eye']

def create_sample_confusion_matrix():
    """Create a sample confusion matrix for demonstration"""
    
    # Sample predictions (simulating 100 test samples)
    np.random.seed(42)  # For reproducible results
    
    # Simulate model predictions with good performance
    # True labels: 50 healthy, 50 infected
    true_labels = np.array([0]*50 + [1]*50)
    
    # Predictions: 45 correct healthy, 48 correct infected
    # 5 false positives, 2 false negatives
    pred_labels = np.array([0]*45 + [1]*5 + [1]*48 + [0]*2)
    
    return true_labels, pred_labels

def plot_confusion_matrix_enhanced(y_true, y_pred, class_names, save_path=None):
    """Plot enhanced confusion matrix with metrics"""
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Add text annotations for TP, TN, FP, FN
    ax1.text(0.5, 0.7, 'TN', ha='center', va='center', fontsize=12, 
             color='white', fontweight='bold')
    ax1.text(1.5, 0.7, 'FP', ha='center', va='center', fontsize=12, 
             color='white', fontweight='bold')
    ax1.text(0.5, 1.7, 'FN', ha='center', va='center', fontsize=12, 
             color='white', fontweight='bold')
    ax1.text(1.5, 1.7, 'TP', ha='center', va='center', fontsize=12, 
             color='white', fontweight='bold')
    
    # 2. Normalized Confusion Matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2)
    ax2.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # 3. Metrics Bar Chart
    metrics = ['Precision', 'Recall', 'F1-Score']
    healthy_metrics = [precision[0], recall[0], f1[0]]
    infected_metrics = [precision[1], recall[1], f1[1]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x - width/2, healthy_metrics, width, label='Healthy Eye', color='skyblue')
    ax3.bar(x + width/2, infected_metrics, width, label='Infected Eye', color='lightcoral')
    
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Score')
    ax3.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary Statistics
    ax4.axis('off')
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for positive class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for negative class
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    summary_text = f"""
    MODEL PERFORMANCE SUMMARY
    =========================
    
    Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)
    
    Confusion Matrix:
    ┌─────────────┬─────────────┬─────────────┐
    │             │ Pred Healthy│ Pred Infected│
    ├─────────────┼─────────────┼─────────────┤
    │ True Healthy│     {tn:3d}     │     {fp:3d}     │
    ├─────────────┼─────────────┼─────────────┤
    │ True Infected│     {fn:3d}     │     {tp:3d}     │
    └─────────────┴─────────────┴─────────────┘
    
    Key Metrics:
    - Sensitivity (Recall+): {sensitivity:.4f} ({sensitivity*100:.1f}%)
    - Specificity (Recall-): {specificity:.4f} ({specificity*100:.1f}%)
    - Precision (Positive): {precision_pos:.4f} ({precision_pos*100:.1f}%)
    - Precision (Negative): {precision_neg:.4f} ({precision_neg*100:.1f}%)
    
    Class-wise Performance:
    Healthy Eye:
      - Precision: {precision[0]:.4f}
      - Recall: {recall[0]:.4f}
      - F1-Score: {f1[0]:.4f}
    
    Infected Eye:
      - Precision: {precision[1]:.4f}
      - Recall: {recall[1]:.4f}
      - F1-Score: {f1[1]:.4f}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    
    return cm

def print_detailed_report(y_true, y_pred, class_names):
    """Print detailed classification report"""
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall: {recall:.4f}")
    print(f"Overall F1-Score: {f1:.4f}")
    
    print("\nPer-Class Metrics:")
    print("-" * 40)
    
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name.upper()}:")
        print(f"  Precision: {precision_per_class[i]:.4f}")
        print(f"  Recall: {recall_per_class[i]:.4f}")
        print(f"  F1-Score: {f1_per_class[i]:.4f}")
        print(f"  Support: {support[i]}")
    
    # Confusion matrix interpretation
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix Interpretation:")
    print("-" * 40)
    print(f"True Positives (TP): {tp} - Correctly identified infected eyes")
    print(f"True Negatives (TN): {tn} - Correctly identified healthy eyes")
    print(f"False Positives (FP): {fp} - Healthy eyes incorrectly flagged as infected")
    print(f"False Negatives (FN): {fn} - Infected eyes missed as healthy")
    
    # Clinical implications
    print(f"\nClinical Implications:")
    print("-" * 40)
    if fn > 0:
        print(f"⚠️  {fn} infected cases were missed - requires follow-up screening")
    if fp > 0:
        print(f"⚠️  {fp} healthy cases were flagged - may cause unnecessary concern")
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"✅ Sensitivity: {sensitivity*100:.1f}% - Ability to detect infected eyes")
    print(f"✅ Specificity: {specificity*100:.1f}% - Ability to identify healthy eyes")

def main():
    """Main function to demonstrate confusion matrix"""
    print("🔬 Pink Eye Detection - Confusion Matrix Demo")
    print("=" * 50)
    
    # Create sample data
    y_true, y_pred = create_sample_confusion_matrix()
    
    # Print detailed report
    print_detailed_report(y_true, y_pred, CLASS_NAMES)
    
    # Plot confusion matrix
    print("\n📊 Generating confusion matrix visualization...")
    plot_confusion_matrix_enhanced(y_true, y_pred, CLASS_NAMES, 
                                  save_path='demo_confusion_matrix.png')
    
    print("\n✅ Demo completed! Check 'demo_confusion_matrix.png' for the visualization.")

if __name__ == "__main__":
    main()
