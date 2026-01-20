"""
Custom Confusion Matrix for Pink Eye Detection
Using provided TP, TN, FP, FN values
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                           accuracy_score, precision_recall_fscore_support)

# Provided values
TN = 1232  # True Negative - Healthy eyes correctly identified
FP = 27    # False Positive - Healthy eyes incorrectly flagged as infected
FN = 20    # False Negative - Infected eyes missed as healthy
TP = 1221  # True Positive - Infected eyes correctly identified

# Class names
CLASS_NAMES = ['healthy_eye', 'infected_eye']

def create_confusion_matrix_from_values():
    """Create confusion matrix from provided TP, TN, FP, FN values"""
    
    # Create confusion matrix
    # Format: [[TN, FP], [FN, TP]]
    cm = np.array([[TN, FP], [FN, TP]])
    
    # Create true and predicted labels
    true_labels = np.array([0]* (TN + FP) + [1]* (FN + TP))
    pred_labels = np.array([0]* TN + [1]* FP + [0]* FN + [1]* TP)
    
    return cm, true_labels, pred_labels

def plot_custom_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix with the provided values"""
    
    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confusion Matrix with counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix - Raw Counts', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Add TP, TN, FP, FN labels
    ax1.text(0.5, 0.7, f'TN\n{TN}', ha='center', va='center', fontsize=12, 
             color='white', fontweight='bold')
    ax1.text(1.5, 0.7, f'FP\n{FP}', ha='center', va='center', fontsize=12, 
             color='white', fontweight='bold')
    ax1.text(0.5, 1.7, f'FN\n{FN}', ha='center', va='center', fontsize=12, 
             color='white', fontweight='bold')
    ax1.text(1.5, 1.7, f'TP\n{TP}', ha='center', va='center', fontsize=12, 
             color='white', fontweight='bold')
    
    # 2. Normalized Confusion Matrix
    total_samples = np.sum(cm)
    cm_norm = cm.astype('float') / total_samples
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax2)
    ax2.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # 3. Performance Metrics Bar Chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    values = [accuracy, precision, recall, f1, specificity]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    bars = ax3.bar(metrics, values, color=colors)
    ax3.set_ylabel('Score')
    ax3.set_title('Performance Metrics', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Detailed Statistics Summary
    ax4.axis('off')
    
    summary_text = f"""
    PINK EYE DETECTION MODEL PERFORMANCE
    ====================================
    
    Dataset Summary:
    • Total Samples: {total_samples:,}
    • True Positives: {TP:,} ({TP/total_samples*100:.1f}%)
    • True Negatives: {TN:,} ({TN/total_samples*100:.1f}%)
    • False Positives: {FP:,} ({FP/total_samples*100:.1f}%)
    • False Negatives: {FN:,} ({FN/total_samples*100:.1f}%)
    
    Key Performance Metrics:
    • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
    • Precision: {precision:.4f} ({precision*100:.2f}%)
    • Recall (Sensitivity): {recall:.4f} ({recall*100:.2f}%)
    • Specificity: {specificity:.4f} ({specificity*100:.2f}%)
    • F1-Score: {f1:.4f} ({f1*100:.2f}%)
    
    Clinical Interpretation:
    • Sensitivity: {recall*100:.1f}% - Ability to detect infected eyes
    • Specificity: {specificity*100:.1f}% - Ability to identify healthy eyes
    • False Negative Rate: {(FN/(FN+TP))*100:.1f}% - Missed infections
    • False Positive Rate: {(FP/(FP+TN))*100:.1f}% - Unnecessary referrals
    
    Model Quality Assessment:
    {'✅ EXCELLENT' if accuracy > 0.95 else '✅ GOOD' if accuracy > 0.90 else '⚠️  NEEDS IMPROVEMENT'}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    
    return accuracy, precision, recall, f1, specificity

def print_detailed_analysis():
    """Print detailed analysis of the confusion matrix"""
    
    total_samples = TP + TN + FP + FN
    accuracy = (TP + TN) / total_samples
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    print("\n" + "="*80)
    print("🔬 PINK EYE DETECTION MODEL - CONFUSION MATRIX ANALYSIS")
    print("="*80)
    
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"┌{'─'*25}┬{'─'*25}┐")
    print(f"│{'':^25}│{'Predicted Healthy':^25}│")
    print(f"├{'─'*25}┼{'─'*25}┤")
    print(f"│{'True Healthy':^25}│{TN:^25}│")
    print(f"│{'':^25}│{FP:^25}│")
    print(f"├{'─'*25}┼{'─'*25}┤")
    print(f"│{'True Infected':^25}│{FN:^25}│")
    print(f"│{'':^25}│{TP:^25}│")
    print(f"└{'─'*25}┴{'─'*25}┘")
    print(f"{'':^25}│{'Predicted Infected':^25}│")
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"─" * 50)
    print(f"Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:       {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:          {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:        {f1:.4f} ({f1*100:.2f}%)")
    print(f"Specificity:     {specificity:.4f} ({specificity*100:.2f}%)")
    
    print(f"\n🏥 CLINICAL SIGNIFICANCE:")
    print(f"─" * 50)
    print(f"Sensitivity (True Positive Rate):     {recall*100:.2f}%")
    print(f"Specificity (True Negative Rate):     {specificity*100:.2f}%")
    print(f"False Negative Rate (Missed Cases):    {(FN/(FN+TP))*100:.2f}%")
    print(f"False Positive Rate (False Alarms):   {(FP/(FP+TN))*100:.2f}%")
    
    print(f"\n📋 SAMPLE BREAKDOWN:")
    print(f"─" * 50)
    print(f"Total Samples:           {total_samples:,}")
    print(f"True Positives (TP):     {TP:,} - Correctly detected infections")
    print(f"True Negatives (TN):     {TN:,} - Correctly identified healthy eyes")
    print(f"False Positives (FP):    {FP:,} - Healthy eyes flagged as infected")
    print(f"False Negatives (FN):    {FN:,} - Infected eyes missed as healthy")
    
    print(f"\n⚠️  CLINICAL IMPACT:")
    print(f"─" * 50)
    if FN > 0:
        print(f"• {FN} infected cases were missed - CRITICAL for patient care")
    if FP > 0:
        print(f"• {FP} healthy cases were flagged - May cause unnecessary concern")
    
    print(f"\n✅ MODEL ASSESSMENT:")
    print(f"─" * 50)
    if accuracy > 0.95:
        print("• Model Performance: EXCELLENT")
        print("• Suitable for clinical deployment")
    elif accuracy > 0.90:
        print("• Model Performance: GOOD")
        print("• Suitable for screening with expert oversight")
    else:
        print("• Model Performance: NEEDS IMPROVEMENT")
        print("• Requires further training before clinical use")

def main():
    """Main function to generate confusion matrix with provided values"""
    print("🔬 Pink Eye Detection - Custom Confusion Matrix")
    print("=" * 60)
    print(f"Using provided values:")
    print(f"• True Negatives: {TN}")
    print(f"• False Positives: {FP}")
    print(f"• False Negatives: {FN}")
    print(f"• True Positives: {TP}")
    
    # Create confusion matrix
    cm, true_labels, pred_labels = create_confusion_matrix_from_values()
    
    # Print detailed analysis
    print_detailed_analysis()
    
    # Plot confusion matrix
    print(f"\n📊 Generating confusion matrix visualization...")
    accuracy, precision, recall, f1, specificity = plot_custom_confusion_matrix(
        cm, save_path='pink_eye_confusion_matrix.png'
    )
    
    print(f"\n✅ Confusion matrix completed!")
    print(f"📁 Saved as: pink_eye_confusion_matrix.png")
    print(f"🎯 Model achieved {accuracy*100:.2f}% accuracy")

if __name__ == "__main__":
    main()
