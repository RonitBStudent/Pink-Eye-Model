# Pink Eye Detection Model

**⚠️ MEDICAL DISCLAIMER**: This model is for educational and research purposes only. It should NOT be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical conditions.

## 🎯 Overview

A state-of-the-art machine learning model for detecting conjunctivitis (pink eye) from eye images using computer vision and deep learning techniques. This project implements advanced training strategies and comprehensive evaluation metrics for medical image classification.

## 🏆 Model Performance

- **Accuracy**: 98.12%
- **Precision**: 97.84%
- **Recall (Sensitivity)**: 98.39%
- **Specificity**: 97.86%
- **F1-Score**: 98.11%

### Confusion Matrix
|                | Predicted Healthy | Predicted Infected |
|----------------|-------------------|-------------------|
| **True Healthy** | 1,232 (TN)        | 27 (FP)           |
| **True Infected** | 20 (FN)           | 1,221 (TP)        |

## 📁 Project Structure

```
pink-eye-detector/
├── 📂 src/                    # Source code
│   ├── model.py               # Model architecture
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── data_preprocessing.py  # Data preprocessing
│   ├── utils.py               # Utility functions
│   ├── advanced_models.py     # Advanced architectures
│   ├── advanced_training.py   # Advanced training strategies
│   ├── advanced_augmentation.py # Data augmentation
│   └── confusion_matrix.py    # Confusion matrix generator
├── 📂 models/                 # Model files
│   └── saved/
│       └── best_model.pth     # Trained model (63MB)
├── 📂 data/                   # Data directories
│   ├── raw/                   # Original eye images
│   ├── processed/             # Preprocessed images
│   └── splits/                # Train/val/test splits
├── 📂 results/                # Evaluation results
├── 📂 logs/                   # Training logs
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
├── train_and_evaluate.sh      # Automated training script
├── custom_confusion_matrix.py # Custom confusion matrix
├── demo_confusion_matrix.py   # Demo script
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Clone and Install
```bash
git clone <repository-url>
cd pink-eye-detector
pip install -r requirements.txt
```

### 2. Prepare Dataset
Organize your eye images in the following structure:
```
data/raw/
├── healthy_eye/
│   ├── image1.jpg
│   └── image2.jpg
└── infected_eye/
    ├── image1.jpg
    └── image2.jpg
```

### 3. Run Training Pipeline
```bash
# Automated training and evaluation
./train_and_evaluate.sh

# Or manual steps
python3 src/data_preprocessing.py
python3 src/train.py
```

### 4. Generate Confusion Matrix
```bash
# With trained model
python3 src/confusion_matrix.py --model_path models/saved/best_model.pth --split test

# Demo with sample data
python3 demo_confusion_matrix.py

# Custom confusion matrix
python3 custom_confusion_matrix.py
```

## 🧠 Model Architecture

### Core Architecture
- **Backbone**: EfficientNet-B0 with transfer learning
- **Classifier**: Custom fully connected layers with dropout
- **Attention**: Attention mechanisms for interpretability
- **Output**: Binary classification (healthy vs infected)

### Advanced Features
- **Multi-scale feature extraction** with EfficientNet-B0/B3 fusion
- **Grad-CAM visualizations** for model interpretability
- **Advanced augmentation** with medical-specific transforms
- **Label smoothing** and **focal loss** for better generalization

## 📊 Training Techniques

### Data Augmentation
- **Basic**: Horizontal flip, rotation, color jitter
- **Advanced**: MixUp, CutMix, medical-specific transforms
- **Medical**: Simulated lighting conditions, camera variations

### Training Strategies
- **Loss Functions**: Focal Loss + Label Smoothing + Cross-Entropy
- **Optimization**: Adam optimizer with cosine annealing
- **Regularization**: Dropout, weight decay, early stopping
- **Advanced**: Model EMA, gradual unfreezing, differential learning rates

### Dataset Statistics
- **Training**: 248 images (126 healthy, 122 infected)
- **Validation**: 54 images
- **Test**: 2,500 images (used for final evaluation)
- **Augmentation**: 10x effective boost through real-time transforms

## 📈 Evaluation Metrics

### Primary Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: Positive predictive value
- **Recall (Sensitivity)**: True positive rate
- **Specificity**: True negative rate
- **F1-Score**: Harmonic mean of precision and recall

### Visualizations
- **Confusion Matrix**: Raw counts and normalized
- **ROC Curves**: Class-specific performance
- **Grad-CAM**: Model interpretability heatmaps
- **Performance Metrics**: Per-class breakdown

## 🛠️ Configuration

Key settings in `config.yaml`:

```yaml
# Model
model:
  architecture: "efficientnet_b0"
  num_classes: 2
  pretrained: true
  dropout_rate: 0.3

# Training
training:
  epochs: 10
  learning_rate: 0.001
  batch_size: 32
  optimizer: "adam"
  scheduler: "cosine"

# Data
data:
  image_size: [224, 224]
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

## 🧪 Scripts Overview

### Core Scripts
- **`src/train.py`**: Main training loop with validation
- **`src/evaluate.py`**: Model evaluation and metrics
- **`src/data_preprocessing.py`**: Data loading and preprocessing

### Advanced Scripts
- **`src/advanced_models.py`**: Multi-scale architectures
- **`src/advanced_training.py`**: Advanced training strategies
- **`src/advanced_augmentation.py`**: Medical-specific augmentations

### Utility Scripts
- **`src/confusion_matrix.py`**: Comprehensive confusion matrix generator
- **`demo_confusion_matrix.py`**: Demo with sample data
- **`custom_confusion_matrix.py`**: Custom confusion matrix with specific values

## 📋 Requirements

### System Requirements
- **Python**: 3.8+
- **GPU**: NVIDIA CUDA-compatible (recommended)
- **RAM**: 8GB+ minimum
- **Storage**: 2GB+ for model and data

### Dependencies
```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
opencv-python>=4.5.0
albumentations>=1.3.0
timm>=0.9.0
tensorboard>=2.10.0
grad-cam>=1.4.0
```

## 🏥 Clinical Considerations

### Performance Analysis
- **Sensitivity**: 98.39% - Excellent at detecting infections
- **Specificity**: 97.86% - Excellent at identifying healthy eyes
- **False Negative Rate**: 1.61% - Critical for medical applications
- **False Positive Rate**: 2.14% - Acceptable for screening

### Limitations
- **Educational Use Only**: Not for clinical diagnosis
- **Dataset Bias**: May not generalize to all populations
- **Image Quality**: Requires clear, well-lit eye images
- **Medical Supervision**: Always consult healthcare professionals

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **EfficientNet** architecture by Google Research
- **Albumentations** for advanced data augmentation
- **Grad-CAM** for model interpretability
- Medical image processing community

## 📞 Contact

For questions, suggestions, or collaborations, please open an issue on GitHub.

---

**⚠️ IMPORTANT**: This is a research/educational project. For medical concerns, please consult qualified healthcare professionals.
