#!/bin/bash

# Quick train and evaluate script for pink eye detection

echo "🔬 Pink Eye Detection - Training & Evaluation"
echo "=============================================="

# Check if data exists
if [ ! -d "data/splits/train" ]; then
    echo "❌ Training data not found. Running data preprocessing first..."
    python3 src/data_preprocessing.py
fi

# Train the model
echo "🚀 Training model..."
python3 src/train.py

# Check if model was saved
if [ -f "models/saved/best_model.pth" ]; then
    echo "✅ Model training completed!"
    
    # Generate confusion matrix
    echo "📊 Generating confusion matrix..."
    python3 src/confusion_matrix.py --model_path models/saved/best_model.pth --split test
    
    echo "🎉 Process completed! Check the 'results' directory for confusion matrix plots."
else
    echo "❌ Model training failed. Please check the training logs."
fi
