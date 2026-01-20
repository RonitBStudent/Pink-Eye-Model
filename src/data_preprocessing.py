"""
Data preprocessing pipeline for pink eye detection
"""

import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from sklearn.model_selection import train_test_split
from utils import load_config, create_directories


class DataPreprocessor:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.image_size = tuple(self.config['data']['image_size'])
        self.raw_data_path = self.config['data']['raw_data_path']
        self.processed_data_path = self.config['data']['processed_data_path']
        self.splits_path = self.config['data']['splits_path']
        
        # Create directories
        create_directories(self.config)
        
        # Define augmentation pipeline
        self.augmentation = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.HorizontalFlip(p=self.config['augmentation']['horizontal_flip']),
            A.Rotate(limit=self.config['augmentation']['rotation_range'], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=self.config['augmentation']['brightness_range'],
                contrast_limit=self.config['augmentation']['contrast_range'],
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(self.config['augmentation']['hue_range'][1] * 180),
                sat_shift_limit=int((self.config['augmentation']['saturation_range'][1] - 1) * 100),
                val_shift_limit=int((self.config['augmentation']['brightness_range'][1] - 1) * 100),
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Blur(blur_limit=3, p=0.1),
        ])
        
        # Basic preprocessing (no augmentation)
        self.basic_transform = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1])
        ])

    def validate_image(self, image_path):
        """Validate if image is readable and has proper format"""
        try:
            image = Image.open(image_path)
            image.verify()  # Verify image integrity
            
            # Check if image has minimum size
            if image.size[0] < 50 or image.size[1] < 50:
                return False
            
            return True
        except Exception as e:
            print(f"Invalid image {image_path}: {e}")
            return False

    def preprocess_image(self, image_path, apply_augmentation=False):
        """Preprocess a single image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transformations
            if apply_augmentation:
                transformed = self.augmentation(image=image)
            else:
                transformed = self.basic_transform(image=image)
            
            return transformed['image']
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def collect_image_paths(self):
        """Collect all image paths and their labels"""
        image_paths = []
        labels = []
        
        class_mapping = {v: k for k, v in self.config['classes'].items()}
        
        for class_name in class_mapping.keys():
            class_dir = os.path.join(self.raw_data_path, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist")
                continue
            
            for image_file in os.listdir(class_dir):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_path = os.path.join(class_dir, image_file)
                    
                    if self.validate_image(image_path):
                        image_paths.append(image_path)
                        labels.append(class_mapping[class_name])
        
        return image_paths, labels

    def create_data_splits(self):
        """Create train/validation/test splits"""
        image_paths, labels = self.collect_image_paths()
        
        if len(image_paths) == 0:
            print("No valid images found! Please add images to the data/raw/ directories.")
            return
        
        print(f"Found {len(image_paths)} valid images")
        
        # Print class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            class_name = self.config['classes'][label]
            print(f"Class {class_name}: {count} images")
        
        # Split data
        train_ratio = self.config['data']['train_split']
        val_ratio = self.config['data']['val_split']
        test_ratio = self.config['data']['test_split']
        
        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels, 
            test_size=test_ratio, 
            random_state=42, 
            stratify=labels
        )
        
        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size, 
            random_state=42, 
            stratify=y_temp
        )
        
        # Save splits
        splits = {
            'train': list(zip(X_train, y_train)),
            'val': list(zip(X_val, y_val)),
            'test': list(zip(X_test, y_test))
        }
        
        for split_name, split_data in splits.items():
            split_dir = os.path.join(self.splits_path, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            # Create class subdirectories
            for class_id, class_name in self.config['classes'].items():
                os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
            
            print(f"Processing {split_name} split: {len(split_data)} images")
            
            for image_path, label in split_data:
                class_name = self.config['classes'][label]
                
                # Process image
                processed_image = self.preprocess_image(
                    image_path, 
                    apply_augmentation=(split_name == 'train')
                )
                
                if processed_image is not None:
                    # Save processed image
                    filename = os.path.basename(image_path)
                    save_path = os.path.join(split_dir, class_name, filename)
                    
                    # Convert to PIL and save
                    pil_image = Image.fromarray(processed_image)
                    pil_image.save(save_path)
        
        print("Data preprocessing completed!")
        print(f"Train: {len(X_train)} images")
        print(f"Validation: {len(X_val)} images")
        print(f"Test: {len(X_test)} images")

    def augment_minority_classes(self, target_samples_per_class=1000):
        """Augment minority classes to balance the dataset"""
        train_dir = os.path.join(self.splits_path, 'train')
        
        for class_id, class_name in self.config['classes'].items():
            class_dir = os.path.join(train_dir, class_name)
            
            if not os.path.exists(class_dir):
                continue
            
            existing_images = [f for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            current_count = len(existing_images)
            
            if current_count < target_samples_per_class:
                needed_samples = target_samples_per_class - current_count
                print(f"Augmenting {class_name}: {current_count} -> {target_samples_per_class}")
                
                for i in range(needed_samples):
                    # Randomly select an existing image
                    source_image = random.choice(existing_images)
                    source_path = os.path.join(class_dir, source_image)
                    
                    # Apply augmentation
                    augmented_image = self.preprocess_image(source_path, apply_augmentation=True)
                    
                    if augmented_image is not None:
                        # Save augmented image
                        base_name = os.path.splitext(source_image)[0]
                        aug_filename = f"{base_name}_aug_{i}.jpg"
                        save_path = os.path.join(class_dir, aug_filename)
                        
                        pil_image = Image.fromarray(augmented_image)
                        pil_image.save(save_path)


def main():
    """Main preprocessing pipeline"""
    print("Starting data preprocessing...")
    
    preprocessor = DataPreprocessor()
    
    # Create data splits
    preprocessor.create_data_splits()
    
    # Optionally augment minority classes
    # preprocessor.augment_minority_classes(target_samples_per_class=500)
    
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
