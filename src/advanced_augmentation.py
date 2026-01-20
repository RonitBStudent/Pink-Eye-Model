"""
Advanced data augmentation techniques for pink eye detection
"""

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


class AdvancedAugmentation:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        
    def get_training_augmentation(self):
        """Enhanced augmentation pipeline for training"""
        return A.Compose([
            # Geometric transformations
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),  # Less common but can help
            A.Rotate(limit=20, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.7),
            
            # Perspective and distortion
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            
            # Color and lighting augmentations
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.ChannelShuffle(p=0.1),
            
            # Lighting conditions
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.2),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], per_channel=True),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            ], p=0.3),
            
            A.OneOf([
                A.Blur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
            ], p=0.2),
            
            # Advanced transformations
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.2),
            
            # Medical-specific augmentations
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.1),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, p=0.05),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def get_mixup_augmentation(self, alpha=0.2):
        """MixUp augmentation for better generalization"""
        def mixup(batch_x, batch_y, alpha=alpha):
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1
            
            batch_size = batch_x.size(0)
            index = torch.randperm(batch_size)
            
            mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]
            y_a, y_b = batch_y, batch_y[index]
            
            return mixed_x, y_a, y_b, lam
        
        return mixup
    
    def get_cutmix_augmentation(self, alpha=1.0):
        """CutMix augmentation"""
        def cutmix(batch_x, batch_y, alpha=alpha):
            lam = np.random.beta(alpha, alpha)
            rand_index = torch.randperm(batch_x.size()[0])
            
            y_a = batch_y
            y_b = batch_y[rand_index]
            
            bbx1, bby1, bbx2, bby2 = rand_bbox(batch_x.size(), lam)
            batch_x[:, :, bbx1:bbx2, bby1:bby2] = batch_x[rand_index, :, bbx1:bbx2, bby1:bby2]
            
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_x.size()[-1] * batch_x.size()[-2]))
            
            return batch_x, y_a, y_b, lam
        
        def rand_bbox(size, lam):
            W = size[2]
            H = size[3]
            cut_rat = np.sqrt(1. - lam)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)
            
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
            
            return bbx1, bby1, bbx2, bby2
        
        return cutmix


# Medical-specific augmentation for eye images
class MedicalEyeAugmentation:
    @staticmethod
    def simulate_lighting_conditions():
        """Simulate different clinical lighting"""
        return A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=30, p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.8)
    
    @staticmethod
    def simulate_camera_variations():
        """Simulate different camera/phone qualities"""
        return A.OneOf([
            A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
            A.Downscale(scale_min=0.7, scale_max=0.9, p=1.0),
            A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
        ], p=0.5)
    
    @staticmethod
    def eye_specific_augmentation():
        """Augmentations specific to eye anatomy"""
        return A.Compose([
            # Simulate eyelash shadows
            A.RandomShadow(shadow_roi=(0, 0, 1, 0.3), num_shadows_lower=1, num_shadows_upper=3, p=0.3),
            # Simulate reflections on eye surface
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.4),
            # Simulate different eye positions
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.6),
        ])
