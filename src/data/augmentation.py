#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data augmentation for YOLO training."""

from typing import Dict, Any, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(config: Dict[str, Any], is_train: bool = True) -> A.Compose:
    """
    Get data transformations based on configuration.
    
    Args:
        config: Training configuration
        is_train: Whether transformations are for training
        
    Returns:
        Albumentations composition of transforms
    """
    img_size = config.get("img_size", 640)
    use_augmentation = config.get("use_augmentation", True)
    
    if is_train and use_augmentation:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.GaussianBlur(p=0.1),
            A.GaussNoise(p=0.1),
            A.Perspective(p=0.1),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2),
            # Careful with these as they might affect the data matrices
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
            ], p=0.2),
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))
