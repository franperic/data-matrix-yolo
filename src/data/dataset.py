#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset module for YOLO training."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

from .augmentation import get_transform

logger = logging.getLogger(__name__)


class DataMatrixDataset(Dataset):
    """Dataset for DataMatrix detection."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 640,
        transform = None,
        format: str = "yolo"
    ):
        """
        Initialize DataMatrix dataset.
        
        Args:
            data_dir: Directory containing dataset
            split: Data split (train, val, test)
            img_size: Target image size
            transform: Transformations to apply
            format: Dataset format (yolo, coco)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.format = format.lower()
        
        # Set up paths
        self.split_dir = self.data_dir / split
        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")
        
        # Load data based on format
        if self.format == "yolo":
            self._load_yolo_format()
        elif self.format == "coco":
            self._load_coco_format()
        else:
            self._load_custom_format()
            
        logger.info(f"Loaded {len(self.image_paths)} images for {split} split")
            
    def _load_yolo_format(self):
        """Load data in YOLO format."""
        self.image_paths = list(self.split_dir.glob("*.jpg")) + list(self.split_dir.glob("*.png"))
        # For each image, there should be a corresponding .txt file with annotations
        self.label_paths = [p.with_suffix(".txt") for p in self.image_paths]
        
    def _load_coco_format(self):
        """Load data in COCO format."""
        # Find annotations file
        anno_file = list(self.split_dir.glob("*.json"))
        if not anno_file:
            raise ValueError(f"No COCO annotation file found in {self.split_dir}")
        
        with open(anno_file[0], "r") as f:
            self.coco_data = json.load(f)
            
        # Create image id to annotations mapping
        self.img_id_to_annotations = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_annotations:
                self.img_id_to_annotations[img_id] = []
            self.img_id_to_annotations[img_id].append(ann)
            
        # Get image paths
        self.image_paths = []
        self.image_info = []
        for img in self.coco_data["images"]:
            img_path = self.split_dir / img["file_name"]
            if img_path.exists():
                self.image_paths.append(img_path)
                self.image_info.append(img)
                
    def _load_custom_format(self):
        """Load data in custom format."""
        # Fallback to loading just images
        self.image_paths = list(self.split_dir.glob("**/*.jpg")) + list(self.split_dir.glob("**/*.png"))
        
    def _read_yolo_annotation(self, label_path):
        """Read YOLO annotation format."""
        boxes = []
        classes = []
        
        if not label_path.exists():
            return np.zeros((0, 4)), np.zeros(0)
        
        with open(label_path, "r") as f:
            for line in f:
                data = line.strip().split()
                if len(data) >= 5:  # class, x_center, y_center, width, height
                    class_id = int(data[0])
                    x_center = float(data[1])
                    y_center = float(data[2])
                    width = float(data[3])
                    height = float(data[4])
                    
                    # Convert to xyxy format (for albumentations)
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    boxes.append([x1, y1, x2, y2])
                    classes.append(class_id)
                    
        return np.array(boxes), np.array(classes)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get image path
        img_path = self.image_paths[idx]
        
        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotations based on format
        if self.format == "yolo":
            label_path = self.label_paths[idx]
            boxes, class_labels = self._read_yolo_annotation(label_path)
        elif self.format == "coco":
            img_info = self.image_info[idx]
            img_id = img_info["id"]
            img_width, img_height = img_info["width"], img_info["height"]
            
            boxes = []
            class_labels = []
            
            if img_id in self.img_id_to_annotations:
                for ann in self.img_id_to_annotations[img_id]:
                    # COCO format: [x, y, width, height]
                    x, y, w, h = ann["bbox"]
                    # Convert to normalized xyxy
                    x1 = x / img_width
                    y1 = y / img_height
                    x2 = (x + w) / img_width
                    y2 = (y + h) / img_height
                    boxes.append([x1, y1, x2, y2])
                    class_labels.append(ann["category_id"])
                    
            boxes = np.array(boxes, dtype=np.float32)
            class_labels = np.array(class_labels, dtype=np.int64)
        else:
            # Custom format: assume no annotations for inference only
            boxes = np.zeros((0, 4), dtype=np.float32)
            class_labels = np.zeros(0, dtype=np.int64)
        
        # Apply transformations
        if self.transform:
            # Create transformed sample with bounding boxes
            sample = {
                "image": img,
                "bboxes": boxes,
                "class_labels": class_labels
            }
            
            transformed = self.transform(**sample)
            img = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32)
            class_labels = np.array(transformed["class_labels"], dtype=np.int64)
        
        # Return transformed data
        return {
            "image": img,
            "boxes": boxes,
            "class_labels": class_labels,
            "image_path": str(img_path)
        }


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create dataloaders for training, validation, and testing.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from .augmentation import get_transform
    
    # Get parameters from config
    data_dir = config["data_dir"]
    batch_size = config.get("batch_size", 16)
    img_size = config.get("img_size", 640)
    num_workers = config.get("num_workers", 4)
    data_format = config.get("data_format", "yolo")
    
    # Create transforms
    transform_train = get_transform(config, is_train=True)
    transform_val = get_transform(config, is_train=False)
    
    # Create datasets
    train_dataset = DataMatrixDataset(
        data_dir=data_dir,
        split="train",
        img_size=img_size,
        transform=transform_train,
        format=data_format
    )
    
    val_dataset = DataMatrixDataset(
        data_dir=data_dir,
        split="val",
        img_size=img_size,
        transform=transform_val,
        format=data_format
    )
    
    # Create test dataset if directory exists
    test_dataset = None
    test_dir = Path(data_dir) / "test"
    if test_dir.exists():
        test_dataset = DataMatrixDataset(
            data_dir=data_dir,
            split="test",
            img_size=img_size,
            transform=transform_val,
            format=data_format
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader
