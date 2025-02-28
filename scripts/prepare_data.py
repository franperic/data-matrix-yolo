#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Preparation Script for YOLO Training

This script helps convert various formats to YOLO format and
prepares the dataset structure for training.
"""

import os
import sys
import json
import shutil
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import project modules
from src.utils import setup_logging, get_logger

# Initialize logger
logger = get_logger(__name__)


def convert_coco_to_yolo(
    coco_file: str,
    output_dir: str,
    image_dir: str = None,
    class_mapping: Dict[int, int] = None
) -> None:
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        coco_file: Path to COCO JSON file
        output_dir: Directory to save YOLO format annotations
        image_dir: Directory containing images (if different from COCO file location)
        class_mapping: Mapping from COCO class ids to YOLO class ids (Default: identity mapping)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COCO data
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get image directory
    if image_dir is None:
        image_dir = str(Path(coco_file).parent)
    
    # Create mapping from image_id to image data
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Process each image
    for image_id, image_info in tqdm(images_dict.items(), desc="Converting to YOLO format"):
        # Get image filename and dimensions
        filename = image_info['file_name']
        img_width = image_info['width']
        img_height = image_info['height']
        
        # Skip if no annotations for this image
        if image_id not in annotations_by_image:
            continue
        
        # Create YOLO annotation file
        yolo_filename = Path(filename).stem + '.txt'
        yolo_filepath = os.path.join(output_dir, yolo_filename)
        
        # Copy image to output directory
        src_image_path = os.path.join(image_dir, filename)
        dst_image_path = os.path.join(output_dir, filename)
        
        if os.path.exists(src_image_path) and src_image_path != dst_image_path:
            shutil.copy(src_image_path, dst_image_path)
        
        # Convert annotations to YOLO format
        with open(yolo_filepath, 'w') as f:
            for ann in annotations_by_image[image_id]:
                # Get category id
                category_id = ann['category_id']
                if class_mapping is not None:
                    category_id = class_mapping.get(category_id, category_id)
                
                # Get bounding box in COCO format [x, y, width, height]
                bbox = ann['bbox']
                x, y, w, h = bbox
                
                # Convert to YOLO format [x_center, y_center, width, height] (normalized)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                
                # Write to file
                f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")
    
    logger.info(f"Converted COCO annotations to YOLO format in {output_dir}")


def split_dataset(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    copy_files: bool = True
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        input_dir: Directory containing images and annotations
        output_dir: Directory to save split dataset
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        copy_files: Whether to copy files to output directories
        
    Returns:
        Tuple of (train_files, val_files, test_files) lists
    """
    # Normalize ratios
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total
    
    # Get all image files
    input_dir = Path(input_dir)
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpeg'))
    
    # Shuffle files
    random.shuffle(image_files)
    
    # Calculate split indices
    n_train = int(len(image_files) * train_ratio)
    n_val = int(len(image_files) * val_ratio)
    
    # Split files
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    if copy_files:
        # Create output directories
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')
        test_dir = os.path.join(output_dir, 'test')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Copy files
        for files, dir_name in [
            (train_files, train_dir),
            (val_files, val_dir),
            (test_files, test_dir)
        ]:
            for img_file in tqdm(files, desc=f"Copying to {dir_name}"):
                # Copy image
                shutil.copy(img_file, os.path.join(dir_name, img_file.name))
                
                # Copy annotation if exists
                ann_file = img_file.with_suffix('.txt')
                if ann_file.exists():
                    shutil.copy(ann_file, os.path.join(dir_name, ann_file.name))
    
    logger.info(f"Split dataset: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    return train_files, val_files, test_files


def create_yolo_dataset_yaml(
    output_path: str,
    train_dir: str,
    val_dir: str,
    test_dir: str = None,
    class_names: List[str] = None
) -> None:
    """
    Create YAML file for YOLO dataset.
    
    Args:
        output_path: Path to save YAML file
        train_dir: Path to training data
        val_dir: Path to validation data
        test_dir: Path to test data (optional)
        class_names: List of class names
    """
    # Default class names
    if class_names is None:
        class_names = ["DataMatrix"]
    
    # Create dataset dict
    dataset_dict = {
        "train": train_dir,
        "val": val_dir,
        "nc": len(class_names),
        "names": class_names
    }
    
    if test_dir:
        dataset_dict["test"] = test_dir
    
    # Save YAML file
    with open(output_path, 'w') as f:
        yaml.dump(dataset_dict, f)
    
    logger.info(f"Created YOLO dataset YAML file: {output_path}")


def extract_dataset_statistics(data_dir: str) -> Dict:
    """
    Extract dataset statistics.
    
    Args:
        data_dir: Directory containing dataset
    
    Returns:
        Dictionary of statistics
    """
    data_dir = Path(data_dir)
    
    # Initialize statistics
    stats = {
        "total_images": 0,
        "total_objects": 0,
        "objects_per_class": {},
        "splits": {}
    }
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        # Count images and objects
        image_files = list(split_dir.glob('*.jpg')) + list(split_dir.glob('*.png')) + list(split_dir.glob('*.jpeg'))
        object_count = 0
        objects_per_class = {}
        
        for img_file in image_files:
            label_file = img_file.with_suffix('.txt')
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            object_count += 1
                            
                            if class_id not in objects_per_class:
                                objects_per_class[class_id] = 0
                            objects_per_class[class_id] += 1
        
        # Update statistics
        stats["total_images"] += len(image_files)
        stats["total_objects"] += object_count
        
        # Update class statistics
        for class_id, count in objects_per_class.items():
            if class_id not in stats["objects_per_class"]:
                stats["objects_per_class"][class_id] = 0
            stats["objects_per_class"][class_id] += count
        
        # Add split statistics
        stats["splits"][split] = {
            "images": len(image_files),
            "objects": object_count,
            "objects_per_class": objects_per_class
        }
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prepare data for YOLO training")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Convert COCO to YOLO
    convert_parser = subparsers.add_parser("convert", help="Convert COCO to YOLO format")
    convert_parser.add_argument("--coco_file", required=True, help="Path to COCO JSON file")
    convert_parser.add_argument("--output_dir", required=True, help="Directory to save YOLO format annotations")
    convert_parser.add_argument("--image_dir", help="Directory containing images (if different from COCO file location)")
    
    # Split dataset
    split_parser = subparsers.add_parser("split", help="Split dataset into train, val, and test sets")
    split_parser.add_argument("--input_dir", required=True, help="Directory containing images and annotations")
    split_parser.add_argument("--output_dir", required=True, help="Directory to save split dataset")
    split_parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data")
    split_parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validation data")
    split_parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test data")
    
    # Create YAML
    yaml_parser = subparsers.add_parser("yaml", help="Create YAML file for YOLO dataset")
    yaml_parser.add_argument("--output_path", required=True, help="Path to save YAML file")
    yaml_parser.add_argument("--train_dir", required=True, help="Path to training data")
    yaml_parser.add_argument("--val_dir", required=True, help="Path to validation data")
    yaml_parser.add_argument("--test_dir", help="Path to test data (optional)")
    yaml_parser.add_argument("--class_names", nargs="+", default=["DataMatrix"], help="List of class names")
    
    # Statistics
    stats_parser = subparsers.add_parser("stats", help="Extract dataset statistics")
    stats_parser.add_argument("--data_dir", required=True, help="Directory containing dataset")
    
    # Debug mode
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug=args.debug)
    
    # Execute command
    if args.command == "convert":
        convert_coco_to_yolo(args.coco_file, args.output_dir, args.image_dir)
    elif args.command == "split":
        split_dataset(args.input_dir, args.output_dir, args.train_ratio, args.val_ratio, args.test_ratio)
    elif args.command == "yaml":
        create_yolo_dataset_yaml(args.output_path, args.train_dir, args.val_dir, args.test_dir, args.class_names)
    elif args.command == "stats":
        stats = extract_dataset_statistics(args.data_dir)
        print(json.dumps(stats, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
