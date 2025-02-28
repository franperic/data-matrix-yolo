#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Visualization utilities for YOLO predictions."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def visualize_detections(
    image: Union[str, np.ndarray, Image.Image],
    detections: List[Dict],
    output_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    line_width: int = 2,
    font_size: int = 12,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> Image.Image:
    """
    Visualize detections on image.
    
    Args:
        image: Image path, numpy array, or PIL Image
        detections: List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2]
            - confidence: float
            - class_id: int
        output_path: Path to save output image
        class_names: List of class names
        line_width: Line width for bounding boxes
        font_size: Font size for labels
        colors: Dictionary mapping class IDs to colors
        
    Returns:
        PIL Image with detections visualized
    """
    # Load image if needed
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Create draw object
    draw = ImageDraw.Draw(image)
    
    # Default colors if not provided
    if colors is None:
        colors = {
            0: (0, 255, 0),    # Green for DataMatrix
            1: (255, 0, 0),    # Red for other classes
            2: (0, 0, 255),    # Blue for other classes
        }
    
    # Default class names if not provided
    if class_names is None:
        class_names = ["DataMatrix"]
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw detections
    for det in detections:
        # Get bounding box and class info
        x1, y1, x2, y2 = det["bbox"]
        class_id = det["class_id"]
        confidence = det["confidence"]
        
        # Get class name
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"Class {class_id}"
        
        # Get color for this class
        color = colors.get(class_id, (0, 255, 0))
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        
        # Prepare label text
        label_text = f"{class_name}: {confidence:.2f}"
        
        # Draw label background
        text_size = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        
        draw.rectangle(
            [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
            fill=color
        )
        
        # Draw label text
        draw.text((x1 + 2, y1 - text_height - 2), label_text, fill=(255, 255, 255), font=font)
    
    # Save image if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        image.save(output_path)
        logger.info(f"Saved visualization to {output_path}")
    
    return image


def plot_metrics(
    metrics: Dict[str, List[float]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot training metrics.
    
    Args:
        metrics: Dictionary of metrics with keys as metric names and values as lists of values
        output_path: Path to save output image
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=figsize)
        
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
        
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.title("Training Metrics")
        plt.legend()
        plt.grid(True)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            plt.savefig(output_path)
            logger.info(f"Saved metrics plot to {output_path}")
        
        plt.close()
    except ImportError:
        logger.warning("Matplotlib not installed. Skipping metrics plot.")
