#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions for YOLO models."""

import os
import logging
from typing import Dict, Any, Optional, Union

import torch
import yaml

logger = logging.getLogger(__name__)


def export_model(model, output_dir: str, formats: list = ["onnx", "torchscript"]):
    """
    Export model to various formats.
    
    Args:
        model: YOLO model
        output_dir: Directory to save exported models
        formats: List of formats to export
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle different model types
    if hasattr(model, "model") and hasattr(model.model, "export"):
        # Ultralytics model
        for format_name in formats:
            try:
                model.model.export(format=format_name, save_dir=output_dir)
                logger.info(f"Exported model to {format_name}")
            except Exception as e:
                logger.error(f"Failed to export model to {format_name}: {e}")
    else:
        # Transformers model
        try:
            # Save the PyTorch model
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
            logger.info(f"Saved model to {os.path.join(output_dir, 'model.pt')}")
            
            # Save model configuration
            if hasattr(model, "config"):
                with open(os.path.join(output_dir, "config.yaml"), "w") as f:
                    yaml.dump(model.config.to_dict(), f)
                logger.info(f"Saved model config to {os.path.join(output_dir, 'config.yaml')}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")


def load_yolo_model(config: Dict[str, Any]):
    """
    Load YOLO model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        YOLO model
    """
    model_name = config.get("model_name", "yolov8n")
    model_weights = config.get("model_weights", None)
    use_ultralytics = config.get("use_ultralytics", True)
    
    # Check if ultralytics is available
    ultralytics_available = False
    try:
        import ultralytics
        ultralytics_available = True
    except ImportError:
        logger.warning("Ultralytics not installed. Using transformers implementation.")
    
    # Adjust configuration if ultralytics is not available
    if use_ultralytics and not ultralytics_available:
        use_ultralytics = False
        logger.warning("Ultralytics requested but not available. Using transformers implementation.")
    
    # Load model
    if use_ultralytics:
        from ultralytics import YOLO
        
        if model_weights:
            model = YOLO(model_weights)
        else:
            model = YOLO(f"{model_name}.pt")
    else:
        from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
        
        # Map YOLO model names to huggingface models
        model_mapping = {
            "yolov8n": "hustvl/yolos-small",
            "yolov8s": "hustvl/yolos-small",
            "yolov8m": "hustvl/yolos-base",
            "yolov8l": "hustvl/yolos-base",
            "yolov8x": "hustvl/yolos-large",
        }
        
        hf_model_name = model_mapping.get(model_name, "hustvl/yolos-small")
        
        # Initialize model and feature extractor
        classes = config.get("classes", ["DataMatrix"])
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            hf_model_name,
            size=config.get("img_size", 640),
            max_size=int(config.get("img_size", 640) * 1.5)
        )
        
        model = AutoModelForObjectDetection.from_pretrained(
            hf_model_name,
            num_labels=len(classes),
            ignore_mismatched_sizes=True
        )
        
        # Set class mappings
        id2label = {i: label for i, label in enumerate(classes)}
        label2id = {label: i for i, label in enumerate(classes)}
        model.config.id2label = id2label
        model.config.label2id = label2id
        
        # Bundle with feature extractor
        model.feature_extractor = feature_extractor
    
    return model