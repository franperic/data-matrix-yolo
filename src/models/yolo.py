#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""YOLO model implementations."""

import os
import logging
from typing import Dict, Any, Optional, Union, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class YOLOLightningModule(pl.LightningModule):
    """PyTorch Lightning module for YOLO training."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize YOLO lightning module.
        
        Args:
            config: Training configuration
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Set up model based on implementation
        ultralytics_available = self._check_ultralytics()
        use_ultralytics = config.get("use_ultralytics", True) and ultralytics_available
        
        if use_ultralytics:
            self._setup_ultralytics_model()
        else:
            self._setup_transformers_model()
    
    def _check_ultralytics(self) -> bool:
        """Check if ultralytics is available."""
        try:
            import ultralytics
            return True
        except ImportError:
            logger.warning("Ultralytics not installed. Using transformers implementation.")
            return False
    
    def _setup_ultralytics_model(self):
        """Set up ultralytics YOLO model."""
        from ultralytics import YOLO
        
        model_name = self.config.get("model_name", "yolov8n")
        model_weights = self.config.get("model_weights", None)
        
        if model_weights:
            self.model = YOLO(model_weights)
        else:
            self.model = YOLO(f"{model_name}.pt")
        
        # Set model parameters
        self.model.overrides['optimizer'] = self.config.get("optimizer", "Adam").lower()
        self.model.overrides['lr0'] = self.config.get("learning_rate", 0.001)
        self.model.overrides['weight_decay'] = self.config.get("weight_decay", 0.0005)
        self.model.overrides['momentum'] = self.config.get("momentum", 0.937)
    
    def _setup_transformers_model(self):
        """Set up transformers YOLO model."""
        from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
        
        model_name = self.config.get("model_name", "yolov8n")
        
        # Map YOLO model names to huggingface models
        model_mapping = {
            "yolov8n": "hustvl/yolos-small",
            "yolov8s": "hustvl/yolos-small",
            "yolov8m": "hustvl/yolos-base",
            "yolov8l": "hustvl/yolos-base",
            "yolov8x": "hustvl/yolos-large",
        }
        
        hf_model_name = model_mapping.get(model_name, "hustvl/yolos-small")
        
        # Initialize feature extractor
        img_size = self.config.get("img_size", 640)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            hf_model_name,
            size=img_size,
            max_size=int(img_size * 1.5)
        )
        
        # Initialize model
        classes = self.config.get("classes", ["DataMatrix"])
        self.model = AutoModelForObjectDetection.from_pretrained(
            hf_model_name,
            num_labels=len(classes),
            ignore_mismatched_sizes=True
        )
        
        # Set class mappings
        self.id2label = {i: label for i, label in enumerate(classes)}
        self.label2id = {label: i for i, label in enumerate(classes)}
        self.model.config.id2label = self.id2label
        self.model.config.label2id = self.label2id
        
        # Freeze backbone if specified
        if self.config.get("freeze_backbone", False):
            for param in self.model.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        if hasattr(self, "feature_extractor"):
            # For transformers implementation
            return self.model(pixel_values=x)
        else:
            # For ultralytics implementation
            return self.model(x)
    
    def training_step(self, batch, batch_idx):
        if not hasattr(self, "feature_extractor"):
            # For ultralytics, handled internally
            pass
        else:
            # For transformers implementation
            pixel_values = batch["image"]
            
            # Format labels for DETR/YOLOS
            labels = []
            for i, (boxes, class_labels) in enumerate(zip(batch["boxes"], batch["class_labels"])):
                if len(boxes) > 0:
                    # Convert to COCO format (x, y, width, height)
                    boxes_coco = []
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        w = x2 - x1
                        h = y2 - y1
                        boxes_coco.append([x1, y1, w, h])
                        
                    boxes_tensor = torch.tensor(boxes_coco, device=self.device)
                    class_labels_tensor = torch.tensor(class_labels, device=self.device)
                    
                    # Original image size for normalization
                    img_size = self.config.get("img_size", 640)
                    orig_size = torch.tensor([img_size, img_size], device=self.device)
                    
                    labels.append({
                        "boxes": boxes_tensor,
                        "class_labels": class_labels_tensor,
                        "image_id": torch.tensor([i], device=self.device),
                        "area": (boxes_tensor[:, 2] * boxes_tensor[:, 3]),
                        "iscrowd": torch.zeros_like(class_labels_tensor),
                        "orig_size": orig_size,
                        "size": orig_size
                    })
                else:
                    # Empty sample with no boxes
                    img_size = self.config.get("img_size", 640)
                    orig_size = torch.tensor([img_size, img_size], device=self.device)
                    labels.append({
                        "boxes": torch.zeros((0, 4), device=self.device),
                        "class_labels": torch.zeros(0, dtype=torch.long, device=self.device),
                        "image_id": torch.tensor([i], device=self.device),
                        "area": torch.zeros(0, device=self.device),
                        "iscrowd": torch.zeros(0, dtype=torch.long, device=self.device),
                        "orig_size": orig_size,
                        "size": orig_size
                    })
            
            # Run model
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            
            # Log losses
            loss = outputs.loss
            loss_dict = outputs.loss_dict
            
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            for k, v in loss_dict.items():
                self.log(f"train_{k}", v, on_step=True, on_epoch=True)
                
            return loss
    
    def validation_step(self, batch, batch_idx):
        if not hasattr(self, "feature_extractor"):
            # For ultralytics, handled internally
            pass
        else:
            # For transformers implementation (same as training_step)
            pixel_values = batch["image"]
            
            # Format labels for DETR/YOLOS (same as training_step)
            labels = []
            for i, (boxes, class_labels) in enumerate(zip(batch["boxes"], batch["class_labels"])):
                if len(boxes) > 0:
                    # Convert to COCO format (x, y, width, height)
                    boxes_coco = []
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        w = x2 - x1
                        h = y2 - y1
                        boxes_coco.append([x1, y1, w, h])
                        
                    boxes_tensor = torch.tensor(boxes_coco, device=self.device)
                    class_labels_tensor = torch.tensor(class_labels, device=self.device)
                    
                    # Original image size for normalization
                    img_size = self.config.get("img_size", 640)
                    orig_size = torch.tensor([img_size, img_size], device=self.device)
                    
                    labels.append({
                        "boxes": boxes_tensor,
                        "class_labels": class_labels_tensor,
                        "image_id": torch.tensor([i], device=self.device),
                        "area": (boxes_tensor[:, 2] * boxes_tensor[:, 3]),
                        "iscrowd": torch.zeros_like(class_labels_tensor),
                        "orig_size": orig_size,
                        "size": orig_size
                    })
                else:
                    # Empty sample with no boxes
                    img_size = self.config.get("img_size", 640)
                    orig_size = torch.tensor([img_size, img_size], device=self.device)
                    labels.append({
                        "boxes": torch.zeros((0, 4), device=self.device),
                        "class_labels": torch.zeros(0, dtype=torch.long, device=self.device),
                        "image_id": torch.tensor([i], device=self.device),
                        "area": torch.zeros(0, device=self.device),
                        "iscrowd": torch.zeros(0, dtype=torch.long, device=self.device),
                        "orig_size": orig_size,
                        "size": orig_size
                    })
            
            # Run model
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            
            # Log losses
            loss = outputs.loss
            loss_dict = outputs.loss_dict
            
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)
            for k, v in loss_dict.items():
                self.log(f"val_{k}", v, on_epoch=True)
                
            return loss
    
    def configure_optimizers(self):
        if not hasattr(self, "feature_extractor"):
            # For ultralytics, optimizer is configured internally
            return None
        
        # Select optimizer
        optimizer_name = self.config.get("optimizer", "Adam").lower()
        learning_rate = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 0.0005)
        momentum = self.config.get("momentum", 0.937)
        
        if optimizer_name == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            optimizer = optim.Adam(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # Select scheduler
        scheduler_name = self.config.get("scheduler", "onecycle").lower()
        epochs = self.config.get("epochs", 100)
        
        if scheduler_name == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                epochs=epochs,
                steps_per_epoch=100,  # This will be updated in train_model
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=10000.0
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        
        elif scheduler_name == "cosine":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=epochs,
                T_mult=1,
                eta_min=learning_rate / 100
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
        else:
            # No scheduler
            return optimizer
