#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training functions for YOLO."""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

logger = logging.getLogger(__name__)


def train_with_ultralytics(model, config: Dict[str, Any]):
    """
    Train YOLO using ultralytics.
    
    Args:
        model: Ultralytics YOLO model
        config: Training configuration
    """
    # Create data YAML if not already exists
    data_dir = config.get("data_dir")
    output_dir = config.get("output_dir")
    
    # Create YAML config for ultralytics
    data_yaml = {
        "train": str(Path(data_dir) / "train"),
        "val": str(Path(data_dir) / "val"),
        "nc": len(config.get("classes", ["DataMatrix"])),
        "names": config.get("classes", ["DataMatrix"])
    }
    
    data_yaml_path = Path(output_dir) / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f)
    
    # Start training
    model.train(
        data=str(data_yaml_path),
        epochs=config.get("epochs", 100),
        imgsz=config.get("img_size", 640),
        batch=config.get("batch_size", 16),
        patience=config.get("patience", 10),
        optimizer=config.get("optimizer", "Adam").lower(),
        project=config.get("project_name", "datamatrix-detection"),
        name=config.get("run_name", "yolo-datamatrix"),
        device=0 if torch.cuda.is_available() else "cpu",
        pretrained=True,
        lr0=config.get("learning_rate", 0.001),
        lrf=0.01,  # Final learning rate factor
        weight_decay=config.get("weight_decay", 0.0005),
        momentum=config.get("momentum", 0.937),
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        amp=config.get("use_mixed_precision", True),
    )
    
    # Export model
    model.export(format="onnx")
    model.export(format="torchscript")
    
    return model


def train_with_lightning(model, train_loader, val_loader, config: Dict[str, Any]):
    """
    Train YOLO using PyTorch Lightning.
    
    Args:
        model: Lightning module
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
    """
    # Set up loggers
    output_dir = config.get("output_dir", "outputs")
    project_name = config.get("project_name", "datamatrix-detection")
    run_name = config.get("run_name", "yolo-datamatrix")
    
    loggers = [TensorBoardLogger(save_dir=output_dir, name=project_name, version=run_name)]
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/{run_name}/checkpoints",
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=config.get("patience", 10),
        mode="min",
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    progress_bar = TQDMProgressBar(refresh_rate=10)
    
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor, progress_bar]
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=config.get("epochs", 100),
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=loggers,
        callbacks=callbacks,
        precision="16-mixed" if config.get("use_mixed_precision", True) and torch.cuda.is_available() else 32,
        gradient_clip_val=0.1,
        log_every_n_steps=10,
        deterministic=True
    )
    
    # Update steps_per_epoch in OneCycleLR scheduler if used
    if hasattr(model, "configure_optimizers"):
        opt_config = model.configure_optimizers()
        if isinstance(opt_config, dict) and "lr_scheduler" in opt_config:
            scheduler_config = opt_config["lr_scheduler"]
            scheduler = scheduler_config["scheduler"]
            if hasattr(scheduler, "total_steps"):
                # Update total_steps for OneCycleLR
                steps_per_epoch = len(train_loader)
                epochs = config.get("epochs", 100)
                scheduler.total_steps = steps_per_epoch * epochs
                logger.info(f"Updated scheduler total_steps to {scheduler.total_steps}")
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Export model
    from ..models.utils import export_model
    export_dir = Path(output_dir) / run_name / "exported"
    export_model(model, str(export_dir))
    
    return model


def train_model(model, train_loader, val_loader, config: Dict[str, Any]):
    """
    Train YOLO model based on configuration.
    
    Args:
        model: YOLO model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
    """
    # Check if model is ultralytics or lightning
    if hasattr(model, "model") and hasattr(model.model, "train"):
        # Ultralytics model
        return train_with_ultralytics(model.model, config)
    else:
        # Lightning model
        return train_with_lightning(model, train_loader, val_loader, config)
