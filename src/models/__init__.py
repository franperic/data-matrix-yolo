#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Models package for YOLO training."""

from .yolo import YOLOLightningModule
from .utils import export_model, load_yolo_model


def create_model(config):
    """
    Create model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        YOLO model
    """
    return YOLOLightningModule(config)


__all__ = ["YOLOLightningModule", "export_model", "load_yolo_model", "create_model"]
