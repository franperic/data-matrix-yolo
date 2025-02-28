#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Custom callbacks for YOLO training."""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class ModelExportCallback(Callback):
    """Callback to export model during training."""
    
    def __init__(self, output_dir: str, export_every_n_epochs: int = 10):
        """
        Initialize model export callback.
        
        Args:
            output_dir: Directory to save exported models
            export_every_n_epochs: Export model every N epochs
        """
        super().__init__()
        self.output_dir = output_dir
        self.export_every_n_epochs = export_every_n_epochs
    
    def on_epoch_end(self, trainer, pl_module):
        """Call on epoch end."""
        epoch = trainer.current_epoch
        if epoch > 0 and epoch % self.export_every_n_epochs == 0:
            try:
                from ..models.utils import export_model
                
                # Create export directory
                export_dir = Path(self.output_dir) / f"epoch_{epoch}"
                os.makedirs(export_dir, exist_ok=True)
                
                # Export model
                export_model(pl_module, str(export_dir))
                logger.info(f"Exported model at epoch {epoch}")
            except Exception as e:
                logger.error(f"Failed to export model at epoch {epoch}: {e}")


class VisualizationCallback(Callback):
    """Callback to visualize predictions during training."""
    
    def __init__(self, output_dir: str, val_samples: list, visualize_every_n_epochs: int = 10):
        """
        Initialize visualization callback.
        
        Args:
            output_dir: Directory to save visualizations
            val_samples: List of validation samples to visualize
            visualize_every_n_epochs: Visualize every N epochs
        """
        super().__init__()
        self.output_dir = output_dir
        self.val_samples = val_samples
        self.visualize_every_n_epochs = visualize_every_n_epochs
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Call on validation epoch end."""
        epoch = trainer.current_epoch
        if epoch > 0 and epoch % self.visualize_every_n_epochs == 0:
            try:
                # Create visualization directory
                vis_dir = Path(self.output_dir) / f"epoch_{epoch}_vis"
                os.makedirs(vis_dir, exist_ok=True)
                
                # Visualize predictions
                if not hasattr(pl_module, "feature_extractor"):
                    # Ultralytics model
                    self._visualize_ultralytics(pl_module, vis_dir, epoch)
                else:
                    # Transformers model
                    self._visualize_transformers(pl_module, vis_dir, epoch)
                    
                logger.info(f"Visualized predictions at epoch {epoch}")
            except Exception as e:
                logger.error(f"Failed to visualize predictions at epoch {epoch}: {e}")
    
    def _visualize_ultralytics(self, model, vis_dir, epoch):
        """Visualize predictions for ultralytics model."""
        # Implementation depends on ultralytics API
        pass
    
    def _visualize_transformers(self, model, vis_dir, epoch):
        """Visualize predictions for transformers model."""
        # Implementation depends on transformers API
        pass
