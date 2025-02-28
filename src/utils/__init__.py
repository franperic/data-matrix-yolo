#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities package for YOLO training."""

import os
import random
import logging
import numpy as np
import torch

from .logging import setup_logging, get_logger
from .visualization import visualize_detections, plot_metrics

logger = logging.getLogger(__name__)


def setup_environment(seed: int = 42) -> None:
    """
    Set up environment for training.
    
    Args:
        seed: Random seed
    """
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set default tensor type
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Environment setup completed with seed {seed}")


__all__ = [
    "setup_logging",
    "get_logger",
    "visualize_detections",
    "plot_metrics",
    "setup_environment"
]
