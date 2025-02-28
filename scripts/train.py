#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO Training Script for DataMatrix Detection

This script serves as the main entry point for training YOLO models
for DataMatrix detection.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import project modules
from src.config import load_config, save_config
from src.data import create_dataloaders
from src.models import create_model
from src.training import train_model
from src.utils import setup_logging, setup_environment


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train YOLO for DataMatrix detection")
    
    # Configuration arguments
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    
    # Model arguments
    parser.add_argument("--model", type=str, choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"], 
                        help="YOLO model variant")
    parser.add_argument("--weights", type=str, help="Path to pre-trained weights")
    parser.add_argument("--img_size", type=int, help="Image size for training")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--workers", type=int, help="Number of worker threads")
    parser.add_argument("--device", type=str, help="Device to use (cuda, cpu)")
    
    # Logging arguments
    parser.add_argument("--name", type=str, help="Run name")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    
    # Setup environment
    setup_environment()
    
    # Generate run name if not provided
    if not args.name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model or "yolov8n"
        args.name = f"{model_name}_{timestamp}"
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != "config":
            if key in config:
                logger.info(f"Overriding config {key}: {config[key]} -> {value}")
                config[key] = value
            elif key == "lr":
                logger.info(f"Setting learning_rate to {value}")
                config["learning_rate"] = value
            elif key == "model":
                logger.info(f"Setting model_name to {value}")
                config["model_name"] = value
            elif key == "weights":
                logger.info(f"Setting model_weights to {value}")
                config["model_weights"] = value
            elif key == "workers":
                logger.info(f"Setting num_workers to {value}")
                config["num_workers"] = value
            elif key == "wandb":
                logger.info(f"Setting use_wandb to {value}")
                config["use_wandb"] = value
    
    # Create output directory
    output_dir = Path(config.get("output_dir", "outputs")) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output_dir)
    config["run_name"] = args.name
    
    # Save configuration
    config_path = output_dir / "config.yaml"
    save_config(config, config_path)
    logger.info(f"Configuration saved to {config_path}")
    
    try:
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(config)
        logger.info(f"Created dataloaders with {len(train_loader)} training batches")
        
        # Create model
        model = create_model(config)
        logger.info(f"Created model: {config.get('model_name', 'yolov8n')}")
        
        # Train model
        train_model(model, train_loader, val_loader, config)
        logger.info(f"Training completed. Model saved to {output_dir}")
        
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
