#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO Inference Script for DataMatrix Detection

This script provides functionality to run inference on images
using trained YOLO models for DataMatrix detection.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import project modules
from src.config import load_config
from src.models.utils import load_yolo_model
from src.utils import setup_logging, visualize_detections


def detect_with_ultralytics(
    model,
    image_path: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str = None
) -> Dict:
    """
    Detect DataMatrix codes using Ultralytics YOLO.
    
    Args:
        model: Ultralytics YOLO model
        image_path: Path to input image
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        device: Device to run inference on
        
    Returns:
        Dictionary of detection results
    """
    # Run inference
    results = model(
        image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device
    )
    
    # Process results
    detections = []
    
    for result in results:
        boxes = result.boxes
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            confidence = boxes.conf[i].item()
            class_id = int(boxes.cls[i].item())
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "class_id": class_id,
                "class_name": model.names[class_id]
            })
    
    return {
        "image_path": image_path,
        "detections": detections,
        "inference_time": results[0].speed["inference"]
    }


def detect_with_transformers(
    model,
    image_path: str,
    conf_threshold: float = 0.25,
    device: str = None
) -> Dict:
    """
    Detect DataMatrix codes using transformers YOLO.
    
    Args:
        model: transformers model
        image_path: Path to input image
        conf_threshold: Confidence threshold
        device: Device to run inference on
        
    Returns:
        Dictionary of detection results
    """
    import torch
    from PIL import Image
    import time
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare inputs
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Extract features
    feature_extractor = model.feature_extractor
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    
    # Run inference
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Convert outputs to detections
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = feature_extractor.post_process_object_detection(
        outputs,
        threshold=conf_threshold,
        target_sizes=target_sizes
    )[0]
    
    # Process results
    detections = []
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = box.tolist()
        confidence = score.item()
        class_id = label.item()
        class_name = model.config.id2label[class_id]
        
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence,
            "class_id": class_id,
            "class_name": class_name
        })
    
    return {
        "image_path": image_path,
        "detections": detections,
        "inference_time": inference_time
    }


def batch_inference(
    model,
    input_dir: str,
    output_dir: str = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str = None,
    extensions: List[str] = ['.jpg', '.jpeg', '.png'],
    visualize: bool = True,
    class_names: List[str] = None
) -> List[Dict]:
    """
    Run batch inference on a directory of images.
    
    Args:
        model: YOLO model
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        device: Device to run inference on
        extensions: List of valid file extensions
        visualize: Whether to visualize detections
        class_names: List of class names
        
    Returns:
        List of detection results
    """
    logger = logging.getLogger(__name__)
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create output directory
    if output_dir and visualize:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images
    input_dir = Path(input_dir)
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(list(input_dir.glob(f"*{ext}")))
    
    logger.info(f"Found {len(image_paths)} images in {input_dir}")
    
    # Run inference on each image
    results = []
    
    for img_path in image_paths:
        logger.info(f"Processing {img_path}")
        
        try:
            # Run detection
            if hasattr(model, "predict"):
                # Ultralytics model
                result = detect_with_ultralytics(
                    model,
                    str(img_path),
                    conf_threshold,
                    iou_threshold,
                    device
                )
            else:
                # Transformers model
                result = detect_with_transformers(
                    model,
                    str(img_path),
                    conf_threshold,
                    device
                )
            
            # Visualize if requested
            if visualize and output_dir:
                output_path = os.path.join(output_dir, img_path.name)
                visualize_detections(
                    str(img_path),
                    result["detections"],
                    output_path,
                    class_names=class_names
                )
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    return results


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(description="Run inference with YOLO models")
    
    parser.add_argument("--model", required=True, help="Path to model weights or directory")
    parser.add_argument("--input", required=True, help="Path to input image or directory")
    parser.add_argument("--output", help="Path to output image or directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--device", help="Device to run inference on (e.g., cpu, cuda:0)")
    parser.add_argument("--batch", action="store_true", help="Run batch inference on a directory")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")
    parser.add_argument("--classes", nargs="+", default=["DataMatrix"], help="List of class names")
    parser.add_argument("--config", help="Path to configuration file (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--save-json", action="store_true", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    
    # Load configuration if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override configuration with command line arguments
    config["model_weights"] = args.model
    config["conf_threshold"] = args.conf
    config["iou_threshold"] = args.iou
    config["device"] = args.device
    config["classes"] = args.classes
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = load_yolo_model(config)
    logger.info("Model loaded successfully")
    
    # Determine if input is a file or directory
    input_path = args.input
    is_dir = os.path.isdir(input_path)
    
    # Run inference
    if is_dir or args.batch:
        # Batch inference
        results = batch_inference(
            model,
            input_path,
            args.output,
            args.conf,
            args.iou,
            args.device,
            visualize=not args.no_visualize,
            class_names=args.classes
        )
        
        # Print summary
        total_detections = sum(len(r["detections"]) for r in results)
        avg_time = sum(r["inference_time"] for r in results) / len(results) if results else 0
        
        logger.info(f"Processed {len(results)} images")
        logger.info(f"Found {total_detections} detections")
        logger.info(f"Average inference time: {avg_time:.2f} ms")
        
        # Save results to JSON if requested
        if args.save_json and args.output:
            json_path = os.path.join(args.output, "results.json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved results to {json_path}")
    else:
        # Single image inference
        try:
            # Run detection
            if hasattr(model, "predict"):
                # Ultralytics model
                result = detect_with_ultralytics(
                    model,
                    input_path,
                    args.conf,
                    args.iou,
                    args.device
                )
            else:
                # Transformers model
                result = detect_with_transformers(
                    model,
                    input_path,
                    args.conf,
                    args.device
                )
            
            # Print results
            logger.info(f"Found {len(result['detections'])} detections")
            logger.info(f"Inference time: {result['inference_time']:.2f} ms")
            
            for i, det in enumerate(result["detections"]):
                logger.info(f"Detection {i+1}: {det['class_name']} ({det['confidence']:.2f}) at {det['bbox']}")
            
            # Visualize if requested
            if not args.no_visualize:
                output_path = args.output if args.output else "output.jpg"
                visualize_detections(
                    input_path,
                    result["detections"],
                    output_path,
                    class_names=args.classes
                )
                logger.info(f"Visualization saved to {output_path}")
            
            # Save results to JSON if requested
            if args.save_json and args.output:
                json_path = os.path.splitext(output_path)[0] + ".json"
                with open(json_path, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Saved results to {json_path}")
                
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
