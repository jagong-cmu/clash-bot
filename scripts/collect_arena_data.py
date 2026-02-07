#!/usr/bin/env python3
"""
Collect arena troop detection training data using Roboflow detector.

Captures arena screenshots and uses Roboflow to auto-label them with bounding boxes,
saving in COCO format for training RetinaNet.

Usage:
  python scripts/collect_arena_data.py --output-dir data/arena_dataset/collect
  python scripts/collect_arena_data.py --output-dir data/arena_dataset/collect --loop 2.0 --threshold 0.5
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coords import get_window_coordinates
from src.capture import capture_screen_region
from src.detection import get_arena_region, _get_roboflow_arena_detector
import cv2
import numpy as np

# Reduce terminal noise
os.environ.setdefault("QWEN_2_5_ENABLED", "False")
os.environ.setdefault("QWEN_3_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM3_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_YOLO_WORLD_ENABLED", "False")


def extract_bbox_from_roboflow_prediction(prediction: Any, img_w: int, img_h: int) -> tuple:
    """
    Extract bounding box from Roboflow prediction.
    Returns (x, y, width, height) in pixels, or None if not available.
    """
    def _get(o, key, default=None):
        return getattr(o, key, default) if not isinstance(o, dict) else o.get(key, default)
    
    x = _get(prediction, "x")
    y = _get(prediction, "y")
    width = _get(prediction, "width")
    height = _get(prediction, "height")
    
    # Try to get bbox directly
    if width is not None and height is not None:
        if width <= 1 and height <= 1:
            # Normalized
            x_px = int(float(x) * img_w) if x is not None else 0
            y_px = int(float(y) * img_h) if y is not None else 0
            w_px = int(float(width) * img_w)
            h_px = int(float(height) * img_h)
        else:
            # Already in pixels
            x_px = int(float(x)) if x is not None else 0
            y_px = int(float(y)) if y is not None else 0
            w_px = int(float(width))
            h_px = int(float(height))
        return (x_px, y_px, w_px, h_px)
    
    # Try bbox/box/xyxy format
    bbox = _get(prediction, "bbox") or _get(prediction, "box") or _get(prediction, "xyxy")
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        if x2 <= 1 and y2 <= 1:
            # Normalized
            x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
        x_px = int(x1)
        y_px = int(y1)
        w_px = int(x2 - x1)
        h_px = int(y2 - y1)
        return (x_px, y_px, w_px, h_px)
    
    return None


def get_roboflow_predictions_with_bbox(model: Any, image_bgr: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
    """
    Get Roboflow predictions with full bounding box information.
    Returns list of dicts with: unit_id, confidence, bbox (x, y, width, height)
    """
    image_rgb = image_bgr[:, :, ::-1].copy()
    try:
        results = model.infer(image_rgb)
    except Exception:
        results = model.infer(image_bgr)
    
    if not results:
        return []
    
    r0 = results[0] if isinstance(results, (list, tuple)) else results
    predictions = getattr(r0, "predictions", None)
    if not predictions:
        if hasattr(r0, "dict"):
            d = r0.dict(by_alias=True, exclude_none=True)
            predictions = d.get("predictions", [])
        else:
            return []
    
    h, w = image_bgr.shape[:2]
    out = []
    
    def _get(o, key, default=None):
        return getattr(o, key, default) if not isinstance(o, dict) else o.get(key, default)
    
    for p in predictions:
        conf = float(_get(p, "confidence") or 0)
        if conf < confidence_threshold:
            continue
        
        class_name = _get(p, "class_name") or _get(p, "class")
        if not class_name:
            continue
        
        bbox = extract_bbox_from_roboflow_prediction(p, w, h)
        if bbox is None:
            # Fallback: create bbox around center point (estimate size)
            cx = _get(p, "x")
            cy = _get(p, "y")
            if cx is None or cy is None:
                continue
            if abs(float(cx)) <= 1 and abs(float(cy)) <= 1:
                cx, cy = int(float(cx) * w), int(float(cy) * h)
            else:
                cx, cy = int(float(cx)), int(float(cy))
            # Estimate bbox size (you may need to adjust these defaults)
            est_size = min(w, h) // 10  # Rough estimate
            bbox = (max(0, cx - est_size // 2), max(0, cy - est_size // 2), est_size, est_size)
        
        out.append({
            "unit_id": str(class_name),
            "confidence": conf,
            "bbox": bbox  # (x, y, width, height)
        })
    
    return out


def main():
    parser = argparse.ArgumentParser(description="Collect arena training data using Roboflow auto-labeling")
    parser.add_argument("--output-dir", default="data/arena_dataset/collect", help="Output directory for images and annotations")
    parser.add_argument("--window", default="iPhone Mirroring", help="Window name to capture")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--loop", type=float, metavar="SEC", default=0, help="Capture every SEC seconds (0 = once)")
    args = parser.parse_args()
    
    # Check Roboflow is available
    detector = _get_roboflow_arena_detector()
    if not detector:
        print("Error: Roboflow detector not available.")
        print("Configure config/roboflow_arena_config.py and set ROBOFLOW_API_KEY")
        return 1
    
    print("Using Roboflow Universe model for auto-labeling")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO data structures
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Category mapping (will be built as we encounter units)
    category_map = {}  # unit_id -> category_id
    next_category_id = 0
    next_image_id = 1
    next_annotation_id = 1
    
    coords = get_window_coordinates(args.window)
    if not coords:
        print(f"Window '{args.window}' not found.")
        return 1
    game_x, game_y, game_width, game_height = coords
    
    if args.loop > 0:
        print(f"Collecting data every {args.loop}s. Press Ctrl+C to stop.")
        time.sleep(1)
    else:
        print("Capturing once in 2s...")
        time.sleep(2)
    
    def capture_and_label():
        nonlocal next_image_id, next_annotation_id
        
        screen = capture_screen_region(game_x, game_y, game_width, game_height)
        if screen is None or screen.size == 0:
            return
        
        h, w = screen.shape[:2]
        x1, y1, x2, y2 = get_arena_region()
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        arena_crop = screen[py1:py2, px1:px2]
        
        if arena_crop.size == 0:
            print("Arena crop is empty")
            return
        
        # Get predictions with bboxes
        # Access the underlying model from RoboflowArenaDetector
        model = detector._model if hasattr(detector, '_model') else detector
        predictions = get_roboflow_predictions_with_bbox(model, arena_crop, args.threshold)
        
        if not predictions:
            print("No detections in this frame")
            return
        
        # Save image
        image_filename = f"frame_{next_image_id:04d}.png"
        image_path = images_dir / image_filename
        cv2.imwrite(str(image_path), arena_crop)
        
        # Add image to COCO
        crop_h, crop_w = arena_crop.shape[:2]
        image_info = {
            "id": next_image_id,
            "file_name": image_filename,
            "width": crop_w,
            "height": crop_h
        }
        coco_data["images"].append(image_info)
        
        # Process annotations
        for pred in predictions:
            unit_id = pred["unit_id"]
            bbox = pred["bbox"]  # (x, y, width, height)
            
            # Add category if new
            if unit_id not in category_map:
                category_map[unit_id] = next_category_id
                coco_data["categories"].append({
                    "id": next_category_id,
                    "name": unit_id
                })
                next_category_id += 1
            
            category_id = category_map[unit_id]
            
            # Add annotation (COCO format: [x, y, width, height])
            annotation = {
                "id": next_annotation_id,
                "image_id": next_image_id,
                "category_id": category_id,
                "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],  # [x, y, width, height]
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation)
            next_annotation_id += 1
        
        print(f"Saved frame_{next_image_id:04d}.png with {len(predictions)} detections")
        next_image_id += 1
        
        # Save annotations.json after each capture
        annotations_path = output_dir / "annotations.json"
        with open(annotations_path, "w") as f:
            json.dump(coco_data, f, indent=2)
    
    capture_and_label()
    
    if args.loop > 0:
        try:
            while True:
                time.sleep(args.loop)
                capture_and_label()
        except KeyboardInterrupt:
            print("\nStopped collecting.")
    
    # Final save
    annotations_path = output_dir / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nCollected {len(coco_data['images'])} images")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Categories: {[c['name'] for c in coco_data['categories']]}")
    print(f"\nSaved to: {output_dir}")
    print(f"  - images/: {len(coco_data['images'])} images")
    print(f"  - annotations.json: COCO format")
    print(f"\nTo train: python \"image detector/train_arena_detector.py\" --data-dir {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
