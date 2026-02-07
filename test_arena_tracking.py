#!/usr/bin/env python3
"""
Test arena deployment detection and position tracking.

Detects troops on the arena (player and enemy side), assigns stable track IDs,
and keeps their positions updated across frames.

Uses, in order: Roboflow Universe model (if configured), local arena_detector.pth,
or template matching with assets/arena/.

Usage:
  python test_arena_tracking.py
  python test_arena_tracking.py --loop 1.0
  python test_arena_tracking.py --window "iPhone Mirroring"
  python test_arena_tracking.py --save-troops troops_out --troop-crop-size 128
"""

import sys
import os
import time
import argparse

# Reduce terminal noise: disable optional inference models we don't use for arena detection
os.environ.setdefault("QWEN_2_5_ENABLED", "False")
os.environ.setdefault("QWEN_3_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM3_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_YOLO_WORLD_ENABLED", "False")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.coords import get_window_coordinates
from src.capture import capture_screen_region
from src.detection import load_card_templates, ArenaTracker, _get_roboflow_arena_detector, _arena_detector_available, _get_arena_detector, get_arena_region, ARENA_TOP, ARENA_BOTTOM, ARENA_LEFT, ARENA_RIGHT
from src.roboflow_arena_detector import get_unavailable_reason
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARENA_DIR = os.path.join(PROJECT_ROOT, "assets", "arena")


def main():
    parser = argparse.ArgumentParser(description="Test arena unit detection and tracking")
    parser.add_argument("--window", default="iPhone Mirroring", help="Window name to capture")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (detector or templates)")
    parser.add_argument("--loop", type=float, metavar="SEC", default=0, help="Run every SEC seconds (0 = once)")
    parser.add_argument("--show-region", action="store_true", help="Visualize the arena region and save to arena_region.png")
    parser.add_argument("--save-troops", metavar="DIR", help="Save detected troop crops to directory (organized by unit_id)")
    parser.add_argument("--troop-crop-size", type=int, default=128, help="Size of square crop around each detected troop (default: 128)")
    args = parser.parse_args()

    arena_detector = _get_roboflow_arena_detector()
    if not arena_detector:
        arena_detector = _get_arena_detector() if _arena_detector_available() else None
    arena_templates = load_card_templates(ARENA_DIR) if not arena_detector else None

    if arena_detector:
        name = getattr(arena_detector, "__class__", type(arena_detector)).__name__
        if "Roboflow" in name:
            print("Using Roboflow Universe arena model (config/roboflow_arena_config.py)")
            # Check inference version
            try:
                import inference
                import platform
                version = getattr(inference, "__version__", "unknown")
                if version.startswith("1."):
                    print(f"  ✓ inference {version} (should fix resize method errors)")
                elif version.startswith("0."):
                    print(f"  ⚠ inference {version} detected. Upgrade to 1.0.x with 'pip install --upgrade --pre inference' to fix resize method errors.")
                # Note about GPU on Mac
                if platform.system() == "Darwin":
                    print(f"  ℹ On Mac: CoreML/Apple Neural Engine will be used for GPU acceleration (if available)")
            except:
                pass
        else:
            print("Using local arena detector (arena_detector.pth)")
    elif arena_templates:
        print(f"Using template matching: {len(arena_templates)} templates in assets/arena/")
        reason = get_unavailable_reason()
        if reason:
            print(f"  (Roboflow not used: {reason})")
    else:
        print("No arena detector available.")
        print("Option 1: Configure Roboflow model in config/roboflow_arena_config.py and set ROBOFLOW_API_KEY")
        print("Option 2: Train a model: python \"image detector/train_arena_detector.py\" --data-dir data/arena_dataset")
        print("Option 3: Add template images to assets/arena/ (e.g. hog_rider.png)")
        return 1

    coords = get_window_coordinates(args.window)
    if not coords:
        print(f"Window '{args.window}' not found.")
        return 1
    game_x, game_y, game_width, game_height = coords

    # Show arena region visualization if requested
    if args.show_region:
        screen = capture_screen_region(game_x, game_y, game_width, game_height)
        if screen is None or screen.size == 0:
            print("Failed to capture screen.")
            return 1
        h, w = screen.shape[:2]
        x1, y1, x2, y2 = get_arena_region()
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        
        # Draw rectangle on screenshot
        vis = screen.copy()
        cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 0), 3)
        cv2.putText(vis, "Arena Region", (px1 + 10, py1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, f"Top: {ARENA_TOP:.2f}, Bottom: {ARENA_BOTTOM:.2f}", (px1 + 10, py1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f"Left: {ARENA_LEFT:.2f}, Right: {ARENA_RIGHT:.2f}", (px1 + 10, py1 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f"Size: {px2-px1}x{py2-py1} px", (px1 + 10, py1 + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        output_path = os.path.join(PROJECT_ROOT, "arena_region.png")
        cv2.imwrite(output_path, vis)
        print(f"Arena region visualization saved to: {output_path}")
        print(f"Arena region: Top={ARENA_TOP:.2f}, Bottom={ARENA_BOTTOM:.2f}, Left={ARENA_LEFT:.2f}, Right={ARENA_RIGHT:.2f}")
        print(f"Arena crop size: {px2-px1} x {py2-py1} pixels (from {w}x{h} frame)")
        print(f"To adjust, edit config/arena_slots.py")
        return 0

    # Setup save directory if requested
    save_dir = None
    crop_size = args.troop_crop_size
    if args.save_troops:
        save_dir = os.path.abspath(os.path.expanduser(args.save_troops))
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving troop crops to: {save_dir}")
        print(f"Crop size: {crop_size}x{crop_size} pixels")

    tracker = ArenaTracker(max_distance_px=60, max_frames_lost=10)
    if args.loop > 0:
        print(f"Tracking every {args.loop}s. Ctrl+C to stop.")
        time.sleep(1)
    else:
        print("Capturing once in 2s...")
        time.sleep(2)

    # Counter for saved crops per unit
    saved_counts = {}

    def run():
        screen = capture_screen_region(game_x, game_y, game_width, game_height)
        if screen is None or screen.size == 0:
            return
        tracks = tracker.update(
            screen,
            arena_templates=arena_templates,
            threshold=args.threshold,
            scales=[0.7, 1.0, 1.3],
            arena_detector=arena_detector,
        )
        if tracks:
            print("Tracked units (id, unit, side, position):")
            for t in tracks:
                print(f"  [{t.track_id}] {t.unit_id} ({t.side}) at ({t.x}, {t.y}) conf={t.confidence:.2f}")
                
                # Save crop if requested
                if save_dir:
                    h, w = screen.shape[:2]
                    half_size = crop_size // 2
                    x1 = max(0, t.x - half_size)
                    y1 = max(0, t.y - half_size)
                    x2 = min(w, t.x + half_size)
                    y2 = min(h, t.y + half_size)
                    
                    crop = screen[y1:y2, x1:x2]
                    if crop.size > 0:
                        # Resize to exact size if needed (padding if too small)
                        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                            crop = cv2.resize(crop, (crop_size, crop_size))
                        
                        # Create subdirectory for this unit type
                        unit_dir = os.path.join(save_dir, t.unit_id)
                        os.makedirs(unit_dir, exist_ok=True)
                        
                        # Save with unique filename
                        count = saved_counts.get(t.unit_id, 0) + 1
                        saved_counts[t.unit_id] = count
                        filename = f"{t.unit_id}_{count:04d}.png"
                        path = os.path.join(unit_dir, filename)
                        cv2.imwrite(path, crop)
        else:
            print("No units on arena this frame.")
        return screen

    run()
    if args.loop > 0:
        try:
            while True:
                time.sleep(args.loop)
                print("---")
                run()
        except KeyboardInterrupt:
            print("\nStopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
