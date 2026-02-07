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
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from src.coords import get_window_coordinates
from src.capture import capture_screen_region
from src.detection import (
    load_card_templates,
    ArenaTracker,
    get_arena_region,
    _get_roboflow_arena_detector,
    _arena_detector_available,
    _get_arena_detector,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARENA_DIR = os.path.join(PROJECT_ROOT, "assets", "arena")


def main():
    parser = argparse.ArgumentParser(description="Test arena unit detection and tracking")
    parser.add_argument("--window", default="iPhone Mirroring", help="Window name to capture")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (detector or templates)")
    parser.add_argument("--loop", type=float, metavar="SEC", default=0, help="Run every SEC seconds (0 = once)")
    parser.add_argument("--save-arena", metavar="PATH", default=None, help="Save the arena crop to PATH and exit (to verify crop region)")
    parser.add_argument("--save-troops", metavar="DIR", default=None, help="Save a crop of each detected troop to DIR (filename: troop_<id>_<unit>_<conf>.png)")
    parser.add_argument("--troop-crop-size", type=int, default=96, help="Side length of the square crop around each troop (default 96)")
    args = parser.parse_args()

    arena_detector = _get_roboflow_arena_detector()
    if not arena_detector:
        arena_detector = _get_arena_detector() if _arena_detector_available() else None
    arena_templates = load_card_templates(ARENA_DIR) if not arena_detector else None

    if arena_detector:
        name = getattr(arena_detector, "__class__", type(arena_detector)).__name__
        if "Roboflow" in name:
            print("Using Roboflow Universe arena model (config/roboflow_arena_config.py)")
        else:
            print("Using local arena detector (arena_detector.pth)")
    elif arena_templates:
        print(f"Using template matching: {len(arena_templates)} templates in assets/arena/")
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

    if args.save_arena:
        print("Capturing in 2s to save arena crop...")
        time.sleep(2)
        screen = capture_screen_region(game_x, game_y, game_width, game_height)
        if screen is None or screen.size == 0:
            print("Capture failed.")
            return 1
        x1, y1, x2, y2 = get_arena_region()
        h, w = screen.shape[:2]
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        arena_crop = screen[py1:py2, px1:px2]
        path = os.path.abspath(os.path.expanduser(args.save_arena))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cv2.imwrite(path, arena_crop)
        print("Saved arena crop to:", path)
        print("Open it and check: it should show only the battlefield (no hand, no top UI).")
        print("If not, edit config/arena_region.py (ARENA_TOP/BOTTOM/LEFT/RIGHT).")
        return 0

    tracker = ArenaTracker(max_distance_px=60, max_frames_lost=10)
    save_troops_dir = os.path.abspath(os.path.expanduser(args.save_troops)) if args.save_troops else None
    if save_troops_dir:
        os.makedirs(save_troops_dir, exist_ok=True)
        print("Saving troop crops to:", save_troops_dir)
    half = max(1, args.troop_crop_size // 2)
    run_counter = 0

    if args.loop > 0:
        print(f"Tracking every {args.loop}s. Ctrl+C to stop.")
        time.sleep(1)
    else:
        print("Capturing once in 2s...")
        time.sleep(2)

    def run():
        nonlocal run_counter
        screen = capture_screen_region(game_x, game_y, game_width, game_height)
        if screen is None or screen.size == 0:
            return None
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
                if save_troops_dir and screen is not None:
                    h, w = screen.shape[:2]
                    x1 = max(0, t.x - half)
                    y1 = max(0, t.y - half)
                    x2 = min(w, t.x + half)
                    y2 = min(h, t.y + half)
                    if x2 > x1 and y2 > y1:
                        crop = screen[y1:y2, x1:x2]
                        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(t.unit_id))
                        fname = f"troop_{run_counter}_{t.track_id}_{safe_name}_{t.confidence:.2f}.png"
                        path = os.path.join(save_troops_dir, fname)
                        cv2.imwrite(path, crop)
                        print(f"    -> saved: {fname}")
        else:
            print("No units on arena this frame.")
        run_counter += 1
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
    elif save_troops_dir:
        print("Troop images saved to:", save_troops_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
