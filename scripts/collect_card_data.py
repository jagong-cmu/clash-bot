#!/usr/bin/env python3
"""
Collect card images from the game window for training the classifier.

Uses the same window detection as test_window_detection (src.coords) and the same
hand-slot regions as detection. Captures the game window, crops the 4 hand slots,
and saves them under data/card_dataset/collect/<class_name>/ so you can later
move images to train/ and val/ for train_card_classifier.py.

Usage:
  python scripts/collect_card_data.py
  python scripts/collect_card_data.py --window "iPhone Mirroring"

Commands at prompt:
  <card_name>  - Save the current 4 slot crops as that class (e.g. knight, fireball).
  list         - List all open windows (to find the right --window name).
  quit         - Exit.
"""

import sys
import os
import argparse

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
from src.coords import get_window_coordinates, list_open_windows
from src.capture import capture_screen_region
from src.detection import DEFAULT_SLOT_REGIONS


COLLECT_DIR = os.path.join(ROOT, "data", "card_dataset", "collect")


def _normalize_class_name(name: str) -> str:
    """Lowercase, replace spaces with underscores."""
    return name.strip().lower().replace(" ", "_") or "unknown"


def save_slot_crops(screen, slot_regions, out_dir: str, prefix: str) -> int:
    """Save one image per slot; return count saved."""
    h, w = screen.shape[:2]
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for slot, (x1, y1, x2, y2) in enumerate(slot_regions):
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        crop = screen[py1:py2, px1:px2]
        if crop.size == 0:
            continue
        # Unique filename: prefix_slot_N.png (N = existing count in dir)
        existing = [f for f in os.listdir(out_dir) if f.startswith(prefix) and f.endswith(".png")]
        n = len(existing)
        path = os.path.join(out_dir, f"{prefix}_slot{n:04d}.png")
        cv2.imwrite(path, crop)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Collect card crops from game window for training")
    parser.add_argument("--window", default="iPhone Mirroring", help="Window name to capture (partial match)")
    args = parser.parse_args()

    print("Card data collector (uses same window detection as test_window_detection)")
    print("=" * 60)
    print(f"Collect dir: {COLLECT_DIR}")
    print("Type a card/class name to save current 4 slots under that class.")
    print("Type 'list' to see open windows, 'quit' to exit.\n")

    coords = get_window_coordinates(args.window)
    if not coords:
        print(f"Window '{args.window}' not found. Try --window with a different name, or type 'list' after starting.")
        print("Starting anyway; you can type 'list' to find the window name.\n")
        coords = None

    while True:
        cmd = input("Class name to save (or 'list' / 'quit'): ").strip()
        if not cmd:
            continue
        if cmd.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        if cmd.lower() == "list":
            print("\nFetching open windows...")
            pairs = list_open_windows()
            if pairs:
                for app, title in pairs:
                    print(f"  App: {app}  |  Window: {title}")
            else:
                print("Could not list windows (Accessibility permission?).")
            print()
            continue

        if not coords:
            coords = get_window_coordinates(args.window)
            if not coords:
                print("Still no window. Use --window or type 'list' to find a name.\n")
                continue
        game_x, game_y, game_width, game_height = coords
        screen = capture_screen_region(game_x, game_y, game_width, game_height)
        if screen is None or screen.size == 0:
            print("Capture failed.\n")
            continue
        class_name = _normalize_class_name(cmd)
        out_dir = os.path.join(COLLECT_DIR, class_name)
        n = save_slot_crops(screen, DEFAULT_SLOT_REGIONS, out_dir, class_name)
        print(f"  Saved {n} crops to {out_dir}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
