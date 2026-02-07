#!/usr/bin/env python3
"""
Collect card images from the game window for training the classifier.

You specify which slot (1–4, left to right) each card is in so labels are correct.
Saves one crop per entry under data/card_dataset/collect/<class_name>/.

Usage:
  python scripts/collect_card_data.py
  python scripts/collect_card_data.py --window "iPhone Mirroring"

At prompt:
  <card_name> <slot>  - Save the crop from that slot only (e.g. knight 1, fireball 3).
  <card_name>         - You'll be asked "Which slot (1-4)?"
  list                - List open windows.
  quit                - Exit.

Slots: 1 = leftmost card, 4 = rightmost.
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


def save_one_slot(screen, slot_regions, slot_1based: int, out_dir: str, class_name: str) -> bool:
    """Save only the specified slot (1-4) as one image. Returns True if saved."""
    if not 1 <= slot_1based <= 4:
        return False
    slot = slot_1based - 1  # 0-based index
    h, w = screen.shape[:2]
    x1, y1, x2, y2 = slot_regions[slot]
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)
    crop = screen[py1:py2, px1:px2]
    if crop.size == 0:
        return False
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir) if f.endswith(".png")]
    n = len(existing)
    path = os.path.join(out_dir, f"{class_name}_{n:04d}.png")
    cv2.imwrite(path, crop)
    return True


def main():
    parser = argparse.ArgumentParser(description="Collect card crops from game window for training")
    parser.add_argument("--window", default="iPhone Mirroring", help="Window name to capture (partial match)")
    args = parser.parse_args()

    print("Card data collector (slot = which card position, 1=left .. 4=right)")
    print("=" * 60)
    print(f"Collect dir: {COLLECT_DIR}")
    print("Enter:  <card_name> <slot>   e.g.  knight 1   fireball 3")
    print("Or:     <card_name>          then you'll be asked which slot (1-4).")
    print("        list / quit\n")

    coords = get_window_coordinates(args.window)
    if not coords:
        print(f"Window '{args.window}' not found. Try --window with a different name, or type 'list' after starting.")
        print("Starting anyway; you can type 'list' to find the window name.\n")
        coords = None

    while True:
        cmd = input("Card and slot (e.g. knight 1) or 'list'/'quit': ").strip()
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

        parts = cmd.split()
        if not parts:
            continue
        class_name = _normalize_class_name(parts[0])
        slot_input = None
        if len(parts) >= 2:
            try:
                slot_input = int(parts[1])
            except ValueError:
                pass
        if slot_input is None or not (1 <= slot_input <= 4):
            while True:
                slot_str = input(f"  Which slot is '{class_name}' in? (1=left, 4=right): ").strip()
                try:
                    slot_input = int(slot_str)
                    if 1 <= slot_input <= 4:
                        break
                except ValueError:
                    pass
                print("  Enter a number 1, 2, 3, or 4.")

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
        out_dir = os.path.join(COLLECT_DIR, class_name)
        if save_one_slot(screen, DEFAULT_SLOT_REGIONS, slot_input, out_dir, class_name):
            print(f"  Saved 1 image to {out_dir}\n")
        else:
            print("  Save failed.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
