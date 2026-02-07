#!/usr/bin/env python3
"""
Debug card detection: visualize slot regions, save crops, and show match scores.

Run with the game window visible to see:
  - Where the code thinks the 4 card slots are (saved as debug_slots.png)
  - What each slot crop looks like (debug_slot_0.png, debug_slot_1.png, ...)
  - Full-frame search: does hog_rider appear anywhere and at what score?
  - Match score for each template in each slot

Usage:
  python debug_detection.py [--window "iPhone Mirroring"]
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from src.coords import get_window_coordinates
from src.capture import capture_screen_region
from src.detection import (
    load_card_templates,
    DEFAULT_SLOT_REGIONS,
    _match_in_region,
    get_slot_best_matches,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CARDS_DIR = os.path.join(PROJECT_ROOT, "assets", "cards")


def main():
    parser = argparse.ArgumentParser(description="Debug card detection regions and matching")
    parser.add_argument("--window", default="iPhone Mirroring", help="Window name to capture")
    args = parser.parse_args()

    templates = load_card_templates(CARDS_DIR)
    if not templates:
        print("No templates in", CARDS_DIR)
        return 1

    window_coords = get_window_coordinates(args.window)
    if not window_coords:
        print(f"Window '{args.window}' not found.")
        return 1

    gx, gy, gw, gh = window_coords
    print(f"Window: ({gx}, {gy}) size {gw}x{gh}")

    print("\nCapturing in 2 seconds... Focus the game window.")
    import time
    time.sleep(2)

    screen = capture_screen_region(gx, gy, gw, gh)
    if screen is None or screen.size == 0:
        print("Capture failed.")
        return 1

    h, w = screen.shape[:2]
    print(f"Frame size: {w}x{h}")

    # 1. Draw slot regions on the frame
    debug_frame = screen.copy()
    for i, region in enumerate(DEFAULT_SLOT_REGIONS):
        x1, y1, x2, y2 = region
        px1 = int(x1 * w)
        py1 = int(y1 * h)
        px2 = int(x2 * w)
        py2 = int(y2 * h)
        cv2.rectangle(debug_frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cv2.putText(debug_frame, f"Slot {i}", (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite("debug_slots.png", debug_frame)
    print("\nSaved debug_slots.png — green boxes show where the code searches for cards.")
    print("  If the boxes DON'T overlap your cards, the HAND_TOP/BOTTOM/LEFT/RIGHT in detection.py are wrong for your setup.")

    # 2. Save each slot crop
    for i, region in enumerate(DEFAULT_SLOT_REGIONS):
        x1, y1, x2, y2 = region
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        crop = screen[py1:py2, px1:px2]
        cv2.imwrite(f"debug_slot_{i}.png", crop)
    print(f"Saved debug_slot_0.png .. debug_slot_3.png — the actual crops being searched.")

    # 3. Per-slot, per-template scores
    print("\n--- Match scores per slot (template -> score) ---")
    for slot, region in enumerate(DEFAULT_SLOT_REGIONS):
        print(f"\nSlot {slot}:")
        for card_id, template in templates.items():
            score = _match_in_region(screen, h, w, template, region)
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {card_id:15} {score:.3f} {bar}")

    # 4. Best match per slot
    best_per_slot = get_slot_best_matches(screen, templates)
    print("\n--- Best match per slot ---")
    for slot, card_id, score in best_per_slot:
        print(f"  Slot {slot}: {card_id or '?'} = {score:.3f}")

    # 5. Full-frame search: where does hog_rider appear with best score?
    if "hog_rider" in templates:
        tmpl = templates["hog_rider"]
        th, tw = tmpl.shape[:2]
        if h >= th and w >= tw:
            result = cv2.matchTemplate(screen, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            cx = max_loc[0] + tw // 2
            cy = max_loc[1] + th // 2
            print(f"\n--- Full-frame search for hog_rider ---")
            print(f"  Best score: {max_val:.3f} at pixel ({cx}, {cy})")
            print(f"  Is that inside a slot? Slot regions use normalized coords 0-1.")
            for i, region in enumerate(DEFAULT_SLOT_REGIONS):
                x1, y1, x2, y2 = region
                nx, ny = cx / w, cy / h
                in_slot = x1 <= nx <= x2 and y1 <= ny <= y2
                print(f"  Slot {i}: x=[{x1:.2f}-{x2:.2f}] y=[{y1:.2f}-{y2:.2f}] -> point ({nx:.2f},{ny:.2f}) {'INSIDE' if in_slot else 'outside'}")

    print("\nDone. Check debug_slots.png — do the green boxes cover your card hand?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
