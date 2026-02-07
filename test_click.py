#!/usr/bin/env python3
"""Test that click_in_window and click_at work with the detected window."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.coords import get_window_coordinates, list_open_windows
from src.control import click_at, click_in_window


def test_click():
    print("=" * 60)
    print("Mouse click test (window-relative)")
    print("=" * 60)
    print("\nThis will find a window, wait 5 seconds, then click ONCE inside it.")
    print("Use a safe window (e.g. TextEdit, Notes, or a blank area in Cursor).")
    print("Move mouse to screen corner to trigger PyAutoGUI failsafe and stop.\n")

    name = input("Window name to target (or Enter for 'Cursor'): ").strip() or "Cursor"
    if name.lower() in ("list", "l"):
        print("\nFetching open windows...")
        pairs = list_open_windows()
        if pairs:
            for app, title in pairs:
                print(f"  {app}  →  {title}")
        print("\nRe-run and type one of the names above.")
        return

    print(f"\nLooking for window containing '{name}'...")
    window_coords = get_window_coordinates(name)
    if not window_coords:
        print("Window not found. Try 'list' to see names, or run test_window_detection.py")
        return

    x, y, w, h = window_coords
    # Click near center of window (visible and safe)
    rel_x, rel_y = w // 2, h // 2
    abs_x, abs_y = x + rel_x, y + rel_y

    print(f"Window at ({x}, {y}), size {w}x{h}")
    print(f"Will click at relative ({rel_x}, {rel_y}) → screen ({abs_x}, {abs_y})")
    print("\nFocus that window, then wait for the countdown...\n")

    for i in range(5, 0, -1):
        print(f"  Click in {i}...")
        time.sleep(1)
    print("  Clicking now.")
    click_in_window(rel_x, rel_y, window_coords, delay=0)
    print("Done. Did the click land inside the window?")
    print("(If not, check that the window wasn't covered or moved.)")


if __name__ == "__main__":
    try:
        test_click()
    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
