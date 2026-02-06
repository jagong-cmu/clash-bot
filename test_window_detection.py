#!/usr/bin/env python3
"""Simple test script for window detection function."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.coords import get_window_coordinates, list_open_windows


def test_window_detection():
    """Test the window detection function interactively."""
    print("=" * 60)
    print("Window Detection Test")
    print("=" * 60)
    print("\nThis script will test the window detection function.")
    print("Make sure the window you want to detect is open!\n")
    print("What to type: the APP name (e.g. Cursor, Safari) or part of the WINDOW title.")
    print("Either works. Type 'list' to see all apps/windows, or 'quit' to exit.\n")

    while True:
        window_name = input("App name or window title (or 'list' / 'quit'): ").strip()

        if not window_name:
            print("No input. Type a name to search, 'list', or 'quit'.")
            continue

        if window_name.lower() in ("quit", "exit", "q"):
            print("Bye.")
            return

        if window_name.lower() == "list":
            print("\nFetching open windows (requires Accessibility permission)...")
            print("-" * 60)
            pairs = list_open_windows()
            print("-" * 60)
            if pairs:
                print("\nOpen windows (use any part of App or Window name to search):\n")
                for app, title in pairs:
                    print(f"  App: {app}")
                    print(f"  Window: {title}")
                    print()
            else:
                print("Could not list windows. Grant Terminal/Cursor Accessibility access in:")
                print("  System Settings → Privacy & Security → Accessibility")
            continue

        print(f"\nSearching for window containing '{window_name}'...")
        print("-" * 60)

        result = get_window_coordinates(window_name)

        print("-" * 60)

        if result:
            x, y, width, height = result
            print(f"\n✓ SUCCESS! Window detected:")
            print(f"  Position: ({x}, {y})")
            print(f"  Size: {width} x {height}")
            print(f"  Bottom-right corner: ({x + width}, {y + height})")
        else:
            print(f"\n✗ FAILED: Window not found")
            print("\nTips:")
            print("  - Type 'list' to see exact app and window names")
            print("  - Search by app name (e.g. 'Cursor') or window title")
            print("  - Matching is case-insensitive")
            print("  - Grant Accessibility to Terminal/Cursor: System Settings → Privacy & Security → Accessibility")
        print()


if __name__ == "__main__":
    try:
        test_window_detection()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
