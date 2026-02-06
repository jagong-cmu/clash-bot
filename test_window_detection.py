#!/usr/bin/env python3
"""Simple test script for window detection function."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.coords import get_window_coordinates


def test_window_detection():
    """Test the window detection function interactively."""
    print("=" * 60)
    print("Window Detection Test")
    print("=" * 60)
    print("\nThis script will test the window detection function.")
    print("Make sure the window you want to detect is open!\n")
    
    # Get window name from user
    window_name = input("Enter the window name (or partial name) to search for: ").strip()
    
    if not window_name:
        print("No window name provided. Exiting.")
        return
    
    print(f"\nSearching for window containing '{window_name}'...")
    print("-" * 60)
    
    # Test the function
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
        print("  - Make sure the window is open and visible")
        print("  - Try using a partial name (e.g., 'Safari' instead of 'Safari - Google')")
        print("  - Window names are case-insensitive")
        print("  - Check the exact window title in the window's title bar")


if __name__ == "__main__":
    try:
        test_window_detection()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
