"""
Arena crop region (the playable battlefield where troops are detected).

Values are fractions of frame width/height (0.0 = top/left, 1.0 = bottom/right).
The detector only sees this crop; if it's wrong, detection will fail.

To tune: run "python test_arena_tracking.py --save-arena arena_crop.png" and open
the saved image. It should show only the battlefield (no hand, no top UI).
Adjust these until the crop matches. Defaults are in src/detection.py.
"""

# Vertical range of the arena (top and bottom as fraction of frame height)
ARENA_TOP = 0.12
ARENA_BOTTOM = 0.70

# Horizontal range (left and right as fraction of frame width)
ARENA_LEFT = 0.02
ARENA_RIGHT = 0.98
