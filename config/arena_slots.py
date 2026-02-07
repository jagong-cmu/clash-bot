"""
Arena region (playable battlefield where troops appear).

Values are fractions of frame width/height (0.0 = top/left, 1.0 = bottom/right).
Adjust these to match your game window's arena area.

Default (if this file is missing): see src/detection.py
ARENA_TOP = 0.12, ARENA_BOTTOM = 0.70, ARENA_LEFT = 0.02, ARENA_RIGHT = 0.98
"""

# Vertical range of the arena (top and bottom as fraction of frame height)
ARENA_TOP = 0.15
ARENA_BOTTOM = 0.75

# Horizontal range (left and right as fraction of frame width)
ARENA_LEFT = 0.02
ARENA_RIGHT = 0.98
