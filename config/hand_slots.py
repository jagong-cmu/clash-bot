"""
Hand slot region (where the 4 cards appear at the bottom of the screen).

Values are fractions of frame width/height (0.0 = top/left, 1.0 = bottom/right).
Tune these until "python test_detection.py --save-slots slots/" shows
one card per image in slot_0.png .. slot_3.png, then detection will use them.

Default (if this file is missing or values are wrong): see src/detection.py
"""

# Vertical range of the hand strip (top and bottom as fraction of frame height)
HAND_TOP = 0.72
HAND_BOTTOM = 0.98

# Horizontal range (left and right as fraction of frame width)
HAND_LEFT = 0.05
HAND_RIGHT = 0.95
