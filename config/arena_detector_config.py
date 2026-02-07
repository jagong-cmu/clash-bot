"""
Optional: custom path for the arena unit detector model.

If ARENA_MODEL_PATH is None, the detector uses:
  image detector/arena_detector.pth
(where the trainer saves by default).

To use a different checkpoint, set ARENA_MODEL_PATH to a full path or a path
relative to the project root.
"""

# None = use default (image detector/arena_detector.pth)
ARENA_MODEL_PATH = None

# Example: path relative to project root
# ARENA_MODEL_PATH = "models/arena_detector.pth"
