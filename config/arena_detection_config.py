"""
Arena detection behavior.

- USE_TEMPLATE_MATCHING_ONLY: If True, skip all detectors and use only
  template matching (assets/arena/*.png). Add one image per troop to assets/arena/.

- USE_PYCLASHBOT_ARENA: If True, prefer PyClashBot-style detection (same algorithm as
  https://github.com/pyclashbot/py-clash-bot detection/image_rec.py): grayscale
  template matching over assets/arena/. No ML model. Set True to "replace" with
  their approach.

- PYCLASHBOT_ARENA_REFERENCE_FOLDER: Folder of reference images (default: assets/arena).
  Use an absolute path or path relative to project root.
"""
USE_TEMPLATE_MATCHING_ONLY = False

# Use py-clash-bot style template matching for arena (no Roboflow/local model needed)
USE_PYCLASHBOT_ARENA = True

# Folder with one image per troop (filename without extension = unit_id). None = assets/arena
PYCLASHBOT_ARENA_REFERENCE_FOLDER = None
