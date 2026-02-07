"""
Arena detection using the same algorithm as py-clash-bot.

Uses [py-clash-bot](https://github.com/pyclashbot/py-clash-bot)'s image_rec approach:
grayscale template matching (cv2.matchTemplate, TM_CCOEFF_NORMED) over reference
images. One image = one template; use a subfolder per troop to add multiple
templates for the same unit (e.g. assets/arena/hog_rider/frame1.png, frame2.png).

Based on: pyclashbot/detection/image_rec.py (find_references, compare_images).
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _load_one_image(path: str) -> Optional[np.ndarray]:
    """Load a single image (BGR). Supports png, jpg, webp."""
    img = cv2.imread(path)
    if img is None and path.lower().endswith(".webp"):
        try:
            from PIL import Image
            pil_img = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    return img


def _load_reference_images(folder: str) -> List[Tuple[str, np.ndarray]]:
    """
    Load reference images from folder. Returns list of (unit_id, BGR image).

    - Top-level: hog_rider.png -> one template with unit_id "hog_rider".
    - Subfolders: hog_rider/frame1.png, hog_rider/frame2.png -> multiple templates
      all with unit_id "hog_rider". Use subfolders to add several images per troop.
    """
    out: List[Tuple[str, np.ndarray]] = []
    if not os.path.isdir(folder):
        return out
    exts = (".png", ".jpg", ".jpeg", ".webp")
    # 1) Single images in the folder root: unit_id = filename without extension
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path) or not name.lower().endswith(exts):
            continue
        img = _load_one_image(path)
        if img is not None:
            base = os.path.splitext(name)[0]
            out.append((base, img))
    # 2) Subfolders: unit_id = subfolder name, all images inside = templates for that unit
    for name in sorted(os.listdir(folder)):
        subdir = os.path.join(folder, name)
        if not os.path.isdir(subdir) or name.startswith("."):
            continue
        for fname in sorted(os.listdir(subdir)):
            if not fname.lower().endswith(exts):
                continue
            path = os.path.join(subdir, fname)
            img = _load_one_image(path)
            if img is not None:
                out.append((name, img))
    return out


def _compare_images(
    image_gray: np.ndarray,
    template_gray: np.ndarray,
    threshold: float,
) -> Optional[Tuple[float, int, int]]:
    """
    PyClashBot-style: match template in image, return (score, x_center, y_center) or None.
    Returns center of match in image coordinates.
    """
    if template_gray.shape[0] > image_gray.shape[0] or template_gray.shape[1] > image_gray.shape[1]:
        return None
    res = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val >= threshold:
        th, tw = template_gray.shape[:2]
        cx = max_loc[0] + tw // 2
        cy = max_loc[1] + th // 2
        return (float(max_val), cx, cy)
    return None


class PyClashBotArenaDetector:
    """
    Arena detector using py-clash-bot's template-matching approach.
    .predict(image_bgr, confidence_threshold) -> list of (unit_id, conf, center_x, center_y).
    """

    def __init__(self, reference_folder: str):
        self.reference_folder = reference_folder
        self._templates: List[Tuple[str, np.ndarray]] = _load_reference_images(reference_folder)
        self._gray: List[Tuple[str, np.ndarray]] = [
            (name, cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY))
            for name, tpl in self._templates
        ]

    def predict(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> List[Tuple[str, float, int, int]]:
        """Return list of (unit_id, confidence, center_x, center_y) in image coordinates."""
        if not self._gray:
            return []
        img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        results: List[Tuple[str, float, int, int]] = []
        for name, tpl_gray in self._gray:
            hit = _compare_images(img_gray, tpl_gray, confidence_threshold)
            if hit is not None:
                score, cx, cy = hit
                results.append((name, score, cx, cy))
        # Deduplicate by position (same unit matched multiple times)
        results.sort(key=lambda x: -x[1])
        seen: List[Tuple[int, int]] = []
        out: List[Tuple[str, float, int, int]] = []
        for unit_id, conf, cx, cy in results:
            if any(abs(cx - sx) < 30 and abs(cy - sy) < 30 for sx, sy in seen):
                continue
            seen.append((cx, cy))
            out.append((unit_id, conf, cx, cy))
        return out


def _get_reference_folder() -> str:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        from config.arena_detection_config import PYCLASHBOT_ARENA_REFERENCE_FOLDER
        folder = PYCLASHBOT_ARENA_REFERENCE_FOLDER
    except (ImportError, AttributeError):
        folder = None
    if not folder:
        return os.path.join(_root, "assets", "arena")
    return folder if os.path.isabs(folder) else os.path.join(_root, folder)


def is_available() -> bool:
    """True if reference folder exists and has at least one image."""
    folder = _get_reference_folder()
    if not os.path.isdir(folder):
        return False
    return len(_load_reference_images(folder)) > 0


def load_detector() -> Optional[PyClashBotArenaDetector]:
    """Load detector if reference folder exists and has images."""
    folder = _get_reference_folder()
    if not os.path.isdir(folder):
        return None
    templates = _load_reference_images(folder)
    if not templates:
        return None
    return PyClashBotArenaDetector(folder)
