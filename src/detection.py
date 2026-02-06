"""
Troop and spell detection for Clash Royale.

- **In hand:** template matching in the 4 card slots (card art).
- **On arena:** template matching in the playable area (unit/spell sprites as they appear on the battlefield).

External assets:
- assets/cards/ — card art for hand detection (filename = card key). PNG, JPG, or WebP.
- assets/arena/ — how each unit/spell looks ON THE BATTLEFIELD (sprite). PNG, JPG, or WebP.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ——— Card slot layout (fractions of game frame width/height) ———
# Clash Royale hand: 4 cards in a row at the bottom.
HAND_TOP    = 0.72
HAND_BOTTOM = 0.98
HAND_LEFT   = 0.05
HAND_RIGHT  = 0.95

# ——— Arena (playable battlefield) ———
# Region where troops/spells appear after placement. Excludes hand and top UI.
ARENA_TOP    = 0.12
ARENA_BOTTOM = 0.70
ARENA_LEFT   = 0.02
ARENA_RIGHT  = 0.98

# Build 4 slot regions (left to right)
def _hand_slot_regions():
    n = 4
    top, bottom = HAND_TOP, HAND_BOTTOM
    left, right = HAND_LEFT, HAND_RIGHT
    w = (right - left) / n
    return [
        (left + i * w, top, left + (i + 1) * w, bottom)
        for i in range(n)
    ]


DEFAULT_SLOT_REGIONS = _hand_slot_regions()


@dataclass
class CardMatch:
    """A detected card in a slot (in-hand)."""
    slot: int
    card_id: str
    confidence: float


@dataclass
class ArenaUnitMatch:
    """A detected troop/spell on the arena (placed on the battlefield)."""
    unit_id: str
    confidence: float
    center_x: int
    center_y: int


def load_card_templates(
    templates_dir: str,
    extensions: tuple = (".png", ".jpg", ".jpeg", ".webp"),
) -> dict[str, np.ndarray]:
    """
    Load all card template images from a directory.
    Filename (without extension) is used as card_id.

    Args:
        templates_dir: Path to folder containing card images
        extensions: Allowed image extensions

    Returns:
        Dict mapping card_id -> BGR image (numpy array)
    """
    out = {}
    if not os.path.isdir(templates_dir):
        return out
    for name in os.listdir(templates_dir):
        base, ext = os.path.splitext(name)
        if ext.lower() not in extensions:
            continue
        path = os.path.join(templates_dir, name)
        img = cv2.imread(path)
        if img is not None:
            out[base] = img
    return out


def _match_in_region(
    frame: np.ndarray,
    frame_h: int,
    frame_w: int,
    template: np.ndarray,
    region: tuple[float, float, float, float],
    method: int = cv2.TM_CCOEFF_NORMED,
) -> float:
    """Run template match in a normalized region; return best score."""
    x1, y1, x2, y2 = region
    # Pixel bounds
    px1 = int(x1 * frame_w)
    py1 = int(y1 * frame_h)
    px2 = int(x2 * frame_w)
    py2 = int(y2 * frame_h)
    crop = frame[py1:py2, px1:px2]
    if crop.size == 0:
        return 0.0
    th, tw = template.shape[:2]
    if crop.shape[0] < th or crop.shape[1] < tw:
        # Scale template down to fit
        scale = min(crop.shape[0] / th, crop.shape[1] / tw)
        new_w = max(1, int(tw * scale))
        new_h = max(1, int(th * scale))
        template_scaled = cv2.resize(template, (new_w, new_h))
        result = cv2.matchTemplate(crop, template_scaled, method)
    else:
        result = cv2.matchTemplate(crop, template, method)
    if result.size == 0:
        return 0.0
    if method in (cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF):
        _, min_val, _, _ = cv2.minMaxLoc(result)
        return 1.0 - min_val  # convert to "higher is better"
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return float(max_val)


def detect_cards_in_hand(
    screenshot: np.ndarray,
    templates: dict[str, np.ndarray],
    threshold: float = 0.75,
    slot_regions: Optional[list[tuple[float, float, float, float]]] = None,
) -> list[CardMatch]:
    """
    Detect which card is in each hand slot using template matching.

    Args:
        screenshot: Full game frame (BGR, from capture_screen_region).
        templates: Dict of card_id -> template image (from load_card_templates).
        threshold: Minimum match score to report a card (0.0–1.0).
        slot_regions: List of (x1, y1, x2, y2) in normalized coords [0,1].
                     Default: 4 slots at bottom of screen.

    Returns:
        List of CardMatch (slot, card_id, confidence), one per slot with match above threshold.
    """
    if not templates:
        return []
    regions = slot_regions or DEFAULT_SLOT_REGIONS
    h, w = screenshot.shape[:2]
    results: list[CardMatch] = []
    for slot, region in enumerate(regions):
        best_card: Optional[str] = None
        best_score = 0.0
        for card_id, template in templates.items():
            score = _match_in_region(screenshot, h, w, template, region)
            if score > best_score:
                best_score = score
                best_card = card_id
        if best_card is not None and best_score >= threshold:
            results.append(CardMatch(slot=slot, card_id=best_card, confidence=best_score))
    return results


def get_arena_region() -> tuple[float, float, float, float]:
    """Normalized (x1, y1, x2, y2) for the playable arena. Use for cropping or search bounds."""
    return (ARENA_LEFT, ARENA_TOP, ARENA_RIGHT, ARENA_BOTTOM)


def detect_units_on_arena(
    screenshot: np.ndarray,
    arena_templates: dict[str, np.ndarray],
    threshold: float = 0.75,
    scales: Optional[list[float]] = None,
) -> list[ArenaUnitMatch]:
    """
    Detect troops/spells as they appear on the battlefield (placed in the arena).

    Use templates that show each unit **on the arena** (the in-game sprite), not the card art.
    Load from a separate folder, e.g. load_card_templates("assets/arena").

    Args:
        screenshot: Full game frame (BGR).
        arena_templates: Dict of unit_id -> template (sprite as seen on battlefield).
        threshold: Minimum match score (0.0–1.0).
        scales: Template scales to try (e.g. [0.7, 1.0, 1.3]). None = [1.0] only.

    Returns:
        List of ArenaUnitMatch (unit_id, confidence, center_x, center_y) in frame coordinates.
    """
    if not arena_templates:
        return []
    if scales is None:
        scales = [1.0]
    h, w = screenshot.shape[:2]
    x1, y1, x2, y2 = get_arena_region()
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)
    arena_crop = screenshot[py1:py2, px1:px2]
    if arena_crop.size == 0:
        return []
    found: list[tuple[str, float, int, int]] = []
    for unit_id, template in arena_templates.items():
        th, tw = template.shape[:2]
        best_score = 0.0
        best_cx, best_cy = 0, 0
        for scale in scales:
            if scale <= 0:
                continue
            tw_s = max(1, int(tw * scale))
            th_s = max(1, int(th * scale))
            if arena_crop.shape[0] < th_s or arena_crop.shape[1] < tw_s:
                continue
            tmpl = cv2.resize(template, (tw_s, th_s))
            result = cv2.matchTemplate(arena_crop, tmpl, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val >= threshold and max_val > best_score:
                best_score = float(max_val)
                best_cx = px1 + max_loc[0] + tw_s // 2
                best_cy = py1 + max_loc[1] + th_s // 2
        if best_score >= threshold:
            found.append((unit_id, best_score, best_cx, best_cy))
    # Deduplicate overlapping detections (same unit at multiple scales): keep highest score per unit
    found.sort(key=lambda x: -x[1])
    seen_centers: list[tuple[int, int]] = []
    results: list[ArenaUnitMatch] = []
    for unit_id, conf, cx, cy in found:
        too_close = any(abs(cx - sx) < 30 and abs(cy - sy) < 30 for sx, sy in seen_centers)
        if not too_close:
            seen_centers.append((cx, cy))
            results.append(ArenaUnitMatch(unit_id=unit_id, confidence=conf, center_x=cx, center_y=cy))
    return results


def detect_any_card_on_screen(
    screenshot: np.ndarray,
    templates: dict[str, np.ndarray],
    threshold: float = 0.8,
) -> list[tuple[str, float, int, int]]:
    """
    Search the entire frame for any card template (e.g. for battlefield or UI).
    Slower than slot-based detection; use for occasional full-screen checks.

    Returns:
        List of (card_id, confidence, center_x, center_y) in frame coordinates.
    """
    if not templates:
        return []
    h, w = screenshot.shape[:2]
    found = []
    for card_id, template in templates.items():
        th, tw = template.shape[:2]
        if h < th or w < tw:
            continue
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            cx = max_loc[0] + tw // 2
            cy = max_loc[1] + th // 2
            found.append((card_id, float(max_val), cx, cy))
    return found
