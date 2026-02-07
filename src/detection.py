"""
Troop and spell detection for Clash Royale.

- **In hand:** trained classifier (image detector/card_classifier.pth) by default; optional template-matching fallback.
- **On arena:** template matching in the playable area (unit/spell sprites as they appear on the battlefield).

Classifier: MobileNetV2 trained on card icons (cards, spells, buildings). No template images needed for hand.
Templates: assets/cards/ and assets/arena/ only needed for template-based fallback or arena detection.
"""

import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from src.classifier import is_available as _classifier_available, predict_card as _predict_card
except ImportError:
    _classifier_available = lambda: False
    _predict_card = None


# ——— Card slot layout (fractions of game frame width/height) ———
# Clash Royale hand: 4 cards in a row at the bottom.
# Override from config/hand_slots.py if it exists (edit that file to fix crop position).
HAND_TOP    = 0.72
HAND_BOTTOM = 0.98
HAND_LEFT   = 0.05
HAND_RIGHT  = 0.95
try:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from config.hand_slots import HAND_TOP as _t, HAND_BOTTOM as _b, HAND_LEFT as _l, HAND_RIGHT as _r
    HAND_TOP, HAND_BOTTOM, HAND_LEFT, HAND_RIGHT = _t, _b, _l, _r
except (ImportError, AttributeError):
    pass

# ——— Arena (playable battlefield) ———
ARENA_TOP    = 0.12
ARENA_BOTTOM = 0.70
ARENA_LEFT   = 0.02
ARENA_RIGHT  = 0.98

# Build 4 slot regions (left to right)
def _hand_slot_regions(custom_top=None, custom_bottom=None, custom_left=None, custom_right=None):
    n = 4
    top = custom_top if custom_top is not None else HAND_TOP
    bottom = custom_bottom if custom_bottom is not None else HAND_BOTTOM
    left = custom_left if custom_left is not None else HAND_LEFT
    right = custom_right if custom_right is not None else HAND_RIGHT
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
        if img is None and ext.lower() == ".webp":
            try:
                from PIL import Image
                pil_img = Image.open(path).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except (ImportError, OSError):
                pass
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
    scales: Optional[list[float]] = None,
) -> float:
    """Run template match in a normalized region; return best score.
    Tries multiple scales to handle resolution mismatches between template and screen."""
    x1, y1, x2, y2 = region
    px1 = int(x1 * frame_w)
    py1 = int(y1 * frame_h)
    px2 = int(x2 * frame_w)
    py2 = int(y2 * frame_h)
    crop = frame[py1:py2, px1:px2]
    if crop.size == 0:
        return 0.0
    crop_h, crop_w = crop.shape[:2]
    th, tw = template.shape[:2]
    if scales is None:
        scales = [0.55, 0.7, 0.85, 1.0, 1.15]
    best_val = 0.0
    for s in scales:
        if s <= 0:
            continue
        new_w = max(1, int(tw * s))
        new_h = max(1, int(th * s))
        if new_h > crop_h or new_w > crop_w:
            continue
        template_scaled = cv2.resize(template, (new_w, new_h))
        result = cv2.matchTemplate(crop, template_scaled, method)
        if result.size == 0:
            continue
        if method in (cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF):
            _, min_val, _, _ = cv2.minMaxLoc(result)
            val = 1.0 - min_val
        else:
            _, max_val, _, _ = cv2.minMaxLoc(result)
            val = float(max_val)
        if val > best_val:
            best_val = val
    return best_val


def detect_cards_in_hand(
    screenshot: np.ndarray,
    templates: Optional[dict[str, np.ndarray]] = None,
    threshold: float = 0.65,
    slot_regions: Optional[list[tuple[float, float, float, float]]] = None,
    scales: Optional[list[float]] = None,
    use_classifier: bool = True,
) -> list[CardMatch]:
    """
    Detect which card is in each hand slot.

    By default uses the trained classifier (image detector/card_classifier.pth) so no
    template images are needed. Set use_classifier=False and pass templates to use
    template matching instead.

    Args:
        screenshot: Full game frame (BGR, from capture_screen_region).
        templates: Optional dict of card_id -> template (only used if use_classifier=False).
        threshold: Minimum confidence to report a card (0.0–1.0).
        slot_regions: List of (x1, y1, x2, y2) in normalized coords [0,1]. Default: 4 slots.
        scales: Template scales when using template matching. None = default.
        use_classifier: If True and classifier is available, use it; else use templates.

    Returns:
        List of CardMatch (slot, card_id, confidence), one per slot with match above threshold.
    """
    regions = slot_regions or DEFAULT_SLOT_REGIONS
    h, w = screenshot.shape[:2]

    # Prefer trained classifier when requested and available
    if use_classifier and _classifier_available() and _predict_card is not None:
        results: list[CardMatch] = []
        for slot, region in enumerate(regions):
            x1, y1, x2, y2 = region
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)
            crop = screenshot[py1:py2, px1:px2]
            if crop.size == 0:
                continue
            card_name, conf = _predict_card(crop)
            if card_name != "unknown":
                results.append(CardMatch(slot=slot, card_id=card_name, confidence=conf))
        return results

    # Fallback: template matching
    if not templates:
        return []
    results = []
    for slot, region in enumerate(regions):
        best_card: Optional[str] = None
        best_score = 0.0
        for card_id, template in templates.items():
            score = _match_in_region(screenshot, h, w, template, region, scales=scales)
            if score > best_score:
                best_score = score
                best_card = card_id
        if best_card is not None and best_score >= threshold:
            results.append(CardMatch(slot=slot, card_id=best_card, confidence=best_score))
    return results


def get_slot_best_matches(
    screenshot: np.ndarray,
    templates: dict[str, np.ndarray],
    slot_regions: Optional[list[tuple[float, float, float, float]]] = None,
) -> list[tuple[int, Optional[str], float]]:
    """
    Return the best card match for each slot, regardless of threshold.
    Useful for debugging: (slot, card_id, score) per slot.
    """
    if not templates:
        return []
    regions = slot_regions or DEFAULT_SLOT_REGIONS
    h, w = screenshot.shape[:2]
    out: list[tuple[int, Optional[str], float]] = []
    for slot, region in enumerate(regions):
        best_card: Optional[str] = None
        best_score = 0.0
        for card_id, template in templates.items():
            score = _match_in_region(screenshot, h, w, template, region)
            if score > best_score:
                best_score = score
                best_card = card_id
        out.append((slot, best_card, best_score))
    return out


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


def _template_match_best(
    screenshot: np.ndarray,
    template: np.ndarray,
    scales: list[float],
    use_grayscale: bool = False,
) -> tuple[float, int, int]:
    """Run multi-scale template matching; return (best_score, center_x, center_y)."""
    h, w = screenshot.shape[:2]
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY) if use_grayscale else screenshot
    tmpl_base = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if use_grayscale else template
    best_score = 0.0
    best_cx, best_cy = 0, 0
    th, tw = tmpl_base.shape[:2]
    for s in scales:
        if s <= 0:
            continue
        tw_s = max(1, int(tw * s))
        th_s = max(1, int(th * s))
        if h < th_s or w < tw_s:
            continue
        tmpl = cv2.resize(tmpl_base, (tw_s, th_s))
        result = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = float(max_val)
            best_cx = max_loc[0] + tw_s // 2
            best_cy = max_loc[1] + th_s // 2
    return best_score, best_cx, best_cy


def get_best_match_scores(
    screenshot: np.ndarray,
    templates: dict[str, np.ndarray],
    scales: Optional[list[float]] = None,
) -> list[tuple[str, float, int, int]]:
    """
    Return best match per template regardless of threshold (for debugging).
    Returns list of (card_id, score, cx, cy).
    """
    if not templates:
        return []
    if scales is None:
        scales = [0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.2, 1.4]
    out: list[tuple[str, float, int, int]] = []
    for card_id, template in templates.items():
        score_c, cx_c, cy_c = _template_match_best(screenshot, template, scales, use_grayscale=False)
        score_g, cx_g, cy_g = _template_match_best(screenshot, template, scales, use_grayscale=True)
        if score_g > score_c:
            out.append((card_id, score_g, cx_g, cy_g))
        else:
            out.append((card_id, score_c, cx_c, cy_c))
    return out


def detect_any_card_on_screen(
    screenshot: np.ndarray,
    templates: dict[str, np.ndarray],
    threshold: float = 0.3,
    scales: Optional[list[float]] = None,
    use_grayscale: bool = True,
) -> list[tuple[str, float, int, int]]:
    """
    Search the entire frame for any card template. Matches anywhere on screen.
    Uses multi-scale matching and optional grayscale for robustness.

    Returns:
        List of (card_id, confidence, center_x, center_y) in frame coordinates.
    """
    if not templates:
        return []
    if scales is None:
        scales = [0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.2, 1.4]
    found: list[tuple[str, float, int, int]] = []
    for card_id, template in templates.items():
        score_c, cx_c, cy_c = _template_match_best(screenshot, template, scales, use_grayscale=False)
        score_g, cx_g, cy_g = _template_match_best(screenshot, template, scales, use_grayscale=True)
        best_score = max(score_c, score_g)
        best_cx, best_cy = (cx_g, cy_g) if score_g >= score_c else (cx_c, cy_c)
        if best_score >= threshold:
            found.append((card_id, best_score, best_cx, best_cy))
    found.sort(key=lambda x: -x[1])
    seen: list[tuple[int, int]] = []
    results: list[tuple[str, float, int, int]] = []
    for card_id, conf, cx, cy in found:
        too_close = any(abs(cx - sx) < 50 and abs(cy - sy) < 50 for sx, sy in seen)
        if not too_close:
            seen.append((cx, cy))
            results.append((card_id, conf, cx, cy))
    return results
