"""
Arena troop detection using a Roboflow Universe model.

Use a pre-trained Clash Royale troop detection model (e.g. from Roboflow Universe)
without training your own. Configure model ID and API key in config/roboflow_arena_config.py.

Example model: https://universe.roboflow.com/crbot-nu0tk/clash-royale-troop-detection-dkcsn/model/1
Model ID: clash-royale-troop-detection-dkcsn/1

Requires: pip install inference
API key: set ROBOFLOW_API_KEY in environment or in config.
"""

from typing import List, Optional, Tuple, Any

import numpy as np

# Lazy load so we don't require 'inference' unless Roboflow is used
_roboflow_model: Optional[Any] = None
_unavailable_reason: Optional[str] = None


def _get_config() -> Tuple[Optional[str], Optional[str]]:
    """Return (model_id, api_key). api_key can come from config or env."""
    import os
    import sys
    model_id = None
    api_key = None
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    try:
        from config.roboflow_arena_config import ROBOFLOW_ARENA_MODEL_ID, ROBOFLOW_API_KEY
        model_id = ROBOFLOW_ARENA_MODEL_ID
        api_key = ROBOFLOW_API_KEY
    except (ImportError, AttributeError):
        pass
    if not api_key:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
    return model_id, api_key


def _load_model() -> Optional[Any]:
    """Load Roboflow model once. Returns the model object or None."""
    global _roboflow_model, _unavailable_reason
    if _roboflow_model is not None:
        return _roboflow_model
    _unavailable_reason = None
    model_id, api_key = _get_config()
    if not model_id:
        _unavailable_reason = "ROBOFLOW_ARENA_MODEL_ID not set in config/roboflow_arena_config.py"
        return None
    if not api_key:
        _unavailable_reason = (
            "ROBOFLOW_API_KEY not set. Set it in config/roboflow_arena_config.py or export ROBOFLOW_API_KEY"
        )
        return None
    try:
        from inference import get_model
        _roboflow_model = get_model(model_id=model_id, api_key=api_key)
        return _roboflow_model
    except Exception as e:
        _unavailable_reason = f"Roboflow model load failed: {e}"
        return None


def get_unavailable_reason() -> Optional[str]:
    """If Roboflow is not available, return a short reason; otherwise None."""
    _load_model()  # ensure we've tried to load and set _unavailable_reason
    return _unavailable_reason


def is_available() -> bool:
    """True if Roboflow arena model is configured and loadable."""
    return _load_model() is not None


def load_detector() -> Optional["RoboflowArenaDetector"]:
    """Return a detector instance if config and inference package are valid."""
    model = _load_model()
    if model is None:
        return None
    return RoboflowArenaDetector(model)


class RoboflowArenaDetector:
    """
    Wraps a Roboflow inference model for arena detection.
    Same interface as the local ArenaDetector: predict(image_bgr, confidence_threshold) -> list of (unit_id, conf, cx, cy).
    """

    def __init__(self, model: Any):
        self._model = model

    def predict(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> List[Tuple[str, float, int, int]]:
        """
        Run detection on a BGR image (e.g. arena crop).
        Returns list of (unit_id, confidence, center_x, center_y) in image coordinates.
        """
        # Roboflow inference often accepts numpy; some models expect RGB
        image_rgb = image_bgr[:, :, ::-1].copy()
        try:
            results = self._model.infer(image_rgb)
        except Exception:
            results = self._model.infer(image_bgr)
        if not results:
            return []
        # First image result
        r0 = results[0] if isinstance(results, (list, tuple)) else results
        predictions = getattr(r0, "predictions", None)
        if not predictions:
            # Try dict-style
            if hasattr(r0, "dict"):
                d = r0.dict(by_alias=True, exclude_none=True)
                predictions = d.get("predictions", [])
            else:
                return []
        out: List[Tuple[str, float, int, int]] = []
        h, w = image_bgr.shape[:2]

        def _get(o, key, default=None):
            return getattr(o, key, default) if not isinstance(o, dict) else o.get(key, default)

        for p in predictions:
            conf = float(_get(p, "confidence") or 0)
            if conf < confidence_threshold:
                continue
            class_name = _get(p, "class_name") or _get(p, "class")
            if not class_name:
                continue
            # Bbox: x,y,width,height (center or top-left; normalized or pixels) or xyxy
            x = _get(p, "x")
            y = _get(p, "y")
            width = _get(p, "width")
            height = _get(p, "height")
            if x is not None and y is not None:
                if width is not None and height is not None:
                    if width <= 1 and height <= 1:
                        cx = int(float(x) * w)
                        cy = int(float(y) * h)
                    else:
                        cx = int(float(x) + float(width) / 2)
                        cy = int(float(y) + float(height) / 2)
                else:
                    if abs(float(x)) <= 1 and abs(float(y)) <= 1:
                        cx = int(float(x) * w)
                        cy = int(float(y) * h)
                    else:
                        cx = int(float(x))
                        cy = int(float(y))
            else:
                bbox = _get(p, "bbox") or _get(p, "box") or _get(p, "xyxy")
                if bbox and len(bbox) >= 4:
                    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    if x2 <= 1 and y2 <= 1:
                        x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                else:
                    continue
            out.append((str(class_name), conf, cx, cy))
        return out
