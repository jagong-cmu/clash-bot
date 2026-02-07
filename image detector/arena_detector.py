"""
Arena unit object detection: load a trained RetinaNet and run inference.

Model expects images in 0–1 range, returns boxes (x1,y1,x2,y2), labels (1-indexed),
and scores. We map labels to class names and return detections in frame coordinates.

Checkpoint format: dict with "state_dict", "class_names", "num_classes".
Default path: image detector/arena_detector.pth (override via config/arena_detector_config.py).
"""

import os
import sys
from typing import List, Optional, Tuple

import numpy as np

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch
    from torchvision.models.detection import retinanet_resnet50_fpn
    from torchvision.models.detection.retinanet import RetinaNet
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    RetinaNet = None


DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "arena_detector.pth")


def _get_model_path() -> str:
    try:
        from config.arena_detector_config import ARENA_MODEL_PATH
        if ARENA_MODEL_PATH is not None:
            p = ARENA_MODEL_PATH
            return p if os.path.isabs(p) else os.path.join(ROOT, p)
    except (ImportError, AttributeError):
        pass
    return DEFAULT_MODEL_PATH


def _load_checkpoint(path: str) -> Tuple[dict, List[str], int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        class_names = ckpt.get("class_names")
        num_classes = ckpt.get("num_classes")
        if not class_names or not num_classes:
            raise ValueError("Checkpoint must contain 'class_names' and 'num_classes'")
        return state, class_names, num_classes
    raise ValueError("Checkpoint must be a dict with 'state_dict', 'class_names', 'num_classes'")


class ArenaDetector:
    """
    Wraps a trained RetinaNet for arena unit detection.
    Input: BGR image (numpy, uint8). Output: list of (unit_id, confidence, center_x, center_y).
    Coordinates are in the same frame as the input (arena crop or full frame).
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch and torchvision are required for arena detector")
        path = model_path or _get_model_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Arena detector model not found: {path}")
        state_dict, class_names, num_classes = _load_checkpoint(path)
        self.class_names = class_names
        self.num_classes = num_classes
        # Use MPS (Metal) on Mac, CUDA on Linux/Windows, else CPU
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"  # Mac GPU acceleration
        else:
            self.device = "cpu"
        self.model: RetinaNet = retinanet_resnet50_fpn(num_classes=num_classes, weights=None)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        # Print device info for debugging
        if self.device == "mps":
            print(f"Arena detector using MPS (Mac GPU) for acceleration")
        elif self.device == "cuda":
            print(f"Arena detector using CUDA (GPU) for acceleration")
        else:
            print(f"Arena detector using CPU (no GPU acceleration)")

    def predict(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> List[Tuple[str, float, int, int]]:
        """
        Run detection on a single BGR image (e.g. arena crop).

        Returns:
            List of (unit_id, confidence, center_x, center_y) in image coordinates.
        """
        # HWC BGR [0,255] -> CHW RGB [0,1]
        rgb = image_bgr[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        assert len(out) == 1
        pred = out[0]
        boxes = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        results: List[Tuple[str, float, int, int]] = []
        for i in range(len(scores)):
            if scores[i] < confidence_threshold:
                continue
            label = int(labels[i])
            # Labels are 1-indexed; 0 is background (usually not in output)
            if label < 1 or label > len(self.class_names):
                continue
            unit_id = self.class_names[label - 1]
            x1, y1, x2, y2 = boxes[i]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            results.append((unit_id, float(scores[i]), cx, cy))
        return results


def is_available() -> bool:
    """True if torch/torchvision are available and a model file exists."""
    if not _TORCH_AVAILABLE:
        return False
    path = _get_model_path()
    return os.path.isfile(path)


def load_detector(model_path: Optional[str] = None) -> Optional[ArenaDetector]:
    """Load arena detector if possible; returns None on failure."""
    if not _TORCH_AVAILABLE:
        return None
    path = model_path or _get_model_path()
    if not os.path.isfile(path):
        return None
    try:
        return ArenaDetector(model_path=path)
    except Exception:
        return None
