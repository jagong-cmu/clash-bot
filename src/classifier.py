"""
Trained card classifier for Clash Royale (cards, spells, buildings in hand).

Uses the MobileNetV2 model from the image detector folder. Lazy-loads on first use.
Requires: torch, torchvision, Pillow. Model file: image detector/card_classifier.pth
"""

from __future__ import annotations

import os
from typing import Tuple, Optional

# Lazy-loaded state
_model = None
_transform = None
_class_names = None
_device = None
_load_attempted = False
_load_error: Optional[str] = None

IMG_SIZE = 128


def _get_model_path() -> str:
    """Path to card_classifier.pth relative to project root."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "image detector", "card_classifier.pth")


def load_classifier(model_path: Optional[str] = None) -> bool:
    """
    Load the trained card classifier. Called automatically on first predict_card().
    Returns True if loaded successfully, False otherwise.
    """
    global _model, _transform, _class_names, _device, _load_attempted, _load_error
    if _model is not None:
        return True
    if _load_attempted:
        return False
    _load_attempted = True
    try:
        import torch
        from torch import nn
        from torchvision import models, transforms
        from PIL import Image
    except ImportError as e:
        _load_error = f"Missing dependencies: {e}"
        return False
    path = model_path or _get_model_path()
    if not os.path.isfile(path):
        _load_error = f"Model file not found: {path}"
        return False
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, map_location=device)
        classes = checkpoint["classes"]
        if not isinstance(classes, (list, tuple)):
            classes = list(classes)
        class_names = [str(c) for c in classes]
        num_classes = len(class_names)
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        _model = model
        _transform = transform
        _class_names = class_names
        _device = device
        return True
    except Exception as e:
        _load_error = str(e)
        return False


def is_available() -> bool:
    """Return True if the classifier can be used (model loaded or loadable)."""
    if _model is not None:
        return True
    if not _load_attempted:
        load_classifier()
    return _model is not None


def get_load_error() -> Optional[str]:
    """Return the last load error message if loading failed."""
    return _load_error


def predict_card(bgr_img) -> Tuple[str, float]:
    """
    Predict the card (troop/spell/building) in a BGR image (e.g. a hand-slot crop).

    Args:
        bgr_img: OpenCV-style image (numpy array, BGR).

    Returns:
        (card_name, confidence) where confidence is in [0, 1].
        If the classifier is not available or prediction fails, returns ("unknown", 0.0).
    """
    global _model, _transform, _class_names, _device
    if _model is None and not load_classifier():
        return ("unknown", 0.0)
    try:
        import numpy as np
        from PIL import Image
        import torch
        rgb = bgr_img[:, :, ::-1]
        pil_img = Image.fromarray(rgb)
        x = _transform(pil_img).unsqueeze(0).to(_device)
        with torch.no_grad():
            logits = _model(x)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(1)
        name = _class_names[idx.item()]
        return (name, conf.item())
    except Exception:
        return ("unknown", 0.0)
