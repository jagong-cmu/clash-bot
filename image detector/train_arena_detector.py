"""
Train RetinaNet for arena unit detection (object detection: boxes + class per unit).

Dataset: COCO-style annotations.json + images/ (see data/arena_dataset/README.md).
Saves a checkpoint with state_dict, class_names, and num_classes for arena_detector.py.

Usage:
  python "image detector/train_arena_detector.py" --data-dir data/arena_dataset
  python "image detector/train_arena_detector.py" --data-dir data/arena_dataset --epochs 20 --batch-size 4
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models import ResNet50_Weights
import numpy as np
from PIL import Image
from tqdm import tqdm


def _find_annotations(data_dir: str):
    """Locate annotations (annotations.json or _annotations.coco.json in data_dir or data_dir/train)."""
    for name in ("annotations.json", "_annotations.coco.json"):
        single = os.path.join(data_dir, name)
        if os.path.isfile(single):
            return single, None
    for name in ("annotations.json", "_annotations.coco.json"):
        train_a = os.path.join(data_dir, "train", name)
        val_a = os.path.join(data_dir, "val", name)
        if os.path.isfile(train_a):
            return train_a, val_a if os.path.isfile(val_a) else None
    # Roboflow export: data_dir may contain "Project.v1i.coco" with train/ and valid/ inside
    for sub in os.listdir(data_dir):
        subdir = os.path.join(data_dir, sub)
        if not os.path.isdir(subdir):
            continue
        for name in ("annotations.json", "_annotations.coco.json"):
            train_a = os.path.join(subdir, "train", name)
            if os.path.isfile(train_a):
                for val_folder, val_name in [("valid", "_annotations.coco.json"), ("val", "annotations.json"), ("val", "_annotations.coco.json")]:
                    val_a = os.path.join(subdir, val_folder, val_name)
                    if os.path.isfile(val_a):
                        return train_a, val_a
                return train_a, None
    return None, None


class ArenaDetectionDataset(Dataset):
    """COCO-style detection dataset. Returns image tensor (C,H,W) and target dict (boxes, labels)."""

    def __init__(self, annotations_path: str, images_dir: Optional[str] = None, transform=None):
        with open(annotations_path) as f:
            data = json.load(f)
        self.images = {im["id"]: im for im in data["images"]}
        self.img_list = list(self.images.values())
        self.annotations = data["annotations"]
        self.categories = {c["id"]: c["name"] for c in data["categories"]}
        self.cat_ids = sorted(self.categories.keys())
        self.id_to_idx = {cid: i for i, cid in enumerate(self.cat_ids)}
        base = os.path.dirname(annotations_path)
        if images_dir:
            self.img_dir = images_dir
        else:
            # Prefer images in same folder as JSON (Roboflow); else images/ subfolder
            first_fn = self.img_list[0]["file_name"] if self.img_list else ""
            same_dir = os.path.join(base, first_fn)
            self.img_dir = base if first_fn and os.path.isfile(same_dir) else os.path.join(base, "images")
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        im_info = self.img_list[idx]
        img_id = im_info["id"]
        path = os.path.join(self.img_dir, im_info["file_name"])
        img = Image.open(path).convert("RGB")
        img = np.array(img)
        # Annotations for this image: bbox [x,y,w,h] -> xyxy; category_id 0-based -> 1-based label
        boxes = []
        labels = []
        for ann in self.annotations:
            if ann["image_id"] != img_id:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            x2 = x + w
            y2 = y + h
            cat = ann["category_id"]
            if cat not in self.id_to_idx:
                continue
            label = self.id_to_idx[cat] + 1
            boxes.append([x, y, x2, y2])
            labels.append(label)
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor(img_id)}
        if self.transform:
            img, target = self.transform(img, target)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        return img, target


def collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


def main():
    parser = argparse.ArgumentParser(description="Train arena unit object detector (RetinaNet)")
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "data", "arena_dataset"), help="Dataset root")
    parser.add_argument("--model-out", default=os.path.join(SCRIPT_DIR, "arena_detector.pth"), help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    train_path, val_path = _find_annotations(args.data_dir)
    if not train_path or not os.path.isfile(train_path):
        print("Expected annotations at:")
        print("  {}/annotations.json  or  {}/train/annotations.json".format(args.data_dir, args.data_dir))
        print("See data/arena_dataset/README.md for format.")
        return 1

    train_ds = ArenaDetectionDataset(train_path)
    if not train_ds.img_list:
        print("No images in dataset.")
        return 1
    class_names = [train_ds.categories[cid] for cid in train_ds.cat_ids]
    num_classes = len(class_names) + 1

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_path and os.path.isfile(val_path):
        val_ds = ArenaDetectionDataset(val_path)
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: RetinaNet = retinanet_resnet50_fpn(
        num_classes=num_classes,
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
    )
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print("Classes:", class_names)
    print("Device:", device)
    print("Epochs:", args.epochs)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.1)
            optimizer.step()
            total_loss += losses.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{losses.item():.4f}")
        avg = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1} train loss: {avg:.4f}")
        if val_loader:
            model.eval()
            val_loss = 0.0
            nv = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    val_loss += sum(loss_dict.values()).item()
                    nv += 1
            if nv:
                print(f"  val loss: {val_loss / nv:.4f}")
        scheduler.step()

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "num_classes": num_classes,
    }
    torch.save(ckpt, args.model_out)
    print("Saved:", args.model_out)
    return 0
