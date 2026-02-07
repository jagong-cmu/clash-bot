#!/usr/bin/env python3
"""
Split collected card images into train/ and val/ for the classifier.

Reads from data/card_dataset/collect/<class>/ and copies images into
data/card_dataset/train/<class>/ and val/<class>/ with a configurable ratio.

Usage:
  python scripts/prepare_dataset.py
  python scripts/prepare_dataset.py --val-ratio 0.2
"""

import sys
import os
import argparse
import random
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLLECT = os.path.join(ROOT, "data", "card_dataset", "collect")
DATA_DIR = os.path.join(ROOT, "data", "card_dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-dir", default=COLLECT, help="Source: collect/<class>/")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Parent of train/ and val/")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of each class for val (0.2 = 20%%)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isdir(args.collect_dir):
        print("No collect dir:", args.collect_dir)
        print("Run: python scripts/collect_card_data.py")
        return 1

    random.seed(args.seed)
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    classes = [d for d in os.listdir(args.collect_dir) if os.path.isdir(os.path.join(args.collect_dir, d))]
    if not classes:
        print("No class folders in", args.collect_dir)
        return 1

    total_train, total_val = 0, 0
    for class_name in classes:
        src = os.path.join(args.collect_dir, class_name)
        images = [f for f in os.listdir(src) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        if not images:
            continue
        random.shuffle(images)
        n_val = max(1, int(len(images) * args.val_ratio))
        n_train = len(images) - n_val
        val_images = set(images[:n_val])
        train_images = [f for f in images if f not in val_images]
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        for f in train_images:
            shutil.copy2(os.path.join(src, f), os.path.join(train_dir, class_name, f))
            total_train += 1
        for f in val_images:
            shutil.copy2(os.path.join(src, f), os.path.join(val_dir, class_name, f))
            total_val += 1
        print(f"  {class_name}: {n_train} train, {n_val} val")

    print(f"\nDone: {total_train} train, {total_val} val.")
    print("Run: python \"image detector/train_card_classifier.py\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
