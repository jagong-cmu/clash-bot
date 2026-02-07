"""
Train the card classifier (troops, spells, buildings).

Uses the same window/capture setup as the rest of the project: data is expected
under data/card_dataset/ (train/<class>/ and val/<class>/). Collect images with
scripts/collect_card_data.py, then organize into train/ and val/ subfolders by class.

Usage:
  python "image detector/train_card_classifier.py"
  python "image detector/train_card_classifier.py" --epochs 30 --batch-size 16
  python "image detector/train_card_classifier.py" --fresh   # start fresh, ignore existing model

By default, resumes from card_classifier.pth if it exists (adds to training).

Data layout (ImageFolder):
  data/card_dataset/
    train/
      knight/
      fireball/
      hog_rider/
      ...
    val/
      knight/
      fireball/
      ...
"""

import os
import sys
import argparse

# Project root (parent of "image detector")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)

import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ================= CONFIG =================
DATA_DIR = os.path.join(ROOT, "data", "card_dataset")
MODEL_OUT = os.path.join(SCRIPT_DIR, "card_classifier.pth")
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-3
# ==========================================


def main():
    parser = argparse.ArgumentParser(description="Train card classifier for Clash Royale detection")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Dataset root (contains train/ and val/)")
    parser.add_argument("--model-out", default=MODEL_OUT, help="Output .pth path")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore existing model)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print("Expected dataset layout:")
        print("  data/card_dataset/train/<class_name>/  (images per class)")
        print("  data/card_dataset/val/<class_name>/")
        print("\nCollect data with:  python scripts/collect_card_data.py")
        print("Then copy images from data/card_dataset/collect/<class>/ into train/ and val/.")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Data:", args.data_dir)
    print("Model out:", args.model_out)

    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    num_classes = len(train_ds.classes)
    print("Classes:", train_ds.classes)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    best_val_acc = 0.0
    if os.path.isfile(args.model_out) and not args.fresh:
        checkpoint = torch.load(args.model_out, map_location=device)
        old_classes = checkpoint.get("classes", [])
        if set(old_classes) == set(train_ds.classes):
            model.load_state_dict(checkpoint["model_state"], strict=True)
            print("✅ Resumed from existing model (same classes)")
        else:
            state = checkpoint["model_state"]
            backbone_state = {k: v for k, v in state.items() if not k.startswith("classifier")}
            model.load_state_dict(backbone_state, strict=False)
            model.classifier[1] = nn.Linear(model.last_channel, num_classes).to(device)
            print(f"✅ Resumed backbone; classifier rebuilt for {num_classes} classes (was {len(old_classes)})")
    else:
        if args.fresh:
            print("Starting fresh (--fresh flag)")
        else:
            print("No existing model found, starting from ImageNet pretrained")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}: loss={epoch_loss:.3f}, val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
            }, args.model_out)
            print("✅ Best model saved")

    print("\nTraining finished.")
    print("Best validation accuracy:", best_val_acc)
    print("Model saved at:", args.model_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
