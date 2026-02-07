#!/usr/bin/env python3
"""
Train the card classifier on your data and create card_classifier.pth.

Uses data under data/card_dataset/train/ and val/ (one folder per card).
The .pth file stores the trained model (learned weights), not the raw images.
After this runs, test_detection.py and the bot will use the new model automatically.

Usage:
  python scripts/train_from_data.py
  python scripts/train_from_data.py --epochs 40
"""

import sys
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINER = os.path.join(ROOT, "image detector", "train_card_classifier.py")

def main():
    if not os.path.isfile(TRAINER):
        print("Trainer not found:", TRAINER)
        return 1
    # Run the trainer with default data dir (data/card_dataset)
    cmd = [sys.executable, TRAINER, "--data-dir", os.path.join(ROOT, "data", "card_dataset")]
    cmd.extend(sys.argv[1:])  # pass through --epochs etc.
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(main())
