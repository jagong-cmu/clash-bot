# Card classifier dataset

Used by `image detector/train_card_classifier.py` to train the in-hand card detector (troops, spells, buildings).

## Layout

```
data/card_dataset/
  collect/          ← Output of scripts/collect_card_data.py (raw crops by class)
    knight/
    fireball/
    hog_rider/
    ...
  train/            ← Move 70–80% of images per class here (for training)
    knight/
    fireball/
    ...
  val/              ← Move 20–30% per class here (for validation)
    knight/
    fireball/
    ...
```

## Workflow

1. **Collect data** (uses window detection to find the game window, crops the 4 hand slots):
   ```bash
   python scripts/collect_card_data.py
   python scripts/collect_card_data.py --window "iPhone Mirroring"
   ```
   Type a card name (e.g. `knight`) and press Enter to save the current frame’s 4 slot crops under that class. Use `list` to see open windows, `quit` to exit.

2. **Split into train/val** (optional; or move files manually):
   ```bash
   python scripts/prepare_dataset.py          # 80% train, 20% val per class
   python scripts/prepare_dataset.py --val-ratio 0.25
   ```
   This copies from `collect/<class>/` into `train/<class>/` and `val/<class>/`.

3. **Train** the classifier (saves to `image detector/card_classifier.pth`):
   ```bash
   python "image detector/train_card_classifier.py"
   python "image detector/train_card_classifier.py" --epochs 30 --batch-size 16
   ```

4. **Use** the new model: the bot and `test_detection.py` already load `image detector/card_classifier.pth` for in-hand detection.

## Window detection

The same window detection used in `test_window_detection.py` (and the main bot) is used by `scripts/collect_card_data.py`: it finds the game window by name (e.g. "iPhone Mirroring") so crops are taken from the correct region.
