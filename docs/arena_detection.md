# Arena unit detection (trained model only)

Arena detection uses **only** the trained RetinaNet saved as **image detector/arena_detector.pth**. No template matching or Roboflow.

---

## Train the model

Use COCO-style data in **data/arena_dataset** (e.g. the Roboflow export in `Clash Royale troop detection.v1i.coco/` with train/ and valid/):

```bash
python "image detector/train_arena_detector.py" --data-dir data/arena_dataset
python "image detector/train_arena_detector.py" --data-dir data/arena_dataset --epochs 25 --batch-size 4
```

This produces **image detector/arena_detector.pth**. Override the path in **config/arena_detector_config.py** (ARENA_MODEL_PATH) if needed.

---

## Testing

```bash
python test_arena_tracking.py
python test_arena_tracking.py --loop 1.0 --threshold 0.5
```

You should see “Using trained arena detector (arena_detector.pth)”. If the model is missing, the script prints the train command.
