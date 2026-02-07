# Arena unit detection dataset

Object detection for troops/spells on the battlefield. Each image has bounding boxes and class labels.

## Layout

```
data/arena_dataset/
  annotations.json   # COCO-style: images[], annotations[], categories[]
  images/            # All images referenced in annotations (train + val)
    frame_001.png
    frame_002.png
    ...
```

Or split by train/val:

```
data/arena_dataset/
  train/
    annotations.json
    images/
  val/
    annotations.json
    images/
```

## annotations.json format (COCO-style)

- **images**: `[{"id": 1, "file_name": "frame_001.png", "width": 800, "height": 600}, ...]`
- **annotations**: `[{"id": 1, "image_id": 1, "bbox": [x, y, width, height], "category_id": 0}, ...]`  
  `bbox` is in pixels: top-left x, top-left y, width, height.
- **categories**: `[{"id": 0, "name": "hog_rider"}, {"id": 1, "name": "knight"}, ...]`  
  IDs must be 0-based; the trainer maps them to 1-indexed labels for the model.

## Creating labels

1. Capture screenshots of the game with units on the arena (crop to arena or full frame; keep consistent).
2. Use a labeling tool that exports COCO JSON, e.g.:
   - [Label Studio](https://labelstud.io/)
   - [CVAT](https://www.cvat.ai/)
   - [Roboflow](https://roboflow.com/) (export COCO)
   - [labelimg](https://github.com/HumanSignal/labelImg) (Pascal VOC; convert to COCO or add a small script)
3. Place `annotations.json` and images as above, then run the trainer with `--data-dir data/arena_dataset` (or point to the folder that contains `annotations.json` and `images/`).

## Training

From project root:

```bash
python "image detector/train_arena_detector.py" --data-dir data/arena_dataset
```

See `image detector/train_arena_detector.py` for options (`--epochs`, `--batch-size`, `--model-out`, etc.).
