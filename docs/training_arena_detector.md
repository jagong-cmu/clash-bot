# Training Your Own Arena Troop Detector

Train a custom RetinaNet model on your own screenshots to improve accuracy for your specific game setup, screen resolution, and troop appearances.

---

## Overview

The training process:
1. **Collect screenshots** of the game with troops on the arena
2. **Label them** with bounding boxes and class names (e.g., "hog_rider", "knight")
3. **Organize** into COCO-style format
4. **Train** RetinaNet model
5. **Use** the trained model (it will automatically replace Roboflow/template matching)

---

## Step 1: Collect Screenshots

Capture screenshots during actual gameplay when troops are visible on the arena.

**Option A: Manual capture**
- Play matches and take screenshots (Cmd+Shift+3 on Mac, or use a screenshot tool)
- Save to a folder like `data/arena_dataset/raw/`

**Option B: Use the test script to capture**
```bash
# Modify test_arena_tracking.py temporarily to save screenshots
# Or use a simple capture script
```

**Tips:**
- Capture **diverse scenarios**: different troops, positions, lighting, game states
- Include **multiple angles/sizes** of the same troop
- Aim for **50-200+ images** minimum (more = better accuracy)
- Use the **arena region** you've configured (see `arena_region.png`)

---

## Step 2: Label Your Data

You need to draw bounding boxes around each troop and label them.

### Recommended Tools:

**1. Roboflow (easiest, free tier available)**
- Upload images to [Roboflow](https://roboflow.com/)
- Draw boxes and label each troop
- Export as **COCO JSON** format
- Download: `annotations.json` + `images/` folder

**2. Label Studio (open source)**
```bash
pip install label-studio
label-studio start
```
- Import images
- Use object detection template
- Export as COCO format

**3. CVAT (powerful, more complex)**
- Install: https://www.cvat.ai/
- Create project → Add images → Annotate → Export COCO 1.0

**4. labelImg (simple, but exports Pascal VOC - needs conversion)**
- Download: https://github.com/HumanSignal/labelImg
- Draw boxes, save as XML
- Convert to COCO (see conversion script below)

### Labeling Tips:

- **Draw tight boxes** around each troop (not too loose, not cutting off parts)
- **Use consistent class names**: `hog_rider`, `knight`, `archer`, `balloon`, etc.
- **Label all visible troops** in each image (don't skip any)
- **Include partial/occluded troops** if at least 50% visible
- **Split train/val**: Use 80% for training, 20% for validation

---

## Step 3: Organize Dataset

Your dataset should look like this:

```
data/arena_dataset/
  train/
    annotations.json
    images/
      frame_001.png
      frame_002.png
      ...
  val/
    annotations.json
    images/
      frame_050.png
      frame_051.png
      ...
```

Or single split:

```
data/arena_dataset/
  annotations.json
  images/
    frame_001.png
    frame_002.png
    ...
```

### COCO JSON Format

`annotations.json` should have this structure:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "frame_001.png",
      "width": 418,
      "height": 920
    },
    ...
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "bbox": [100, 200, 50, 60],  // [x, y, width, height] in pixels
      "category_id": 0
    },
    ...
  ],
  "categories": [
    {"id": 0, "name": "hog_rider"},
    {"id": 1, "name": "knight"},
    {"id": 2, "name": "archer"},
    ...
  ]
}
```

**Important:**
- `bbox` format: `[x, y, width, height]` (top-left corner + dimensions)
- `category_id` must be **0-based** (0, 1, 2, ...)
- `image_id` and `annotation id` should be unique integers

---

## Step 4: Train the Model

Once your dataset is ready:

```bash
cd /Users/rk/Documents/GitHub/clash-bot
source .venv/bin/activate

# Basic training (15 epochs, batch size 4)
python "image detector/train_arena_detector.py" --data-dir data/arena_dataset

# With custom settings
python "image detector/train_arena_detector.py" \
  --data-dir data/arena_dataset \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4 \
  --model-out "image detector/arena_detector.pth"
```

**Training options:**
- `--epochs`: Number of training epochs (default: 15, try 20-30 for better results)
- `--batch-size`: Images per batch (default: 4, reduce if you run out of memory)
- `--lr`: Learning rate (default: 1e-4)
- `--model-out`: Where to save the model (default: `image detector/arena_detector.pth`)

**Training time:**
- ~5-30 minutes depending on dataset size and hardware
- Uses CPU by default (CoreML/CPU on Mac)
- If you have CUDA GPU, it will use that automatically

---

## Step 5: Use Your Trained Model

Once training completes, the model is saved to `image detector/arena_detector.pth`.

**The detection system will automatically use it:**
- Priority order: **Your trained model** → Roboflow → Template matching
- Just run: `python test_arena_tracking.py --loop 1.0`
- It will detect and use `arena_detector.pth` automatically

**To verify it's using your model:**
- You should see: `"Using local arena detector (arena_detector.pth)"`
- Not: `"Using Roboflow Universe arena model"` or `"Using template matching"`

---

## Improving Accuracy

**1. More data**
- Collect 100-500+ images (more diverse = better)
- Include edge cases: small troops, overlapping, different game states

**2. Better labels**
- Tight, accurate bounding boxes
- Consistent class names
- Label all visible troops

**3. Training settings**
- Increase epochs: `--epochs 30`
- Try different learning rates: `--lr 5e-5` or `--lr 2e-4`
- Use train/val split to monitor overfitting

**4. Data augmentation** (advanced)
- The trainer doesn't include augmentation by default
- You could add rotation, brightness, etc. in `train_arena_detector.py`

**5. Fine-tune from checkpoint**
- Save checkpoints during training
- Resume training from a checkpoint (requires code modification)

---

## Troubleshooting

**"No images in dataset"**
- Check that `annotations.json` references images that exist
- Verify `file_name` paths match actual filenames

**"Expected annotations at..."**
- Ensure `annotations.json` is in `data/arena_dataset/` or `data/arena_dataset/train/`
- Check JSON syntax is valid

**Low accuracy after training**
- Collect more diverse training data
- Check labels are accurate
- Increase training epochs
- Verify arena region matches your screenshots

**Out of memory errors**
- Reduce `--batch-size` to 2 or 1
- Use smaller images (resize before labeling)

---

## Example Workflow

```bash
# 1. Collect screenshots (manual or script)
# Save to: data/arena_dataset/raw/

# 2. Label with Roboflow
# Upload to roboflow.com → Annotate → Export COCO → Download

# 3. Organize dataset
mkdir -p data/arena_dataset/train/images
mkdir -p data/arena_dataset/val/images
# Move train images and annotations.json to train/
# Move val images and annotations.json to val/

# 4. Train
python "image detector/train_arena_detector.py" --data-dir data/arena_dataset --epochs 20

# 5. Test
python test_arena_tracking.py --loop 1.0
```

---

## Converting from Other Formats

If you labeled with a tool that exports Pascal VOC (XML) or YOLO format, you'll need to convert to COCO. Here's a simple Python script:

```python
# convert_voc_to_coco.py (example - you may need to adapt)
import json
import xml.etree.ElementTree as ET
from pathlib import Path

# ... conversion logic ...
# See: https://github.com/facebookresearch/Detectron2/tree/main/tools
```

Or use online converters or Roboflow's import feature.
