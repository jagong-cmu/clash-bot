# Improving arena detection accuracy

If the model misses troops, mislabels them, or gives too many false positives, try these in order.

---

## When the model doesn’t work at all: alternatives

### 1. Force template matching (no model)

Use only image templates and skip Roboflow/local model:

1. In **config/arena_detection_config.py** set **`USE_TEMPLATE_MATCHING_ONLY = True`**.
2. Add one image per troop to **assets/arena/** (e.g. `knight.png`, `hog_rider.png`). Each file should be a crop of that unit as it appears on the battlefield.
3. Run **`python test_arena_tracking.py`**. You should see “Using template matching only”.
4. Use **`--threshold 0.6`** (or lower) if you get no matches; template matching is sensitive to scale, so the script tries several scales.

**Pros:** No API, no training, works offline. **Cons:** One image per unit; fragile to big resolution or aspect changes.

### 2. See what the model actually returns (debug)

Check whether the model produces any detections at all:

```bash
python test_arena_tracking.py --debug-detection
```

This saves the image sent to the model as **debug_detection_input.png** and prints every detection with confidence ≥ 0.1. If the list is empty, the model isn’t detecting on your input (wrong image size, wrong crop, or model mismatch). If there are detections but they’re outside the arena, the arena filter may be too strict or the coordinates wrong.

### 3. Try another Roboflow model

Search [Roboflow Universe](https://universe.roboflow.com) for “Clash Royale” or “troop detection”. Pick another model and set **ROBOFLOW_ARENA_MODEL_ID** in **config/roboflow_arena_config.py** to that model’s ID (`project_id/version`).

### 4. Use your trained model and disable Roboflow

If you trained **arena_detector.pth** on the same kind of images you see at runtime:

1. In **config/roboflow_arena_config.py** set **ROBOFLOW_API_KEY = None** (or remove it) so the Roboflow model isn’t loaded.
2. Set **USE_FULL_FRAME** to match your training data: **True** if training images were full screens, **False** if they were arena crops only.
3. Run with **`--threshold 0.3`** or **0.4** and then tune.

### 5. Train a new model on your own screenshots

Capture screens from your actual setup (same device/mirror), label troops, export COCO, and train:

```bash
python "image detector/train_arena_detector.py" --data-dir data/arena_dataset --epochs 25
```

Then use that **arena_detector.pth** with Roboflow disabled and **USE_FULL_FRAME** set to match how you cropped (or didn’t) in the dataset.

---

---

## 1. Match what the model was trained on

**Roboflow model:**  
- Many are trained on **full game screens**. In `config/roboflow_arena_config.py` set **`USE_FULL_FRAME = True`** so the detector sees the full window; we still only keep detections inside the arena.  
- If you get no detections, try **`--threshold 0.25`** when testing to see if the model outputs anything at all.

**Your trained model (arena_detector.pth):**  
- It was trained on **arena crops** from your dataset. At runtime we feed an **arena crop** by default. Set **`USE_FULL_FRAME = False`** (or leave it unset) so we keep using the crop.  
- If your training images were **full screens** (e.g. Roboflow export), set **`USE_FULL_FRAME = True`** so inference matches training.

---

## 2. Tune the confidence threshold

When testing:

```bash
python test_arena_tracking.py --threshold 0.25   # more detections, more false positives
python test_arena_tracking.py --threshold 0.6    # fewer, more confident only
```

Lower threshold = more detections but more noise. Raise it until false positives are acceptable while still catching real troops.

---

## 3. Improve your own model (if you train locally)

**Train longer**

```bash
python "image detector/train_arena_detector.py" --data-dir data/arena_dataset --epochs 25
# or 30–40 if val loss is still decreasing
```

**Use a learning rate schedule (optional)**  
The trainer can be extended with a scheduler (e.g. step or cosine) so the learning rate drops over time; often improves final accuracy.

**More / better data**

- Add more screenshots from **your** setup (same device, same mirror resolution) and label them.  
- Fix bad labels: wrong boxes or wrong class names hurt accuracy.  
- Balance classes: if one troop has 10x more examples than another, the model will be biased. Add more images for rare troops or use augmentation.

**Same crop at train and test**

- If the trainer’s data is **full-frame** images, set **`USE_FULL_FRAME = True`** at inference.  
- If the trainer’s data is **arena-only crops**, keep **`USE_FULL_FRAME = False`** and ensure `config/arena_region.py` (or the default arena bounds) matches how you cropped training images.

---

## 4. Data augmentation (advanced)

The current trainer uses no augmentation. Adding random flips, small brightness/contrast changes, or slight scaling to the training pipeline can help the model generalize. That would require editing `train_arena_detector.py` and the dataset/transform logic.

---

## 5. Try a different model (Roboflow)

If the current Roboflow model stays weak, search [Roboflow Universe](https://universe.roboflow.com) for another Clash Royale / troop detection model. Update **ROBOFLOW_ARENA_MODEL_ID** in `config/roboflow_arena_config.py` to the new project’s model ID (format: `project_id/version`).

---

## Quick checklist

| Issue | What to try |
|-------|-------------|
| No detections at all | `USE_FULL_FRAME = True` (Roboflow), or `--threshold 0.25` |
| Too many false positives | Raise `--threshold` (e.g. 0.5 → 0.6) |
| Misses real troops | Lower threshold; train longer; add more data |
| Wrong labels | More/better labeled data; balance classes |
| Works in training data, fails live | Match crop/full-frame and resolution to training; add live screenshots to the dataset |
