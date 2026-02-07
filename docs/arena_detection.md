# Arena unit detection (object detection)

Arena detection can use, in order: **PyClashBot-style** (template matching from [py-clash-bot](https://github.com/pyclashbot/py-clash-bot)), **Roboflow Universe**, **local RetinaNet**, or **template matching**.

---

## PyClashBot-style (no training, no API)

Uses the same algorithm as [py-clash-bot](https://github.com/pyclashbot/py-clash-bot)’s [image_rec.py](https://github.com/pyclashbot/py-clash-bot/blob/master/pyclashbot/detection/image_rec.py): grayscale template matching over reference images. No ML model or API key.

1. Set **USE_PYCLASHBOT_ARENA = True** in **config/arena_detection_config.py** (default).
2. Put one image per troop in **assets/arena/** (e.g. `knight.png`, `hog_rider.png`) — the unit as it appears on the battlefield.
3. Run **python test_arena_tracking.py**. You should see “Using PyClashBot-style arena detection”.

Templates are matched at a single scale; if your game resolution differs a lot, add multiple reference images or tune threshold with **--threshold 0.6**.

---

## Roboflow Universe (pre-trained, no training)

Use a public Clash Royale troop detection model from Roboflow without training.

1. **Install** the inference package:
   ```bash
   pip install inference
   ```
2. **Get an API key** from [Roboflow](https://app.roboflow.com/settings/api).
3. **Configure** `config/roboflow_arena_config.py`:
   - `ROBOFLOW_ARENA_MODEL_ID` is already set to `clash-royale-troop-detection-dkcsn/1` (from [this model](https://universe.roboflow.com/crbot-nu0tk/clash-royale-troop-detection-dkcsn/model/1)).
   - Set `ROBOFLOW_API_KEY = "rf_xxxx..."` or export in the shell: `export ROBOFLOW_API_KEY=rf_xxxx...`
4. Run `python test_arena_tracking.py`; the app will use the Roboflow model automatically.

---

## Local object detection (train your own)

1. **Collect labeled data**  
   Screenshots of the game with units on the arena, with bounding boxes and class labels. Use a COCO-style `annotations.json` and an `images/` folder (see **data/arena_dataset/README.md**).

2. **Train**  
   From project root:
   ```bash
   python "image detector/train_arena_detector.py" --data-dir data/arena_dataset --epochs 20
   ```
   This produces **image detector/arena_detector.pth**.

3. **Use**  
   `detect_units_on_arena()` and `ArenaTracker.update()` will automatically use the detector when `arena_detector.pth` exists. No template images are required.

4. **Config**  
   Optional override: set **ARENA_MODEL_PATH** in **config/arena_detector_config.py** to point to another checkpoint.

---

## Template matching (fallback)

If no arena detector model is found, detection falls back to template matching:

- Put one image per unit in **assets/arena/** (e.g. `hog_rider.png`, `skeleton.png`).
- Each image should show the unit **as it appears on the battlefield** (in-game sprite).
- Load with `load_card_templates("assets/arena")` and pass to `detect_units_on_arena(screenshot, arena_templates=...)`.

---

## Testing

```bash
python test_arena_tracking.py
python test_arena_tracking.py --loop 1.0 --threshold 0.5
```

If a detector is loaded, you’ll see “Using arena object detector”. Otherwise “Using template matching” with templates from **assets/arena/**.
