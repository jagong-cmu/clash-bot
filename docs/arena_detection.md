# Arena unit detection (object detection)

Arena detection can use, in order: **Roboflow Universe** (pre-trained model), **local RetinaNet** (trained by you), or **template matching**.

---

## Roboflow Universe (pre-trained, no training)

Use a public Clash Royale troop detection model from Roboflow without training.

**Important:** The `inference` package requires **Python 3.9–3.12**. Python 3.13+ is not supported. If `pip install inference` fails with "Could not find a version that satisfies the requirement", you are likely using Python 3.14 or 3.13.

1. **Install** the inference package. Use Python 3.11 or 3.12 (create a fresh venv if needed):
   ```bash
   # If you have Python 3.14, install Python 3.12 first:
   #   brew install python@3.12

   # Create venv with Python 3.12:
   python3.12 -m venv .venv
   source .venv/bin/activate   # or: .venv\Scripts\activate on Windows

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

If a detector is loaded, you’ll see “Using Roboflow Universe arena model" or "Using local arena detector". Otherwise "Using template matching" with templates from **assets/arena/**.

---

## Troubleshooting (terminal messages and accuracy)

- **"Unknown resize method, defaulting to 'Stretch'"**  
  This can hurt accuracy. Upgrade the inference package so the model's preprocessing is handled correctly. 1.0.x is still a pre-release, so use `--pre`:
  ```bash
  pip install --upgrade --pre inference
  ```
  Or install a specific version: `pip install inference==1.0.0rc1 --pre`. Use Python 3.9-3.12; 1.0.x supports the expected resize methods.

- **"ModelDependencyMissing" (Qwen, SAM, Gaze, YoloWorld)**  
  These are optional models not used for Clash Royale arena detection. The test script disables them to reduce noise. You can ignore or suppress via env vars (e.g. `CORE_MODEL_SAM_ENABLED=False`).

- **"Specified provider 'CUDAExecutionProvider' is not in available provider names"**  
  On Mac, CUDA is not available; CoreML or CPU is used instead. This is expected and not an error.

- **Few troops detected or low accuracy**  
  1. Upgrade inference (see above).  
  2. Try a lower confidence threshold: `python test_arena_tracking.py --threshold 0.4`.  
  3. Ensure the game window (or iPhone Mirroring) is clearly visible and not minimized.

If a detector is loaded, you'll see "Using Roboflow Universe arena model" or "Using local arena detector". Otherwise "Using template matching" with templates from **assets/arena/**.

