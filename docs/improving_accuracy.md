# Improving in-hand card detection accuracy

If the classifier gives **wrong cards** or misses cards, try these in order.

---

## 1. Check that slot crops are correct

The model only sees the **four cropped hand slots**. If those regions are off (e.g. half arena, or wrong aspect), accuracy will suffer.

**Save the four crops and inspect them:**

```bash
python test_detection.py --save-slots slots/
```

Open `slots/slot_0.png` … `slot_3.png`. Each should show **one card face** with minimal extra UI. If not:

- Edit **`src/detection.py`** and adjust the hand constants at the top:
  - `HAND_TOP`, `HAND_BOTTOM` – vertical range of the hand (default 0.72–0.98).
  - `HAND_LEFT`, `HAND_RIGHT` – horizontal range (default 0.05–0.95).
- Run `--save-slots` again until the four images look like four clean card faces.

---

## 2. See what the model is considering (--show-top)

When the top prediction is wrong, the correct card might be #2 or #3. To check:

```bash
python test_detection.py --show-top 3
```

This prints the top 3 predictions per slot with confidence. If the right card often appears as #2 or #3, the model is confused and **retraining on your data** (step 4) will help.

---

## 3. Tune the confidence threshold

- **Too many wrong cards** → raise the threshold, e.g. `--threshold 0.7`.
- **Too many slots empty** → lower it, e.g. `--threshold 0.35`.

```bash
python test_detection.py --threshold 0.6
```

---

## 4. Retrain on your own data (best fix for wrong cards)

The current model was likely trained on different resolution or card art. Training on **your** game (e.g. iPhone Mirroring) usually fixes wrong-card issues.

The current model may have been trained on different resolution or layout. Training on **your** game (e.g. iPhone Mirroring) usually helps a lot.

1. **Collect data** (with the game in a battle, hand visible):
   ```bash
   python scripts/collect_card_data.py
   ```
   Type each card name and press Enter whenever that card is visible in hand (repeat for many games so you get variety).

2. **Split into train/val:**
   ```bash
   python scripts/prepare_dataset.py
   ```

3. **Train** (more epochs often help):
   ```bash
   python "image detector/train_card_classifier.py" --epochs 40 --batch-size 16
   ```

The script saves to `image detector/card_classifier.pth`; the bot and test script will use this new model automatically.

---

## 5. Improve training data quality

- **More images per card** – aim for at least ~50–100+ per class in `train/`.
- **Balance classes** – avoid one card with 500 images and another with 10.
- **Variety** – different games, elixir states, and lighting so the model generalizes.
- **Correct labels** – wrong labels in `train/` or `val/` will cap accuracy; remove or fix bad images.

---

## 6. Train longer / slightly stronger augmentation

The trainer now uses a bit more augmentation (rotation, scale, color jitter, flip). You can also:

- Increase epochs: `--epochs 50`
- Optionally lower learning rate for fine-tuning: `--lr 5e-4`

```bash
python "image detector/train_card_classifier.py" --epochs 50 --lr 5e-4
```

---

## Quick checklist (wrong or missing cards)

| Step | Action |
|------|--------|
| 1 | Run `python test_detection.py --show-top 3` to see if the correct card is in the top 3. |
| 2 | Try `--threshold 0.7` to only show high-confidence predictions (fewer wrong, maybe more misses). |
| 3 | **Retrain on your game:** run `scripts/collect_card_data.py`, then `scripts/prepare_dataset.py`, then `python "image detector/train_card_classifier.py" --epochs 40`. |
| 4 | Use 50–100+ images per card and similar counts across cards for best accuracy. |
