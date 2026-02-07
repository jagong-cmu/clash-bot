# Card template images (in-hand detection)

Place **one image per card** here for **detecting cards in your hand**. The filename (without extension) is used as the **card ID** in code.

To detect **troops/spells as they are placed on the arena**, use **assets/arena/** with images of each unit as it appears on the battlefield (see assets/arena/README.md).

- **Format:** PNG or JPG.
- **Naming:** Use a consistent key, e.g. `knight.png`, `fireball.png`, `hog_rider.png`.
- **Content:** Crop to the **card art/icon as it appears in your game** (in-hand card face). Matching works best when the template looks like a single card at roughly the size it appears on screen.
- **Size:** Any size is fine; the detector scales templates to the hand slot region. For best results, use a crop from a 1080p (or similar) game frame so detail is preserved.

## How to get templates

1. Take a screenshot of the game with the card visible in your hand (or in the card collection).
2. Crop each card to just the face (no border needed, but consistent lighting helps).
3. Save as `card_id.png` in this folder.

You only need templates for the cards you want to detect. The detector will only report cards that have a template and score above the confidence threshold.
