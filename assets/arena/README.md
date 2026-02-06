# Arena unit templates (troops/spells as placed on the battlefield)

Use this folder to detect **troops and spells after they are placed in the arena**, not the cards in your hand.

- **Difference from `assets/cards/`:**
  - **cards/** = how the card looks in your hand (card face/art). Used for `detect_cards_in_hand()`.
  - **arena/** = how the **unit or spell looks on the battlefield** (the in-game sprite). Used for `detect_units_on_arena()`.

- **Naming:** Same card/unit id as in `assets/cards/` so you can match them (e.g. `knight.png`, `fireball.png`).

- **Content:** Screenshot or crop of the **troop or spell as it appears on the playable arena** — the character, building, or spell effect. One image per unit. Size can vary; the detector tries multiple scales.

- **Format:** PNG or JPG.

## How to get arena templates

1. Start a game (or friendly battle) and place a unit.
2. Pause or take a screenshot when the unit is clearly visible on the arena.
3. Crop to just that unit (or a small region around it) and save as `unit_id.png` here.

Spells (e.g. Fireball, Zap) are often easier to capture at the moment of impact; you can use that frame as the template.
