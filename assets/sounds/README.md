# Card sound effects (optional audio verification)

Optional: place **one WAV file per card** here to double-verify card plays using audio. The filename (without extension) should match the **card ID** used in `assets/cards/` (e.g. `knight.wav`, `fireball.wav`).

- **Format:** WAV (mono or stereo; the loader converts to mono).
- **Content:** The short sound that plays when that card is placed (the in-game “card drop” sound). Record from the game or extract from game assets.
- **Length:** Up to ~1 second is used for matching; longer files are trimmed.

## Capturing game audio (macOS)

To use game audio as input for verification:

1. Install a virtual audio device (e.g. [BlackHole](https://github.com/ExistentialAudio/BlackHole)) so you can route system output into a capture input.
2. In **System Settings → Sound**, set the game’s output to go to BlackHole (or your loopback).
3. In your script, pass the BlackHole device index to `record_snippet(..., device=...)` (use `sounddevice.query_devices()` to find the index).

If you don’t use a loopback, the microphone will pick up whatever your speakers play; that can work but is noisier.

## Dependencies

Audio features require:

```bash
pip install sounddevice numpy scipy
```

If `sounddevice` or `scipy` are missing, the rest of the bot still works; only audio verification is skipped.
