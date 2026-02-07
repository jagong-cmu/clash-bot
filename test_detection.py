"""
Test troop/spell detection: capture the game window and print:
  - Cards in hand (trained classifier by default; use --templates for template matching).
  - Units on the arena (from assets/arena/ templates).
Optionally run with --audio to record and match sound signatures.

Usage:
  python test_detection.py              # classifier for hand (no card assets needed)
  python test_detection.py --templates  # template matching for hand (needs assets/cards/)
  python test_detection.py --audio

Requires:
  - Game window visible (iPhone Mirroring or set via --window).
  - For classifier: torch, torchvision, and image detector/card_classifier.pth.
  - For template fallback: assets/cards/. For arena: assets/arena/.
  - For --audio: assets/sounds/ and pip install sounddevice numpy scipy
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from src.coords import get_window_coordinates
from src.capture import capture_screen_region
from src.detection import (
    load_card_templates,
    detect_cards_in_hand,
    detect_any_card_on_screen,
    get_best_match_scores,
)
from src.classifier import is_available as classifier_available, get_load_error as classifier_load_error

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CARDS_DIR = os.path.join(PROJECT_ROOT, "assets", "cards")
ARENA_DIR = os.path.join(PROJECT_ROOT, "assets", "arena")
SOUNDS_DIR = os.path.join(PROJECT_ROOT, "assets", "sounds")


def main():
    parser = argparse.ArgumentParser(description="Test card detection on Clash Royale window")
    parser.add_argument("--audio", action="store_true", help="Record and match audio after one frame")
    parser.add_argument("--window", default="iPhone Mirroring", help="Window name to capture")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for classifier / match threshold (0-1)")
    parser.add_argument("--templates", action="store_true", help="Use template matching for hand instead of classifier")
    parser.add_argument("--save", metavar="FILE", help="Save captured frame to FILE (e.g. frame.png) to inspect what the bot sees")
    parser.add_argument("--loop", type=float, metavar="SEC", help="Run detection every SEC seconds until Ctrl+C (e.g. --loop 2)")
    args = parser.parse_args()

    use_classifier = not args.templates
    if use_classifier:
        if classifier_available():
            print("Using trained card classifier (image detector/card_classifier.pth) for hand detection.")
        else:
            print("Classifier not available:", classifier_load_error() or "unknown")
            print("Falling back to template matching. Add card images to assets/cards/ or install torch/torchvision.")
            use_classifier = False
    card_templates = load_card_templates(CARDS_DIR) if not use_classifier else {}
    if not use_classifier and not card_templates:
        print("No card templates found in", CARDS_DIR)
        print("Add PNG/JPG images named by card, or use classifier (install torch, ensure image detector/card_classifier.pth exists).")
        return 1
    if card_templates:
        print(f"Loaded {len(card_templates)} card templates: {sorted(card_templates.keys())}")

    window_coords = get_window_coordinates(args.window)
    if not window_coords:
        print(f"Window '{args.window}' not found. Try --window with a partial name.")
        return 1
    game_x, game_y, game_width, game_height = window_coords

    sound_signatures = None
    record_snippet = None
    match_sound = None
    if args.audio:
        try:
            from src.audio_detection import (
                load_sound_signatures,
                record_snippet as _record_snippet,
                match_sound as _match_sound,
            )
            record_snippet = _record_snippet
            match_sound = _match_sound
            sound_signatures = load_sound_signatures(SOUNDS_DIR)
            if sound_signatures:
                print(f"Loaded {len(sound_signatures)} sound signatures for verification")
            else:
                print("No WAVs in assets/sounds/; skipping audio")
        except ImportError as e:
            print("Audio deps missing (pip install sounddevice numpy scipy); skipping audio:", e)

    arena_templates = load_card_templates(ARENA_DIR)
    if arena_templates:
        print(f"Loaded {len(arena_templates)} arena templates: {sorted(arena_templates.keys())}")
    if args.loop and args.loop > 0:
        print(f"Looping every {args.loop}s. Focus the game window; press Ctrl+C to stop.")
        time.sleep(2)
        loop_interval = args.loop
    else:
        print("Capturing in 2 seconds... Focus the game window.")
        time.sleep(2)
        loop_interval = None

    def run_one_capture():
        screen = capture_screen_region(game_x, game_y, game_width, game_height)
        if screen is None or screen.size == 0:
            return
        if args.save and not getattr(run_one_capture, "_saved", False):
            cv2.imwrite(args.save, screen)
            print(f"Saved frame to {args.save}")
            run_one_capture._saved = True
        # Cards in hand (classifier or template matching)
        hand_matches = detect_cards_in_hand(
            screen,
            templates=card_templates if not use_classifier else None,
            threshold=args.threshold,
            use_classifier=use_classifier,
        )
        if hand_matches:
            print("Cards in hand:")
            for m in hand_matches:
                print(f"  Slot {m.slot}: {m.card_id} ({m.confidence:.2f})")
        else:
            print("No cards in hand detected above threshold.")
            if use_classifier:
                print("  → Try lowering --threshold or check image detector/card_classifier.pth")
            else:
                print("  → Try --threshold 0.25 or add templates. Use --save frame.png to inspect.")
        # Arena detection (full-screen search for battlefield units)
        if arena_templates:
            arena_matches = detect_any_card_on_screen(
                screen, arena_templates, threshold=args.threshold
            )
            if arena_matches:
                print("Units on arena (battlefield):")
                for unit_id, conf, cx, cy in arena_matches:
                    print(f"  {unit_id} - similarity: {conf:.3f} at ({cx}, {cy})")
            else:
                print("No units on arena detected.")
                best = get_best_match_scores(screen, arena_templates)
                for unit_id, score, cx, cy in sorted(best, key=lambda x: -x[1]):
                    print(f"  Best: {unit_id} - similarity: {score:.3f} at ({cx}, {cy})")
        return screen

    screen = run_one_capture()
    if loop_interval:
        try:
            while True:
                time.sleep(loop_interval)
                print("---")
                run_one_capture()
        except KeyboardInterrupt:
            print("\nStopped.")

    if args.audio and sound_signatures and record_snippet and match_sound and not loop_interval:
        print("\nRecording 1s for audio match...")
        snippet = record_snippet(1.0)
        if snippet is not None:
            audio_matches = match_sound(snippet, sound_signatures)
            print("Top audio matches:", audio_matches[:5])

    return 0


if __name__ == "__main__":
    sys.exit(main())
