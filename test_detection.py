"""
Test troop/spell detection: capture the game window and print:
  - Cards in hand (from assets/cards/).
  - Units on the arena / placed on the battlefield (from assets/arena/).
Optionally run with --audio to record and match sound signatures.

Usage:
  python test_detection.py
  python test_detection.py --arena     # also detect units on the battlefield
  python test_detection.py --audio

Requires:
  - Game window visible (name containing "iPhone Mirroring" or set via --window).
  - assets/cards/ for in-hand detection; assets/arena/ for on-arena detection (see READMEs).
  - For --audio: assets/sounds/ with WAVs and: pip install sounddevice numpy scipy
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
    detect_units_on_arena,
    CardMatch,
    ArenaUnitMatch,
)

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CARDS_DIR = os.path.join(PROJECT_ROOT, "assets", "cards")
ARENA_DIR = os.path.join(PROJECT_ROOT, "assets", "arena")
SOUNDS_DIR = os.path.join(PROJECT_ROOT, "assets", "sounds")


def main():
    parser = argparse.ArgumentParser(description="Test card detection on Clash Royale window")
    parser.add_argument("--audio", action="store_true", help="Record and match audio after one frame")
    parser.add_argument("--arena", action="store_true", help="Also detect units on the arena (placed troops/spells)")
    parser.add_argument("--window", default="iPhone Mirroring", help="Window name to capture")
    parser.add_argument("--threshold", type=float, default=0.75, help="Image match threshold (0-1)")
    parser.add_argument("--save", metavar="FILE", help="Save captured frame to FILE (e.g. frame.png) to inspect what the bot sees")
    parser.add_argument("--loop", type=float, metavar="SEC", help="Run detection every SEC seconds until Ctrl+C (e.g. --loop 2)")
    args = parser.parse_args()

    templates = load_card_templates(CARDS_DIR)
    if not templates:
        print("No card templates found in", CARDS_DIR)
        print("Add PNG/JPG images named by card (e.g. knight.png). See assets/cards/README.md")
        return 1
    print(f"Loaded {len(templates)} card templates: {sorted(templates.keys())}")

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

    arena_templates = load_card_templates(ARENA_DIR) if args.arena else {}
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
        # In-hand detection
        matches: list[CardMatch] = detect_cards_in_hand(screen, templates, threshold=args.threshold)
        if not matches:
            print("No cards in hand detected above threshold. Try lowering --threshold or adding templates to assets/cards/.")
        else:
            print("Cards in hand:")
            for m in matches:
                print(f"  Slot {m.slot}: {m.card_id} ({m.confidence:.2f})")
        # Arena detection
        if arena_templates:
            arena_matches: list[ArenaUnitMatch] = detect_units_on_arena(
                screen, arena_templates, threshold=args.threshold, scales=[0.7, 1.0, 1.3]
            )
            if arena_matches:
                print("Units on arena:")
                for m in arena_matches:
                    print(f"  {m.unit_id} at ({m.center_x}, {m.center_y}) conf={m.confidence:.2f}")
            else:
                print("No units on arena detected.")
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
