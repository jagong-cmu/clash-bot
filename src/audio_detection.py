"""
Optional audio-based card verification for Clash Royale.

Each card has a distinct play sound. This module records a short snippet and
matches it against reference WAV files to double-verify a card (e.g. after
image detection or when you hear a card being played).

Setup:
- Install: pip install sounddevice numpy scipy
- Place reference WAVs in assets/sounds/ named by card_id (e.g. knight.wav, fireball.wav).
  These should be the in-game "card placed" sound for each card (record or extract from game).
- On macOS, to capture game audio: use a virtual loopback (e.g. BlackHole) and set
  that as the input device, or record from the same device the game outputs to.
"""

from __future__ import annotations

import os
import wave
import struct
from typing import Optional

try:
    import numpy as np
    import sounddevice as sd
    HAS_AUDIO_DEPS = True
except ImportError:
    HAS_AUDIO_DEPS = False

# Optional: scipy for better correlation (faster and more robust)
try:
    from scipy import signal as scipy_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_sound_signatures(
    sounds_dir: str,
    sample_rate: int = 44100,
    max_duration_sec: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Load reference WAV files from a directory.
    Filename without extension = card_id. Each is resampled to sample_rate and trimmed.

    Returns:
        Dict of card_id -> 1D float array (mono, normalized).
    """
    if not HAS_AUDIO_DEPS:
        return {}
    out = {}
    if not os.path.isdir(sounds_dir):
        return out
    max_samples = int(sample_rate * max_duration_sec)
    for name in os.listdir(sounds_dir):
        if not name.lower().endswith(".wav"):
            continue
        path = os.path.join(sounds_dir, name)
        card_id = os.path.splitext(name)[0]
        try:
            data, sr = _read_wav_mono(path, target_sr=sample_rate, max_samples=max_samples)
            if data is not None and len(data) > 0:
                # Normalize
                peak = np.abs(data).max()
                if peak > 0:
                    data = data / peak
                out[card_id] = data
        except Exception:
            continue
    return out


def _read_wav_mono(
    path: str,
    target_sr: int = 44100,
    max_samples: Optional[int] = None,
) -> tuple[Optional[np.ndarray], int]:
    """Read WAV to mono float in [-1, 1]; resample to target_sr if needed."""
    if not HAS_AUDIO_DEPS:
        return None, target_sr
    with wave.open(path, "rb") as w:
        nch = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if sampwidth == 2:  # 16-bit
        fmt = f"{n * nch}h"
        samples = np.array(struct.unpack(fmt, raw), dtype=np.float64) / 32768.0
    else:
        # Fallback: treat as bytes and normalize
        samples = np.frombuffer(raw, dtype=np.int8).astype(np.float64) / 128.0
    if nch == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    if sr != target_sr and len(samples) > 0:
        # Simple linear resample
        old_len = len(samples)
        new_len = int(old_len * target_sr / sr)
        indices = np.linspace(0, old_len - 1, new_len)
        samples = np.interp(indices, np.arange(old_len), samples)
        sr = target_sr
    if max_samples is not None and len(samples) > max_samples:
        samples = samples[:max_samples]
    return samples, sr


def record_snippet(
    duration_sec: float = 1.0,
    sample_rate: int = 44100,
    device: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Record a short mono snippet from the default (or given) input device.

    Args:
        duration_sec: Recording length in seconds.
        sample_rate: Sample rate (should match reference WAVs).
        device: sounddevice device index; None = default input.

    Returns:
        1D float array in [-1, 1], or None if recording failed or deps missing.
    """
    if not HAS_AUDIO_DEPS:
        return None
    n = int(duration_sec * sample_rate)
    try:
        rec = sd.rec(n, samplerate=sample_rate, channels=1, dtype="float32", device=device)
        sd.wait()
        return rec.squeeze()
    except Exception:
        return None


def match_sound(
    snippet: np.ndarray,
    signatures: dict[str, np.ndarray],
    sample_rate: int = 44100,
) -> list[tuple[str, float]]:
    """
    Match a recorded snippet against reference signatures.
    Uses cross-correlation (numpy or scipy). Snippet can be longer than refs.

    Returns:
        List of (card_id, score) sorted by score descending. Score is normalized correlation [0, 1].
    """
    if not HAS_AUDIO_DEPS or not snippet.size or not signatures:
        return []
    snippet = snippet.astype(np.float64)
    if np.abs(snippet).max() > 0:
        snippet = snippet / np.abs(snippet).max()
    results = []
    for card_id, ref in signatures.items():
        ref = ref.astype(np.float64)
        if ref.size == 0:
            continue
        if HAS_SCIPY:
            corr = scipy_signal.correlate(snippet, ref, mode="valid")
        else:
            corr = np.correlate(snippet, ref, mode="valid")
        if corr.size == 0:
            continue
        # Normalize by ref energy so score is ~[0,1]
        ref_energy = np.sqrt(np.sum(ref ** 2))
        if ref_energy > 0:
            peak = np.abs(corr).max() / ref_energy
            # Scale to rough 0-1 (peak can be > 1 with noise)
            score = min(1.0, float(peak) / max(1, len(ref) ** 0.5 * 0.1))
        else:
            score = 0.0
        results.append((card_id, score))
    results.sort(key=lambda x: -x[1])
    return results


def verify_card_with_audio(
    candidate_card_id: str,
    signatures: dict[str, np.ndarray],
    record_duration_sec: float = 1.0,
    sample_rate: int = 44100,
    min_score: float = 0.3,
) -> bool:
    """
    Record a snippet and check if the best match is the given candidate card.
    Use after image detection to double-verify with audio.

    Returns:
        True if the best-matching card is candidate_card_id and score >= min_score.
    """
    if candidate_card_id not in signatures or not HAS_AUDIO_DEPS:
        return False
    snippet = record_snippet(record_duration_sec, sample_rate)
    if snippet is None:
        return False
    matches = match_sound(snippet, signatures, sample_rate)
    if not matches:
        return False
    best_id, best_score = matches[0]
    return best_id == candidate_card_id and best_score >= min_score
