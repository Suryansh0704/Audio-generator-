"""
generate_voice.py — Kokoro-82M Voiceover Engine
================================================
Senior Automation Engineer Build | Free-tier GitHub Actions compatible.

Features:
  - Sentence-level audio generation with Kokoro-82M (ONNX, CPU-only)
  - Speed set to 1.15x  (high energy, never 1.0x)
  - Pitch shifted +2%   (sounds 'younger', brighter)
  - 0.3s silence gap    between sentences (momentum, no swiping)
  - 0.5s silence buffer at start & end    (professional bookends)
  - UPPERCASE detection  → per-word speed nudge for emphasis
  - Exports 24-bit / 24kHz WAV  (high quality, small size)
"""

import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample as scipy_resample

# ── Kokoro import (graceful error) ──────────────────────────────────────────
try:
    from kokoro import KPipeline
except ImportError:
    sys.exit(
        "[ERROR] kokoro not found.\n"
        "Install with:  pip install kokoro soundfile scipy\n"
        "System deps:   sudo apt-get install -y espeak-ng"
    )

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  — tweak these to dial in your 'Viral' sound
# ══════════════════════════════════════════════════════════════════════════════

VOICE          = "af_bella"          # American English female — energetic & clear
BASE_SPEED     = 1.15                # 1.15x = punchy, modern attention span
PITCH_FACTOR   = 1.02                # +2% pitch lift  → sounds younger / brighter
SAMPLE_RATE    = 24_000              # Kokoro native sample rate
BIT_DEPTH      = "PCM_24"           # 24-bit WAV output

GAP_SECONDS    = 0.30               # gap between sentences  (DO: 0.3s, DON'T: 1.0s)
BUFFER_SECONDS = 0.50               # silence at start / end (professional bookend)

EMPHASIS_SPEED = 1.05               # tiny speed-UP on ALL-CAPS words for punch

INPUT_FILE     = Path("script.txt")
OUTPUT_FILE    = Path("output_voice.wav")

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_silence(seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a float32 silence array."""
    return np.zeros(int(seconds * sr), dtype=np.float32)


def pitch_shift_audio(audio: np.ndarray, factor: float = PITCH_FACTOR) -> np.ndarray:
    """
    Shift pitch by `factor` (1.02 = +2%) via rational resampling.
    Strategy:
      1. Resample to N/factor samples  → same duration, higher pitch
      Done in one scipy call; negligible quality loss at 2%.
    """
    if factor == 1.0:
        return audio
    target_len = int(len(audio) / factor)
    shifted = scipy_resample(audio, target_len).astype(np.float32)
    return shifted


def split_into_sentences(text: str) -> list[str]:
    """
    Split on sentence-ending punctuation.
    Preserves the punctuation mark. Strips blanks.
    """
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


def has_emphasis(sentence: str) -> bool:
    """Return True if sentence contains at least one ALL-CAPS word (≥3 chars)."""
    words = sentence.split()
    return any(w.isupper() and len(w) >= 3 for w in words)


def load_script(path: Path) -> str:
    if not path.exists():
        sys.exit(f"[ERROR] Input file not found: {path.resolve()}")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        sys.exit("[ERROR] script.txt is empty.")
    print(f"[INFO] Loaded script ({len(raw)} chars) from '{path}'")
    return raw


def preview_script(sentences: list[str]) -> None:
    print(f"\n[INFO] {len(sentences)} sentence(s) detected:\n")
    for i, s in enumerate(sentences, 1):
        preview = textwrap.shorten(s, width=72, placeholder="…")
        marker = "  ★ EMPHASIS" if has_emphasis(s) else ""
        print(f"  [{i:02d}] {preview}{marker}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  CORE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def generate_sentence_audio(pipeline: "KPipeline", sentence: str) -> np.ndarray:
    """
    Run Kokoro on a single sentence.
    Applies EMPHASIS_SPEED boost if sentence contains ALL-CAPS word(s).
    Kokoro yields (graphemes, phonemes, audio_chunk) tuples.
    """
    speed = BASE_SPEED * (EMPHASIS_SPEED if has_emphasis(sentence) else 1.0)
    chunks: list[np.ndarray] = []

    for _, _, audio in pipeline(sentence, voice=VOICE, speed=speed):
        if audio is not None and len(audio) > 0:
            chunks.append(audio.astype(np.float32))

    if not chunks:
        # Kokoro returned nothing — return tiny silence as fallback
        return make_silence(0.05)

    return np.concatenate(chunks)


def build_full_audio(pipeline: "KPipeline", sentences: list[str]) -> np.ndarray:
    """
    Generate audio for every sentence, weave in silence gaps, apply pitch shift.
    Layout:
      [0.5s buffer] + [sent1] + [0.3s gap] + [sent2] + ... + [0.5s buffer]
    """
    segments: list[np.ndarray] = [make_silence(BUFFER_SECONDS)]  # opening pad

    total = len(sentences)
    for idx, sentence in enumerate(sentences, 1):
        print(f"  [TTS {idx}/{total}] Generating → {textwrap.shorten(sentence, 60)}")
        raw_audio = generate_sentence_audio(pipeline, sentence)

        # Apply pitch shift to each sentence independently
        pitched = pitch_shift_audio(raw_audio, PITCH_FACTOR)
        segments.append(pitched)

        # Add gap after every sentence except the last
        if idx < total:
            segments.append(make_silence(GAP_SECONDS))

    segments.append(make_silence(BUFFER_SECONDS))  # closing pad

    full = np.concatenate(segments)
    duration = len(full) / SAMPLE_RATE
    print(f"\n[INFO] Total audio duration: {duration:.2f}s  ({len(full)} samples)")
    return full


def normalize_audio(audio: np.ndarray, target_peak: float = 0.92) -> np.ndarray:
    """Peak-normalize to avoid clipping while maximizing loudness."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio.astype(np.float32)


def export_audio(audio: np.ndarray, path: Path) -> None:
    """Write high-quality 24-bit WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SAMPLE_RATE, subtype=BIT_DEPTH)
    size_kb = path.stat().st_size / 1024
    print(f"[✓] Audio saved → '{path}'  ({size_kb:.1f} KB, 24-bit / {SAMPLE_RATE//1000}kHz WAV)")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  Kokoro-82M Voiceover Engine  |  Brainrot Edition")
    print("=" * 60)
    print(f"  Voice    : {VOICE}")
    print(f"  Speed    : {BASE_SPEED}x  (emphasis: {BASE_SPEED * EMPHASIS_SPEED:.3f}x)")
    print(f"  Pitch    : +{(PITCH_FACTOR - 1) * 100:.0f}%")
    print(f"  Gap      : {GAP_SECONDS}s between sentences")
    print(f"  Buffer   : {BUFFER_SECONDS}s at start/end")
    print("=" * 60 + "\n")

    # 1. Load & split script
    raw_text  = load_script(INPUT_FILE)
    sentences = split_into_sentences(raw_text)
    preview_script(sentences)

    # 2. Boot Kokoro pipeline (downloads model on first run, ~330MB)
    print("[INFO] Initialising Kokoro pipeline (lang='a' = American English)…")
    pipeline = KPipeline(lang_code="a")
    print("[INFO] Pipeline ready.\n")

    # 3. Generate, stitch, pitch-shift
    full_audio = build_full_audio(pipeline, sentences)

    # 4. Normalise & export
    full_audio = normalize_audio(full_audio)
    export_audio(full_audio, OUTPUT_FILE)

    print("\n[✓] Done. Upload the artifact in GitHub Actions or grab output_voice.wav.")


if __name__ == "__main__":
    main()
