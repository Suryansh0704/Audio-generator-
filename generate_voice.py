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

try:
    from kokoro import KPipeline
except ImportError:
    sys.exit(
        "[ERROR] kokoro not found.\n"
        "Install with:  pip install kokoro soundfile scipy\n"
        "System deps:   sudo apt-get install -y espeak-ng"
    )

VOICE          = "af_bella"
BASE_SPEED     = 1.15
PITCH_FACTOR   = 1.02
SAMPLE_RATE    = 24_000
BIT_DEPTH      = "PCM_24"

GAP_SECONDS    = 0.30
BUFFER_SECONDS = 0.50
EMPHASIS_SPEED = 1.05

INPUT_FILE     = Path("script.txt")
OUTPUT_FILE    = Path("output_voice.wav")


def make_silence(seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def pitch_shift_audio(audio: np.ndarray, factor: float = PITCH_FACTOR) -> np.ndarray:
    if factor == 1.0:
        return audio
    target_len = int(len(audio) / factor)
    shifted = scipy_resample(audio, target_len).astype(np.float32)
    return shifted


def split_into_sentences(text: str) -> list[str]:
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


def has_emphasis(sentence: str) -> bool:
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


def generate_sentence_audio(pipeline: "KPipeline", sentence: str) -> np.ndarray:
    speed = BASE_SPEED * (EMPHASIS_SPEED if has_emphasis(sentence) else 1.0)
    chunks: list[np.ndarray] = []

    for _, _, audio in pipeline(sentence, voice=VOICE, speed=speed):
        if audio is not None and len(audio) > 0:
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.array(audio, dtype=np.float32))

    if not chunks:
        return make_silence(0.05)

    return np.concatenate(chunks)


def build_full_audio(pipeline: "KPipeline", sentences: list[str]) -> np.ndarray:
    segments: list[np.ndarray] = [make_silence(BUFFER_SECONDS)]

    total = len(sentences)
    for idx, sentence in enumerate(sentences, 1):
        print(f"  [TTS {idx}/{total}] Generating → {textwrap.shorten(sentence, 60)}")
        raw_audio = generate_sentence_audio(pipeline, sentence)
        pitched = pitch_shift_audio(raw_audio, PITCH_FACTOR)
        segments.append(pitched)
        if idx < total:
            segments.append(make_silence(GAP_SECONDS))

    segments.append(make_silence(BUFFER_SECONDS))

    full = np.concatenate(segments)
    duration = len(full) / SAMPLE_RATE
    print(f"\n[INFO] Total audio duration: {duration:.2f}s  ({len(full)} samples)")
    return full


def normalize_audio(audio: np.ndarray, target_peak: float = 0.92) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio.astype(np.float32)


def export_audio(audio: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SAMPLE_RATE, subtype=BIT_DEPTH)
    size_kb = path.stat().st_size / 1024
    print(f"[✓] Audio saved → '{path}'  ({size_kb:.1f} KB, 24-bit / {SAMPLE_RATE//1000}kHz WAV)")


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

    raw_text  = load_script(INPUT_FILE)
    sentences = split_into_sentences(raw_text)
    preview_script(sentences)

    print("[INFO] Initialising Kokoro pipeline (lang='a' = American English)…")
    pipeline = KPipeline(lang_code="a")
    print("[INFO] Pipeline ready.\n")

    full_audio = build_full_audio(pipeline, sentences)
    full_audio = normalize_audio(full_audio)
    export_audio(full_audio, OUTPUT_FILE)

    print("\n[✓] Done. Upload the artifact in GitHub Actions or grab output_voice.wav.")


if __name__ == "__main__":
    main()
