"""
generate_voice.py — Microsoft Edge TTS Voice Engine
=====================================================
Voice  : en-GB-RyanNeural (deep British male)
Pitch  : -2Hz (deeper, mature sound)
Target : max 60 seconds — hard cap enforced
Free   : No API key needed
"""

import re
import sys
import os
import asyncio
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

try:
    from pydub import AudioSegment
    from pydub.effects import compress_dynamic_range
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import edge_tts
except ImportError:
    sys.exit("[ERROR] edge-tts not found. Install: pip install edge-tts")

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

VOICE           = "en-GB-RyanNeural"
PITCH           = "-2Hz"          # Deeper, mature, expensive sound
RATE_MIN        = "+5%"           # Slowest allowed
RATE_MAX        = "+50%"          # Fastest allowed
WORDS_PER_MIN   = 150
TARGET_MAX_SEC  = 60.0
TARGET_MID_SEC  = 52.0
SAMPLE_RATE     = 24000
BIT_DEPTH       = "PCM_24"
BASS_FREQ       = 180
BASS_GAIN_DB    = 3.5
INPUT_FILE      = Path("script.txt")
OUTPUT_MP3      = Path("output_raw.mp3")
OUTPUT_FILE     = Path("output_voice.wav")


# ══════════════════════════════════════════════════════════════════
#  SCRIPT CLEANING
# ══════════════════════════════════════════════════════════════════

def clean_script(raw: str) -> str:
    text = raw
    text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
    text = re.sub(r'-{2,}.*?-{2,}', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[\/\\@#\$%\^&\*\(\)\[\]\{\}\|<>~`_+=]', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    return text.strip()


def count_words(text: str) -> int:
    return len(text.split())


def calculate_rate(word_count: int) -> str:
    """
    Returns Edge TTS rate string like '+20%'
    Auto-calculated to hit TARGET_MID_SEC.
    """
    natural_duration = (word_count / WORDS_PER_MIN) * 60
    needed_rate = natural_duration / TARGET_MID_SEC
    # Convert to percentage: 1.0 = +0%, 1.2 = +20%, 1.5 = +50%
    pct = int((needed_rate - 1.0) * 100)
    pct = max(5, min(50, pct))  # Clamp between +5% and +50%
    estimated = natural_duration / needed_rate
    print(f"[INFO] Words: {word_count} | Natural: {natural_duration:.1f}s | "
          f"Rate: +{pct}% | Estimated: {estimated:.1f}s")
    return f"+{pct}%"


# ══════════════════════════════════════════════════════════════════
#  EDGE TTS SYNTHESIS
# ══════════════════════════════════════════════════════════════════

async def synthesize(text: str, rate: str) -> None:
    """Generate speech and save as MP3."""
    print(f"[TTS] Voice: {VOICE} | Pitch: {PITCH} | Rate: {rate}")
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        pitch=PITCH,
        rate=rate
    )
    await communicate.save(str(OUTPUT_MP3))
    size_kb = OUTPUT_MP3.stat().st_size / 1024
    print(f"[TTS] ✅ Generated {size_kb:.0f} KB MP3")


def mp3_to_wav(mp3_path: str, wav_path: str) -> np.ndarray:
    """Convert MP3 to WAV at target sample rate using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", mp3_path,
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-f", "wav",
        wav_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"[ERROR] ffmpeg conversion failed: {result.stderr[:300]}")
    # Read back as numpy
    audio, _ = sf.read(wav_path, dtype='float32')
    print(f"[TTS] Converted to WAV: {len(audio)/SAMPLE_RATE:.2f}s")
    return audio


# ══════════════════════════════════════════════════════════════════
#  HARD TRIM TO 60s
# ══════════════════════════════════════════════════════════════════

def hard_trim_to_60s(audio: np.ndarray) -> np.ndarray:
    max_samples = int(TARGET_MAX_SEC * SAMPLE_RATE)
    if len(audio) <= max_samples:
        return audio
    print(f"[TRIM] {len(audio)/SAMPLE_RATE:.1f}s → trimming to {TARGET_MAX_SEC}s")
    trimmed = audio[:max_samples].copy()
    fade_samples = int(0.3 * SAMPLE_RATE)
    trimmed[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
    return trimmed


# ══════════════════════════════════════════════════════════════════
#  MASTERING CHAIN
# ══════════════════════════════════════════════════════════════════

def bass_boost(audio: np.ndarray) -> np.ndarray:
    gain_linear = 10 ** (BASS_GAIN_DB / 20)
    nyq = SAMPLE_RATE / 2
    b, a = butter(2, BASS_FREQ / nyq, btype='low', analog=False)
    from scipy.signal import lfilter
    bass = lfilter(b, a, audio)
    out = audio + bass * (gain_linear - 1.0)
    peak = np.max(np.abs(out))
    if peak > 0.98:
        out = out * (0.95 / peak)
    return out.astype(np.float32)


def compress_audio(audio: np.ndarray) -> np.ndarray:
    if not PYDUB_AVAILABLE:
        return audio
    try:
        pcm = (audio * 32767).astype(np.int16).tobytes()
        seg = AudioSegment(
            data=pcm, sample_width=2,
            frame_rate=SAMPLE_RATE, channels=1
        )
        comp = compress_dynamic_range(
            seg, threshold=-18.0, ratio=3.5,
            attack=8.0, release=120.0
        )
        samples = np.frombuffer(comp.raw_data, dtype=np.int16).astype(np.float32)
        return (samples / 32767.0).astype(np.float32)
    except Exception as e:
        print(f"[WARN] Compression failed: {e}")
        return audio


def normalize(audio: np.ndarray, target: float = 0.92) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target / peak)
    return audio.astype(np.float32)


# ══════════════════════════════════════════════════════════════════
#  EXPORT
# ══════════════════════════════════════════════════════════════════

def export(audio: np.ndarray):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUTPUT_FILE), audio, SAMPLE_RATE, subtype=BIT_DEPTH)
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    duration = len(audio) / SAMPLE_RATE
    print(f"[✓] Saved → '{OUTPUT_FILE}'")
    print(f"    {duration:.2f}s | {size_kb:.0f} KB | 24-bit {SAMPLE_RATE//1000}kHz WAV")


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  Edge TTS Premium Voice Engine")
    print(f"  {VOICE} | Pitch {PITCH} | Max {TARGET_MAX_SEC}s | FREE")
    print("=" * 62)

    if not INPUT_FILE.exists():
        sys.exit(f"[ERROR] {INPUT_FILE} not found")

    raw = INPUT_FILE.read_text(encoding="utf-8").strip()
    clean = clean_script(raw)

    if not clean:
        sys.exit("[ERROR] Nothing left after cleaning script")

    word_count = count_words(clean)
    print(f"\n[INFO] Cleaned: {word_count} words")
    print(f"[INFO] Preview: {clean[:120]}...\n")

    rate = calculate_rate(word_count)

    # Generate MP3 via Edge TTS
    asyncio.run(synthesize(clean, rate))

    # Convert to WAV numpy array
    temp_wav = "temp_raw.wav"
    audio = mp3_to_wav(str(OUTPUT_MP3), temp_wav)

    # Hard trim
    audio = hard_trim_to_60s(audio)

    # Mastering
    print("\n[MASTER] Bass boost (180Hz +3.5dB)...")
    audio = bass_boost(audio)
    print("[MASTER] Vocal compressor...")
    audio = compress_audio(audio)
    audio = normalize(audio)

    final_dur = len(audio) / SAMPLE_RATE
    print(f"[✓] Final duration: {final_dur:.2f}s")

    export(audio)


if __name__ == "__main__":
    main()
