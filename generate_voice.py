"""
generate_voice.py — Google Cloud TTS Premium Voice Engine
==========================================================
Voice  : en-GB-Neural2-B (deep British male)
Pitch  : -2.0 semitones (mature, expensive sound)
Target : max 60 seconds — hard cap enforced
Chain  : Bass Boost 180Hz + Vocal Compressor + Normalize
"""

import re
import sys
import base64
import requests
import os
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

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

GOOGLE_API_KEY  = os.environ.get("GOOGLE_TTS_API_KEY", "")
VOICE_NAME      = "en-GB-Neural2-B"
LANGUAGE_CODE   = "en-GB"
PITCH           = -2.0
RATE_MIN        = 1.05
RATE_MAX        = 1.50
WORDS_PER_MIN   = 150
TARGET_MAX_SEC  = 60.0
TARGET_MID_SEC  = 52.0
VOLUME_GAIN     = 2.0
SAMPLE_RATE     = 24000
BIT_DEPTH       = "PCM_24"
BASS_FREQ       = 180
BASS_GAIN_DB    = 3.5
INPUT_FILE      = Path("script.txt")
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


def calculate_rate(word_count: int) -> float:
    natural_duration = (word_count / WORDS_PER_MIN) * 60
    needed_rate = natural_duration / TARGET_MID_SEC
    rate = max(RATE_MIN, min(RATE_MAX, needed_rate))
    estimated = natural_duration / rate
    print(f"[INFO] Words: {word_count} | Natural: {natural_duration:.1f}s | "
          f"Rate: {rate:.2f}x | Estimated: {estimated:.1f}s")
    return rate


# ══════════════════════════════════════════════════════════════════
#  GOOGLE CLOUD TTS
# ══════════════════════════════════════════════════════════════════

def synthesize_speech(text: str, rate: float) -> bytes:
    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_API_KEY}"
    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": LANGUAGE_CODE,
            "name": VOICE_NAME
        },
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": SAMPLE_RATE,
            "pitch": PITCH,
            "speakingRate": rate,
            "volumeGainDb": VOLUME_GAIN,
            "effectsProfileId": ["headphone-class-device"]
        }
    }
    print(f"[TTS] Calling Google Neural2 API...")
    print(f"      Voice: {VOICE_NAME} | Pitch: {PITCH} | Rate: {rate:.2f}x")
    res = requests.post(url, json=payload, timeout=60)
    if res.status_code != 200:
        print(f"[ERROR] Google TTS failed: {res.status_code}")
        print(res.text[:500])
        sys.exit(1)
    data = res.json()
    audio_b64 = data.get("audioContent", "")
    if not audio_b64:
        sys.exit("[ERROR] No audio content in response")
    audio_bytes = base64.b64decode(audio_b64)
    print(f"[TTS] Received {len(audio_bytes)/1024:.0f} KB of audio")
    return audio_bytes


def bytes_to_numpy(audio_bytes: bytes) -> np.ndarray:
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


def hard_trim_to_60s(audio: np.ndarray) -> np.ndarray:
    max_samples = int(TARGET_MAX_SEC * SAMPLE_RATE)
    if len(audio) <= max_samples:
        return audio
    print(f"[TRIM] Audio is {len(audio)/SAMPLE_RATE:.1f}s — trimming to {TARGET_MAX_SEC}s")
    trimmed = audio[:max_samples].copy()
    fade_samples = int(0.3 * SAMPLE_RATE)
    fade = np.linspace(1.0, 0.0, fade_samples)
    trimmed[-fade_samples:] *= fade
    return trimmed


# ══════════════════════════════════════════════════════════════════
#  MASTERING CHAIN
# ══════════════════════════════════════════════════════════════════

def bass_boost(audio: np.ndarray) -> np.ndarray:
    gain_linear = 10 ** (BASS_GAIN_DB / 20)
    nyq = SAMPLE_RATE / 2
    b, a = butter(2, BASS_FREQ / nyq, btype='low', analog=False)
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
    print("  Google Neural2 Premium Voice Engine")
    print(f"  {VOICE_NAME} | Pitch {PITCH} | Max {TARGET_MAX_SEC}s")
    print("=" * 62)

    if not GOOGLE_API_KEY:
        sys.exit("[ERROR] GOOGLE_TTS_API_KEY secret not set in GitHub repo")

    if not INPUT_FILE.exists():
        sys.exit(f"[ERROR] {INPUT_FILE} not found")

    raw = INPUT_FILE.read_text(encoding="utf-8").strip()
    clean = clean_script(raw)

    if not clean:
        sys.exit("[ERROR] Nothing left after cleaning script")

    word_count = count_words(clean)
    print(f"\n[INFO] Cleaned script: {word_count} words")
    print(f"[INFO] Preview: {clean[:120]}...\n")

    rate = calculate_rate(word_count)

    audio_bytes = synthesize_speech(clean, rate)
    audio = bytes_to_numpy(audio_bytes)

    duration = len(audio) / SAMPLE_RATE
    print(f"\n[INFO] Raw audio: {duration:.2f}s")

    audio = hard_trim_to_60s(audio)

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
