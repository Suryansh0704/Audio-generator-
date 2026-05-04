"""
generate_voice.py — Kokoro-82M Premium Male Narrator Engine
============================================================
Upgrade: Variable Pacing · Punctuation Pauses · Audio Mastering
         CAPS Boost · Sentence-End Pitch Decay

All GitHub Actions structure and file-naming logic unchanged.
"""

import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample as scipy_resample, butter, lfilter

# ── Optional pydub (for mastering) ──────────────────────────────────────────
try:
    from pydub import AudioSegment
    from pydub.effects import compress_dynamic_range
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("[WARN] pydub not found — mastering step will be skipped.")

try:
    from kokoro import KPipeline
except ImportError:
    sys.exit(
        "[ERROR] kokoro not found.\n"
        "Install with:  pip install kokoro soundfile scipy pydub\n"
        "System deps:   sudo apt-get install -y espeak-ng ffmpeg"
    )

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

VOICE               = "am_adam"      # Male voice — deep, authoritative narrator
SAMPLE_RATE         = 24_000
BIT_DEPTH           = "PCM_24"

# ── Variable Pacing ──────────────────────────────────────────────────────────
SPEED_LONG          = 1.10           # Sentences > 10 words  → flowing narration
SPEED_SHORT         = 0.95           # Sentences ≤ 10 words  → dramatic punch
WORD_COUNT_THRESHOLD = 10

# ── Pitch ────────────────────────────────────────────────────────────────────
BASE_PITCH_FACTOR   = 1.02           # Baseline +2% pitch lift (brighter tone)
CAPS_PITCH_FACTOR   = 1.05           # ALL-CAPS sentences get extra +3% on top

# ── Volume ───────────────────────────────────────────────────────────────────
CAPS_VOLUME_BOOST   = 1.05           # ALL-CAPS → +5% amplitude boost

# ── Silence Gaps ─────────────────────────────────────────────────────────────
BUFFER_SECONDS      = 0.50           # Start / end pad
SENTENCE_GAP        = 0.30           # Between normal sentences
COMMA_GAP           = 0.20           # Comma → short breath pause
SEMICOLON_GAP       = 0.40           # Semicolon → longer reflective pause
PARAGRAPH_GAP       = 0.60           # Between paragraphs (detected by blank line)

# ── Sentence-end Decay ───────────────────────────────────────────────────────
DECAY_FADE_MS       = 180            # Last N ms of a paragraph sentence fades down
DECAY_FLOOR         = 0.72           # Pitch floor at decay end (−28% lower)

# ── Bass Boost (Butterworth Low-shelf) ───────────────────────────────────────
BASS_BOOST_FREQ     = 180            # Hz — shelf cutoff
BASS_BOOST_GAIN     = 3.5            # dB

# ── Files ────────────────────────────────────────────────────────────────────
INPUT_FILE          = Path("script.txt")
OUTPUT_FILE         = Path("output_voice.wav")

# ══════════════════════════════════════════════════════════════════════════════
#  SILENCE & BASIC AUDIO HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_silence(seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def word_count(sentence: str) -> int:
    return len(sentence.split())


def has_caps(sentence: str) -> bool:
    """True if any word is ALL-CAPS and ≥ 3 chars."""
    return any(w.isupper() and len(w) >= 3 for w in sentence.split())


def to_numpy(audio) -> np.ndarray:
    """Safely convert Kokoro output (Tensor or ndarray) to float32 numpy."""
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.array(audio, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  PITCH & VOLUME PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def pitch_shift(audio: np.ndarray, factor: float) -> np.ndarray:
    """Shift pitch via rational resampling. factor > 1 = higher pitch."""
    if abs(factor - 1.0) < 0.001:
        return audio
    target_len = max(1, int(len(audio) / factor))
    return scipy_resample(audio, target_len).astype(np.float32)


def volume_boost(audio: np.ndarray, factor: float) -> np.ndarray:
    """Scale amplitude by factor, then soft-clip to prevent distortion."""
    boosted = audio * factor
    return np.tanh(boosted * 0.95).astype(np.float32)


def apply_sentence_end_decay(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Pitch-decay on the last DECAY_FADE_MS of audio.
    Simulates a human 'finishing a thought' — pitch drops toward DECAY_FLOOR.
    Strategy: resample small tail segments with increasing downshift.
    """
    fade_samples = int((DECAY_FADE_MS / 1000) * sr)
    if len(audio) <= fade_samples * 2:
        return audio  # Too short to decay safely

    tail   = audio[-fade_samples:].copy()
    body   = audio[:-fade_samples]

    # Apply a gradual pitch bend: split tail into 8 micro-segments
    n_segs = 8
    seg_len = len(tail) // n_segs
    bent_segs = []
    for i in range(n_segs):
        start = i * seg_len
        end   = start + seg_len if i < n_segs - 1 else len(tail)
        seg   = tail[start:end]
        # Interpolate pitch factor from 1.0 → DECAY_FLOOR across segments
        t      = i / (n_segs - 1)
        factor = 1.0 - t * (1.0 - DECAY_FLOOR)
        bent_segs.append(pitch_shift(seg, factor))

    decayed_tail = np.concatenate(bent_segs)
    return np.concatenate([body, decayed_tail])


# ══════════════════════════════════════════════════════════════════════════════
#  PUNCTUATION PAUSE INJECTION
# ══════════════════════════════════════════════════════════════════════════════

def split_on_punctuation(sentence: str) -> list[tuple[str, str]]:
    """
    Split a sentence into (chunk, pause_type) pairs based on commas/semicolons.
    pause_type: 'comma' | 'semicolon' | 'none'
    Example:
      "Wait, think about it; then act." →
      [("Wait", "comma"), ("think about it", "semicolon"), ("then act.", "none")]
    """
    parts = re.split(r'([,;])', sentence)
    result = []
    i = 0
    while i < len(parts):
        chunk = parts[i].strip()
        if not chunk:
            i += 1
            continue
        if i + 1 < len(parts) and parts[i + 1] in (',', ';'):
            pause_type = 'comma' if parts[i + 1] == ',' else 'semicolon'
            i += 2
        else:
            pause_type = 'none'
            i += 1
        result.append((chunk, pause_type))
    return result if result else [(sentence, 'none')]


def gap_for_pause(pause_type: str) -> np.ndarray:
    if pause_type == 'comma':
        return make_silence(COMMA_GAP)
    elif pause_type == 'semicolon':
        return make_silence(SEMICOLON_GAP)
    return np.array([], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO MASTERING  (Bass Boost + Vocal Compression)
# ══════════════════════════════════════════════════════════════════════════════

def butter_lowshelf(audio: np.ndarray, sr: int, cutoff: float, gain_db: float) -> np.ndarray:
    """
    Simple low-shelf bass boost via Butterworth IIR filter.
    Adds warmth and body to a male voice without muddying highs.
    """
    gain_linear = 10 ** (gain_db / 20)
    nyq = sr / 2
    norm_cutoff = cutoff / nyq
    # Low-pass component to extract bass, boost it, then blend back
    b, a = butter(2, norm_cutoff, btype='low', analog=False)
    bass = lfilter(b, a, audio)
    boosted = audio + bass * (gain_linear - 1.0)
    # Normalise to prevent clipping
    peak = np.max(np.abs(boosted))
    if peak > 0.98:
        boosted = boosted * (0.95 / peak)
    return boosted.astype(np.float32)


def pydub_compress(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Soft vocal compression via pydub compress_dynamic_range.
    Makes loud parts less piercing, quiet parts more audible.
    Result: a 'studio' evenness that feels polished and authoritative.
    """
    if not PYDUB_AVAILABLE:
        return audio

    # Convert float32 numpy → pydub AudioSegment (16-bit PCM)
    pcm_16 = (audio * 32767).astype(np.int16).tobytes()
    seg = AudioSegment(
        data=pcm_16,
        sample_width=2,
        frame_rate=sr,
        channels=1
    )
    compressed = compress_dynamic_range(
        seg,
        threshold=-18.0,   # dB — starts compressing above this
        ratio=3.5,          # 3.5:1 ratio — gentle but effective
        attack=8.0,         # ms — fast enough to catch transients
        release=120.0       # ms — smooth release = no pumping artefacts
    )
    # Convert back to float32
    samples = np.frombuffer(compressed.raw_data, dtype=np.int16).astype(np.float32)
    return (samples / 32767.0).astype(np.float32)


def master_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Full mastering chain:
      1. Bass Boost  (low-shelf filter @ 180Hz, +3.5dB)
      2. Compression (3.5:1, -18dB threshold)
      3. Final peak normalise to 0.92
    """
    print("[MASTER] Applying bass boost (low-shelf +3.5dB @ 180Hz)…")
    audio = butter_lowshelf(audio, sr, BASS_BOOST_FREQ, BASS_BOOST_GAIN)

    print("[MASTER] Applying vocal compressor (3.5:1 ratio, -18dB threshold)…")
    audio = pydub_compress(audio, sr)

    # Final normalise
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (0.92 / peak)
    print("[MASTER] Mastering complete.")
    return audio.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  SCRIPT LOADING & PARAGRAPH DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def load_script(path: Path) -> str:
    if not path.exists():
        sys.exit(f"[ERROR] Input file not found: {path.resolve()}")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        sys.exit("[ERROR] script.txt is empty.")
    print(f"[INFO] Loaded script ({len(raw)} chars) from '{path}'")
    return raw


def parse_paragraphs(text: str) -> list[list[str]]:
    """
    Split text into paragraphs (blank line = paragraph break).
    Each paragraph is a list of sentences.
    Returns list of paragraphs, each a list of sentence strings.
    """
    blocks = re.split(r'\n\s*\n', text.strip())
    paragraphs = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        raw_sentences = re.split(r'(?<=[.!?])\s+', block)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if sentences:
            paragraphs.append(sentences)
    return paragraphs


def preview_structure(paragraphs: list[list[str]]) -> None:
    total = sum(len(p) for p in paragraphs)
    print(f"\n[INFO] {len(paragraphs)} paragraph(s), {total} sentence(s) total:\n")
    for pi, para in enumerate(paragraphs, 1):
        print(f"  ── Paragraph {pi} ──")
        for si, s in enumerate(para, 1):
            wc    = word_count(s)
            speed = SPEED_SHORT if wc <= WORD_COUNT_THRESHOLD else SPEED_LONG
            tags  = []
            if has_caps(s):
                tags.append("CAPS↑")
            if wc <= WORD_COUNT_THRESHOLD:
                tags.append("SLOW")
            else:
                tags.append("FAST")
            tag_str = "  [" + " · ".join(tags) + "]" if tags else ""
            print(f"    [{si}] {textwrap.shorten(s, 65)}{tag_str}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  CORE TTS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def generate_chunk_audio(pipeline, chunk: str, speed: float) -> np.ndarray:
    """Run Kokoro on a single text chunk (sub-sentence split at comma/semicolon)."""
    parts = []
    for _, _, audio in pipeline(chunk, voice=VOICE, speed=speed):
        if audio is not None and len(audio) > 0:
            parts.append(to_numpy(audio))
    return np.concatenate(parts) if parts else make_silence(0.04)


def generate_sentence_audio(pipeline, sentence: str, is_paragraph_end: bool = False) -> np.ndarray:
    """
    Full sentence pipeline:
      1. Variable speed (long vs short sentence)
      2. Split on commas/semicolons → inject micro-pauses
      3. Pitch & volume adjustments for ALL-CAPS
      4. Sentence-end pitch decay if paragraph-final
    """
    wc    = word_count(sentence)
    speed = SPEED_SHORT if wc <= WORD_COUNT_THRESHOLD else SPEED_LONG
    caps  = has_caps(sentence)

    # Pitch factor
    pitch = CAPS_PITCH_FACTOR if caps else BASE_PITCH_FACTOR

    # Split on internal punctuation
    chunks_with_pauses = split_on_punctuation(sentence)
    segments = []

    for chunk_text, pause_type in chunks_with_pauses:
        if not chunk_text.strip():
            continue
        raw = generate_chunk_audio(pipeline, chunk_text, speed)
        raw = pitch_shift(raw, pitch)

        if caps:
            raw = volume_boost(raw, CAPS_VOLUME_BOOST)

        segments.append(raw)

        pause = gap_for_pause(pause_type)
        if len(pause) > 0:
            segments.append(pause)

    audio = np.concatenate(segments) if segments else make_silence(0.04)

    # Paragraph-end decay
    if is_paragraph_end:
        audio = apply_sentence_end_decay(audio)

    return audio


def build_full_audio(pipeline, paragraphs: list[list[str]]) -> np.ndarray:
    """
    Stitch all paragraphs with proper gap structure:
      [0.5s pad] + para1_sent1 + [0.3s] + para1_sent2 ... + [0.6s para gap] + para2 ... + [0.5s pad]
    """
    master_segments = [make_silence(BUFFER_SECONDS)]
    total_paras = len(paragraphs)

    for pi, para in enumerate(paragraphs, 1):
        total_sents = len(para)
        for si, sentence in enumerate(para, 1):
            is_last_in_para = (si == total_sents)
            print(f"  [Para {pi}/{total_paras} · Sent {si}/{total_sents}] "
                  f"{'↘DECAY ' if is_last_in_para else ''}"
                  f"{textwrap.shorten(sentence, 55)}")

            audio = generate_sentence_audio(pipeline, sentence, is_paragraph_end=is_last_in_para)
            master_segments.append(audio)

            # Gap after sentence (not after very last sentence in whole script)
            if not (pi == total_paras and si == total_sents):
                if is_last_in_para:
                    master_segments.append(make_silence(PARAGRAPH_GAP))
                else:
                    master_segments.append(make_silence(SENTENCE_GAP))

    master_segments.append(make_silence(BUFFER_SECONDS))

    full = np.concatenate(master_segments)
    duration = len(full) / SAMPLE_RATE
    print(f"\n[INFO] Raw audio duration: {duration:.2f}s")
    return full


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_audio(audio: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SAMPLE_RATE, subtype=BIT_DEPTH)
    size_kb = path.stat().st_size / 1024
    print(f"[✓] Saved → '{path}'  ({size_kb:.1f} KB · 24-bit / {SAMPLE_RATE // 1000}kHz WAV)")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 62)
    print("  Kokoro-82M  |  Premium Male Narrator Engine")
    print("=" * 62)
    print(f"  Voice         : {VOICE}  (male)")
    print(f"  Speed (long)  : {SPEED_LONG}x   (>{WORD_COUNT_THRESHOLD} words)")
    print(f"  Speed (short) : {SPEED_SHORT}x   (≤{WORD_COUNT_THRESHOLD} words — dramatic pause)")
    print(f"  Pitch base    : +{(BASE_PITCH_FACTOR - 1) * 100:.0f}%  "
          f"| CAPS: +{(CAPS_PITCH_FACTOR - 1) * 100:.0f}%")
    print(f"  CAPS boost    : +{(CAPS_VOLUME_BOOST - 1) * 100:.0f}% volume")
    print(f"  Gaps          : sentence={SENTENCE_GAP}s · comma={COMMA_GAP}s · "
          f"semicolon={SEMICOLON_GAP}s · paragraph={PARAGRAPH_GAP}s")
    print(f"  Decay         : last {DECAY_FADE_MS}ms of each paragraph")
    print(f"  Mastering     : Bass Boost +{BASS_BOOST_GAIN}dB @ {BASS_BOOST_FREQ}Hz + "
          f"Compressor {'✓' if PYDUB_AVAILABLE else '✗ (pydub missing)'}")
    print("=" * 62 + "\n")

    raw_text   = load_script(INPUT_FILE)
    paragraphs = parse_paragraphs(raw_text)
    preview_structure(paragraphs)

    print("[INFO] Initialising Kokoro pipeline…")
    pipeline = KPipeline(lang_code="a")
    print("[INFO] Pipeline ready.\n")

    raw_audio = build_full_audio(pipeline, paragraphs)

    print("\n[MASTER] Starting audio mastering chain…")
    final_audio = master_audio(raw_audio)

    export_audio(final_audio, OUTPUT_FILE)
    print("\n[✓] Done. Download output_voice.wav from GitHub Artifacts.")


if __name__ == "__main__":
    main()
