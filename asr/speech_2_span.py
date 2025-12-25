from typing import List, Dict
import pandas as pd
import whisperx
import re
import subprocess
import tempfile
from pathlib import Path
import imageio_ffmpeg

def normalize_word(w: str) -> str:
    pattern_cleanse = re.compile(r"^[^\w']+|[^\w']+$")

    w = w.lower().strip()
    w = pattern_cleanse.sub("", w)

    return w

def run_whisperx(
    audio_path: str,
    device: str = "cuda",
    language: str = "en") -> Dict:

    model = whisperx.load_model(
        "large-v3",
        device=device,
        compute_type="float16" if device == "cuda" else "float32",
    )

    audio = whisperx.load_audio(audio_path)

    result = model.transcribe(audio, language=language)
    segments = result.get("segments", [])
    if not segments:
        return {}

    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
    )

    aligned = whisperx.align(
        segments,
        align_model,
        metadata,
        audio,
        device=device,
        return_char_alignments=False,
    )

    return aligned

def whisperx_output_to_word_df(
    aligned: Dict,
    audio_id: str) -> pd.DataFrame | None:

    aligned_segments = aligned.get("segments", []) if aligned else []
    if not aligned_segments:
        return

    rows: List[Dict] = []
    word_id = 0
    last_end = 0.0

    for seg in aligned_segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])

        words = seg.get("words")

        if not words:
            text_words = seg["text"].split()
            if not text_words:
                continue

            duration = max(seg_end - seg_start, 1e-6)
            step = duration / len(text_words)

            for i, raw_word in enumerate(text_words):
                word = normalize_word(raw_word)
                if not word:
                    continue

                start = seg_start + i * step
                end = start + step

                start = max(start, last_end)
                end = max(end, start)

                rows.append({
                    "audio_id": audio_id,
                    "word_id": word_id,
                    "word": word,
                    "start_time": start,
                    "end_time": end,
                })

                last_end = end
                word_id += 1

            continue

        for w in words:
            raw_word = w.get("word")
            if not raw_word:
                continue

            word = normalize_word(raw_word)
            if not word:
                continue

            start = float(w.get("start", seg_start))
            end = float(w.get("end", start))

            start = max(start, last_end)
            end = max(end, start)

            rows.append({
                "audio_id": audio_id,
                "word_id": word_id,
                "word": word,
                "start_time": start,
                "end_time": end,
            })

            last_end = end
            word_id += 1

    if not rows:
        return

    words_df = pd.DataFrame(rows)

    assert words_df["start_time"].is_monotonic_increasing
    assert (words_df["end_time"] >= words_df["start_time"]).all()

    return words_df

def _media_to_audio_path(media_path: str) -> tuple[str, tempfile.TemporaryDirectory | None]:

    path = Path(media_path)
    if not path.exists():
        raise FileNotFoundError(f"Input media does not exist: {path}")

    if path.suffix.lower() not in {".mp4", ".mkv", ".mov", ".avi"}:
        return str(path), None

    #ffmpeg_exe = shutil.which("ffmpeg")
    # if not ffmpeg_exe:
    #     raise FileNotFoundError("NO ffmpeg")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    tmpdir = tempfile.TemporaryDirectory()
    wav_path = Path(tmpdir.name) / "extracted.wav"

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(path),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        str(wav_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        tmpdir.cleanup()
        raise RuntimeError(f"{e.filename}")

    return str(wav_path), tmpdir

def whisperx_to_word_df(
    audio_path: str,
    audio_id: str,
    device: str = "cuda",
    language: str = "en") -> pd.DataFrame | None:

    media_audio_path, tmpdir = _media_to_audio_path(audio_path)
    try:
        aligned = run_whisperx(media_audio_path, device=device, language=language)
        if not aligned:
            return
        return whisperx_output_to_word_df(aligned, audio_id)
    finally:
        if tmpdir:
            tmpdir.cleanup()