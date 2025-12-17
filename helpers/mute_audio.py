from typing import List, Tuple
import numpy as np
import soundfile as sf


def mute_audio(
    audio_path: str,
    output_path: str,
    intervals: List[Tuple[float, float]],
) -> None:

    if not intervals:
        data, sr = sf.read(audio_path)
        sf.write(output_path, data, sr)
        return

    data, sr = sf.read(audio_path)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    num_samples = data.shape[0]
    audio_duration = num_samples / sr

    intervals = sorted(intervals, key=lambda x: x[0])

    for start_sec, end_sec in intervals:
        start_sec = max(0.0, start_sec)
        end_sec = min(audio_duration, end_sec)

        if start_sec >= end_sec:
            continue

        start_idx = int(round(start_sec * sr))
        end_idx = int(round(end_sec * sr))

        start_idx = max(0, min(start_idx, num_samples))
        end_idx = max(0, min(end_idx, num_samples))

        data[start_idx:end_idx, :] = 0.0

    if data.shape[1] == 1:
        data = data[:, 0]

    sf.write(output_path, data, sr)