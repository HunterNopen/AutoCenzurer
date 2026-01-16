from typing import List, Tuple
import numpy as np
import soundfile as sf
import logging

logger = logging.getLogger(__name__)


def mute_audio(
    audio_path: str,
    output_path: str,
    intervals: List[Tuple[float, float]]) -> None:
    
    logger.info(f"mute_audio: Reading {audio_path}")
    
    try:
        if not intervals:
            data, sr = sf.read(audio_path)
            sf.write(output_path, data, sr)
            logger.info(f"mute_audio: No intervals, copied audio to {output_path}")
            return

        data, sr = sf.read(audio_path)
        logger.info(f"mute_audio: Loaded audio - shape={data.shape}, sr={sr}")
        
        if data.ndim == 1:
            data = data[:, np.newaxis]

        num_samples = data.shape[0]
        audio_duration = num_samples / sr
        logger.info(f"mute_audio: Duration={audio_duration:.2f}s, samples={num_samples}")

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
            logger.info(f"mute_audio: Muted [{start_sec:.2f}s - {end_sec:.2f}s]")

        if data.shape[1] == 1:
            data = data[:, 0]

        logger.info(f"mute_audio: Writing to {output_path}")
        sf.write(output_path, data, sr)
        logger.info(f"mute_audio: Done!")
        
    except Exception as e:
        logger.error(f"mute_audio ERROR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise