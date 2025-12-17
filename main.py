from asr.speech_2_span import whisperx_to_word_df
from helpers.build_span import build_spans
from helpers.merge_intervals import merge_intervals

if __name__ == '__main__':

    words_df = whisperx_to_word_df(
    audio_path="input.wav",
    audio_id="audio_001"
    )

    spans_df = build_spans(words_df)
    spans_df.to_csv("spans.csv", index=False)

    test_intervals = [
        (1.0, 3.5),
        (2.0, 4.0),
        (5.0, 7.5),
        (6.0, 8.0),
        (10.0, 11.0),
    ]

    result = merge_intervals(test_intervals, pad_before=0.5, pad_after=0.8)
    print("Input intervals:", test_intervals)
    print("Merged intervals:", result)
