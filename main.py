import pandas as pd

from asr.speech_2_span import whisperx_to_word_df
from helpers.build_span import build_spans
from llm_pipeline.call_llm import batch_classify_dataframe, classify_span_with_llm
from helpers.merge_intervals import merge_intervals
from asr.mute_audio import mute_audio


def main():
    AUDIO_ID = "audio_001"
    AUDIO_PATH = "C:/Users/User/Downloads/hate_videos/hate_videos/hate_video_3.mp4"
    OUTPUT_PATH = "C:/Users/User/Downloads/hate_videos/hate_videos/hate_video_3_output.mp4"
    SPANS_CSV = "artifacts/spans.csv"
    WORDS_CSV = "artifacts/words.csv"   

    # 1 Audio / Video - Audio extraction (.wav) - ASR (Whisper) - words_df - 
    # 2 build_spans() - span_df - 
    # 3 deterministic signals - min_allowed_label - 
    # 4 LLM classification - final_enforced_label - 
    # 5 filter label != NONE - extract intervals - 
    # 6 merge_intervals() - mute_audio() ###

    print("1 STEP: Running ASR")
    # words_df = whisperx_to_word_df(
    #     audio_path=AUDIO_PATH,
    #     audio_id=AUDIO_ID
    # )

    # words_df.to_csv(WORDS_CSV, index=False)

    # words_df = pd.read_csv(WORDS_CSV)

    print("2 STEP: Building spans")
    # spans_df = build_spans(words_df)
    # spans_df.to_csv("artifacts/spans_async.csv", index=False)

    spans_df = pd.read_csv(SPANS_CSV)

    print(f"LOG: Spans saved to {SPANS_CSV}")

    print("3 STEP: Deterministic Signals Analysis !!!PASS MOCK!!!")
    has_excessive_profanity, has_slur, has_targeted_insult = False, False, False #analyse_signals(spans_df: pd.Dataframe) -> List[bool]

    print("4 STEP: Classifying spans with CALL LLM")

    ### SYNC WORKING STEP###
    # for idx, row in spans_df.iterrows():
    #     result = classify_span_with_llm(
    #         span_text=row["span_text"],
    #         has_excessive_profanity=has_excessive_profanity,
    #         has_slur=has_slur,
    #         has_targeted_insult=has_targeted_insult
    #     )

    #     for k, v in result.items():
    #         spans_df.at[idx, k] = v

    ### ASYNC IMPLEMENTATION ###
    spans_llm_df = batch_classify_dataframe(spans_df)

    spans_llm_df.to_csv("artifacts/spans_async_llm.csv", index=False)

    print("5 STEP Filtering and Extracy harmful spans")
    harmful_spans = spans_df[
        spans_df["final_enforced_label"] != "NONE"
    ]

    if harmful_spans.empty:
        print("No harmful spans detected. Nice.")
        return
    
    intervals = list(zip(harmful_spans["start_time"], harmful_spans["end_time"]))

    print("6 STEP: Merge & MUTE")
    merged_intervals = merge_intervals(
        intervals,
        pad_before=0.5,
        pad_after=0.8
    )

    print("Merged intervals:")
    for i in merged_intervals:
        print("  ", i)
        
    mute_audio(
        audio_path=AUDIO_PATH,
        output_path=OUTPUT_PATH,
        intervals=merged_intervals
    )

    print(f"Muted audio written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()