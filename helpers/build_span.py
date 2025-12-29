import pandas as pd

from static.config import MAX_WORDS, OVERLAP_WORDS, PAUSE_THRESHOLD

def add_overlap_words(span_text: list, span_overlap_words: pd.DataFrame, prepend: bool = True) -> list:
    if prepend:
        for j in range(len(span_overlap_words)):
            span_text.insert(0, span_overlap_words.iloc[j]['word'])
    elif not prepend :
        for j in range(len(span_overlap_words)):
            span_text.append(span_overlap_words.iloc[j]['word'])
    return span_text

def build_spans(words_df: pd.DataFrame,
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS,
    pause_threshold: float = PAUSE_THRESHOLD) -> pd.DataFrame:

    span_df = pd.DataFrame(columns=[
        "span_id",
        "span_text",
        "start_time",
        "end_time",
        "has_excessive_profanity",
        "has_slur",
        "has_targeted_insult",
        "profanity_hits",
        "slur_hits",
        "insult_hits",
        "llm_label",
        "llm_confidence",
        "llm_rationale",
        "final_enforced_label"
    ])

    ### Assuming that words will be grouped by audio_id beforehand and invoked iteratively this if needed
    cursor = 0
    span_counter = 0
    len_df = len(words_df)
    while cursor < len_df:
        span_start_idx = cursor
        main_span_chunk = words_df.iloc[span_start_idx:span_start_idx+max_words]
        span_text = []

        for i in range(len(main_span_chunk)):
            word_curr = main_span_chunk.iloc[i]
            span_text.append(word_curr['word'])

            if i + 1 >= len(main_span_chunk):
                break

            word_next = main_span_chunk.iloc[i + 1]
            if word_next['start_time'] - word_curr['end_time'] >= pause_threshold:
                break

        effective_len = len(span_text)
        spand_end_idx = span_start_idx + effective_len - 1

        if span_start_idx - overlap_words > 0:
            span_text = add_overlap_words(span_text, words_df.iloc[span_start_idx-overlap_words:span_start_idx], prepend=True)
        if spand_end_idx + overlap_words < len_df:
            span_text = add_overlap_words(span_text, words_df.iloc[spand_end_idx+1:spand_end_idx+1+overlap_words], prepend=False)

        span_row = {
            "span_id": span_counter,
            "span_text": ' '.join(span_text),
            "start_time": words_df.iloc[span_start_idx-overlap_words]['start_time'] if span_start_idx - overlap_words > 0 else words_df.iloc[span_start_idx]['start_time'],
            "end_time": words_df.iloc[spand_end_idx+overlap_words]['end_time'] if spand_end_idx + overlap_words < len_df else words_df.iloc[spand_end_idx]['end_time']
        }
        span_df = pd.concat([span_df, pd.DataFrame([span_row])], ignore_index=True)
        cursor += max(effective_len - overlap_words, 1)
        span_counter += 1

    return span_df