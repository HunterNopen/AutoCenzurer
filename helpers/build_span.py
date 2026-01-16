import pandas as pd

from static.config import MAX_WORDS, OVERLAP_WORDS, PAUSE_THRESHOLD, LABEL_ORDER


def build_spans(words_df: pd.DataFrame,
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS,
    pause_threshold: float = PAUSE_THRESHOLD) -> pd.DataFrame:
    
    spans = []
    cursor = 0
    span_counter = 0
    len_df = len(words_df)
    
    while cursor < len_df:
        span_start_idx = cursor
        
        is_beginning = (cursor == 0)
        
        main_span_chunk = words_df.iloc[span_start_idx:span_start_idx + max_words]
        main_words = []
        
        for i in range(len(main_span_chunk)):
            word_curr = main_span_chunk.iloc[i]
            main_words.append(word_curr['word'])
            
            if i + 1 < len(main_span_chunk):
                word_next = main_span_chunk.iloc[i + 1]
                if word_next['start_time'] - word_curr['end_time'] >= pause_threshold:
                    break
        
        effective_len = len(main_words)
        span_end_idx = span_start_idx + effective_len - 1
        is_end = (span_end_idx >= len_df - 1)
        
        overlap_before_start = max(0, span_start_idx - overlap_words)
        overlap_after_end = min(len_df - 1, span_end_idx + overlap_words)
        
        span_text = []
        
        if not is_beginning and overlap_before_start < span_start_idx:
            overlap_before_words = words_df.iloc[overlap_before_start:span_start_idx]
            for j in range(len(overlap_before_words)):
                span_text.append(overlap_before_words.iloc[j]['word'])
        
        span_text.extend(main_words)
        
        if not is_end and span_end_idx + 1 <= overlap_after_end:
            overlap_after_words = words_df.iloc[span_end_idx + 1:overlap_after_end + 1]
            for j in range(len(overlap_after_words)):
                span_text.append(overlap_after_words.iloc[j]['word'])
        
        actual_start_idx = overlap_before_start if not is_beginning else span_start_idx
        actual_end_idx = overlap_after_end if not is_end else span_end_idx
        
        span_row = {
            "span_id": span_counter,
            "span_text": ' '.join(span_text),
            "start_time": words_df.iloc[actual_start_idx]['start_time'],
            "end_time": words_df.iloc[actual_end_idx]['end_time'],
            "has_excessive_profanity": None,
            "has_slur": None,
            "has_targeted_insult": None,
            "has_threat_or_violence": None,
            "profanity_hits": None,
            "slur_hits": None,
            "insult_hits": None,
            "threat_or_violence_hits": None
        }
        spans.append(span_row)
        cursor += max(effective_len, 1)
        span_counter += 1
    
    if not spans:
        return pd.DataFrame(columns=[
            'span_id', 'span_text', 'start_time', 'end_time',
            'has_excessive_profanity', 'has_slur', 'has_targeted_insult', 
            'has_threat_or_violence', 'profanity_hits', 'slur_hits', 
            'insult_hits', 'threat_or_violence_hits'
        ])

    span_df = pd.DataFrame(spans)
    
    span_df = span_df.astype({
        'span_id': 'int64',
        'span_text': 'object',
        'start_time': 'float64',
        'end_time': 'float64',
        'has_excessive_profanity': 'boolean',
        'has_slur': 'boolean',
        'has_targeted_insult': 'boolean',
        'has_threat_or_violence': 'boolean',
        'profanity_hits': 'object',
        'slur_hits': 'object',
        'insult_hits': 'object',
        'threat_or_violence_hits': 'object',
    })

    return span_df


def deduplicate_harmful_spans(spans_df: pd.DataFrame, time_tolerance: float = 0.5) -> pd.DataFrame:
    
    if spans_df.empty:
        return spans_df
    
    if 'final_enforced_label' not in spans_df.columns:
        return spans_df
    
    harmful = spans_df[spans_df['final_enforced_label'] != 'NONE'].copy()
    
    if harmful.empty:
        return spans_df
    
    harmful['_severity'] = harmful['final_enforced_label'].map(LABEL_ORDER).fillna(0)
    harmful = harmful.sort_values(['end_time', '_severity'], ascending=[True, False])
    
    keep_indices = []
    last_end_time = None
    
    for idx, row in harmful.iterrows():
        current_end = row['end_time']
        
        if last_end_time is None or abs(current_end - last_end_time) > time_tolerance:
            keep_indices.append(idx)
            last_end_time = current_end
        
    deduplicated_harmful = harmful.loc[keep_indices].drop(columns=['_severity'])
    
    non_harmful = spans_df[spans_df['final_enforced_label'] == 'NONE']
    result = pd.concat([non_harmful, deduplicated_harmful], ignore_index=False)
    result = result.sort_values('span_id').reset_index(drop=True)
    
    return result


def deduplicate_by_overlap(spans_df: pd.DataFrame, overlap_threshold: float = 0.8) -> pd.DataFrame:
    
    if spans_df.empty or 'final_enforced_label' not in spans_df.columns:
        return spans_df
    
    harmful = spans_df[spans_df['final_enforced_label'] != 'NONE'].copy()
    
    if harmful.empty:
        return spans_df
    
    harmful['_severity'] = harmful['final_enforced_label'].map(LABEL_ORDER).fillna(0)
    harmful = harmful.sort_values('_severity', ascending=False)
    
    keep_indices = []
    kept_intervals = []
    
    for idx, row in harmful.iterrows():
        start, end = row['start_time'], row['end_time']
        span_duration = end - start
        
        if span_duration <= 0:
            continue
        
        is_duplicate = False
        for kept_start, kept_end in kept_intervals:
            overlap_start = max(start, kept_start)
            overlap_end = min(end, kept_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            overlap_ratio = overlap_duration / span_duration
            
            if overlap_ratio >= overlap_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep_indices.append(idx)
            kept_intervals.append((start, end))
    
    deduplicated_harmful = harmful.loc[keep_indices].drop(columns=['_severity'])
    non_harmful = spans_df[spans_df['final_enforced_label'] == 'NONE']
    
    result = pd.concat([non_harmful, deduplicated_harmful], ignore_index=False)
    result = result.sort_values('span_id').reset_index(drop=True)
    
    return result