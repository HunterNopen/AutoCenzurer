from typing import List, Tuple

def merge_intervals(intervals: List[Tuple[float, float]],
    pad_before: float = 0.5,
    pad_after: float = 0.8) -> List[Tuple[float, float]]:

    # Merged on raw timestamps only + MERGE_GAP (padding applied afterward, so could be still overlap in result)
    if not intervals:
        return []

    intervals = [(s, max(s, e)) for s, e in intervals]
    intervals.sort(key=lambda x: (x[0], x[1]))

    merged = []
    current_start, current_end = intervals[0]

    for next_start, next_end in intervals[1:]:
        if next_start <= current_end + MERGE_GAP:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    merged.append((current_start, current_end))

    return [
        (max(0, s - pad_before), e + pad_after)
        for s, e in merged
    ]

    # df_intervals = pd.DataFrame(intervals, columns=["start", "end"])
    
    # df_intervals['end'] = df_intervals.apply(lambda row: max(row['end'], row['start']), axis=1)

    # df_intervals.sort_values(by=['start', 'end'], ascending=True, inplace=True)

    # df_intervals['next_interval_start'] = df_intervals['start'].shift(1)
    # df_intervals['next_interval_end'] = df_intervals['end'].shift(1)
    # df_intervals['is_nested_interval'] = df_intervals['next_interval_start'] <= df_intervals['end'] + MERGE_GAP
    
    # for row in df_intervals.items():
    #     if row['is_nested_interval']:
    #         row['end'] = row['next_interval_end']

    #     row['start'] -= pad_before
    #     row['end'] -= pad_after   
