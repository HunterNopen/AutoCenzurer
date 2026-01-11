import datasets

def extract_spans(df, highlight_field):
    def extract_span(row):
        text = row['text']
        highlights = row[highlight_field]
        
        if len(highlights) == 0:
            return '' 

        words = text.split()
        
        max_len = 0
        selected_highlight = ''
        for hl in highlights:
            hl_text = hl[0]  # since each is [str]
            hl_words = hl_text.split()
            if len(hl_words) > max_len and len(hl_words) <= 10:
                max_len = len(hl_words)
                selected_highlight = hl_text
        
        if not selected_highlight:
            return ''
        
        hl_words = selected_highlight.split()
        
        start_idx = -1
        for i in range(len(words) - len(hl_words) + 1):
            if words[i:i+len(hl_words)] == hl_words:
                start_idx = i
                break
        
        if start_idx == -1:
            return selected_highlight
        
        current_span = hl_words[:]
        left = start_idx - 1
        right = start_idx + len(hl_words)
        
        while len(current_span) < 10:
            added = False
            if left >= 0:
                current_span.insert(0, words[left])
                left -= 1
                added = True
            if len(current_span) < 10 and right < len(words):
                current_span.append(words[right])
                right += 1
                added = True
            if not added:
                break
        
        return ' '.join(current_span)
    df = df.copy()
    df['span'] = df.apply(extract_span, axis=1)
    df = df[df['span'] != '']
    df['original_index'] = df.index
    new_df = df[['original_index', 'span']].copy()
    
    return new_df

data = datasets.load_dataset('cardiffnlp/x_sensitive', split='train')
df = data.to_pandas()

df = df[(df['profanity'] == 1) & (df['conflictual'] != 1) & (df['selfharm'] != 1)]

new_df = extract_spans(df, 'profanity_highlight')
new_df.to_csv('unharmful_profanity.csv', sep=';', index=False)