import datasets

data = datasets.load_dataset('cardiffnlp/x_sensitive', split='train')
df = data.to_pandas()

df = df[df['conflictual'] == 1]

def extract_span(row):
    text = row['text']
    highlights = row['conflictual_highlight']
    
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
    
    # Now expand to 10 words
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

# Apply to each row
df['span'] = df.apply(extract_span, axis=1)

# Drop rows with empty spans
df = df[df['span'] != '']

# Add original index as column
df['original_index'] = df.index

# Create new df with spans
new_df = df[['original_index', 'span']].copy()

# Save to CSV with semicolon separator
new_df.to_csv('extracted_spans.csv', sep=';', index=False)