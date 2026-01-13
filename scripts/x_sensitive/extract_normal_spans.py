import datasets
import re

def cut_span(record):
    text = record['text']
    # Remove leading "@user" prefixes (one or more)
    text = re.sub(r'^(@user\s*)+', '', text)
    words = text.split()[:10]
    return ' '.join(words)

data = datasets.load_dataset('cardiffnlp/x_sensitive', split='train')
df = data.to_pandas()

df = df[df['labels'].apply(len) == 0]

df['span'] = df.apply(cut_span, axis=1)
df['original_index'] = df.index
new_df = df[['original_index', 'span']].copy()

new_df.to_csv('none.csv', sep=';', index=False)