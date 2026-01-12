import pandas as pd

df = pd.read_csv('x_sensitive_train_v2.csv')

new_df = df[df['label'] == 'CONFLICTUAL'].copy()

new_df['label'] = 1
negative_size = len(new_df)
unharmful_sample = df[df['label'] == 'UNHARMFUL_PROFANITY'][:int(negative_size/2)]
unharmful_sample['label'] = 0
none_sample = df[df['label'] == "NONE"][:int(negative_size/2)]
none_sample['label'] = 0

new_df = pd.concat([new_df, unharmful_sample, none_sample])
new_df = new_df.rename(columns={"span": "span_text", "original_index": "id"})

new_df.to_csv('binary_classification.csv', index=False, sep=';')

