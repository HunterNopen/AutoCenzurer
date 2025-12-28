import re

def normalize(text: str) -> list[str]:
    REPEAT_PATTERN = re.compile(r'(.)\1{2,}') # removes >2 repeated characters

    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    tokens = text.split()

    norm_tokens = []
    for t in tokens:
        t = REPEAT_PATTERN.sub(r'\1\1', t)
        t = t.strip("'")

        if t:
            norm_tokens.append(t)

    return norm_tokens
