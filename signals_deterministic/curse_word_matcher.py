import re
from static.config import CURSE_VOCAB
from .normalize_span import normalize

def find_curse_words(span: str):
    results = []

    for match in re.finditer(r"\b\w[\w']*\b", span):
        orig = match.group(0)
        norm = normalize(orig)
        if not norm:
            continue

        for norm_tok in norm:
            if norm_tok in CURSE_VOCAB:
                results.append({
                    "token": orig,
                    "normalized": norm_tok,
                    "start_idx": match.start(),
                    "end_idx": match.end()
                })

    return results
