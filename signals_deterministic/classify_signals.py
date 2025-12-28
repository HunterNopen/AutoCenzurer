from .normalize_span import normalize

PROFANITY = {"fuck", "shit", "bitch", "asshole", "dick", "bastard"}
def has_excessive_profanity(span: str, k: int = 2, r: float = 0.15):
    tokens = normalize(span)
    hits = [t for t in tokens if t in PROFANITY]
    return len(hits) >= k or (len(hits) / max(len(tokens), 1)) >= r, hits


SLURS = {"nigger", "faggot", "retard", "kike", "chink"}
def has_slur(span: str):
    tokens = normalize(span)
    hits = [t for t in tokens if t in SLURS]
    return len(hits) > 0, hits

INSULTS = {"idiot", "dumbass", "moron", "stupid", "loser"}
TARGETS = {"you", "your", "he", "she", "they", "this", "that"}
def has_targeted_insult(span: str):
    tokens = normalize(span)
    hits = []

    for i, t in enumerate(tokens):
        if t in INSULTS:
            window = tokens[max(0, i-3):i]
            if any(w in TARGETS for w in window):
                hits.append(t)

    return len(hits) > 0, hits
