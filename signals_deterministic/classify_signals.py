from static.config import PROFANITY, SLURS, INSULTS, TARGETS, VIOLENCE_VERBS, THREAT_VERBS, VIOLENT_OUTCOMES
from .normalize_span import normalize

def has_excessive_profanity(span: str, k: int = 2, r: float = 0.15):
    tokens = normalize(span)
    hits = [t for t in tokens if t in PROFANITY]
    return len(hits) >= k or (len(hits) / max(len(tokens), 1)) >= r, hits


def has_slur(span: str):
    tokens = normalize(span)
    hits = [t for t in tokens if t in SLURS]
    return len(hits) > 0, hits

def has_targeted_insult(span: str):
    tokens = normalize(span)
    hits = []

    for i, t in enumerate(tokens):
        if t in INSULTS:
            window = tokens[max(0, i-3):i]
            if any(w in TARGETS for w in window):
                hits.append(t)

    return len(hits) > 0, hits

def has_threat_or_violence(span: str, window_size: int = 4):

    tokens = normalize(span)
    hits = []
    
    for i, token in enumerate(tokens):
        if token in VIOLENCE_VERBS:
            window_before = tokens[max(0, i-window_size):i]
            window_after = tokens[i+1:min(len(tokens), i+window_size+1)]
            window = window_before + window_after
            
            if any(w in TARGETS for w in window):
                hits.append(f"targeted_{token}")
            else:
                hits.append(token)
        
        if token in THREAT_VERBS:
            window_after = tokens[i+1:min(len(tokens), i+window_size+1)]
            if any(w in VIOLENCE_VERBS for w in window_after):
                hits.append(f"threat_{token}")
        
        if token in {"hope", "wish", "want"}:
            window_after = tokens[i+1:min(len(tokens), i+window_size+1)]
            has_target = any(w in TARGETS for w in window_after)
            has_violence = any(w in VIOLENT_OUTCOMES | VIOLENCE_VERBS for w in window_after)
            
            if has_target and has_violence:
                hits.append(f"violent_wish_{token}")
    
    return len(hits) > 0, hits
