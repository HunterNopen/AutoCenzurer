from .classify_signals import has_targeted_insult, has_slur, has_excessive_profanity, has_threat_or_violence

def analyze_span(span_text: str) -> dict:
    prof_flag, prof_hits = has_excessive_profanity(span_text)
    slur_flag, slur_hits = has_slur(span_text)
    insult_flag, insult_hits = has_targeted_insult(span_text)
    harassment_flag, harassment_hits = has_threat_or_violence(span_text)

    return {
        "has_excessive_profanity": prof_flag,
        "has_slur": slur_flag,
        "has_targeted_insult": insult_flag,
        "has_threat_or_violence": harassment_flag,

        "profanity_hits": prof_hits,
        "slur_hits": slur_hits,
        "insult_hits": insult_hits,
        "threat_or_violence_hits": harassment_hits
    }