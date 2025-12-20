from static.config import LABEL_ORDER
from abstraction.span_schema import SpanSchema

def max_label(llm_label: str, min_allowed_label: str) -> str:
    return max(llm_label, min_allowed_label, key=lambda l: LABEL_ORDER[l])

def resolve_min_label(
    has_excessive_profanity: bool,
    has_slur: bool,
    has_targeted_insult: bool,
    span_text: str) -> str:

    violence_keywords = ["kill", "shoot", "burn", "stab", "bomb"] ### MOCKED STUPID

    if any(k in span_text.lower() for k in violence_keywords):
        return "HARASSMENT_OBSCENITY"

    if has_slur or has_targeted_insult:
        return "HATE_SPEECH_GENERAL"

    return "NONE"

