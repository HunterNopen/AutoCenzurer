from static.config import LABEL_ORDER
from abstraction.span_schema import SpanSchema

def max_label(llm_label: str, min_allowed_label: str) -> str:
    return max(llm_label, min_allowed_label, key=lambda l: LABEL_ORDER[l])

def resolve_min_label(
        has_excessive_profanity: bool,
        has_slur: bool,
        has_targeted_insult: bool,
        explicit_violence: bool = False,
        mass_harm_endorsement: bool = False
    ) -> str:

    min_label = "NONE"

    if has_targeted_insult or has_slur or (
        has_excessive_profanity
    ):
        min_label = "HATE_SPEECH_GENERAL"

    if mass_harm_endorsement:
        min_label = "EXTREMISM_PROMOTION"

    if explicit_violence:
        min_label = "HARASSMENT_OBSCENITY"

    return min_label

