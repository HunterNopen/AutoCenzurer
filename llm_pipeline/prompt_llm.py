from static.config import LLM_BASE_PROMPT_LABEL_SPAN
from abstraction.span_schema import SpanSchema

def build_llm_prompt(preprocessed_span: SpanSchema) -> str:
    return f"""
    {LLM_BASE_PROMPT_LABEL_SPAN}

    Span Input:
    {{
    "span_text": "{preprocessed_span["span_text"]}",
    "has_excessive_profanity": {str(preprocessed_span["has_excessive_profanity"]).lower()},
    "has_slur": {str(preprocessed_span["has_slur"]).lower()},
    "has_targeted_insult": {str(preprocessed_span["has_targeted_insult"]).lower()},
    "minimum_allowed_label": "{preprocessed_span["min_allowed_label"]}"
    }}
    """