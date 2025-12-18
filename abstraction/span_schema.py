from typing import TypedDict

class SpanSchema(TypedDict):
   span_id: int
   span_text: str
   start_time: float
   end_time: float
   has_excessive_profanity: bool
   has_slur: bool
   has_targeted_insult: bool
   min_allowed_label: str
   llm_label: str
   llm_confidence: float
   llm_rationale: str
   final_enforced_label: str