LABEL_ORDER = {
    "NONE": 0,
    "HATE_SPEECH_GENERAL": 1,
    "EXTREMISM_PROMOTION": 2,
    "HARASSMENT_OBSCENITY": 3
}

MERGE_GAP = 0.3
MAX_WORDS = 8 #20
OVERLAP_WORDS = 2 #4
PAUSE_THRESHOLD = 0.6 #0.7
PAD_BEFORE = 0.5
PAD_AFTER = 0.8

LLM_BASE_PROMPT_LABEL_SPAN ="""
Role:
You are an Executive Judge responsible for classifying text spans for harmful content under a fixed policy.

Scope:
Each span is independent. No external context, history, intent, framing, or hypothetical interpretation is allowed.

Task:
Given the Span Input, assign exactly one label that reflects the highest severity of harmful content present in the text.

Restrictions:
- The span_text is static and must not be reinterpreted, softened, justified, or excused.
- Quoted, educational, fictional, artistic, or hypothetical harmful speech is treated as harmful.
- Do not invent assumptions beyond the span_text and provided signals.
- Do not output multiple labels.
- Do not deviate from the specified output format.

Labels (ordered by severity):
NONE < HATE_SPEECH_GENERAL < EXTREMISM_PROMOTION < HARASSMENT_OBSCENITY

Output Format:
MUST return ONLY valid JSON WITHOUT any artifacts with exact the following structure:

{
  "label": "NONE | HATE_SPEECH_GENERAL | EXTREMISM_PROMOTION | HARASSMENT_OBSCENITY",
  "confidence": "LOW | MEDIUM | HIGH",
  "rationale": "Concise empirical justification citing explicit phrases or patterns from span_text"
}
"""

pipeline = """
Audio / Video
   -
Audio extraction (.wav)
   -
ASR (Whisper)
   -
words_df
   -
build_spans()
   -
span_df
   -
deterministic signals
   -
min_allowed_label
   -
LLM classification
   -
final_enforced_label
   -
filter label != NONE
   -
extract intervals
   -
merge_intervals()
   -
mute_audio()
"""