from text_processing.preprocessing_span import preprocess_span
from ..llm_pipeline.prompt_llm import build_llm_prompt
from text_processing.postprocess_enforcement import validate_llm_output, enforce_final_label

def classify_span(span_row: dict, base_prompt: str, llm_client) -> dict:
    
    pre = preprocess_span(span_row)
    prompt = build_llm_prompt(pre, base_prompt)

    raw = llm_client(prompt)

    try:
        llm_out = validate_llm_output(raw)
    except ValueError:
        return {
            "final_enforced_label": pre["min_allowed_label"],
            "llm_label": None,
            "llm_confidence": "LOW",
            "llm_rationale": "Invalid LLM output; enforced deterministic minimum."
        }

    return enforce_final_label(llm_out, pre["min_allowed_label"])
