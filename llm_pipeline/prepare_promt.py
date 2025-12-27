from abstraction.span_schema import SpanSchema
from llm_pipeline.prompt_llm import build_llm_prompt
from text_processing.postprocess_enforcement import enforce_final_label, validate_llm_output
from text_processing.preprocessing_span import resolve_min_label

def prepare_classification_prompt(row) -> dict:
    span_text = row["span_text"]
    
    min_label = resolve_min_label(
        row["has_excessive_profanity"],
        row["has_slur"],
        row["has_targeted_insult"],
        span_text
    )

    preprocessed_span: SpanSchema = {
        'span_text': span_text,
        'has_excessive_profanity': row["has_excessive_profanity"],
        'has_slur': row["has_slur"],
        'has_targeted_insult': row["has_targeted_insult"],
        'min_allowed_label': min_label
    }
    
    prompt = build_llm_prompt(preprocessed_span)
    
    return {
        "prompt": prompt,
        "min_label": min_label
    }

def finalize_classification(llm_raw_output: str, min_label: str) -> dict:
    llm_out_json = validate_llm_output(llm_raw_output)
    
    final_label = enforce_final_label(
        llm_out_json,
        min_label
    )
    return final_label