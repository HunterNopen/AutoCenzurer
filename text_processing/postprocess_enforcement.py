import json
import re

from static.config import LABEL_ORDER, BINARY_LABEL_TO_CLASS_VALUES
from .preprocessing_span import max_label

def validate_llm_output(raw_output: str) -> dict:
    try:
        cleaned = raw_output.strip()
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)

        cleaned = cleaned.strip()
        if cleaned.startswith("{{") and cleaned.endswith("}}"):
            cleaned = cleaned[1:-1]
        
        parsed = json.loads(cleaned)
    except Exception:
        raise ValueError("Invalid JSON")

    required_keys = {"label", "confidence", "rationale"}
    if set(parsed.keys()) != required_keys:
        raise ValueError("Invalid schema")

    if parsed["label"] not in LABEL_ORDER:
        raise ValueError("Invalid label")

    if parsed["confidence"] not in {"LOW", "MEDIUM", "HIGH"}:
        raise ValueError("Invalid confidence")

    if not isinstance(parsed["rationale"], str):
        raise ValueError("Invalid rationale")

    return parsed

def parse_llm_output(raw_output: str, label_to_value_map: dict[str, int]) -> dict:
    try:
        cleaned = raw_output.strip()
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)

        cleaned = cleaned.strip()
        if cleaned.startswith("{{") and cleaned.endswith("}}"):
            cleaned = cleaned[1:-1]
        
        parsed = json.loads(cleaned)
    except Exception:
        raise ValueError("Invalid JSON")

    required_keys = {"label", "confidence", "rationale"}
    if set(parsed.keys()) != required_keys:
        raise ValueError("Invalid schema")

    if parsed["label"] not in label_to_value_map:
        raise ValueError("Invalid label")

    if parsed["confidence"] not in {"LOW", "MEDIUM", "HIGH"}:
        raise ValueError("Invalid confidence")

    if not isinstance(parsed["rationale"], str):
        raise ValueError("Invalid rationale")

    return parsed

def enforce_final_label(
    llm_output: dict,
    min_allowed_label: str) -> dict:
    
    final_label = max_label(llm_output["label"], min_allowed_label)

    overridden = final_label != llm_output["label"]

    return {
        "final_enforced_label": final_label,
        "llm_label": llm_output["label"],
        "llm_confidence": llm_output["confidence"] if not overridden else "LOW",
        "llm_rationale": (
            llm_output["rationale"]
            if not overridden
            else llm_output["rationale"] + " | Overridden by deterministic minimum."
        )
    }