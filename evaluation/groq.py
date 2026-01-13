from abstraction.span_schema import SpanSchema
from signals_deterministic.determine_span_signals import analyze_span
from llm_pipeline.prompt_llm import build_flexible_llm_prompt
from llm_pipeline.async_groq_call_llm import run_groq_batch_concurrently
from evaluation.pipeline import Results, Batch
from text_processing.postprocess_enforcement import validate_llm_output_for_binary
from static.config import BINARY_LABEL_TO_CLASS_VALUES
from pandas import DataFrame
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

def build_prompts_from_batch(batch: list[str], system_prompt: str) -> DataFrame:
    prompts = []
    for span in batch:
        deterministic_signals = analyze_span(span)
        schema: SpanSchema = {
            'span_text': span,
            'has_excessive_profanity': deterministic_signals['has_excessive_profanity'], 
            'has_slur': deterministic_signals['has_slur'],
            'has_targeted_insult': deterministic_signals['has_targeted_insult']
        }
        
        prompt = build_flexible_llm_prompt(schema, system_prompt)
        prompts.append(prompt)
    
    return prompts

@dataclass
class RecordMetadata:
    id: str
    span_text: str
    raw_output: str
    pred_label: int
    true_label: int
    confidence: str
    rationale: str
    

class GroqBinaryProcessor():
    def __init__(self, system_prompt: str, max_concurrent: int = 2):
        self.system_prompt = system_prompt
        self.max_concurrent = max_concurrent
        self.metadata = []

    def process_batch(self, batch: Batch) -> Results:
        prompts = build_prompts_from_batch(batch.span_text, system_prompt=self.system_prompt)

        raw_results = run_groq_batch_concurrently(prompts, max_concurrent=self.max_concurrent)

        y_pred = []
        y_true = []
        for i, result in enumerate(raw_results):
            try:
                parsed = validate_llm_output_for_binary(result)
            except:
                logger.exception(f"validate output error for id:{batch.id[i]}")
                self.metadata.append(RecordMetadata(
                    id=batch.id[i],
                    span_text=batch.span_text[i],
                    raw_output=result,
                    pred_label=-1,
                    true_label=batch.label[i],
                    confidence="-",
                    rationale="-"
                ))
                
                continue
            
            label_value = BINARY_LABEL_TO_CLASS_VALUES[parsed['label']]
            y_pred.append(label_value)
            y_true.append(batch.label[i])
            
            self.metadata.append(RecordMetadata(
                id=batch.id[i],
                span_text=batch.span_text[i],
                raw_output=result,
                pred_label=label_value,
                true_label=batch.label[i],
                confidence=parsed['confidence'],
                rationale=parsed['rationale']
            ))

        return Results(y_true=y_true, y_pred=y_pred)
    
    def export_metadata(self, file_path: str) -> None:
        df = DataFrame([asdict(r) for r in self.metadata])
        df.to_csv(file_path, sep=';', index=False)