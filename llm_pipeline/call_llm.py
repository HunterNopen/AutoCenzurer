import os
import google.generativeai as genai
from groq import Groq
import pandas as pd

from llm_pipeline.prepare_promt import finalize_classification, prepare_classification_prompt
from text_processing.preprocessing_span import resolve_min_label
from abstraction.span_schema import SpanSchema
from .prompt_llm import build_llm_prompt
from text_processing.postprocess_enforcement import validate_llm_output, enforce_final_label
from .async_groq_call_llm import async_groq_call, run_groq_batch_concurrently, run_semaphore_groq_call
from signals_deterministic.determine_span_signals import analyze_span

def call_llm_gemini(prompt: str) -> str:

    key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"
    
def call_llm_groq_llama(prompt: str) -> str:

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=512,
        stream=False,
        response_format={"type": "json_object"}
    )
    return completion.choices[0].message.content

def mock_call_llm(prompt: str) -> str:
    ### CALL LLM API WITH BUILT PROMPT ###

    raw_output = """
    {
      "label": "NONE",
      "confidence": "LOW",
      "rationale": "No explicit harmful language detected."
    }
    """

    return raw_output
    
def span_classify_llm(
    span_text: str,
    has_excessive_profanity: bool,
    has_slur: bool,
    has_targeted_insult: bool) -> dict:

    row = {
        "span_text": span_text,
        "has_excessive_profanity": has_excessive_profanity,
        "has_slur": has_slur,
        "has_targeted_insult": has_targeted_insult
    }
    prompt = prepare_classification_prompt(row)

    #llm_out = mock_call_llm(prompt)
    llm_out = call_llm_groq_llama(prompt['prompt'])

    return finalize_classification(llm_out, prompt["min_label"])

def batch_classify_async_llm(spans_df: pd.DataFrame) -> pd.DataFrame:

    batch_data = []
    for row_idx, row in spans_df.iterrows():
        signals = analyze_span(row["span_text"])
        for signal_key, signal_value in signals.items():
            spans_df.at[row_idx, signal_key] = signal_value
        row.update(signals)

        batch_data.append(prepare_classification_prompt(row))

    spans_df.to_csv("artifacts/spans.csv", index=False)
    
    prompts_only = [item["prompt"] for item in batch_data]
    
    raw_results = run_groq_batch_concurrently(prompts_only, max_concurrent=2)
    
    for idx, (row_idx, _) in enumerate(spans_df.iterrows()):
        context = batch_data[idx]
        raw_result = raw_results[idx]
        
        final_result = finalize_classification(raw_result, context["min_label"])
        
        for k, v in final_result.items():
            spans_df.at[row_idx, k] = v
            
    return spans_df

