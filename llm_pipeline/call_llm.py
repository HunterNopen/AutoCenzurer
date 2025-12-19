import os
import google.generativeai as genai
from groq import Groq

from text_processing.preprocessing_span import resolve_min_label
from abstraction.span_schema import SpanSchema
from .prompt_llm import build_llm_prompt
from text_processing.postprocess_enforcement import validate_llm_output, enforce_final_label

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
    
def classify_span_with_llm(
    span_text: str,
    has_excessive_profanity: bool,
    has_slur: bool,
    has_targeted_insult: bool) -> dict:

    ### MIN_LABEL as in severity ###
    min_label = resolve_min_label(
        has_excessive_profanity,
        has_slur,
        has_targeted_insult,
        span_text
    )

    ### Build prompt with SpanSchema ###
    preprocessed_span: SpanSchema = {
        'span_text': span_text,
        'has_excessive_profanity': has_excessive_profanity,
        'has_slur': has_slur,
        'has_targeted_insult': has_targeted_insult,
        'min_allowed_label': min_label
    }
    
    prompt = build_llm_prompt(preprocessed_span)
    #llm_out = mock_call_llm(prompt)
    llm_out = call_llm_groq_llama(prompt)

    llm_out_json = validate_llm_output(llm_out)

    final_label = enforce_final_label(
        llm_out_json,
        min_label
    )

    return final_label

