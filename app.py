import gradio as gr
import pandas as pd
import tempfile
import os
import sys
import traceback
import logging
from pathlib import Path

from pipeline_runner import run_pipeline, PipelineResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


CUSTOM_CSS = """
.step-header {
    font-size: 1.2em;
    font-weight: bold;
    margin-bottom: 10px;
    padding: 10px;
    border-radius: 5px;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    padding: 10px;
    border-radius: 5px;
    color: #155724;
}
.error-box {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    padding: 10px;
    border-radius: 5px;
    color: #721c24;
}
.info-box {
    background-color: #e7f3ff;
    border: 1px solid #b6d4fe;
    padding: 10px;
    border-radius: 5px;
    color: #084298;
}
"""


def format_dataframe_for_display(df: pd.DataFrame, max_cols: list = None) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame()
    
    df_display = df.copy()
    
    for col in df_display.columns:
        if df_display[col].dtype == 'object':
            df_display[col] = df_display[col].astype(str).str[:100]
    
    for col in df_display.select_dtypes(include=['float64', 'float32']).columns:
        df_display[col] = df_display[col].round(3)
    
    if max_cols:
        available_cols = [c for c in max_cols if c in df_display.columns]
        df_display = df_display[available_cols]
    
    return df_display


def process_audio(audio_file, device: str, language: str, pad_before: float, pad_after: float, save_intermediate: bool):
    
    logger.info("=" * 50)
    logger.info("STARTING AUDIO PROCESSING")
    logger.info("=" * 50)
    
    if audio_file is None:
        logger.warning("No audio file provided")
        return (
            "❌ Please upload an audio file.",
            None, "", None, "", None, "", None, "", None, None, ""
        )
    
    try:
        audio_path = audio_file
        logger.info(f"Audio file: {audio_path}")
        logger.info(f"Device: {device}, Language: {language}")
        
        output_dir = Path("artifacts/gradio_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Calling run_pipeline...")
        
        result = run_pipeline(
            audio_path=audio_path,
        output_dir=str(output_dir),
        device=device,
        language=language,
        pad_before=pad_before,
        pad_after=pad_after,
        save_intermediate=save_intermediate
    )
    
        logger.info("Pipeline completed successfully")
        
        if result.error:
            status = f"❌ Pipeline failed!\n\n{result.error}"
            logger.error(f"Pipeline error: {result.error}")
        elif result.success:
            status = f"✅ Pipeline completed successfully!\n\nAudio ID: {result.audio_id}\nOutput: {result.output_path}"
            logger.info(f"Success! Output: {result.output_path}")
        else:
            status = "⚠️ Pipeline finished with warnings."
            logger.warning("Pipeline finished with warnings")
        
        # Step 1: ASR - Words DataFrame
        words_df_display = format_dataframe_for_display(
            result.words_df,
            max_cols=["word_id", "word", "start_time", "end_time"]
        )
        
        # Step 2: Spans DataFrame
        spans_df_display = format_dataframe_for_display(
            result.spans_df,
            max_cols=["span_id", "span_text", "start_time", "end_time"]
        )
        
        # Step 3: LLM Results DataFrame
        llm_cols = [
            "span_id", "span_text", "has_excessive_profanity", "has_slur", 
            "has_targeted_insult", "has_threat_or_violence", "min_allowed_label",
            "llm_label", "llm_confidence", "final_enforced_label"
        ]
        spans_llm_display = format_dataframe_for_display(
            result.spans_llm_df,
            max_cols=llm_cols
        )
    
        # Step 4: Harmful spans
        harmful_display = format_dataframe_for_display(
            result.harmful_spans_df,
            max_cols=["span_id", "span_text", "start_time", "end_time", "final_enforced_label"]
        )
        
        if result.merged_intervals:
            intervals_text = "🔇 Muted Regions:\n\n"
            for i, (start, end) in enumerate(result.merged_intervals, 1):
                intervals_text += f"  Region {i}: {start:.2f}s → {end:.2f}s (duration: {end-start:.2f}s)\n"
        else:
            intervals_text = "✅ No regions muted - audio is clean!"
        
        output_audio = None
        if result.success and result.output_path:
            output_path = Path(result.output_path).resolve()
            if output_path.exists():
                output_audio = str(output_path)
                logger.info(f"Output audio path: {output_audio}")
            else:
                logger.warning(f"Output file not found: {output_path}")
        
        full_log = f"""
{'='*60}
AUTOCENZURER PIPELINE LOG
{'='*60}

{result.asr_log}

{result.span_log}

{result.llm_log}

{result.filter_log}

{result.mute_log}

{'='*60}
"""
        
        return (
            status,
            words_df_display if not words_df_display.empty else None,
            result.asr_log,
            spans_df_display if not spans_df_display.empty else None,
            result.span_log,
            spans_llm_display if not spans_llm_display.empty else None,
            result.llm_log,
            harmful_display if harmful_display is not None and not harmful_display.empty else None,
            result.filter_log,
            intervals_text,
            output_audio,
            full_log
        )
    
    except Exception as e:
        error_msg = f"❌ CRASH ERROR:\n\n{str(e)}\n\n{traceback.format_exc()}"
        logger.error(f"CRASH: {e}")
        logger.error(traceback.format_exc())
        return (
            error_msg,
            None, f"Crashed: {e}", None, "", None, "", None, "", "Crashed before muting", None, error_msg
        )


def create_interface():
    
    with gr.Blocks(
        title="🎙️ AutoCenzurer",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
        # 🎙️ AutoCenzurer - Automatic Audio Content Moderation
        
        Upload an audio file (.ogg, .wav, .mp3) to automatically detect and mute harmful content.
        The pipeline shows you each processing step in detail.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")
                
                audio_input = gr.Audio(
                    label="📁 Upload Audio File",
                    type="filepath",
                    sources=["upload"]
                )
                
                with gr.Accordion("Advanced Options", open=False):
                    device = gr.Radio(
                        choices=["cuda", "cpu"],
                        value="cuda",
                        label="🖥️ Device",
                        info="Use CUDA for GPU acceleration"
                    )
                    
                    language = gr.Dropdown(
                        choices=["en", "ru", "de", "fr", "es", "it", "pt", "nl", "pl", "uk"],
                        value="en",
                        label="🌐 Language"
                    )
                    
                    pad_before = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.5,
                        step=0.1,
                        label="⏪ Padding Before (seconds)"
                    )
                    
                    pad_after = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        label="⏩ Padding After (seconds)"
                    )
                    
                    save_intermediate = gr.Checkbox(
                        value=True,
                        label="💾 Save Intermediate CSVs"
                    )
                
                process_btn = gr.Button(
                    "🚀 Process Audio",
                    variant="primary",
                    size="lg"
                )
                
                status_output = gr.Textbox(
                    label="📊 Status",
                    lines=4,
                    interactive=False
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 🎵 Output")
                
                output_audio = gr.Audio(
                    label="🔇 Processed Audio (Download/Play)",
                    type="filepath",
                    interactive=False
                )
                
                intervals_output = gr.Textbox(
                    label="📍 Muted Intervals",
                    lines=5,
                    interactive=False
                )
        
        gr.Markdown("---")
        gr.Markdown("## 📋 Pipeline Steps")
        
        with gr.Tabs():
            with gr.TabItem("1️⃣ ASR (Speech Recognition)"):
                asr_log = gr.Textbox(
                    label="Log",
                    lines=3,
                    interactive=False
                )
                words_table = gr.Dataframe(
                    label="Detected Words",
                    interactive=False,
                    wrap=True
                )
            
            with gr.TabItem("2️⃣ Span Building"):
                span_log = gr.Textbox(
                    label="Log",
                    lines=3,
                    interactive=False
                )
                spans_table = gr.Dataframe(
                    label="Text Spans",
                    interactive=False,
                    wrap=True
                )
            
            with gr.TabItem("3️⃣ LLM Classification"):
                llm_log = gr.Textbox(
                    label="Log",
                    lines=6,
                    interactive=False
                )
                llm_table = gr.Dataframe(
                    label="Classification Results",
                    interactive=False,
                    wrap=True
                )
            
            with gr.TabItem("4️⃣ Harmful Content"):
                filter_log = gr.Textbox(
                    label="Log",
                    lines=3,
                    interactive=False
                )
                harmful_table = gr.Dataframe(
                    label="Harmful Spans",
                    interactive=False,
                    wrap=True
                )
            
            with gr.TabItem("📜 Full Log"):
                full_log = gr.Textbox(
                    label="Complete Pipeline Log",
                    lines=25,
                    interactive=False
                )
        
        process_btn.click(
            fn=process_audio,
            inputs=[
                audio_input,
                device,
                language,
                pad_before,
                pad_after,
                save_intermediate
            ],
            outputs=[
                status_output,
                words_table,
                asr_log,
                spans_table,
                span_log,
                llm_table,
                llm_log,
                harmful_table,
                filter_log,
                intervals_output,
                output_audio,
                full_log
            ]
        )
        
        gr.Markdown("""
        ---
        ### ℹ️ About
        
        **AutoCenzurer** uses a 6-stage pipeline:
        1. **ASR** - WhisperX transcribes audio to word-level timestamps
        2. **Span Building** - Groups words into overlapping text chunks
        3. **Deterministic Signals** - Detects profanity, slurs, threats via pattern matching
        4. **LLM Classification** - AI classifies severity (NONE, HATE_SPEECH, EXTREMISM, HARASSMENT)
        5. **Filtering** - Identifies content that needs muting
        6. **Muting** - Applies silence to harmful regions with padding
        
        Supported formats: `.ogg`, `.wav`, `.mp3`, `.mp4`, `.mkv`, `.mov`, `.avi`
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.queue()
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        show_error=True
    )
