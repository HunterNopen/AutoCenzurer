import os
import sys
import sys
import uuid
import tempfile
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import pandas as pd

from asr.speech_2_span import whisperx_to_word_df
from helpers.build_span import build_spans, deduplicate_by_overlap
from llm_pipeline.call_llm import batch_classify_async_llm
from helpers.merge_intervals import merge_intervals
from asr.mute_audio import mute_audio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    audio_id: str = ""
    input_path: str = ""
    output_path: str = ""
    
    # Step 1: ASR
    words_df: Optional[pd.DataFrame] = None
    asr_log: str = ""
    
    # Step 2: Span building
    spans_df: Optional[pd.DataFrame] = None
    span_log: str = ""
    
    # Step 3+4: Signals + LLM
    spans_llm_df: Optional[pd.DataFrame] = None
    llm_log: str = ""
    
    # Step 5: Filtering
    harmful_spans_df: Optional[pd.DataFrame] = None
    intervals: List[Tuple[float, float]] = field(default_factory=list)
    filter_log: str = ""
    
    # Step 6: Merging & Muting
    merged_intervals: List[Tuple[float, float]] = field(default_factory=list)
    mute_log: str = ""
    
    success: bool = False
    error: str = ""


def run_pipeline(
    audio_path: str,
    output_dir: str = "artifacts",
    audio_id: Optional[str] = None,
    device: str = "cuda",
    language: str = "en",
    pad_before: float = 0.5,
    pad_after: float = 0.8,
    save_intermediate: bool = True) -> PipelineResult:
    
    result = PipelineResult()
    
    if audio_id is None:
        audio_id = f"audio_{uuid.uuid4().hex[:8]}"
    
    result.audio_id = audio_id
    result.input_path = audio_path
    
    input_path = Path(audio_path)
    if not input_path.exists():
        result.error = f"Input file not found: {audio_path}"
        return result
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = (Path(output_dir) / f"{input_path.stem}_censored.wav").resolve()
    result.output_path = str(output_path)
    
    try:
        # ========== STEP 1: ASR ==========
        logger.info("STEP 1: Starting ASR...")
        result.asr_log = f"🎤 Running ASR on: {input_path.name}\n"
        result.asr_log += f"   Device: {device}, Language: {language}\n"
        
        words_df = whisperx_to_word_df(
            audio_path=audio_path,
            audio_id=audio_id,
            device=device,
            language=language
        )
        logger.info(f"STEP 1: ASR complete, got {len(words_df) if words_df is not None else 0} words")
        
        if words_df is None or words_df.empty:
            result.asr_log += "   ⚠️ No words detected in audio.\n"
            result.words_df = pd.DataFrame(columns=["audio_id", "word_id", "word", "start_time", "end_time"])
        else:
            result.words_df = words_df
            result.asr_log += f"   ✅ Detected {len(words_df)} words\n"
            
            if save_intermediate:
                words_csv = Path(output_dir) / f"{audio_id}_words.csv"
                words_df.to_csv(words_csv, index=False)
                result.asr_log += f"   💾 Saved to: {words_csv}\n"
        
        if result.words_df.empty:
            result.success = True
            result.mute_log = "No speech detected - copying original audio."
            # Just copy the original
            import shutil
            shutil.copy(audio_path, result.output_path)
            return result
        
        # ========== STEP 2: Build Spans ==========
        logger.info("STEP 2: Building spans...")
        result.span_log = "📝 Building text spans with overlap...\n"
        
        spans_df = build_spans(result.words_df)
        result.spans_df = spans_df
        logger.info(f"STEP 2: Created {len(spans_df)} spans")
        
        if spans_df.empty:
            result.span_log += "   ⚠️ No spans created.\n"
            result.success = True
            import shutil
            shutil.copy(audio_path, result.output_path)
            return result
        
        result.span_log += f"   ✅ Created {len(spans_df)} spans\n"
        
        if save_intermediate:
            spans_csv = Path(output_dir) / f"{audio_id}_spans.csv"
            spans_df.to_csv(spans_csv, index=False)
            result.span_log += f"   💾 Saved to: {spans_csv}\n"
        
        # ========== STEP 3+4: Deterministic Signals + LLM Classification ==========
        logger.info("STEP 3+4: Starting LLM classification...")
        result.llm_log = "🤖 Analyzing signals & classifying with LLM...\n"
        result.llm_log += f"   Processing {len(spans_df)} spans asynchronously...\n"
        
        spans_llm_df = batch_classify_async_llm(spans_df.copy())
        result.spans_llm_df = spans_llm_df
        logger.info("STEP 3+4: LLM classification complete")
        
        result.llm_log += "   ✅ Classification complete\n"
        
        # Log individual span results
        for idx, row in spans_llm_df.iterrows():
            label = row.get("final_enforced_label", "UNKNOWN")
            confidence = row.get("llm_confidence", "N/A")
            span_text = row.get("span_text", "")[:50]
            result.llm_log += f"   [{idx}] {label} (conf: {confidence}) - \"{span_text}...\"\n"
        
        if save_intermediate:
            llm_csv = Path(output_dir) / f"{audio_id}_spans_llm.csv"
            spans_llm_df.to_csv(llm_csv, index=False)
            result.llm_log += f"   💾 Saved to: {llm_csv}\n"
        
        # ========== STEP 5: Filter Harmful Spans ==========
        logger.info("STEP 5: Filtering harmful spans...")
        result.filter_log = "🔍 Filtering harmful spans...\n"
        
        # Count before deduplication
        all_harmful = spans_llm_df[spans_llm_df["final_enforced_label"] != "NONE"]
        result.filter_log += f"   Raw harmful spans: {len(all_harmful)}\n"
        
        # Deduplicate overlapping spans, keeping highest severity
        deduplicated_df = deduplicate_by_overlap(spans_llm_df, overlap_threshold=0.7)
        harmful_spans = deduplicated_df[
            deduplicated_df["final_enforced_label"] != "NONE"
        ]
        result.harmful_spans_df = harmful_spans
        logger.info(f"STEP 5: Found {len(harmful_spans)} unique harmful spans")
        
        if harmful_spans.empty:
            result.filter_log += "   ✅ No harmful content detected! Clean audio.\n"
            result.intervals = []
            result.success = True
            import shutil
            shutil.copy(audio_path, result.output_path)
            return result
        
        result.filter_log += f"   ⚠️ After deduplication: {len(harmful_spans)} unique harmful spans:\n"
        for idx, row in harmful_spans.iterrows():
            start = row["start_time"]
            end = row["end_time"]
            label = row["final_enforced_label"]
            result.filter_log += f"      [{start:.2f}s - {end:.2f}s] {label}\n"
        
        result.intervals = list(zip(harmful_spans["start_time"], harmful_spans["end_time"]))
        
        # ========== STEP 6: Merge Intervals & Mute ==========
        logger.info("STEP 6: Merging and muting...")
        result.mute_log = "🔇 Merging intervals and muting audio...\n"
        result.mute_log += f"   Padding: {pad_before}s before, {pad_after}s after\n"
        
        merged = merge_intervals(
            result.intervals,
            pad_before=pad_before,
            pad_after=pad_after
        )
        result.merged_intervals = merged
        
        result.mute_log += f"   📊 Merged {len(result.intervals)} intervals → {len(merged)} regions:\n"
        for start, end in merged:
            result.mute_log += f"      [{start:.2f}s - {end:.2f}s]\n"
        
        logger.info(f"STEP 6: Merged intervals: {merged}")
        logger.info(f"STEP 6: Calling mute_audio with input={audio_path}, output={result.output_path}")
        
        mute_audio(
            audio_path=audio_path,
            output_path=result.output_path,
            intervals=merged
        )
        
        logger.info("STEP 6: mute_audio completed successfully")
        result.mute_log += f"   ✅ Muted audio saved to: {result.output_path}\n"
        result.success = True
        
    except Exception as e:
        logger.error(f"Pipeline exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        result.error = f"Pipeline error: {str(e)}"
        result.success = False
        result.error += f"\n\n{traceback.format_exc()}"
    
    return result


def run_pipeline_step_by_step(
    audio_path: str,
    output_dir: str = "artifacts",
    audio_id: Optional[str] = None,
    device: str = "cuda",
    language: str = "en",
    pad_before: float = 0.5,
    pad_after: float = 0.8,
    save_intermediate: bool = True):
    
    result = PipelineResult()
    
    # Setup
    if audio_id is None:
        audio_id = f"audio_{uuid.uuid4().hex[:8]}"
    
    result.audio_id = audio_id
    result.input_path = audio_path
    
    input_path = Path(audio_path)
    if not input_path.exists():
        result.error = f"Input file not found: {audio_path}"
        yield result, "error"
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = (Path(output_dir) / f"{input_path.stem}_censored.wav").resolve()
    result.output_path = str(output_path)
    
    try:
        # STEP 1: ASR
        result.asr_log = f"🎤 Running ASR on: {input_path.name}\n"
        result.asr_log += f"   Device: {device}, Language: {language}\n"
        
        words_df = whisperx_to_word_df(
            audio_path=audio_path,
            audio_id=audio_id,
            device=device,
            language=language
        )
        
        if words_df is None or words_df.empty:
            result.asr_log += "   ⚠️ No words detected.\n"
            result.words_df = pd.DataFrame(columns=["audio_id", "word_id", "word", "start_time", "end_time"])
        else:
            result.words_df = words_df
            result.asr_log += f"   ✅ Detected {len(words_df)} words\n"
            if save_intermediate:
                words_csv = Path(output_dir) / f"{audio_id}_words.csv"
                words_df.to_csv(words_csv, index=False)
        
        yield result, "step1_asr"
        
        if result.words_df.empty:
            result.success = True
            import shutil
            shutil.copy(audio_path, result.output_path)
            yield result, "complete"
            return
        
        # STEP 2: Build Spans
        result.span_log = "📝 Building text spans...\n"
        spans_df = build_spans(result.words_df)
        result.spans_df = spans_df
        result.span_log += f"   ✅ Created {len(spans_df)} spans\n"
        
        if save_intermediate:
            spans_csv = Path(output_dir) / f"{audio_id}_spans.csv"
            spans_df.to_csv(spans_csv, index=False)
        
        yield result, "step2_spans"
        
        if spans_df.empty:
            result.success = True
            import shutil
            shutil.copy(audio_path, result.output_path)
            yield result, "complete"
            return
        
        # STEP 3+4: LLM Classification
        result.llm_log = "🤖 Classifying with LLM...\n"
        spans_llm_df = batch_classify_async_llm(spans_df.copy())
        result.spans_llm_df = spans_llm_df
        result.llm_log += f"   ✅ Classified {len(spans_llm_df)} spans\n"
        
        for idx, row in spans_llm_df.iterrows():
            label = row.get("final_enforced_label", "UNKNOWN")
            result.llm_log += f"   [{idx}] → {label}\n"
        
        if save_intermediate:
            llm_csv = Path(output_dir) / f"{audio_id}_spans_llm.csv"
            spans_llm_df.to_csv(llm_csv, index=False)
        
        yield result, "step3_llm"
        
        # STEP 5: Filter with deduplication
        result.filter_log = "🔍 Filtering harmful spans...\n"
        all_harmful = spans_llm_df[spans_llm_df["final_enforced_label"] != "NONE"]
        result.filter_log += f"   Raw harmful: {len(all_harmful)}\n"
        
        deduplicated_df = deduplicate_by_overlap(spans_llm_df, overlap_threshold=0.7)
        harmful_spans = deduplicated_df[deduplicated_df["final_enforced_label"] != "NONE"]
        result.harmful_spans_df = harmful_spans
        result.intervals = list(zip(harmful_spans["start_time"], harmful_spans["end_time"])) if not harmful_spans.empty else []
        result.filter_log += f"   Found {len(harmful_spans)} harmful spans\n"
        
        yield result, "step4_filter"
        
        # STEP 6: Mute
        result.mute_log = "🔇 Muting audio...\n"
        
        if not result.intervals:
            result.mute_log += "   No harmful content - keeping original.\n"
            import shutil
            shutil.copy(audio_path, result.output_path)
        else:
            merged = merge_intervals(result.intervals, pad_before=pad_before, pad_after=pad_after)
            result.merged_intervals = merged
            result.mute_log += f"   Merged to {len(merged)} regions\n"
            mute_audio(audio_path=audio_path, output_path=result.output_path, intervals=merged)
        
        result.mute_log += f"   ✅ Output: {result.output_path}\n"
        result.success = True
        
        yield result, "complete"
        
    except Exception as e:
        import traceback
        result.error = f"Error: {str(e)}\n{traceback.format_exc()}"
        result.success = False
        yield result, "error"


if __name__ == "__main__":
        
    if len(sys.argv) < 2:
        print("Usage: python pipeline_runner.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    result = run_pipeline(audio_file, device="cuda")
    
    print("\n" + "="*50)
    print("PIPELINE RESULT")
    print("="*50)
    print(result.asr_log)
    print(result.span_log)
    print(result.llm_log)
    print(result.filter_log)
    print(result.mute_log)
    
    if result.error:
        print(f"\n❌ Error: {result.error}")
    else:
        print(f"\n✅ Success! Output: {result.output_path}")
