# 🎙️ AutoCenzurer

> **Intelligent audio content moderation & POLICY-ENFORCEMENT AI tool** — Automatically detect and mute harmful speech in audio and video files ASR (Automated Speech Recognition) & LLM/SLM

---

## 🌟 Meet AutoCenzurer

AutoCenzurer - content moderation tool that listens to your audio or video files, identifies harmful language (hate speech, profanity, threats, slurs) and mutes it. Think of it as a smart audio filter that keeps content clean and safe

## 🚨 Disclaimer!

The system prioritizes **recall** and policy enforcement over contextual nuance. The current model is a policy-enforcing and filtering tool with a limited context. Due to the problem being overly subjective and broad, lack of data and in sake of consistency - the following topics will be included as harmful: quoted speech is treated as harmful, artistic or informational intent is ignored, endorsement detection is conservative. As for now model will treat all of them equally harmful, which will be further investigated and tackled in the next iterations  
> **KEY GOALS / WHY** Reproducibility - Transperency - Deterministicism - Data Availability - Model Explainability

### ✨ Key Features

- 🎯 **ASR** — Uses WhisperX for accurate speech-to-text transcription with word-level timestamps
- 🤖 **AI-Powered Classification** — Combines deterministic pattern matching with LLM intelligence (Gemini, OpenAI, Groq)
- 📊 **Multi-Level Severity** — Classifies content into severity tiers:  General Hate Speech, Extremism, Harassment
- ⚡ **Async Processing** — Fast batch processing with concurrent API calls
- 🔇 **Smart Muting** — Merges nearby harmful segments and applies precise audio muting with padding
- 🔍 **Transparent Pipeline** — Full visibility into detection signals, LLM reasoning, and final decisions

---

## 🛠️ How It Works

AutoCenzurer follows a sophisticated 6-stage pipeline:

```
📹 Audio/Video Input
    ↓
🎵 Audio Extraction (.wav)
    ↓
🗣️ ASR (WhisperX) → Word-level timestamps
    ↓
📝 Span Building → Overlapping text chunks
    ↓
🔍 Deterministic Signals → Pattern detection (profanity, slurs, threats)
    ↓
🤖 LLM Classification → Contextual severity labeling
    ↓
⚖️ Label Enforcement → Minimum severity threshold
    ↓
🔇 Audio Muting → Merged intervals with padding
```

---

## 🚀 Quick Start

 ```bash
 # Edit requirements.txt to uncomment heavy dependencies if needed
 pip install -r requirements.txt
 ```

 Create a `.env` file in the project root:
 ```env
 GROQ_API_KEY=your_groq_api_key_here
 GOOGLE_API_KEY=your_google_api_key_here
 ```

> **Note:** Currently, you'll need to modify `main.py` with your file paths (see lines 11-16). A CLI interface is coming soon! 

---

## 🎯 Detection Categories

AutoCenzurer classifies content into **4 severity levels**: (Hoping GitHub forgives expicit language)

| Label | Description | Examples | Edge case (will be considered label)
|-------|-------------|----------|
| `NONE` | Clean content | Normal conversation | "just a fucking hard piece of shit job" |
| `HATE_SPEECH_GENERAL` | Profanity, slurs, targeted insults | "she is a total whore" | "you are such a dumbass for thinking so" |
| `EXTREMISM_PROMOTION` | Endorsement of mass harm | "nazi were totally right" | "communism is a great concept for humanity" |
| `HARASSMENT_OBSCENITY` | Explicit violence, threats | "all deserve to die" | "would kill for a snack" |

### Deterministic Signals

The tool uses pattern matching to detect: 
- **Excessive Profanity** (≥2 curse words or >15% profanity density)
- **Slurs** (racial, homophobic, profanity language)
- **Targeted Insults** (insults directed at people using pronouns)
- **Threats & Violence** (violent verbs near target pronouns)

---

## 🏗️ Project Structure

```
AutoCenzurer/
├── asr/                      # Speech recognition & audio processing
│   ├── speech_2_span.py      # WhisperX transcription
│   └── mute_audio.py          # Audio muting with intervals
├── llm_pipeline/             # LLM classification logic
│   ├── call_llm.py            # Sync/async LLM calls
│   ├── async_groq_call_llm.py # Concurrent Groq API handling
│   ├── prepare_promt.py       # Prompt engineering
│   └── prompt_llm.py          # Prompt templates
├── signals_deterministic/    # Pattern-based detection
│   ├── classify_signals.py    # Rule-based classifiers
│   ├── determine_span_signals.py
│   └── normalize_span.py      # Text normalization
├── text_processing/          # Span preprocessing & enforcement
│   ├── preprocessing_span.py  # Minimum label resolution
│   └── postprocess_enforcement.py  # LLM output validation
├── helpers/                  # Utility functions
│   ├── build_span.py          # Span construction with overlap
│   └── merge_intervals.py     # Interval merging algorithm
├── abstraction/              # Data schemas
│   ├── word_schema.py
│   └── span_schema.py
├── static/                   # Configuration & prompts
│   └── config.py              # Labels, thresholds, word lists
├── artifacts/                # Output storage (CSV, audio files)
├── main.py                   # Entry point
└── requirements.txt
```

---

## ⚙️ Config (Modify up to your needs)

```python
# Span building parameters
MAX_WORDS = 8              # Words per span
OVERLAP_WORDS = 2          # Overlap between spans
PAUSE_THRESHOLD = 0.3      # Max silence before span break (seconds)

# Audio muting parameters
PAD_BEFORE = 0.25          # Pre-padding (seconds)
PAD_AFTER = 0.4            # Post-padding (seconds)
MERGE_GAP = 0.3            # Max gap between merged intervals
```

---

## 🔬 Technical Details

### Speech Recognition
- **Model:** WhisperX (large-v3) with forced alignment
- **Output:** Word-level timestamps with precise start/end times
- **Languages:** Currently optimized for English

### LLM Classification
- **Supported APIs:** Groq (Llama 3.1), Google Gemini, OpenAI
- **Async Processing:** Concurrent API calls with semaphore rate limiting
- **Prompt Engineering:** Structured prompts with deterministic signal context

### Label Enforcement
- **Hybrid Approach:** LLM labels are enforced to be ≥ deterministic minimum
- **Safety First:** If uncertain, defaults to higher severity
- **Transparency:** Full reasoning and confidence scores preserved

---

## 🛣️ Roadmap

- [ ] SLM Local Solution
- [ ] Quantized & Distilled
- [ ] Static & Real-Time Audio Filtering
- [ ] Interface UI/UX

---

## 🙏 Have FUN!
