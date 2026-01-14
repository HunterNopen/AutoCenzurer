from evaluation.pipeline import Results, Batch
from local_llm.fine_tune import MODEL_NAME, bnb_config, ID2LABEL, LABEL2ID

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from dataclasses import dataclass

@dataclass
class LabelsConfig:
    num: int
    id2label: dict[int, str]
    label2id: dict[str, int]

MULTICLASS_LABEL_CONFIG = LabelsConfig(
    num=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

BINARY_LABEL_CONFIG = LabelsConfig(
    num=2,
    id2label={0: 'NONE', 1:'CONFLICTUAL'},
    label2id={'NONE': 0, 'CONFLICTUAL':1}
)


class QwenProcessor():
    def __init__(self, checkpoint_dir_path: str, tokenizer_path: str, labels_config: LabelsConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=labels_config.num,
            id2label=labels_config.id2label,
            label2id=labels_config.label2id,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True
        )

        self.model = PeftModel.from_pretrained(base_model, checkpoint_dir_path)
        self.model.eval()
    
    def build_prompt(self, span: str) -> str:
        return f"""
        You are a policy enforcement classifier.

        Text span:
        "{span}"

        Choose exactly one label:
        CONFLICTUAL, NONE, UNHARMFUL_PROFANITY
        """


    def classify(self, span: str):
        text = self.build_prompt(span)
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        pred_id = int(logits.argmax(dim=-1).item())
        
        return pred_id
    
    def process_batch(self, batch: Batch) -> Results:
        y_pred = []
        for span in batch.span_text:
            pred_id = self.classify(span)
            y_pred.append(pred_id)

        return Results(y_true=batch.label, y_pred=y_pred)


