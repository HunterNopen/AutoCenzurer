import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import classification_report

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

LABEL2ID = {
    "NONE": 0,
    "UNHARMFUL_PROFANITY": 1,
    "CONFLICTUAL": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def build_prompt(example):
    return f"""
    You are a policy enforcement classifier.

    Text span:
    "{example['span']}"

    Choose exactly one label:
    CONFLICTUAL, NONE, UNHARMFUL_PROFANITY
    """

### ================== BitsAndBytes & LoRA Configs ==================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_proj", "v_proj"]
)

### =============================================================

if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    def preprocess(example):
        text = build_prompt(example)
        tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256
        )
        tokens["labels"] = LABEL2ID[example["label"]]
        return tokens

    def compute_metrics(eval_pred):

        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)

        report = classification_report(
            labels,
            preds,
            output_dict=True,
            zero_division=0
        )

        return {
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"]
        }

    ### ================== Dataset Prep ==================

    dataset = load_dataset(
        "./artifacts",
        data_files="x_sensitive_train_v2.csv"
    )

    dataset = dataset.map(preprocess)

    dataset = dataset.remove_columns(
        [col for col in dataset["train"].column_names if col not in ["input_ids", "attention_mask", "labels"]]
    )

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )


    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ### ================== TRAINING ==================

    training_args = TrainingArguments(
        output_dir="./artifacts/qwen-toxic-classifier",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["train"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()

    trainer.save_model("./artifact/qwen-toxic-classifier-mine")
    tokenizer.save_pretrained("./artifacts/qwen-toxic-classifier-mine")