import torch
import numpy as np
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                          WhisperFeatureExtractor, WhisperTokenizer,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List
from jiwer import wer, cer
import evaluate
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

model_id = "openai/whisper-small"
device   = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
train_ds   = load_dataset("google/fleurs", "az_az", split="train")
val_ds     = load_dataset("google/fleurs", "az_az", split="validation")
test_ds    = load_dataset("google/fleurs", "az_az", split="test")

train_small = train_ds.select(range(200)).cast_column("audio", Audio(sampling_rate=16000))
val_small   = val_ds.select(range(50)).cast_column("audio", Audio(sampling_rate=16000))
test_small  = test_ds.cast_column("audio", Audio(sampling_rate=16000))

# Model
processor = WhisperProcessor.from_pretrained(model_id, language="az", task="transcribe")
model     = WhisperForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
).to(device)
model.config.forced_decoder_ids = None

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
tokenizer         = WhisperTokenizer.from_pretrained(model_id, language="az", task="transcribe")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features[0]
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

train_prepared = train_small.map(prepare_dataset, remove_columns=train_small.column_names)
val_prepared   = val_small.map(prepare_dataset,   remove_columns=val_small.column_names)
test_prepared  = test_small.map(prepare_dataset,  remove_columns=test_small.column_names)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict]) -> Dict:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    result    = metric.compute(predictions=pred_str, references=label_str)
    wer_score = result if isinstance(result, float) else result["wer"]
    return {"wer": round(wer_score * 100, 2)}

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-az-finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=50,
    num_train_epochs=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    fp16=False,
    bf16=False,
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=10,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_prepared,
    eval_dataset=val_prepared,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

trainer.train()
print("Fine-tuning tamamlandı!")
