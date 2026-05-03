import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
from jiwer import wer, cer
import pandas as pd
import os

os.makedirs("results", exist_ok=True)

model_id  = "openai/whisper-small"
device    = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained(model_id, language="az", task="transcribe")
model     = WhisperForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
).to(device)
model.config.forced_decoder_ids = None

test_ds    = load_dataset("google/fleurs", "az_az", split="test")
test_small = test_ds.cast_column("audio", Audio(sampling_rate=16000))

references, hypotheses, results = [], [], []

for i, sample in enumerate(test_small):
    audio_array   = np.array(sample["audio"]["array"], dtype=np.float32)
    sampling_rate = sample["audio"]["sampling_rate"]
    ref_text      = sample["transcription"].strip().lower()

    inputs = processor(
        audio_array, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(inputs, language="az", task="transcribe")

    pred_text  = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip().lower()
    sample_wer = wer(ref_text, pred_text)
    sample_cer = cer(ref_text, pred_text)

    references.append(ref_text)
    hypotheses.append(pred_text)
    results.append({"id": i, "reference": ref_text, "hypothesis": pred_text,
                    "wer": sample_wer, "cer": sample_cer})

    if i % 50 == 0:
        print(f"[{i}] WER: {sample_wer:.3f}")

df      = pd.DataFrame(results)
avg_wer = wer(references, hypotheses)
avg_cer = cer(references, hypotheses)

print(f"Ortalama WER: {avg_wer*100:.2f}%")
print(f"Ortalama CER: {avg_cer*100:.2f}%")
print("\n-- En yaxsi 5 numune --")
print(df.nsmallest(5, "wer")[["id","reference","hypothesis","wer","cer"]].to_string())
print("\n-- En pis 5 numune --")
print(df.nlargest(5, "wer")[["id","reference","hypothesis","wer","cer"]].to_string())
df.to_csv("results/part_a_results.csv", index=False)
