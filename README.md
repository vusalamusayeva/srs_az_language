# Azərbaycan Dili üçün Nitq Tanıma Sistemi

## Layihənin qısa izahatı
Bu layihə Google FLEURS Azərbaycan dataseti üzərində OpenAI Whisper modelindən
istifadə edərək Azərbaycan dili üçün ASR pipeline qurur.
Part A baza inferensi, Part B isə fine-tuning pipeline-ını əhatə edir.

## İstifadə olunan model və parametrlər
- Model: openai/whisper-small
- Dataset: google/fleurs (az_az) — 923 test, 200 train, 50 validation
- Fine-tuning: 5 epoch, learning rate: 1e-5, batch size: 2

## WER/CER nəticələri

| Model                | WER     | CER     |
|----------------------|---------|---------|
| Baza (whisper-small) | 59.20%  | 15.31%  |
| Fine-tuned           | 53.50%  | 14.20%  |

## Kodu işə salmaq

```bash
pip install -r requirements.txt
```

### Part A — Baza inferens
```bash
cd part_a
python inference.py
```

### Part B — Fine-tuning
```bash
cd part_b
python finetune.py
```

## Nəticələr
- Fine-tuning ilə WER 59.2% → 53.5% endi (5.7% yaxşılaşma)
- Overfitting müşahidə edilmədi
- Qrafiklər: results/training_curves.png
