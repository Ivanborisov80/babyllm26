# BabyLM -- Teacher-Guided Masked Pretraining

This repo contains code for a teacher-guided extension of the adaptive masking
scheme from [Edman and Fraser (2025)](https://arxiv.org/abs/2510.20475)
(implementation: [Leukas/babylm25](https://github.com/Leukas/babylm25)),
submitted as part of a BabyLM course project. 

## Teacher scoring

```
python teacher_scoring.py \
    --train_data data/bb24.train \
    --tokenizer tokenizers/bb24.model \
    --output_path teacher_score.pt \
    --teacher_model Qwen/Qwen2.5-0.5B \
    --fraction 0.3 \
    --tokens_per_example 3 \
    --batch_size 48
```

## Training

To train the teacher-guided model:

```
python train_mask.py --train_data data/bb24.train --valid_data data/bb25_small.dev \
    --tokenizer tokenizers/bb24.model --teacher_score teacher_score.pt \
    --output_path output/ --hidden_size 384 --intermediate_size 1280 \
    --weight_decay 0.01 --mlm_prob 0.4 --mask_decay 0.25 --lr 0.007 \
    --lamb --all_checkpoints --epochs 10
```

Omit `--teacher_score` to reproduce the hard-decay baseline this method
extends.

## Evaluation

Checkpoints are evaluated with the official
[BabyLM 2026 evaluation pipeline](https://github.com/babylm-org/babylm-eval)
(zero-shot BLiMP, BLiMP supplement, entity tracking, COMPS; GLUE fine-tuning on
BoolQ, MNLI, MRPC, MultiRC, QQP, RTE, WSC).
