# LLM Project: Crypto News Modeling

This repository contains training and evaluation scripts for crypto-news direction modeling with text and numeric features.

## Preliminary Results (Test Split)

| Model | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Numeric-only XGB | 0.7098 | 0.6638 | 0.5406 | 0.8597 |
| CLS+Numeric XGB (frozen `boltuix/bert-lite` CLS + numeric) | 0.7881 | 0.7378 | 0.6278 | 0.8947 |
| Pure CLS linear baseline | 0.5252 | 0.2860 | 0.2866 | 0.2854 |

Key deltas:

- Delta test F1 (`CLS+Numeric XGB - Numeric-only XGB`): `+0.0740`
- Test prediction diff count (`CLS+Numeric` vs `Numeric-only`): `1413`
- Test probability MAE (`CLS+Numeric` vs `Numeric-only`): `0.0902`

Source artifact:

- `outputs_compare_models/metrics_xgb_cls_vs_numeric.json`

## Highlights

- End-to-end BERT training utilities
- XGBoost baselines with numeric-only and CLS+numeric feature fusion
- Comparison pipeline for frozen or fine-tuned text embeddings
- Hugging Face publishing utilities for one-repo artifact release

## Main Code Files

- `bert.py`
- `xgb.py`
- `compare_xgb_text_vs_numeric.py`
- `pipeline_common.py`
- `neural_utils.py`
- `xgb_utils.py`
- `text_embeddings.py`

## Run Guide

See:

- `run_end_to_end_bert.md`

## Hugging Face Artifacts

Model/artifact card source used for Hub publishing:

- `hf_one_repo_release/README.md`

Target model repo:

- https://huggingface.co/JamieYuu/slm-bert-emb
