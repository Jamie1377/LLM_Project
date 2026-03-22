# LLM Project: Crypto News Modeling

This repository contains training and evaluation scripts for crypto-news direction modeling with text and numeric features.

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
