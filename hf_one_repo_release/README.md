---
language: en
tags:
- finance
- quant-finance
- algo-trading
- trading
- crypto
- sentiment-analysis
- feature-engineering
- alpha-research
- machine-learning
- xgboost
- bert
- finbert
- embeddings
- preliminary-results
datasets:
- SahandNZ/cryptonews-articles-with-price-momentum-labels
base_model:
- boltuix/bert-lite
license: mit
---

# Crypto News Alpha Features: BERT-Lite Embeddings + Benchmark Results

Precomputed crypto-news CLS embeddings and benchmark outputs for fast quant experimentation.

If you want to test whether text alpha adds signal over market/FnG-style numeric features, this repo gives you ready-to-use artifacts without rerunning expensive encoding.

Primary text encoder used in this release: `boltuix/bert-lite`.

## Data Source

This work uses the public Hugging Face dataset:

- https://huggingface.co/datasets/SahandNZ/cryptonews-articles-with-price-momentum-labels

Please cite and credit the original dataset creator (`SahandNZ`) when reusing these artifacts.

## Three Benchmark Tracks

1. Numeric-only XGBoost baseline
2. Frozen CLS embedding + numeric XGBoost
3. Pure CLS linear baseline

## Why Download This

- Start modeling immediately with ready-made `.npy` embedding tensors.
- Reproduce a strong baseline quickly for crypto text+tabular fusion.
- Compare pure numeric alpha vs text-augmented alpha with a clean metric summary.
- Use as a drop-in feature pack for your own classifier/regressor experiments.

## What Is Inside

### 1) Results artifacts
Stored under `results/`:

- `results/metrics_xgb_cls_vs_numeric.json`
- `results/results_summary.csv`

### 2) Embedding artifacts
Stored under `embeddings/`:

- `embeddings/bertlite_full_fresh__train_cls_embeddings.npy`
- `embeddings/bertlite_full_fresh__val_cls_embeddings.npy`
- `embeddings/bertlite_full_fresh__test_cls_embeddings.npy`
- `embeddings/finbert_full_fresh__train_cls_embeddings.npy`
- `embeddings/finbert_full_fresh__val_cls_embeddings.npy`
- `embeddings/finbert_full_fresh__test_cls_embeddings.npy`

### 3) Integrity manifest
Stored under `embeddings/`:

- `embeddings/embeddings_manifest.csv`

## Quick Start (Embedding Usage)

```python
import numpy as np
from xgboost import XGBClassifier

# 1) Load precomputed text embeddings
X_text_train = np.load("embeddings/bertlite_full_fresh__train_cls_embeddings.npy")
X_text_val = np.load("embeddings/bertlite_full_fresh__val_cls_embeddings.npy")

# 2) Load your numeric features aligned to the same row order
# X_num_train, X_num_val = ...

# 3) Fuse text + numeric features
# X_train = np.concatenate([X_num_train, X_text_train], axis=1)
# X_val = np.concatenate([X_num_val, X_text_val], axis=1)

# 4) Train a downstream model
# clf = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.1)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_val)
```

Minimal pseudocode:

```text
load CLS embeddings
align with numeric feature rows
concatenate [numeric, CLS]
train XGBoost
compare vs numeric-only baseline
```

## Main Test Metrics (from results summary)

- Numeric-only XGB: accuracy 0.7098, F1 0.6638, precision 0.5406, recall 0.8597
- CLS+Numeric XGB: accuracy 0.7881, F1 0.7378, precision 0.6278, recall 0.8947
- Pure CLS linear: accuracy 0.5252, F1 0.2860, precision 0.2866, recall 0.2854
- Delta test F1 (CLS+Numeric - Numeric-only): +0.0740

## Practical Notes

- Embeddings are frozen feature tensors (`.npy`), not fine-tuned checkpoints.
- Files are split by train/val/test for direct experimental use.
- Check `embeddings/embeddings_manifest.csv` to verify integrity.

## Notes

- Preliminary release only; confidence intervals and repeated-seed aggregates are not included yet.
- Embedding files are derived feature tensors (`.npy`), not raw text records.
- D-1 lagging was used for market/FnG features to avoid look-ahead bias.
- Training and experiment scripts are intentionally not mirrored in this artifact-only repo.
