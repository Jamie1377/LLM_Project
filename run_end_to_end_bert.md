# Run Commands: End-to-End BERT Training

## 1) Go to project folder
```bash
cd /Users/jamie/Downloads/LLM_Project
```

## 2) Activate virtual environment
```bash
source .venv/bin/activate
```

## 3) Train end-to-end BERT (full run)
```bash
python bert.py \
  --model_name boltuix/bert-lite \
  --data_path enriched_news.parquet \
  --output_dir outputs_compare_models \
  --epochs 3 \
  --train_batch_size 16 \
  --max_length 128 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --sample_frac 1.0 \
  --min_per_stratum 1 \
  --log_level INFO
```

## 4) Train end-to-end BERT (faster dev run on laptop)
```bash
python bert.py \
  --model_name boltuix/bert-lite \
  --data_path enriched_news.parquet \
  --output_dir outputs_compare_models \
  --fast_mode \
  --sample_frac 0.35 \
  --min_per_stratum 50 \
  --log_level INFO
```

## 5) Train with LoRA (faster and lighter than full fine-tuning)
```bash
python bert.py \
  --model_name kk08/CryptoBERT \
  --data_path enriched_news.parquet \
  --output_dir outputs_compare_models \
  --peft_mode lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules query,key,value \
  --train_batch_size 8 \
  --grad_accum_steps 2 \
  --num_workers 2 \
  --eval_every_epochs 1 \
  --sample_frac 0.5 \
  --min_per_stratum 50 \
  --log_level INFO
```

## 6) Extra speed-focused run (LoRA + fast mode)
```bash
python bert.py \
  --model_name boltuix/bert-lite \
  --data_path enriched_news.parquet \
  --output_dir outputs_compare_models \
  --peft_mode lora \
  --fast_mode \
  --grad_accum_steps 2 \
  --num_workers 2 \
  --sample_frac 0.35 \
  --min_per_stratum 50 \
  --log_level INFO
```

## 7) Output files to check
- `outputs_compare_models/best_neural_model.pt`
- `outputs_compare_models/metrics_summary_neural.json`
- `outputs_compare_models/numeric_scaler.joblib`
- `outputs_compare_models/fng_onehot_encoder.joblib`

## 8) Optional: freeze BERT (not end-to-end fine-tuning)
```bash
python bert.py --freeze_bert
```

## 9) Train XGBoost only (frozen CLS + numeric)
```bash
python xgb.py \
  --model_name gaunernst/bert-mini-uncased \
  --data_path enriched_news.parquet \
  --output_json outputs_compare_models/metrics_xgb.json \
  --sample_frac 1.0 \
  --min_per_stratum 1 \
  --xgb_random_search \
  --xgb_random_iters 20 \
  --xgb_refit_on_train_val \
  --log_level INFO
```

## 10) Train XGBoost only (faster run)
```bash
python xgb.py \
  --model_name gaunernst/bert-mini-uncased \
  --data_path enriched_news.parquet \
  --sample_frac 0.35 \
  --min_per_stratum 50 \
  --embedding_batch_size 32 \
  --max_length 96 \
  --xgb_n_estimators 200 \
  --xgb_max_depth 4 \
  --xgb_random_search \
  --xgb_random_iters 10 \
  --xgb_refit_on_train_val \
  --log_level INFO
```

## 11) Compare numeric-only XGB vs frozen FinBERT CLS+numeric XGB
```bash
python compare_xgb_text_vs_numeric.py \
  --embedding_source frozen \
  --model_name ProsusAI/finbert \
  --data_path enriched_news.parquet \
  --sample_frac 1.0 \
  --min_per_stratum 1 \
  --xgb_random_search \
  --xgb_random_iters 20 \
  --xgb_refit_on_train_val \
  --log_level INFO
```

## 12) Compare numeric-only XGB vs fine-tuned CLS+numeric XGB
```bash
python compare_xgb_text_vs_numeric.py \
  --embedding_source finetuned \
  --model_name boltuix/bert-lite \
  --finetuned_checkpoint outputs_compare_models/best_neural_model_base_boltuix_bert-lite.pt \
  --tokenizer_path outputs_compare_models \
  --data_path enriched_news.parquet \
  --sample_frac 1.0 \
  --min_per_stratum 1 \
  --xgb_random_search \
  --xgb_random_iters 20 \
  --xgb_refit_on_train_val \
  --log_level INFO
```

## 13) Compare script (faster run)
```bash
python compare_xgb_text_vs_numeric.py \
  --embedding_source frozen \
  --model_name ProsusAI/finbert \
  --sample_frac 0.35 \
  --min_per_stratum 50 \
  --embedding_batch_size 32 \
  --max_length 96 \
  --xgb_random_search \
  --xgb_random_iters 10 \
  --xgb_refit_on_train_val \
  --log_level INFO
```

## 14) Typical workflow (recommended order)
```bash
# Step A: train neural model (full or LoRA)
python bert.py --peft_mode lora --fast_mode --sample_frac 0.35 --min_per_stratum 50 --log_level INFO

# Step B: compare numeric-only vs fine-tuned CLS+numeric
python compare_xgb_text_vs_numeric.py --finetuned_checkpoint outputs_compare_models/best_neural_model.pt --tokenizer_path outputs_compare_models --sample_frac 0.35 --min_per_stratum 50 --log_level INFO

# Optional Step C: run frozen-CLS XGB baseline script directly
python xgb.py --sample_frac 0.35 --min_per_stratum 50 --xgb_random_search --xgb_refit_on_train_val --log_level INFO
```

## 15) Useful output files from other scripts
- `outputs_compare_models/metrics_xgb.json` (from `xgb.py`)
- `outputs_compare_models/metrics_xgb_cls_vs_numeric.json` (from `compare_xgb_text_vs_numeric.py`)

## 16) Embedding cache safety (important)
- Caches now use an auto key derived from model + sample settings + seed + max length.
- This makes cache reuse safer when you change `sample_frac`, model, or other settings.
- Optional manual override (compare script):
```bash
python compare_xgb_text_vs_numeric.py \
  --embedding_source frozen \
  --model_name ProsusAI/finbert \
  --embedding_cache_key my_custom_cache_v1
```
- If you want strict isolation per experiment, use a separate cache folder too:
```bash
python compare_xgb_text_vs_numeric.py \
  --embedding_source frozen \
  --model_name ProsusAI/finbert \
  --embedding_cache_dir embedding_cache_xgb_compare_exp1
```

# After getting correct embeddings (this takes a hour to run!!!)
```bash
python compare_xgb_text_vs_numeric.py \
  --embedding_source frozen \
  --model_name ProsusAI/finbert \
  --data_path enriched_news.parquet \
  --sample_frac 1.0 \
  --min_per_stratum 1 \
  --embedding_cache_dir embedding_cache_xgb_compare_fresh_20260316 \
  --embedding_cache_key finbert_full_fresh \
  --xgb_random_iters 1 \
  --xgb_n_estimators 100 \
  --xgb_max_depth 3 \
  --log_level INFO
```

Copy-safe single line:
```bash
python compare_xgb_text_vs_numeric.py --embedding_source frozen --model_name ProsusAI/finbert --data_path enriched_news.parquet --sample_frac 1.0 --min_per_stratum 1 --embedding_cache_dir embedding_cache_xgb_compare_fresh_20260316 --embedding_cache_key finbert_full_fresh --xgb_random_iters 1 --xgb_n_estimators 100 --xgb_max_depth 3 --log_level INFO
```

# Fast version of frozen cls (bert-lite) - similar performance against FinBert
```bash
python compare_xgb_text_vs_numeric.py \
  --embedding_source frozen \
  --model_name boltuix/bert-lite \
  --data_path enriched_news.parquet \
  --sample_frac 1.0 \
  --min_per_stratum 1 \
  --embedding_cache_dir embedding_cache_xgb_compare_fresh_20260316 \
  --embedding_cache_key bertlite_full_fresh \
  --log_level INFO 
```

## 17) Reproducible run settings (recommended)
- Use fixed `--seed` and fixed cache key.
- Use `--deterministic_run` (forces single-thread XGBoost/CV workers).
- Keep data/sampling settings unchanged across reruns.

```bash
python compare_xgb_text_vs_numeric.py \
  --embedding_source frozen \
  --model_name boltuix/bert-lite \
  --data_path enriched_news.parquet \
  --sample_frac 1.0 \
  --min_per_stratum 1 \
  --seed 42 \
  --embedding_cache_dir embedding_cache_xgb_compare_fresh_20260316 \
  --embedding_cache_key bertlite_full_fresh \
  --deterministic_run \
  --xgb_random_iters 20 \
  --xgb_refit_on_train_val \
  --log_level INFO
```

Optional: if you do not use `--deterministic_run`, set workers manually:
```bash
python compare_xgb_text_vs_numeric.py --xgb_n_jobs 1
```


# not limited xgb bert-lite
```bash
python compare_xgb_text_vs_numeric.py \
  --embedding_source frozen \
  --model_name boltuix/bert-lite \
  --data_path enriched_news.parquet \
  --sample_frac 1.0 \
  --min_per_stratum 1 \
  --seed 42 \
  --embedding_cache_dir embedding_cache_xgb_compare_fresh_20260316 \
  --embedding_cache_key bertlite_full_fresh \
  --deterministic_run \
  --xgb_refit_on_train_val \
  --log_level INFO
```
# not limited xgb ver of finbert
```bash
python compare_xgb_text_vs_numeric.py \
  --embedding_source frozen \
  --model_name ProsusAI/finbert \
  --data_path enriched_news.parquet \
  --sample_frac 1.0 \
  --min_per_stratum 1 \
  --seed 42 \
  --embedding_cache_dir embedding_cache_xgb_compare_fresh_20260316 \
  --embedding_cache_key finbert_full_fresh \
  --deterministic_run \
  --xgb_refit_on_train_val \
  --log_level INFO
```

# upload the HF (only bert-lite)
/Users/jamie/Downloads/LLM_Project/.venv/bin/hf auth login
HF_USERNAME=JamieYuu REPO_NAME=JamieYuu/slm-bert-emb EMBEDDING_FILTER=bertlite ./hf_embedding_release/upload_embeddings_to_hf.sh