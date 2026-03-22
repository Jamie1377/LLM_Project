#!/usr/bin/env bash
set -euo pipefail

# Upload both preliminary metrics + embeddings to ONE Hugging Face repo.
#
# Usage:
#   HF_USERNAME=yourname REPO_NAME=crypto-bert-prelim-all ./hf_one_repo_release/upload_one_repo_to_hf.sh
# Optional:
#   REPO_TYPE=dataset|model            (default: dataset)
#   REPO_VISIBILITY=public|private     (default: public)
#   EMBEDDING_FILTER=all|bertlite|finbert   (default: all)

HF_BIN="/Users/jamie/Downloads/LLM_Project/.venv/bin/hf"
PROJECT_ROOT="/Users/jamie/Downloads/LLM_Project"
RESULTS_DIR="$PROJECT_ROOT/hf_preliminary_results"
EMB_DIR="$PROJECT_ROOT/embedding_cache_xgb_compare_fresh_20260316"
EMB_META="$PROJECT_ROOT/hf_embedding_release/embeddings_manifest.csv"
ROOT_README="$PROJECT_ROOT/hf_one_repo_release/README.md"

: "${HF_USERNAME:?Set HF_USERNAME, e.g. HF_USERNAME=your_hf_username}"
: "${REPO_NAME:?Set REPO_NAME, e.g. REPO_NAME=crypto-bert-prelim-all}"
REPO_TYPE="${REPO_TYPE:-dataset}"
REPO_VISIBILITY="${REPO_VISIBILITY:-public}"
EMBEDDING_FILTER="${EMBEDDING_FILTER:-all}"
REPO_ID="$HF_USERNAME/$REPO_NAME"

if [[ "$REPO_TYPE" != "dataset" && "$REPO_TYPE" != "model" ]]; then
  echo "Invalid REPO_TYPE='$REPO_TYPE'. Use: dataset or model"
  exit 1
fi

if [[ "$REPO_VISIBILITY" == "private" ]]; then
  "$HF_BIN" repos create "$REPO_ID" --repo-type "$REPO_TYPE" --private --exist-ok
else
  "$HF_BIN" repos create "$REPO_ID" --repo-type "$REPO_TYPE" --exist-ok
fi

# Upload one canonical root README for the whole repo.
"$HF_BIN" upload "$REPO_ID" "$ROOT_README" README.md --repo-type "$REPO_TYPE" --commit-message "Add unified repo card"

# Upload results artifacts into subfolder and remove script-centric files.
"$HF_BIN" upload "$REPO_ID" "$RESULTS_DIR" results --repo-type "$REPO_TYPE" --include "metrics_xgb_cls_vs_numeric.json" --include "results_summary.csv" --commit-message "Sync results artifacts"
"$HF_BIN" repos delete-files "$REPO_ID" results/run_end_to_end_bert.md results/requirements.txt --repo-type "$REPO_TYPE" --commit-message "Remove script-centric files from results" || true

# Upload embedding manifest into embeddings/.
"$HF_BIN" upload "$REPO_ID" "$EMB_META" embeddings/embeddings_manifest.csv --repo-type "$REPO_TYPE" --commit-message "Add embedding manifest"

case "$EMBEDDING_FILTER" in
  bertlite)
    "$HF_BIN" upload "$REPO_ID" "$EMB_DIR" embeddings --repo-type "$REPO_TYPE" --include "bertlite_*_cls_embeddings.npy" --commit-message "Upload bert-lite CLS embeddings"
    ;;
  finbert)
    "$HF_BIN" upload "$REPO_ID" "$EMB_DIR" embeddings --repo-type "$REPO_TYPE" --include "finbert_*_cls_embeddings.npy" --commit-message "Upload FinBERT CLS embeddings"
    ;;
  all)
    "$HF_BIN" upload "$REPO_ID" "$EMB_DIR" embeddings --repo-type "$REPO_TYPE" --include "*_cls_embeddings.npy" --commit-message "Upload CLS embeddings"
    ;;
  *)
    echo "Invalid EMBEDDING_FILTER='$EMBEDDING_FILTER'. Use: all, bertlite, finbert"
    exit 1
    ;;
esac

if [[ "$REPO_TYPE" == "dataset" ]]; then
  echo "Uploaded one-repo package to: https://huggingface.co/datasets/$REPO_ID"
else
  echo "Uploaded one-repo package to: https://huggingface.co/$REPO_ID"
fi
