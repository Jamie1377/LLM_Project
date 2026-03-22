import os
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from pipeline_common import LOGGER, PreparedFeatures


def _sanitize_cache_key(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return clean.strip("_")[:120] or "default"


def _cache_file_path(
    cache_dir: Optional[str],
    split_name: str,
    embedding_kind: str,
    cache_key: Optional[str],
) -> Optional[str]:
    if not cache_dir:
        return None
    safe_key = _sanitize_cache_key(cache_key or "default")
    filename = f"{safe_key}__{split_name}_{embedding_kind}.npy"
    return os.path.join(cache_dir, filename)


def _legacy_cache_file_path(
    cache_dir: Optional[str],
    split_name: str,
    embedding_kind: str,
) -> Optional[str]:
    if not cache_dir:
        return None

    # Backward compatibility with pre-key cache filenames.
    if embedding_kind == "cls_embeddings":
        filename = f"{split_name}_cls_embeddings.npy"
    elif embedding_kind == "cls_embeddings_finetuned":
        filename = f"{split_name}_cls_embeddings_finetuned.npy"
    else:
        return None
    return os.path.join(cache_dir, filename)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batched_cls_embeddings(
    texts: List[str],
    tokenizer: AutoTokenizer,
    bert_model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Extract [CLS] vectors in batches from a loaded encoder."""
    bert_model.eval()
    all_embs: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embs.append(cls_vec)

    return np.vstack(all_embs).astype(np.float32)


def prepare_frozen_embeddings_for_xgb(
    prepared: PreparedFeatures,
    model_name: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    cache_dir: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract frozen BERT [CLS] embeddings for train/val/test splits.
    
    Be aware this is FROZEN BERT, not fine-tuned. For fine-tuned embeddings, use prepare_finetuned_embeddings_for_xgb."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name).to(device)

    def get_or_compute(split_name: str, texts: List[str]) -> np.ndarray:
        path = _cache_file_path(
            cache_dir=cache_dir,
            split_name=split_name,
            embedding_kind="cls_embeddings",
            cache_key=cache_key,
        )
        if path and os.path.exists(path):
            LOGGER.info("Loading cached embeddings for %s from %s", split_name, path)
            return np.load(path)

        legacy_path = _legacy_cache_file_path(
            cache_dir=cache_dir,
            split_name=split_name,
            embedding_kind="cls_embeddings",
        )
        if legacy_path and os.path.exists(legacy_path):
            LOGGER.info(
                "Loading legacy cached embeddings for %s from %s",
                split_name,
                legacy_path,
            )
            arr = np.load(legacy_path)
            if path:
                np.save(path, arr)
                LOGGER.info("Migrated legacy cache for %s to %s", split_name, path)
            return arr

        embs = _batched_cls_embeddings(
            texts=texts,
            tokenizer=tokenizer,
            bert_model=bert_model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )

        if path:
            np.save(path, embs)
        return embs

    train_emb = get_or_compute("train", prepared.train_df["text"].astype(str).tolist())
    val_emb = get_or_compute("val", prepared.val_df["text"].astype(str).tolist())
    test_emb = get_or_compute("test", prepared.test_df["text"].astype(str).tolist())
    return train_emb, val_emb, test_emb


def _resolve_checkpoint_path(path: str) -> str:
    if os.path.exists(path):
        return path

    candidates = [
        os.path.join("outputs_compare_models", os.path.basename(path)),
        os.path.join("outputs_compare_model", os.path.basename(path)),
        os.path.join(".", os.path.basename(path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            LOGGER.info("Resolved checkpoint path to %s", candidate)
            return candidate

    raise FileNotFoundError(
        f"Fine-tuned checkpoint not found: {path}. Tried: {candidates}"
    )


def _resolve_tokenizer_path(path: str) -> str:
    if path and os.path.exists(path):
        return path

    candidates = ["outputs_compare_models", "outputs_compare_model"]
    for candidate in candidates:
        if os.path.exists(candidate):
            LOGGER.info("Resolved tokenizer path to %s", candidate)
            return candidate
    return path


def prepare_finetuned_embeddings_for_xgb(
    prepared: PreparedFeatures,
    model_name: str,
    finetuned_checkpoint: str,
    tokenizer_path: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    cache_dir: Optional[str],
    cache_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract [CLS] embeddings using a fine-tuned BERT checkpoint."""
    finetuned_checkpoint = _resolve_checkpoint_path(finetuned_checkpoint)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    tokenizer_path = _resolve_tokenizer_path(tokenizer_path)
    tokenizer_source = (
        tokenizer_path
        if tokenizer_path and os.path.exists(tokenizer_path)
        else model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    bert_model = AutoModel.from_pretrained(model_name).to(device)
    state = torch.load(finetuned_checkpoint, map_location=device)

    bert_state = None
    if isinstance(state, dict) and "bert_encoder_state_dict" in state:
        bert_state = state["bert_encoder_state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        model_state = state["model_state_dict"]
        bert_state = {
            k[len("bert.") :]: v
            for k, v in model_state.items()
            if isinstance(k, str) and k.startswith("bert.")
        }
    elif isinstance(state, dict):
        # Legacy checkpoint: full model state dict with `bert.` prefix keys.
        bert_state = {
            k[len("bert.") :]: v
            for k, v in state.items()
            if isinstance(k, str) and k.startswith("bert.")
        }

    if not bert_state:
        raise ValueError(
            "No reusable BERT encoder state found in checkpoint. "
            "Expected `bert_encoder_state_dict` or legacy `bert.*` keys."
        )

    missing, unexpected = bert_model.load_state_dict(bert_state, strict=False)
    if missing:
        LOGGER.info("Missing keys loading fine-tuned BERT (first 10): %s", missing[:10])
    if unexpected:
        LOGGER.info(
            "Unexpected keys loading fine-tuned BERT (first 10): %s",
            unexpected[:10],
        )

    def get_or_compute(split_name: str, texts: List[str]) -> np.ndarray:
        cache_file = _cache_file_path(
            cache_dir=cache_dir,
            split_name=split_name,
            embedding_kind="cls_embeddings_finetuned",
            cache_key=cache_key,
        )
        if cache_file and os.path.exists(cache_file):
            return np.load(cache_file)

        legacy_cache_file = _legacy_cache_file_path(
            cache_dir=cache_dir,
            split_name=split_name,
            embedding_kind="cls_embeddings_finetuned",
        )
        if legacy_cache_file and os.path.exists(legacy_cache_file):
            arr = np.load(legacy_cache_file)
            if cache_file:
                np.save(cache_file, arr)
                LOGGER.info("Migrated legacy finetuned cache for %s to %s", split_name, cache_file)
            return arr

        embs = _batched_cls_embeddings(
            texts=texts,
            tokenizer=tokenizer,
            bert_model=bert_model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        if cache_file:
            np.save(cache_file, embs)
        return embs

    tr_emb = get_or_compute("train", prepared.train_df["text"].astype(str).tolist())
    va_emb = get_or_compute("val", prepared.val_df["text"].astype(str).tolist())
    te_emb = get_or_compute("test", prepared.test_df["text"].astype(str).tolist())
    return tr_emb, va_emb, te_emb
