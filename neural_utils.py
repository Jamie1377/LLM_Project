import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, BertTokenizer

from pipeline_common import LOGGER, PreparedFeatures, evaluate


def set_torch_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultimodalNewsDataset(Dataset):
    """Dataset for neural multimodal training."""

    def __init__(
        self,
        texts: List[str],
        numeric_features: np.ndarray,
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.numeric_features = numeric_features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "numeric_feats": torch.tensor(
                self.numeric_features[idx], dtype=torch.float32
            ),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class EndToEndBertModel(nn.Module):
    """Text branch (BERT) + numeric branch (MLP/linear) + fusion classifier.""" 
    def __init__(
        self,
        model_name: str, 
        numeric_dim: int,
        use_mlp: bool = True,
        mlp_hidden_1: int = 128,
        mlp_hidden_2: int = 64,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the multimodal model with optional MLP for numeric features.
        Args:
            model_name: Pretrained BERT model name for text encoding.
            numeric_dim: Number of numeric features.
            use_mlp: Whether to use an MLP for numeric feature processing before fusion.
            mlp_hidden_1: Hidden layer size 1 for numeric MLP (if used).
            mlp_hidden_2: Hidden layer size 2 for numeric MLP (if used).
            dropout: Dropout rate for MLP layers (if used).
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        if use_mlp:
            self.numeric_branch = nn.Sequential(
                nn.Linear(numeric_dim, mlp_hidden_1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_1, mlp_hidden_2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            fusion_dim = self.bert.config.hidden_size + mlp_hidden_2
        else:
            self.numeric_branch = nn.Identity()
            fusion_dim = self.bert.config.hidden_size + numeric_dim

        self.classifier = nn.Linear(fusion_dim, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_feats: torch.Tensor,
    ) -> torch.Tensor:
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = bert_out.last_hidden_state[:, 0, :]
        numeric_vector = self.numeric_branch(numeric_feats)
        fused = torch.cat([cls_vector, numeric_vector], dim=1)
        return self.classifier(fused)


def _evaluate_neural(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numeric_feats = batch["numeric_feats"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numeric_feats=numeric_feats,
            )
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return evaluate(np.array(all_labels), np.array(all_preds))


def _apply_lora_to_bert(
    bert_model: nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Optional[List[str]],
) -> nn.Module:
    """Wrap BERT with LoRA adapters for parameter-efficient fine-tuning."""
    try:
        import importlib

        peft_mod = importlib.import_module("peft")
        LoraConfig = getattr(peft_mod, "LoraConfig")
        TaskType = getattr(peft_mod, "TaskType")
        get_peft_model = getattr(peft_mod, "get_peft_model")
    except ImportError as exc:
        raise ImportError(
            "peft is required for LoRA. Install with: pip install peft"
        ) from exc

    targets = lora_target_modules or ["query", "key", "value"]
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=targets,
    )
    peft_model = get_peft_model(bert_model, config)
    LOGGER.info(
        "LoRA enabled | r=%d alpha=%d dropout=%.3f targets=%s",
        lora_r,
        lora_alpha,
        lora_dropout,
        targets,
    )
    return peft_model


def _load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Load either legacy raw state_dict checkpoints or new packaged checkpoints."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else ckpt
    if state_dict is None:
        raise ValueError(f"Invalid checkpoint format in {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        LOGGER.info("Missing keys loading checkpoint (first 10): %s", missing[:10])
    if unexpected:
        LOGGER.info(
            "Unexpected keys loading checkpoint (first 10): %s", unexpected[:10]
        )


def _extract_bert_encoder_state_for_reuse(bert_module: nn.Module) -> Dict[str, torch.Tensor]:
    """Export plain BERT encoder weights for downstream CLS extraction in XGB scripts."""
    if hasattr(bert_module, "merge_and_unload"):
        # LoRA case: merge adapters into base weights before exporting.
        merged = bert_module.merge_and_unload()
        return {k: v.detach().cpu() for k, v in merged.state_dict().items()}

    return {k: v.detach().cpu() for k, v in bert_module.state_dict().items()}


def train_neural(
    prepared: PreparedFeatures,
    model_name: str,
    output_dir: str,
    use_mlp: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_length: int,
    device: torch.device,
    freeze_bert: bool,
    peft_mode: str = "none",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    grad_accum_steps: int = 1,
    num_workers: int = 0,
    eval_every_epochs: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Train end-to-end multimodal neural model and save best checkpoint.
    
    Args:
        prepared: PreparedFeatures object containing train/val/test splits.
        model_name: Pretrained BERT model name for text encoding.
        output_dir: Directory to save model artifacts.
        use_mlp: Whether to use an MLP for numeric feature processing before fusion.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay for optimizer.
        max_length: Maximum token length for BERT tokenizer.
        device: Torch device to use for training.   
        freeze_bert: Whether to freeze BERT encoder parameters during training.
        peft_mode: Parameter-efficient tuning mode. Supported: "none", "lora".
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling alpha.
        lora_dropout: LoRA dropout.
        lora_target_modules: Optional target submodules, default query/key/value.
        grad_accum_steps: Number of gradient accumulation steps.
        num_workers: DataLoader worker count.
        eval_every_epochs: Validation frequency in epochs.
        """
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    if eval_every_epochs < 1:
        raise ValueError("eval_every_epochs must be >= 1")

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = MultimodalNewsDataset(
        texts=prepared.train_df["text"].astype(str).tolist(),
        numeric_features=prepared.X_train_num,
        labels=prepared.y_train,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_ds = MultimodalNewsDataset(
        texts=prepared.val_df["text"].astype(str).tolist(),
        numeric_features=prepared.X_val_num,
        labels=prepared.y_val,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_ds = MultimodalNewsDataset(
        texts=prepared.test_df["text"].astype(str).tolist(),
        numeric_features=prepared.X_test_num,
        labels=prepared.y_test,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = EndToEndBertModel(
        model_name=model_name,
        numeric_dim=prepared.X_train_num.shape[1],
        use_mlp=use_mlp,
    ).to(device)

    if peft_mode == "lora":
        if freeze_bert:
            raise ValueError("Cannot use --freeze_bert with peft_mode='lora'.")
        model.bert = _apply_lora_to_bert(
            model.bert,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )

    if freeze_bert:
        LOGGER.info("Freezing BERT encoder parameters for faster neural training.")
        for param in model.bert.parameters():
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    LOGGER.info("Trainable parameters: %d / %d", trainable_params, total_params)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = max(1, (len(train_loader) + grad_accum_steps - 1) // grad_accum_steps)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_ckpt_path = os.path.join(output_dir, f"best_neural_model_base_{model_name.replace('/', '_')}.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numeric_feats = batch["numeric_feats"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numeric_feats=numeric_feats,
            )
            loss = criterion(logits, labels) / grad_accum_steps
            loss.backward()

            should_step = (step_idx % grad_accum_steps == 0) or (step_idx == len(train_loader))
            if should_step:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item()) * grad_accum_steps

        if epoch % eval_every_epochs != 0 and epoch != epochs:
            avg_loss = running_loss / max(1, len(train_loader))
            LOGGER.info(
                "[Neural] Epoch %d/%d | Loss %.4f | Val skipped (eval_every_epochs=%d)",
                epoch,
                epochs,
                avg_loss,
                eval_every_epochs,
            )
            continue

        val_metrics = _evaluate_neural(model, val_loader, device)
        avg_loss = running_loss / max(1, len(train_loader))
        LOGGER.info(
            "[Neural] Epoch %d/%d | Loss %.4f | Val F1 %.4f | Val Acc %.4f",
            epoch,
            epochs,
            avg_loss,
            val_metrics["f1"],
            val_metrics["accuracy"],
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "meta": {
                        "peft_mode": peft_mode,
                        "freeze_bert": freeze_bert,
                    },
                },
                best_ckpt_path,
            )
            LOGGER.info("Saved new best neural checkpoint with val_f1=%.4f", best_f1)

    _load_checkpoint_into_model(model, best_ckpt_path, device)
    val_metrics = _evaluate_neural(model, val_loader, device)
    test_metrics = _evaluate_neural(model, test_loader, device)

    bert_encoder_state = _extract_bert_encoder_state_for_reuse(model.bert)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "bert_encoder_state_dict": bert_encoder_state,
            "meta": {
                "peft_mode": peft_mode,
                "freeze_bert": freeze_bert,
            },
        },
        best_ckpt_path,
    )

    tokenizer.save_pretrained(output_dir)

    return {
        "val": val_metrics,
        "test": test_metrics,
        "model": model,
    }
