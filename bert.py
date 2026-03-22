import argparse
import json
import os

import torch

from neural_utils import set_torch_seed, train_neural
from pipeline_common import (
    LOGGER,
    ensure_lagged,
    load_data,
    log_stage,
    prepare_numeric_features,
    set_seed,
    setup_logging,
    stratified_sample_df,
)
from text_embeddings import pick_device


def save_neural_artifacts(output_dir: str, neural_metrics: dict, prepared) -> None:
    """Persist neural metrics and preprocessing artifacts for later reuse."""
    os.makedirs(output_dir, exist_ok=True)

    import joblib

    joblib.dump(prepared.scaler, os.path.join(output_dir, "numeric_scaler.joblib"))
    joblib.dump(prepared.encoder, os.path.join(output_dir, "fng_onehot_encoder.joblib"))

    summary = {
        "neural": {
            "val": neural_metrics["val"],
            "test": neural_metrics["test"],
        },
        "note": "All market/FnG features are lagged by one day (D-1) to avoid look-ahead bias.",
    }
    with open(
        os.path.join(output_dir, "metrics_summary_neural.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)


def run(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    set_seed(args.seed)
    set_torch_seed(args.seed)

    if args.fast_mode:
        LOGGER.info(
            "Fast mode enabled: applying lightweight defaults for laptop training."
        )
        args.max_length = min(args.max_length, 96)
        args.train_batch_size = min(args.train_batch_size, 8)
        args.epochs = min(args.epochs, 1)
        args.sample_frac = min(args.sample_frac, 0.35)
        args.eval_every_epochs = max(1, args.eval_every_epochs)

    device = pick_device()
    LOGGER.info("Using device: %s", device)

    log_stage("Load and prepare lagged data")
    df = load_data(args.data_path)
    df = ensure_lagged(df)

    if args.sample_frac < 1.0 or args.min_per_stratum > 0:
        log_stage("Apply stratified sampling")
        strata_cols = [c.strip() for c in args.strata_cols.split(",") if c.strip()]
        df = stratified_sample_df(
            df=df,
            sample_frac=args.sample_frac,
            seed=args.seed,
            strata_cols=strata_cols,
            min_per_stratum=args.min_per_stratum,
        )

    log_stage("Prepare numeric features")
    prepared = prepare_numeric_features(df)

    log_stage("Train end-to-end neural model")
    neural_out = train_neural(
        prepared=prepared,
        model_name=args.model_name,
        output_dir=args.output_dir,
        use_mlp=not args.use_linear_numeric_fusion,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        device=device,
        freeze_bert=args.freeze_bert,
        peft_mode=args.peft_mode,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=[
            m.strip() for m in args.lora_target_modules.split(",") if m.strip()
        ],
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        eval_every_epochs=args.eval_every_epochs,
    )

    LOGGER.info("Neural Validation: %s", neural_out["val"])
    LOGGER.info("Neural Test: %s", neural_out["test"])

    log_stage("Save neural artifacts")
    save_neural_artifacts(args.output_dir, neural_out, prepared)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train end-to-end BERT+numeric (MLP fusion) classifier."
    )
    parser.add_argument("--data_path", type=str, default="enriched_news.parquet")
    parser.add_argument("--model_name", type=str, default="boltuix/bert-lite")
    parser.add_argument("--output_dir", type=str, default="outputs_compare_models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="INFO")

    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--strata_cols", type=str, default="split,label")
    parser.add_argument("--min_per_stratum", type=int, default=1)
    parser.add_argument("--fast_mode", action="store_true")

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--use_linear_numeric_fusion", action="store_true")

    parser.add_argument("--peft_mode", type=str, default="none", choices=["none", "lora"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="query,key,value",
        help="Comma-separated module names for LoRA injection.",
    )
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_every_epochs", type=int, default=1)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(args)
