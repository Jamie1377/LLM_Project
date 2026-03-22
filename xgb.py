import argparse
import hashlib
import json
import os

import numpy as np

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
from text_embeddings import pick_device, prepare_frozen_embeddings_for_xgb
from xgb_utils import train_xgb


def _build_embedding_cache_key(args: argparse.Namespace) -> str:
    payload = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "sample_frac": args.sample_frac,
        "seed": args.seed,
        "strata_cols": args.strata_cols,
        "min_per_stratum": args.min_per_stratum,
        "data_path": args.data_path,
    }
    raw = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"frozen_{digest}"


def run(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    set_seed(args.seed)
    device = pick_device()

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
    embedding_cache_key = _build_embedding_cache_key(args)

    log_stage("Prepare frozen CLS embeddings")
    tr_emb, va_emb, te_emb = prepare_frozen_embeddings_for_xgb(
        prepared=prepared,
        model_name=args.model_name,
        batch_size=args.embedding_batch_size,
        max_length=args.max_length,
        device=device,
        cache_dir=args.embedding_cache_dir,
        cache_key=embedding_cache_key,
    )

    X_train = np.hstack([tr_emb, prepared.X_train_num])
    X_val = np.hstack([va_emb, prepared.X_val_num])
    X_test = np.hstack([te_emb, prepared.X_test_num])

    log_stage("Train XGBoost")
    xgb_out = train_xgb(
        X_train=X_train,
        y_train=prepared.y_train,
        X_val=X_val,
        y_val=prepared.y_val,
        X_test=X_test,
        y_test=prepared.y_test,
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        use_random_search=args.xgb_random_search,
        random_search_iters=args.xgb_random_iters,
        refit_on_train_val=args.xgb_refit_on_train_val,
        random_state=args.seed,
        xgb_n_jobs=args.xgb_n_jobs,
    )

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "xgb": {
                    "val": xgb_out["val"],
                    "test": xgb_out["test"],
                },
                "note": "Frozen CLS + numeric features.",
            },
            f,
            indent=2,
        )

    LOGGER.info("XGB Validation: %s", xgb_out["val"])
    LOGGER.info("XGB Test: %s", xgb_out["test"])
    LOGGER.info("Saved metrics: %s", args.output_json)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train XGBoost on frozen CLS + numeric features."
    )
    parser.add_argument("--data_path", type=str, default="enriched_news.parquet")
    parser.add_argument("--model_name", type=str, default="gaunernst/bert-mini-uncased")
    parser.add_argument(
        "--output_json", type=str, default="outputs_compare_models/metrics_xgb.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="INFO")

    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--strata_cols", type=str, default="split,label")
    parser.add_argument("--min_per_stratum", type=int, default=1)

    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument(
        "--embedding_cache_dir", type=str, default="embedding_cache_xgb"
    )

    parser.add_argument("--xgb_n_estimators", type=int, default=250)
    parser.add_argument("--xgb_max_depth", type=int, default=4)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--xgb_subsample", type=float, default=0.9)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.9)
    parser.add_argument("--xgb_random_search", action="store_true", default=True)
    parser.add_argument("--xgb_random_iters", type=int, default=20)
    parser.add_argument("--xgb_refit_on_train_val", action="store_true", default=True)
    parser.add_argument("--xgb_n_jobs", type=int, default=-1)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(args)
