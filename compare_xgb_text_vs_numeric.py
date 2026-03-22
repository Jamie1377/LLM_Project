import argparse
import hashlib
import json
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from pipeline_common import (
    evaluate,
    ensure_lagged,
    load_data,
    log_stage,
    prepare_numeric_features,
    set_seed,
    setup_logging,
    stratified_sample_df,
)
from text_embeddings import (
    pick_device,
    prepare_finetuned_embeddings_for_xgb,
    prepare_frozen_embeddings_for_xgb,
)
from xgb_utils import train_xgb


def _build_embedding_cache_key(args: argparse.Namespace) -> str:
    """Build a stable cache key so embeddings are isolated by run configuration."""
    if args.embedding_cache_key and args.embedding_cache_key.lower() != "auto":
        return args.embedding_cache_key

    payload = {
        "embedding_source": args.embedding_source,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "sample_frac": args.sample_frac,
        "seed": args.seed,
        "strata_cols": args.strata_cols,
        "min_per_stratum": args.min_per_stratum,
        "data_path": args.data_path,
    }
    if args.embedding_source == "finetuned":
        payload["finetuned_checkpoint"] = args.finetuned_checkpoint

    raw = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{args.embedding_source}_{digest}"


def run(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    """Compare numeric-only XGB vs fine-tuned-CLS+numeric XGB."""
    if args.deterministic_run:
        args.xgb_n_jobs = 1

    setup_logging(args.log_level)
    set_seed(args.seed)
    device = pick_device()

    # ---------------------------
    # Data and lagging section
    # ---------------------------
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

    # ---------------------------
    # Numeric feature section
    # ---------------------------
    log_stage("Prepare numeric features")
    prepared = prepare_numeric_features(df)
    embedding_cache_key = _build_embedding_cache_key(args)

    # Baseline classifier: XGBoost on numeric-only branch.
    log_stage("Train XGBoost on numeric features only")
    numeric_only_out = train_xgb(
        X_train=prepared.X_train_num,
        y_train=prepared.y_train,
        X_val=prepared.X_val_num,
        y_val=prepared.y_val,
        X_test=prepared.X_test_num,
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

    # Text+numeric classifier branch.
    if args.embedding_source == "finetuned":
        log_stage("Prepare fine-tuned CLS embeddings and train XGBoost on CLS+numeric")
        tr_emb, va_emb, te_emb = prepare_finetuned_embeddings_for_xgb(
            prepared=prepared,
            model_name=args.model_name,
            finetuned_checkpoint=args.finetuned_checkpoint,
            tokenizer_path=args.tokenizer_path,
            batch_size=args.embedding_batch_size,
            max_length=args.max_length,
            device=device,
            cache_dir=args.embedding_cache_dir,
            cache_key=embedding_cache_key,
        )
    else:
        log_stage("Prepare frozen CLS embeddings and train XGBoost on CLS+numeric")
        tr_emb, va_emb, te_emb = prepare_frozen_embeddings_for_xgb(
            prepared=prepared,
            model_name=args.model_name,
            batch_size=args.embedding_batch_size,
            max_length=args.max_length,
            device=device,
            cache_dir=args.embedding_cache_dir,
            cache_key=embedding_cache_key,
        )

    # Cache can be stale if dataset/sample settings changed; recompute if mismatch.
    if (
        tr_emb.shape[0] != prepared.X_train_num.shape[0]
        or va_emb.shape[0] != prepared.X_val_num.shape[0]
        or te_emb.shape[0] != prepared.X_test_num.shape[0]
    ):
        if args.embedding_source == "finetuned":
            tr_emb, va_emb, te_emb = prepare_finetuned_embeddings_for_xgb(
                prepared=prepared,
                model_name=args.model_name,
                finetuned_checkpoint=args.finetuned_checkpoint,
                tokenizer_path=args.tokenizer_path,
                batch_size=args.embedding_batch_size,
                max_length=args.max_length,
                device=device,
                cache_dir=None,
                cache_key=None,
            )
        else:
            tr_emb, va_emb, te_emb = prepare_frozen_embeddings_for_xgb(
                prepared=prepared,
                model_name=args.model_name,
                batch_size=args.embedding_batch_size,
                max_length=args.max_length,
                device=device,
                cache_dir=None,
                cache_key=None,
            )

    # Fusion for XGB path: concatenate text embedding vector and numeric vector.
    X_train_cls_num = np.hstack([tr_emb, prepared.X_train_num])
    X_val_cls_num = np.hstack([va_emb, prepared.X_val_num])
    X_test_cls_num = np.hstack([te_emb, prepared.X_test_num])

    # Pure CLS linear classifier branch: one linear layer on CLS embeddings only.
    log_stage("Train pure CLS linear classifier")
    cls_linear = LogisticRegression(
        random_state=args.seed,
        max_iter=args.cls_linear_max_iter,
        C=args.cls_linear_c,
        solver="liblinear",
    )
    cls_linear.fit(tr_emb, prepared.y_train)
    val_pred_cls_linear = cls_linear.predict(va_emb)
    test_pred_cls_linear = cls_linear.predict(te_emb)
    cls_linear_out = {
        "val": evaluate(prepared.y_val, val_pred_cls_linear),
        "test": evaluate(prepared.y_test, test_pred_cls_linear),
        "model": cls_linear,
    }

    cls_plus_num_out = train_xgb(
        X_train=X_train_cls_num,
        y_train=prepared.y_train,
        X_val=X_val_cls_num,
        y_val=prepared.y_val,
        X_test=X_test_cls_num,
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

    # Diagnostics: compare how much predictions/probabilities differ between models.
    num_model = numeric_only_out["model"]
    cls_model = cls_plus_num_out["model"]

    val_pred_num = num_model.predict(prepared.X_val_num)
    test_pred_num = num_model.predict(prepared.X_test_num)
    val_pred_cls = cls_model.predict(X_val_cls_num)
    test_pred_cls = cls_model.predict(X_test_cls_num)

    val_prob_num = num_model.predict_proba(prepared.X_val_num)[:, 1]
    test_prob_num = num_model.predict_proba(prepared.X_test_num)[:, 1]
    val_prob_cls = cls_model.predict_proba(X_val_cls_num)[:, 1]
    test_prob_cls = cls_model.predict_proba(X_test_cls_num)[:, 1]

    val_pred_diff = int(np.sum(val_pred_num != val_pred_cls))
    test_pred_diff = int(np.sum(test_pred_num != test_pred_cls))
    val_prob_mae = float(np.mean(np.abs(val_prob_num - val_prob_cls)))
    test_prob_mae = float(np.mean(np.abs(test_prob_num - test_prob_cls)))

    # Structured output for reproducible experiment tracking.
    summary = {
        "numeric_only_xgb": {
            "val": numeric_only_out["val"],
            "test": numeric_only_out["test"],
        },
        "cls_plus_numeric_xgb": {
            "val": cls_plus_num_out["val"],
            "test": cls_plus_num_out["test"],
        },
        "pure_cls_linear": {
            "val": cls_linear_out["val"],
            "test": cls_linear_out["test"],
        },
        "delta_test_f1": cls_plus_num_out["test"]["f1"]
        - numeric_only_out["test"]["f1"],
        "delta_val_f1": cls_plus_num_out["val"]["f1"] - numeric_only_out["val"]["f1"],
        "delta_test_f1_cls_linear_vs_numeric": cls_linear_out["test"]["f1"]
        - numeric_only_out["test"]["f1"],
        "delta_test_f1_cls_linear_vs_clsnum": cls_linear_out["test"]["f1"]
        - cls_plus_num_out["test"]["f1"],
        "diagnostics": {
            "val_pred_diff_count": val_pred_diff,
            "test_pred_diff_count": test_pred_diff,
            "val_prob_mae": val_prob_mae,
            "test_prob_mae": test_prob_mae,
        },
        "embedding_source": args.embedding_source,
        "text_model": args.model_name,
        "embedding_cache_key": embedding_cache_key,
        "note": "All market/FnG features are D-1 lagged to avoid look-ahead bias.",
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Numeric-only XGB (test):", numeric_only_out["test"])
    print("CLS+Numeric XGB (test):", cls_plus_num_out["test"])
    print("Pure CLS linear (test):", cls_linear_out["test"])
    print("Val pred diff count:", val_pred_diff)
    print("Test pred diff count:", test_pred_diff)
    print("Val prob MAE:", val_prob_mae)
    print("Test prob MAE:", test_prob_mae)
    print("Delta test F1:", summary["delta_test_f1"])
    print("Saved:", args.output_json)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Define CLI options for data, fine-tuned checkpoint, and XGB settings."""
    parser = argparse.ArgumentParser(
        description="Compare XGBoost performance: numeric-only vs fine-tuned CLS+numeric."
    )
    parser.add_argument("--data_path", type=str, default="enriched_news.parquet")
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs_compare_models/metrics_xgb_cls_vs_numeric.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="INFO")

    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--strata_cols", type=str, default="split,label")
    parser.add_argument("--min_per_stratum", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="ProsusAI/finbert")
    parser.add_argument(
        "--embedding_source",
        type=str,
        choices=["frozen", "finetuned"],
        default="frozen",
        help="Use frozen base encoder CLS embeddings or a fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--finetuned_checkpoint",
        type=str,
        default=f"outputs_compare_models/best_neural_model_base_{parser.get_default('model_name').replace('/', '_')}.pt",
    )
    parser.add_argument("--tokenizer_path", type=str, default="outputs_compare_models")
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument(
        "--embedding_cache_dir", type=str, default="embedding_cache_xgb_compare"
    )
    parser.add_argument(
        "--embedding_cache_key",
        type=str,
        default="auto",
        help="Cache namespace key. Use 'auto' to derive from model/sample settings.",
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
    parser.add_argument(
        "--deterministic_run",
        action="store_true",
        help="Set XGBoost and CV workers to 1 for more reproducible reruns.",
    )

    parser.add_argument("--cls_linear_max_iter", type=int, default=2000)
    parser.add_argument("--cls_linear_c", type=float, default=1.0)

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(args)
