import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LOGGER = logging.getLogger("model_pipeline")


def setup_logging(level: str = "INFO") -> None:
    """Configure global logging format/verbosity for all pipeline stages."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def log_stage(message: str) -> None:
    """Emit a standardized stage marker to make long runs easy to follow."""
    LOGGER.info("[STAGE] %s", message)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy/torch/random."""
    random.seed(seed)
    np.random.seed(seed)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard binary classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, average="binary", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="binary", zero_division=0)
        ),
    }


def load_data(parquet_path: str) -> pd.DataFrame:
    """Load enriched dataset or fallback split files produced by wrangling."""
    if os.path.exists(parquet_path):
        LOGGER.info("Loading dataset from %s", parquet_path)
        df = pd.read_parquet(parquet_path)
    else:
        LOGGER.info("Primary dataset not found: %s", parquet_path)
        fallback = {
            "train": "train_after_merge.parquet",
            "val": "validation_after_merge.parquet",
            "test": "test_after_merge.parquet",
        }
        missing = [v for v in fallback.values() if not os.path.exists(v)]
        if missing:
            raise FileNotFoundError(
                f"Missing input data. '{parquet_path}' not found and fallback files missing: {missing}"
            )

        frames = []
        for split_name, path in fallback.items():
            part = pd.read_parquet(path).copy()
            part["split"] = split_name
            frames.append(part)
        df = pd.concat(frames, ignore_index=True)

    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df["split"] = df["split"].replace({"validation": "val"})

    required_cols = [
        "datetime",
        "text",
        "label",
        "date",
        "split",
        "btc_open",
        "btc_high",
        "btc_low",
        "btc_close",
        "btc_volume",
        "fng_value",
        "fng_classification",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    LOGGER.info("Loaded rows: %d | columns: %d", len(df), len(df.columns))
    LOGGER.info("Splits found: %s", dict(df["split"].value_counts().sort_index()))
    return df


def create_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build D-1 market context features and merge to each row by date."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date

    daily = (
        out[
            [
                "date",
                "btc_open",
                "btc_high",
                "btc_low",
                "btc_close",
                "btc_volume",
                "fng_value",
                "fng_classification",
            ]
        ]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    base_market_cols = [
        "btc_open",
        "btc_high",
        "btc_low",
        "btc_close",
        "btc_volume",
        "fng_value",
        "fng_classification",
    ]
    for col in base_market_cols:
        daily[f"{col}_lag1"] = daily[col].shift(1)

    daily["btc_return_lag1"] = (
        daily["btc_close_lag1"] / daily["btc_open_lag1"].replace(0, np.nan)
    ) - 1.0
    daily["btc_volatility_lag1"] = (
        daily["btc_high_lag1"] - daily["btc_low_lag1"]
    ) / daily["btc_open_lag1"].replace(0, np.nan)
    daily["btc_volume_7d_avg_lag1"] = (
        daily["btc_volume"].shift(1).rolling(window=7, min_periods=1).mean()
    )
    daily["btc_volume_change_vs_7d_lag1"] = (
        daily["btc_volume_lag1"] / daily["btc_volume_7d_avg_lag1"].replace(0, np.nan)
    ) - 1.0

    lagged_cols = [
        "date",
        "btc_open_lag1",
        "btc_high_lag1",
        "btc_low_lag1",
        "btc_close_lag1",
        "btc_volume_lag1",
        "fng_value_lag1",
        "fng_classification_lag1",
        "btc_return_lag1",
        "btc_volatility_lag1",
        "btc_volume_change_vs_7d_lag1",
    ]
    merged = out.merge(daily[lagged_cols], on="date", how="left")

    merged = merged.dropna(
        subset=[
            "text",
            "label",
            "btc_open_lag1",
            "btc_high_lag1",
            "btc_low_lag1",
            "btc_close_lag1",
            "btc_volume_lag1",
            "fng_value_lag1",
            "fng_classification_lag1",
            "btc_return_lag1",
            "btc_volatility_lag1",
            "btc_volume_change_vs_7d_lag1",
        ]
    ).copy()

    merged["label"] = merged["label"].astype(int)
    merged = merged[merged["label"].isin([0, 1])].copy()

    valid_splits = {"train", "val", "test"}
    found_splits = set(merged["split"].unique())
    if not found_splits.issubset(valid_splits):
        raise ValueError(
            f"Unexpected split values found: {sorted(found_splits - valid_splits)}"
        )

    LOGGER.info("Lagged dataset rows after dropna: %d", len(merged))
    return merged


def ensure_lagged(df: pd.DataFrame) -> pd.DataFrame:
    required_lagged_cols = {
        "btc_open_lag1",
        "btc_high_lag1",
        "btc_low_lag1",
        "btc_close_lag1",
        "btc_volume_lag1",
        "fng_value_lag1",
        "fng_classification_lag1",
        "btc_return_lag1",
        "btc_volatility_lag1",
        "btc_volume_change_vs_7d_lag1",
    }
    if not required_lagged_cols.issubset(set(df.columns)):
        return create_lagged_features(df)
    return df.dropna(subset=list(required_lagged_cols)).copy()


def stratified_sample_df(
    df: pd.DataFrame,
    sample_frac: float,
    seed: int,
    strata_cols: List[str],
    min_per_stratum: int,
) -> pd.DataFrame:
    """Sample rows per stratum to keep split/label distribution representative."""
    if sample_frac >= 1.0 and min_per_stratum <= 0:
        return df

    missing_cols = [c for c in strata_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Stratified sampling columns not found: {missing_cols}")

    sampled_groups = []
    grouped = df.groupby(strata_cols, dropna=False)
    for idx, (_, group) in enumerate(grouped):
        n = len(group)
        frac_target = int(np.ceil(n * sample_frac))
        target_n = max(1, frac_target, min_per_stratum)
        target_n = min(n, target_n)
        sampled_groups.append(group.sample(n=target_n, random_state=seed + idx))

    return pd.concat(sampled_groups, ignore_index=True)


@dataclass
class PreparedFeatures:
    X_train_num: np.ndarray
    X_val_num: np.ndarray
    X_test_num: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    scaler: StandardScaler
    encoder: OneHotEncoder
    numeric_feature_names: List[str]
    categorical_feature_names: List[str]


def _build_onehot_encoder() -> OneHotEncoder:
    # Handle both newer and older sklearn versions.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_numeric_features(df: pd.DataFrame) -> PreparedFeatures:
    """Build train/val/test numeric matrices without leakage."""
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more splits are empty after lagged feature creation.")

    numeric_feature_names = [
        "btc_open_lag1",
        "btc_high_lag1",
        "btc_low_lag1",
        "btc_close_lag1",
        "btc_volume_lag1",
        "fng_value_lag1",
        "btc_return_lag1",
        "btc_volatility_lag1",
        "btc_volume_change_vs_7d_lag1",
    ]

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[numeric_feature_names])
    X_val_num = scaler.transform(val_df[numeric_feature_names])
    X_test_num = scaler.transform(test_df[numeric_feature_names])

    encoder = _build_onehot_encoder()
    train_cat = encoder.fit_transform(train_df[["fng_classification_lag1"]])
    val_cat = encoder.transform(val_df[["fng_classification_lag1"]])
    test_cat = encoder.transform(test_df[["fng_classification_lag1"]])

    cat_names = encoder.get_feature_names_out(["fng_classification_lag1"]).tolist()

    X_train_num = np.hstack([X_train_num, train_cat]).astype(np.float32)
    X_val_num = np.hstack([X_val_num, val_cat]).astype(np.float32)
    X_test_num = np.hstack([X_test_num, test_cat]).astype(np.float32)

    y_train = train_df["label"].to_numpy(dtype=np.int64)
    y_val = val_df["label"].to_numpy(dtype=np.int64)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    return PreparedFeatures(
        X_train_num=X_train_num,
        X_val_num=X_val_num,
        X_test_num=X_test_num,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        scaler=scaler,
        encoder=encoder,
        numeric_feature_names=numeric_feature_names,
        categorical_feature_names=cat_names,
    )
