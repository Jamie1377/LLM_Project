from typing import Dict

import numpy as np
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from xgboost import XGBClassifier

from pipeline_common import LOGGER, evaluate


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    use_random_search: bool = False,
    random_search_iters: int = 20,
    refit_on_train_val: bool = False,
    random_state: int = 42,
    xgb_n_jobs: int = -1,
) -> Dict[str, Dict[str, float]]:
    """Train XGBoost classifier and return validation/test metrics."""
    LOGGER.info("Training XGBoost model.")

    base_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": xgb_n_jobs,
    }

    best_params = dict(base_params)
    if use_random_search:
        LOGGER.info(
            "Running RandomizedSearchCV on validation split | iters=%d",
            random_search_iters,
        )
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])

        val_fold = np.concatenate(
            [
                np.full(X_train.shape[0], -1, dtype=int),
                np.zeros(X_val.shape[0], dtype=int),
            ]
        )
        split = PredefinedSplit(test_fold=val_fold)

        param_dist = {
            "n_estimators": sorted(
                set([max(50, n_estimators // 2), n_estimators, int(n_estimators * 1.5)])
            ),
            "max_depth": sorted(set([max(2, max_depth - 2), max_depth, max_depth + 2])),
            "learning_rate": sorted(
                set(
                    [
                        max(0.01, learning_rate / 2),
                        learning_rate,
                        min(0.3, learning_rate * 2),
                    ]
                )
            ),
            "subsample": sorted(
                set([max(0.6, subsample - 0.2), subsample, min(1.0, subsample + 0.1)])
            ),
            "colsample_bytree": sorted(
                set(
                    [
                        max(0.6, colsample_bytree - 0.2),
                        colsample_bytree,
                        min(1.0, colsample_bytree + 0.1),
                    ]
                )
            ),
            "min_child_weight": [1, 3, 5],
            "gamma": [0.0, 0.1, 0.3],
            "reg_lambda": [1.0, 3.0, 10.0],
        }

        search = RandomizedSearchCV(
            estimator=XGBClassifier(**base_params),
            param_distributions=param_dist,
            n_iter=random_search_iters,
            scoring="f1",
            cv=split,
            random_state=random_state,
            n_jobs=xgb_n_jobs,
            verbose=0,
            refit=False,
        )
        search.fit(X_train_val, y_train_val)

        best_params.update(search.best_params_)
        LOGGER.info(
            "Best XGBoost params from validation search: %s", search.best_params_
        )

    val_model = XGBClassifier(**best_params)
    val_model.fit(X_train, y_train)

    val_pred = val_model.predict(X_val)

    final_model = val_model
    if refit_on_train_val:
        LOGGER.info("Refitting XGBoost on train+val using selected hyperparameters.")
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])
        final_model = XGBClassifier(**best_params)
        final_model.fit(X_train_val, y_train_val)

    test_pred = final_model.predict(X_test)

    return {
        "val": evaluate(y_val, val_pred),
        "test": evaluate(y_test, test_pred),
        "model": final_model,
    }
