import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_ARTIFACTS_DIR = Path("outputs_compare_models")
NUMERIC_FEATURE_NAMES = [
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


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_default_model_name(artifacts_dir: Path, fallback: str) -> str:
    metrics_path = artifacts_dir / "metrics_xgb_cls_vs_numeric.json"
    if not metrics_path.exists():
        return fallback

    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        model_name = metrics.get("text_model")
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()
    except Exception:
        pass

    return fallback


def infer_required_text_dim(artifacts_dir: Path) -> Optional[int]:
    model_path = artifacts_dir / "xgb_model.joblib"
    scaler_path = artifacts_dir / "numeric_scaler.joblib"
    encoder_path = artifacts_dir / "fng_onehot_encoder.joblib"
    if not model_path.exists() or not scaler_path.exists() or not encoder_path.exists():
        return None

    try:
        xgb_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        total_dim = getattr(xgb_model, "n_features_in_", None)
        num_dim = int(getattr(scaler, "n_features_in_", 0))
        cat_dim = int(sum(len(c) for c in getattr(encoder, "categories_", [])))
        if total_dim is None:
            return None
        text_dim = int(total_dim) - num_dim - cat_dim
        if text_dim <= 0:
            return None
        return text_dim
    except Exception:
        return None


def pick_model_from_required_dim(
    required_text_dim: Optional[int],
    metrics_model_name: str,
) -> str:
    if required_text_dim == 768:
        return "ProsusAI/finbert"
    if required_text_dim == 256:
        return "boltuix/bert-lite"
    return metrics_model_name


class PlaygroundPredictor:
    def __init__(
        self,
        artifacts_dir: Path,
        model_name: str,
        tokenizer_path: Optional[str],
        max_length: int,
        batch_size: int,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = pick_device()

        model_path = artifacts_dir / "xgb_model.joblib"
        scaler_path = artifacts_dir / "numeric_scaler.joblib"
        encoder_path = artifacts_dir / "fng_onehot_encoder.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler artifact: {scaler_path}")
        if not encoder_path.exists():
            raise FileNotFoundError(f"Missing encoder artifact: {encoder_path}")

        self.xgb_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)

        tokenizer_source: str
        if tokenizer_path and Path(tokenizer_path).exists():
            tokenizer_source = tokenizer_path
        elif (artifacts_dir / "tokenizer_config.json").exists() or (
            artifacts_dir / "tokenizer.json"
        ).exists():
            tokenizer_source = str(artifacts_dir)
        else:
            tokenizer_source = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        self.text_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.text_model.eval()

        self.expected_feature_count = getattr(self.xgb_model, "n_features_in_", None)

    @staticmethod
    def _safe_float(value: object, default: float) -> float:
        if value is None:
            return float(default)
        try:
            if isinstance(value, str) and not value.strip():
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def _normalize_row(self, row: Dict[str, object]) -> Dict[str, object]:
        # Allow either user-friendly now-values or explicit lagged values.
        btc_price_now = self._safe_float(row.get("btc_price_now"), 0.0)
        btc_open = self._safe_float(row.get("btc_open_lag1"), btc_price_now)
        btc_high = self._safe_float(row.get("btc_high_lag1"), btc_open)
        btc_low = self._safe_float(row.get("btc_low_lag1"), btc_open)
        btc_close = self._safe_float(row.get("btc_close_lag1"), btc_open)
        btc_volume = self._safe_float(row.get("btc_volume_lag1"), 1.0)
        fng_value = self._safe_float(
            row.get("fng_value_lag1", row.get("fng_value")),
            50.0,
        )

        # If optional engineered fields are missing, derive conservative defaults.
        btc_return = self._safe_float(
            row.get("btc_return_lag1"),
            (btc_close / btc_open - 1.0) if btc_open != 0 else 0.0,
        )
        btc_volatility = self._safe_float(
            row.get("btc_volatility_lag1"),
            ((btc_high - btc_low) / btc_open) if btc_open != 0 else 0.0,
        )
        volume_7d_avg = self._safe_float(row.get("btc_volume_7d_avg_lag1"), btc_volume)
        btc_volume_change = self._safe_float(
            row.get("btc_volume_change_vs_7d_lag1"),
            (btc_volume / volume_7d_avg - 1.0) if volume_7d_avg != 0 else 0.0,
        )

        fng_cls = (
            str(
                row.get(
                    "fng_classification_lag1",
                    row.get("fng_classification", "Neutral"),
                )
            )
            .strip()
            .title()
        )

        text = str(row.get("text", "")).strip()
        if not text:
            raise ValueError("Each row must include non-empty 'text'.")

        return {
            "text": text,
            "btc_open_lag1": btc_open,
            "btc_high_lag1": btc_high,
            "btc_low_lag1": btc_low,
            "btc_close_lag1": btc_close,
            "btc_volume_lag1": btc_volume,
            "fng_value_lag1": fng_value,
            "fng_classification_lag1": fng_cls,
            "btc_return_lag1": btc_return,
            "btc_volatility_lag1": btc_volatility,
            "btc_volume_change_vs_7d_lag1": btc_volume_change,
        }

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        embs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
                enc = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                outputs = self.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                cls_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embs.append(cls_vec)
        return np.vstack(embs).astype(np.float32)

    def _build_numeric_features(self, rows: List[Dict[str, object]]) -> np.ndarray:
        df = pd.DataFrame(rows)
        x_num = self.scaler.transform(df[NUMERIC_FEATURE_NAMES])
        x_cat = self.encoder.transform(df[["fng_classification_lag1"]])
        return np.hstack([x_num, x_cat]).astype(np.float32)

    def predict_rows(self, raw_rows: List[Dict[str, object]]) -> pd.DataFrame:
        normalized = [self._normalize_row(row) for row in raw_rows]
        texts = [r["text"] for r in normalized]

        x_text = self._embed_texts(texts)
        x_num = self._build_numeric_features(normalized)
        x_all = np.hstack([x_text, x_num]).astype(np.float32)

        if self.expected_feature_count is not None and x_all.shape[1] != int(
            self.expected_feature_count
        ):
            raise ValueError(
                "Feature mismatch: model expects "
                f"{self.expected_feature_count} columns but got {x_all.shape[1]}. "
                "Check model_name/tokenizer/artifacts consistency."
            )

        proba_up = self.xgb_model.predict_proba(x_all)[:, 1]
        pred = (proba_up >= 0.5).astype(int)
        confidence = np.maximum(proba_up, 1.0 - proba_up)
        signed_score = 2.0 * proba_up - 1.0

        out = pd.DataFrame(normalized)
        out["pred_class"] = pred
        out["sentiment"] = np.where(pred == 1, "Bullish", "Bearish")
        out["score"] = signed_score
        out["prob_up"] = proba_up
        out["confidence"] = confidence
        return out


def load_rows_from_input_file(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError("Supported input formats: .csv, .parquet, .pq, .json")

    return df.to_dict(orient="records")


def save_predictions(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(output_path, index=False)
    elif suffix == ".json":
        df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    else:
        raise ValueError("Supported output formats: .csv, .parquet, .pq, .json")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Playground inference for crypto-news sentiment using text + BTC + FNG. "
            "Supports single input, dataset batch, and optional Gradio UI."
        )
    )

    parser.add_argument("--artifacts_dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--model_name", type=str, default="auto")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)

    sub = parser.add_subparsers(dest="mode", required=True)

    single = sub.add_parser("single", help="Predict one news row.")
    single.add_argument("--text", type=str, required=True)
    single.add_argument("--btc_price_now", type=float, default=0.0)
    single.add_argument("--fng_value", type=float, required=True)
    single.add_argument("--fng_classification", type=str, required=True)
    single.add_argument("--btc_open_lag1", type=float)
    single.add_argument("--btc_high_lag1", type=float)
    single.add_argument("--btc_low_lag1", type=float)
    single.add_argument("--btc_close_lag1", type=float)
    single.add_argument("--btc_volume_lag1", type=float)
    single.add_argument("--btc_return_lag1", type=float)
    single.add_argument("--btc_volatility_lag1", type=float)
    single.add_argument("--btc_volume_change_vs_7d_lag1", type=float)

    batch = sub.add_parser("batch", help="Predict a full dataset file.")
    batch.add_argument("--input_path", type=str, required=True)
    batch.add_argument("--output_path", type=str, required=True)

    ui = sub.add_parser("ui", help="Launch Gradio playground UI.")
    ui.add_argument("--host", type=str, default="0.0.0.0")
    ui.add_argument("--port", type=int, default=7860)
    ui.add_argument("--share", action="store_true")

    return parser


def run_single(args: argparse.Namespace, predictor: PlaygroundPredictor) -> None:
    raw_row = {
        "text": args.text,
        "btc_price_now": args.btc_price_now,
        "fng_value": args.fng_value,
        "fng_classification": args.fng_classification,
        "btc_open_lag1": args.btc_open_lag1,
        "btc_high_lag1": args.btc_high_lag1,
        "btc_low_lag1": args.btc_low_lag1,
        "btc_close_lag1": args.btc_close_lag1,
        "btc_volume_lag1": args.btc_volume_lag1,
        "btc_return_lag1": args.btc_return_lag1,
        "btc_volatility_lag1": args.btc_volatility_lag1,
        "btc_volume_change_vs_7d_lag1": args.btc_volume_change_vs_7d_lag1,
    }
    pred_df = predictor.predict_rows([raw_row])
    print(pred_df.to_json(orient="records", indent=2))


def run_batch(args: argparse.Namespace, predictor: PlaygroundPredictor) -> None:
    rows = load_rows_from_input_file(Path(args.input_path))
    pred_df = predictor.predict_rows(rows)
    save_predictions(pred_df, Path(args.output_path))
    print(f"Predictions saved: {args.output_path}")
    print(f"Rows processed: {len(pred_df)}")


def create_gradio_app(predictor: PlaygroundPredictor):
    try:
        import gradio as gr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "gradio is not installed. Run: pip install gradio"
        ) from exc

    def predict_one(
        text: str,
        btc_price_now: float,
        fng_value: float,
        fng_classification: str,
    ) -> Tuple[str, float, float, float]:
        rows = [
            {
                "text": text,
                "btc_price_now": btc_price_now,
                "fng_value": fng_value,
                "fng_classification": fng_classification,
            }
        ]
        out = predictor.predict_rows(rows).iloc[0]
        return (
            str(out["sentiment"]),
            float(out["score"]),
            float(out["confidence"]),
            float(out["prob_up"]),
        )

    with gr.Blocks(title="Crypto News Sentiment Playground") as demo:
        gr.Markdown("# Crypto News Sentiment Playground")
        gr.Markdown(
            "Enter a news snippet plus market context to get class, score, and confidence."
        )

        with gr.Row():
            text = gr.Textbox(
                label="News Text",
                lines=6,
                placeholder="Paste a crypto news piece here...",
            )

        with gr.Row():
            btc_price_now = gr.Number(label="BTC Price Now", value=70000)
            fng_value = gr.Number(label="FNG Index", value=50)
            fng_classification = gr.Dropdown(
                choices=["Extreme Fear", "Fear", "Neutral", "Greed"],
                value="Neutral",
                label="FNG Classification",
            )

        run_btn = gr.Button("Generate Sentiment")

        sentiment = gr.Textbox(label="Sentiment")
        score = gr.Number(label="Score (-1 to +1)")
        confidence = gr.Number(label="Confidence (0 to 1)")
        prob_up = gr.Number(label="Probability of Up Move")

        run_btn.click(
            fn=predict_one,
            inputs=[text, btc_price_now, fng_value, fng_classification],
            outputs=[sentiment, score, confidence, prob_up],
        )

    return demo


def run_ui(args: argparse.Namespace, predictor: PlaygroundPredictor) -> None:
    app = create_gradio_app(predictor)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    model_name = args.model_name
    if model_name == "auto":
        metrics_model = resolve_default_model_name(
            artifacts_dir=artifacts_dir,
            fallback="boltuix/bert-lite",
        )
        required_text_dim = infer_required_text_dim(artifacts_dir)
        model_name = pick_model_from_required_dim(required_text_dim, metrics_model)

    tokenizer_path = args.tokenizer_path.strip() or None

    predictor = PlaygroundPredictor(
        artifacts_dir=artifacts_dir,
        model_name=model_name,
        tokenizer_path=tokenizer_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    if args.mode == "single":
        run_single(args, predictor)
    elif args.mode == "batch":
        run_batch(args, predictor)
    elif args.mode == "ui":
        run_ui(args, predictor)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
