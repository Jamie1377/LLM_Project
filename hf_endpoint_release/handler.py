from pathlib import Path
from typing import Any, Dict, List

from hf_playground_inference import (
    PlaygroundPredictor,
    infer_required_text_dim,
    pick_model_from_required_dim,
    resolve_default_model_name,
)


class EndpointHandler:
    def __init__(self, path: str = "") -> None:
        root = Path(path) if path else Path(".")
        artifacts_dir = root / "outputs_compare_models"

        metrics_model = resolve_default_model_name(
            artifacts_dir=artifacts_dir,
            fallback="boltuix/bert-lite",
        )
        required_text_dim = infer_required_text_dim(artifacts_dir)
        model_name = pick_model_from_required_dim(required_text_dim, metrics_model)

        self.predictor = PlaygroundPredictor(
            artifacts_dir=artifacts_dir,
            model_name=model_name,
            tokenizer_path=str(artifacts_dir),
            max_length=96,
            batch_size=32,
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Expected shape: {"inputs": {...single row...}} or {"inputs": [{...}, {...}]}
        payload = data.get("inputs", data)

        if isinstance(payload, dict):
            rows: List[Dict[str, Any]] = [payload]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError("Payload must be a dict row or a list of row dicts.")

        pred_df = self.predictor.predict_rows(rows)
        return {
            "predictions": pred_df.to_dict(orient="records"),
            "count": int(len(pred_df)),
        }
