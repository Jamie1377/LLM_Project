from pathlib import Path

from hf_playground_inference import (
    PlaygroundPredictor,
    infer_required_text_dim,
    pick_model_from_required_dim,
    resolve_default_model_name,
)


def build_predictor() -> PlaygroundPredictor:
    artifacts_dir = Path("outputs_compare_models")
    metrics_model = resolve_default_model_name(
        artifacts_dir=artifacts_dir,
        fallback="boltuix/bert-lite",
    )
    required_text_dim = infer_required_text_dim(artifacts_dir)
    model_name = pick_model_from_required_dim(required_text_dim, metrics_model)

    return PlaygroundPredictor(
        artifacts_dir=artifacts_dir,
        model_name=model_name,
        tokenizer_path=None,
        max_length=96,
        batch_size=32,
    )


def main() -> None:
    predictor = build_predictor()
    app = __import__("hf_playground_inference").create_gradio_app(predictor)
    app.launch()


if __name__ == "__main__":
    main()
