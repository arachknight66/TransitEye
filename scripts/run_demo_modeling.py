"""Execute and replay the frozen B040--B045 development-demo protocol."""

from pathlib import Path

from transiteye.modeling import run_demo_modeling


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = run_demo_modeling(root)
    print(f"model_version={result.model.model_version}")
    print(f"selected_model={result.model.classifier.family}")
    print(f"threshold={result.model.threshold:.17g}")
    print(f"scientific_inference_rows={len(result.scientific_predictions)}")


if __name__ == "__main__":
    main()
