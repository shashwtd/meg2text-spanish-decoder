from pathlib import Path

import torch

from spanish_meg_decoder.predict import DEFAULT_MODEL, Predictor


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    examples = sorted((ROOT / "test_cases" / "inputs").glob("*.npz"))
    assert DEFAULT_MODEL.is_file(), f"Missing {DEFAULT_MODEL}"
    assert examples, "No test examples found"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = Predictor(device=device).predict(examples[0])
    assert isinstance(result["prediction"], str)
    assert result["target_text"]
    print(result)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()

