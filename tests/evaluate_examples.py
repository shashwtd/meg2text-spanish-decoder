import json
from pathlib import Path

import torch

from spanish_meg_decoder.predict import Predictor


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = Predictor(device=device)
    results = [
        predictor.predict(path)
        for path in sorted((ROOT / "test_cases" / "inputs").glob("*.npz"))
    ]
    output = ROOT / "test_cases" / "results.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Examples: {len(results)}")
    print(f"Mean CER: {sum(row['cer'] for row in results) / len(results):.4f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

