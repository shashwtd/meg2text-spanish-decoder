"""Command-line and Python inference for preprocessed Spanish MEG windows."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch

from .architecture import build_model


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PACKAGE_ROOT / "models" / "spanish_meg_ctc_v2.pt"
ALPHABET = "-abcdefghijklmnopqrstuvwxyz "


def character_error_rate(target: str, prediction: str) -> float:
    if not target:
        return float(bool(prediction))
    previous = list(range(len(prediction) + 1))
    for row, target_char in enumerate(target, start=1):
        current = [row]
        for column, predicted_char in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (target_char != predicted_char),
                )
            )
        previous = current
    return previous[-1] / len(target)


def decode_ctc(logits: torch.Tensor, length: int) -> str:
    ids = logits[:length].argmax(dim=-1).detach().cpu().tolist()
    output: list[str] = []
    previous = 0
    for value in ids:
        if value != previous and 0 < value < len(ALPHABET):
            output.append(ALPHABET[value])
        previous = value
    return "".join(output)


def load_npz(source: Path | str | bytes | BinaryIO) -> dict[str, Any]:
    if isinstance(source, bytes):
        handle: Any = BytesIO(source)
    else:
        handle = source
    with np.load(handle, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    neuros = np.asarray(data.get("neuros"), dtype=np.float32)
    chan_pos = np.asarray(data.get("chan_pos"), dtype=np.float32)
    if neuros.ndim != 2 or neuros.shape[1] != 306:
        raise ValueError("neuros must have shape T × 306")
    if chan_pos.shape != (306, 2):
        raise ValueError("chan_pos must have shape 306 × 2")
    if not np.isfinite(neuros).all() or not np.isfinite(chan_pos).all():
        raise ValueError("input contains non-finite values")
    day = int(np.asarray(data.get("day")).item())
    if not 0 <= day <= 18:
        raise ValueError("day must be a subject index from 0 to 18")
    return {
        "neuros": neuros,
        "chan_pos": chan_pos,
        "day": day,
        "target_text": str(np.asarray(data.get("target_text", "")).item()),
        "intended_text": str(np.asarray(data.get("intended_text", "")).item()),
        "sentence_uid": str(np.asarray(data.get("sentence_uid", "")).item()),
        "subject": str(np.asarray(data.get("subject", day)).item()),
    }


class Predictor:
    def __init__(self, model_path: Path = DEFAULT_MODEL, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(device)
        self.payload = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model: torch.nn.Module | None = None

    def _initialize(self, sample: dict[str, Any]) -> None:
        model = build_model(
            n_in_channels=int(self.payload["n_in_channels"]),
            n_outputs=int(self.payload["n_outputs"]),
        )
        neuros = torch.from_numpy(sample["neuros"]).unsqueeze(0)
        positions = torch.from_numpy(sample["chan_pos"]).unsqueeze(0)
        with warnings.catch_warnings(), torch.inference_mode():
            warnings.simplefilter("ignore", UserWarning)
            model(neuros, torch.tensor([18]), positions)
        state_dict = self.payload.pop("state_dict")
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        model.eval().to(self.device)
        self.model = model

    def predict(self, source: Path | str | bytes | BinaryIO) -> dict[str, Any]:
        sample = load_npz(source)
        if self.model is None:
            self._initialize(sample)
        assert self.model is not None
        neuros = torch.from_numpy(sample["neuros"]).unsqueeze(0).to(self.device)
        days = torch.tensor([sample["day"]], dtype=torch.long, device=self.device)
        positions = torch.from_numpy(sample["chan_pos"]).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            logits = self.model(neuros, days, positions)["c_out"][0]
        convolution = self.model.temporal_downsampling.agg
        length = (
            sample["neuros"].shape[0] - int(convolution.kernel_size[0])
        ) // int(convolution.stride[0]) + 1
        prediction = decode_ctc(logits, length)
        result: dict[str, Any] = {
            "prediction": prediction,
            "duration_seconds": sample["neuros"].shape[0] / 100.0,
            "subject": sample["subject"],
            "sentence_uid": sample["sentence_uid"],
            "inference_seconds": time.perf_counter() - started,
            "device": str(self.device),
        }
        if sample["target_text"]:
            result["target_text"] = sample["target_text"]
            result["cer"] = character_error_rate(sample["target_text"], prediction)
        if sample["intended_text"]:
            result["intended_text"] = sample["intended_text"]
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode a preprocessed MEG sentence")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = Predictor(args.model, args.device).predict(args.input)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
