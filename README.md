# Spanish MEG decoder

This model decodes a continuous Spanish MEG sentence window into text. It uses a 352-million-parameter convolutional Conformer with a CTC output head.

Final held-out CER: **0.603 ± 0.014**. The paper reports **0.59 ± 0.02** for the comparable Spanish asynchronous encoder.

## Install

Open PowerShell in this directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

## Run the web UI

```powershell
.\run_ui.ps1
```

The browser opens at `http://127.0.0.1:7861`. Pick one of the included held-out examples and select **Decode**.

## Run from the command line

```powershell
& .\.venv\Scripts\spanish-meg-predict.exe `
  --input .\test_cases\inputs\example_01.npz `
  --device cuda
```

## Input

The predictor accepts a preprocessed `.npz` containing:

- `neuros`: `T × 306` MEG samples at 100 Hz
- `day`: subject index from 0 to 18
- `chan_pos`: `306 × 2` sensor positions
- optional `target_text` and metadata

This model does not accept MRI, DICOM, EEG, or arbitrary MEG sensor layouts.

## Structure

```text
models/                 Inference-only model
src/spanish_meg_decoder Model architecture, predictor, and web UI
test_cases/inputs/      Nine held-out examples
tests/                  Package checks
metadata/               Metrics and export details
```

The large files use Git LFS. The model is derived from Meta's Brain2Qwerty code and is for non-commercial use under CC BY-NC 4.0.

To push the model with the repository, install Git LFS first and then run `git lfs install`. Alternatively, upload `models/spanish_meg_ctc_v2.pt` as a GitHub Release asset.
