$ErrorActionPreference = "Stop"

py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -m pip install `
  torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 `
  --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Ready. Run: & .\.venv\Scripts\spanish-meg-web.exe"
