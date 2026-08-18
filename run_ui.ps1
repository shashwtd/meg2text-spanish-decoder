$ErrorActionPreference = "Stop"

if (-not (Test-Path .\.venv\Scripts\spanish-meg-web.exe)) {
    throw "Run .\setup.ps1 first."
}

& .\.venv\Scripts\spanish-meg-web.exe

