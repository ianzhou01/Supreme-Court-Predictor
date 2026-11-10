# Windows setup. Run from the repo folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
# Safe to re-run: it rebuilds .venv from requirements.txt every time.
Set-Location $PSScriptRoot

function Fail($msg) { Write-Host "`n$msg" -ForegroundColor Red; exit 1 }

# 1. Find Python 3.13 (via the "py" launcher that python.org installs)
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Fail "The 'py' launcher isn't installed. Install Python 3.13 from https://www.python.org/downloads/ and re-run."
}
$null = cmd /c "py -3.13 --version 2>&1"
if ($LASTEXITCODE -ne 0) {
    Fail "Python 3.13 not found. Install it from https://www.python.org/downloads/ (pick 3.13.x), then re-run."
}

# 2. Fresh virtual environment (deletes any old .venv, e.g. a 3.14 one)
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
py -3.13 -m venv .venv
if ($LASTEXITCODE -ne 0) { Fail "Couldn't create .venv." }
$py = ".\.venv\Scripts\python.exe"

& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Fail "pip install failed. See Troubleshooting in SETUP.md, or post the error in the group chat."
}

# 3. Register the Jupyter kernel and check everything
& $py -m ipykernel install --user --name scotus --display-name "Python (scotus)"
& $py check_env.py
exit $LASTEXITCODE