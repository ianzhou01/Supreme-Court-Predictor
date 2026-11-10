#!/usr/bin/env bash
# macOS setup. Run from the repo folder:  bash setup.sh
# Safe to re-run: it rebuilds .venv from requirements.txt every time.
set -e
cd "$(dirname "$0")"

# 1. Find Python 3.13
if ! command -v python3.13 >/dev/null 2>&1; then
  echo "Python 3.13 not found. Install it from https://www.python.org/downloads/ (pick 3.13.x), then re-run."
  exit 1
fi

# 2. Intel Macs can't run current TensorFlow: install everything else
REQS=requirements.txt
CHECK_ARGS=""
if [ "$(uname)" = "Darwin" ] && [ "$(python3.13 -c 'import platform; print(platform.machine())')" = "x86_64" ]; then
  echo "Intel Mac detected: TensorFlow doesn't support it, so it will be skipped."
  echo "Everything else works. Leave the Keras tasks to a teammate (see SETUP.md)."
  grep -v '^tensorflow' requirements.txt > .requirements-no-tf.txt
  REQS=.requirements-no-tf.txt
  CHECK_ARGS="--no-tf"
fi

# 3. Fresh virtual environment (deletes any old .venv, e.g. a 3.14 one)
rm -rf .venv
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r "$REQS"

# 4. Register the Jupyter kernel and check everything
.venv/bin/python -m ipykernel install --user --name scotus --display-name "Python (scotus)"
.venv/bin/python check_env.py $CHECK_ARGS