#!/usr/bin/env bash
# Idempotent Cloud Agent setup for fpvscan.
# Prepares a Python virtualenv with the project dependencies. The bladeRF
# hardware driver is not usable in the cloud VM, so development and testing
# run against the built-in "sim" and "file" IQ sources.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$REPO_ROOT/fpvscan"

cd "$APP_DIR"

# The default image ships Python 3.12 but not the venv/ensurepip package.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3-venv python3-dev build-essential
fi

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo "fpvscan environment ready: $(./.venv/bin/python --version)"
