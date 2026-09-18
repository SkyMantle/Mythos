#!/usr/bin/env bash
# Точка входу інсталятора з архіву: sudo bash install.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/deploy/install_pi.sh" "$@"
