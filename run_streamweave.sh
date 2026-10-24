#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if ! command -v make >/dev/null 2>&1; then
  echo "GNU Make was not found. Install it with your Fedora package manager, then rerun this script." >&2
  exit 127
fi
exec make all "$@"
