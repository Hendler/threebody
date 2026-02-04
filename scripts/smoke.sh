#!/usr/bin/env bash
set -euo pipefail

if ! command -v just >/dev/null 2>&1; then
  echo "just is required for smoke tests" >&2
  exit 1
fi

just example-config
just simulate
just test
