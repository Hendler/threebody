#!/usr/bin/env bash
set -euo pipefail

ts="$(date +%Y%m%d_%H%M%S)"
out_dir="results/autoresearch_${ts}"

cargo run -p threebody-cli -- quickstart --profile autoresearch --out-dir "$out_dir" "$@"

score_tsv="${out_dir}/autoresearch_score.tsv"
score_txt="${out_dir}/autoresearch_score.txt"

if [[ ! -f "${score_tsv}" ]]; then
  echo "missing score file: ${score_tsv}" >&2
  exit 1
fi

if [[ ! -f results.tsv ]]; then
  head -n 1 "${score_tsv}" > results.tsv
fi
tail -n +2 "${score_tsv}" >> results.tsv

echo "out_dir=${out_dir}"
cat "${score_txt}"
