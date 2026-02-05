set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

example-config:
  mkdir -p results/manual
  cargo run -p threebody-cli -- example-config --out results/manual/config.json

example-ic:
  mkdir -p results/manual
  cargo run -p threebody-cli -- example-ic --preset three-body --out results/manual/ic.json

simulate:
  mkdir -p results/manual
  cargo run -p threebody-cli -- simulate --config results/manual/config.json --output results/manual/traj.csv --steps 100 --dt 0.01

simulate-ic:
  mkdir -p results/manual
  cargo run -p threebody-cli -- simulate --config results/manual/config.json --ic results/manual/ic.json --output results/manual/traj.csv --steps 100 --dt 0.01

simulate-em:
  mkdir -p results/manual
  cargo run -p threebody-cli -- simulate --config results/manual/config.json --output results/manual/traj_em.csv --steps 100 --dt 0.01 --em

discover:
  mkdir -p results/manual
  cargo run -p threebody-cli -- discover --solver stls --input results/manual/traj.csv --sidecar results/manual/traj.json --out results/manual/top_equations.json

quickstart steps="200":
  #!/usr/bin/env bash
  set -euo pipefail
  ts="$(date +%Y%m%d_%H%M%S)"
  out="results/quickstart_${ts}"
  mkdir -p "$out"

  echo "Preflight: checking LLM connectivity (fails fast if misconfigured)"
  cargo run -p threebody-cli -- llm-check

  echo "Running quickstart: out_dir=$out steps={{steps}} max_iters=10 llm_mode=auto require_llm=true"
  cargo run -p threebody-cli -- quickstart --out-dir "$out" --steps "{{steps}}" --max-iters 10 --require-llm

  echo "Quickstart complete: $out"

quickstart10 steps="200":
  #!/usr/bin/env bash
  set -euo pipefail
  steps="{{steps}}"

  echo "Running 10 quickstarts: steps=${steps} max_iters=10 llm_mode=auto"
  echo "Preflight: checking LLM connectivity (fails fast if misconfigured)"
  cargo run -p threebody-cli -- llm-check
  for i in $(seq 1 10); do
    ts="$(date +%Y%m%d_%H%M%S)"
    out="results/quickstart_${ts}_${i}"
    echo "Run $i/10: out_dir=$out"
    cargo run -p threebody-cli -- quickstart --out-dir "$out" --steps "$steps" --max-iters 10 --require-llm
  done

  cargo run -p threebody-cli -- findings --results-dir results --out-tex results/findings.tex

findings:
  cargo run -p threebody-cli -- findings --results-dir results --out-tex results/findings.tex

llm-check:
  cargo run -p threebody-cli -- llm-check

test:
  cargo test

bench:
  cargo bench

paper:
  if command -v pdflatex >/dev/null 2>&1; then PDFLATEX="pdflatex"; elif [ -x "/Library/TeX/texbin/pdflatex" ]; then PDFLATEX="/Library/TeX/texbin/pdflatex"; else echo "pdflatex not found. Install BasicTeX or TeX Live." >&2; echo "macOS: brew install --cask basictex" >&2; exit 1; fi; "$PDFLATEX" -interaction=nonstopmode academic_paper.tex
