set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

example-config:
  cargo run -p threebody-cli -- example-config --out config.json

example-ic:
  cargo run -p threebody-cli -- example-ic --preset three-body --out ic.json

simulate:
  cargo run -p threebody-cli -- simulate --config config.json --output traj.csv --steps 100 --dt 0.01

simulate-ic:
  cargo run -p threebody-cli -- simulate --config config.json --ic ic.json --output traj.csv --steps 100 --dt 0.01

simulate-em:
  cargo run -p threebody-cli -- simulate --config config.json --output traj_em.csv --steps 100 --dt 0.01 --em

discover:
  cargo run -p threebody-cli -- discover --solver stls --out top_equations.json

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
