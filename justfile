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

quickstart:
  #!/usr/bin/env bash
  set -euo pipefail
  ts="$(date +%Y%m%d_%H%M%S)"
  out="results/quickstart_${ts}"
  mkdir -p "$out"

  cargo run -p threebody-cli -- example-config --out "$out/config.json"
  cargo run -p threebody-cli -- example-ic --preset three-body --out "$out/ic.json"
  cargo run -p threebody-cli -- simulate --config "$out/config.json" --ic "$out/ic.json" --output "$out/traj.csv" --steps 200 --dt 0.01
  cargo run -p threebody-cli -- discover --solver stls --input "$out/traj.csv" --sidecar "$out/traj.json" --out "$out/top_equations.json"

  factory_dir="$out/factory"
  if [ -f .openai_key ] || [ -n "${OPENAI_API_KEY:-}" ]; then llm_mode=openai; else llm_mode=mock; fi
  echo "Running factory (10 iters): out_dir=$factory_dir llm_mode=$llm_mode"
  cargo run -p threebody-cli -- factory --out-dir "$factory_dir" --max-iters 10 --auto --config "$out/config.json" --steps 200 --dt 0.01 --llm-mode "$llm_mode" --model gpt-5 --solver stls --rollout-integrator euler --fitness mse

  if [ -f "$factory_dir/evaluation.md" ]; then cp "$factory_dir/evaluation.md" "$out/evaluation.md"; fi

  echo "Quickstart complete: $out"
  echo "Key outputs:"
  echo "- $out/top_equations.json"
  echo "- $out/evaluation.md"

test:
  cargo test

bench:
  cargo bench

paper:
  if command -v pdflatex >/dev/null 2>&1; then PDFLATEX="pdflatex"; elif [ -x "/Library/TeX/texbin/pdflatex" ]; then PDFLATEX="/Library/TeX/texbin/pdflatex"; else echo "pdflatex not found. Install BasicTeX or TeX Live." >&2; echo "macOS: brew install --cask basictex" >&2; exit 1; fi; "$PDFLATEX" -interaction=nonstopmode academic_paper.tex
