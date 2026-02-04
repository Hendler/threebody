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
  cargo run -p threebody-cli -- discover --runs 50 --population 20 --out top_equations.json

test:
  cargo test

bench:
  cargo bench

paper:
  if command -v pdflatex >/dev/null 2>&1; then
    PDFLATEX="pdflatex"
  elif [ -x "/Library/TeX/texbin/pdflatex" ]; then
    PDFLATEX="/Library/TeX/texbin/pdflatex"
  else
    echo "pdflatex not found. Install BasicTeX or TeX Live." >&2
    echo "macOS: brew install --cask basictex" >&2
    exit 1
  fi
  "$PDFLATEX" -interaction=nonstopmode academic_paper.tex
