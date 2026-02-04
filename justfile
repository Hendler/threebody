set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

example-config:
  cargo run -p threebody-cli -- example-config --out config.json

simulate:
  cargo run -p threebody-cli -- simulate --config config.json --output traj.csv --steps 100 --dt 0.01

simulate-em:
  cargo run -p threebody-cli -- simulate --config config.json --output traj_em.csv --steps 100 --dt 0.01

discover:
  cargo run -p threebody-cli -- discover --runs 50 --population 20 --out top_equations.json

test:
  cargo test

bench:
  cargo bench
