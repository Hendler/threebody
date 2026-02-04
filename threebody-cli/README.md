# threebody-cli

Command-line interface for running the simulator.

**Examples**
```bash
# Generate an example config
cargo run -p threebody-cli -- example-config --out config.json

# Generate example initial conditions (recommended for experiments)
cargo run -p threebody-cli -- example-ic --preset three-body --out ic.json

# Run a simulation (gravity-only by default)
cargo run -p threebody-cli -- simulate --config config.json --ic ic.json --output traj.csv --steps 100 --dt 0.01

# Enable EM explicitly
cargo run -p threebody-cli -- simulate --config config.json --ic ic.json --output traj_em.csv --steps 100 --dt 0.01 --em

# Run in truth mode (adaptive RK45)
cargo run -p threebody-cli -- simulate --config config.json --ic ic.json --output traj_truth.csv --mode truth

# Run discovery from traj.csv/traj.json in the current directory (top 3 vector candidates)
cargo run -p threebody-cli -- discover --runs 50 --population 20 --out top_equations.json

# Or pass explicit paths:
# cargo run -p threebody-cli -- discover --input run.csv --sidecar run.json --out top_equations.json

# Run one experiment iteration (mock LLM, no API key)
cargo run -p threebody-cli -- factory --max-iters 1 --auto --llm-mode mock

# Run one factory iteration (OpenAI LLM)
export OPENAI_API_KEY="your_key_here"
cargo run -p threebody-cli -- factory --max-iters 1 --auto --llm-mode openai --model gpt-5

# Show help
cargo run -p threebody-cli -- --help
```

**Flags**
- `example-ic` emits an IC JSON; `--ic` loads an IC JSON for `simulate`/`run`.
- `--preset` selects built-in initial conditions (`two-body`, `three-body`, `static`).
- `--em` enables quasi-static EM (off by default); `--no-em` disables it explicitly.
- `--integrator` overrides the integrator (`leapfrog`, `rk45`, `boris`, `implicit_midpoint`).
- `--no-gravity` disables gravity.
- `--dry-run` performs all steps without writing output files.
- `--mode truth` enables adaptive RK45 with strict tolerances.
- Default config uses adaptive RK45; `--mode standard` preserves user config without overrides.
- `--fitness` selects GA fitness heuristic (`mse`, `mse_parsimony`).
- `--rollout-integrator` selects the rollout evaluator (`euler`, `leapfrog`) for scoring.

**LLM**
Set `OPENAI_API_KEY` to enable LLM judge mode for discovery or the factory loop:
```bash
export OPENAI_API_KEY="your_key_here"
cargo run -p threebody-cli -- discover --llm --model gpt-5
cargo run -p threebody-cli -- factory --llm-mode openai --model gpt-5 --max-iters 1 --auto
```
The LLM judge receives equation forms plus numeric metrics (MSE, rollout RMSE, divergence time) and emits JSON-only scorecards.
It may recommend the next rollout evaluator (`euler` or `leapfrog`) and GA fitness heuristic (`mse` or `mse_parsimony`).
