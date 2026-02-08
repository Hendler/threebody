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

# Run one factory iteration (OpenAI LLM, uses .openai_key if present)
cargo run -p threebody-cli -- factory --max-iters 1 --auto --llm-mode openai --model gpt-5.2

# Or override with a key file (ignores OPENAI_API_KEY and .openai_key)
cargo run -p threebody-cli -- factory --max-iters 1 --auto --llm-mode openai --model gpt-5.2 --openai-key-file .openai_key

# Show help
cargo run -p threebody-cli -- --help

# Predictability Takens with symbolic + neural residual candidates (delta_mlp)
cargo run -p threebody-cli -- predictability takens --input traj.csv --column a1_x --sensors "a1_x,r1_x,r1_y,r1_z,r2_x,r2_y,r2_z,r3_x,r3_y,r3_z,v1_x,v1_y,v1_z,v2_x,v2_y,v2_z,v3_x,v3_y,v3_z" --tau 1 --m 2,4 --k 8 --lambda 1e-6 --model all --split-mode chronological --out takens_a1x_all.json

# Neural-only ablation:
# cargo run -p threebody-cli -- predictability takens --input traj.csv --column a1_x --sensors "a1_x,r1_x,r1_y,r1_z,r2_x,r2_y,r2_z,r3_x,r3_y,r3_z,v1_x,v1_y,v1_z,v2_x,v2_y,v2_z,v3_x,v3_y,v3_z" --tau 1 --m 2,4 --k 8 --lambda 1e-6 --model delta_mlp --split-mode chronological --out takens_a1x_nn.json
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
If `.openai_key` exists in the working directory, it is used by default for OpenAI mode.
Create the key file with: `echo "sk-your-key" > .openai_key`.
Otherwise, set `OPENAI_API_KEY` to enable LLM judge mode for discovery or the factory loop:
```bash
export OPENAI_API_KEY="your_key_here"
cargo run -p threebody-cli -- discover --llm --model gpt-5.2
cargo run -p threebody-cli -- factory --llm-mode openai --model gpt-5.2 --max-iters 1 --auto
```
Or override with a key file:
```bash
cargo run -p threebody-cli -- discover --llm --model gpt-5.2 --openai-key-file .openai_key
```
The LLM judge receives equation forms plus numeric metrics (MSE, rollout RMSE, divergence time) and emits JSON-only scorecards.
It may recommend the next rollout evaluator (`euler` or `leapfrog`) and GA fitness heuristic (`mse` or `mse_parsimony`).
