# threebody-cli

Command-line interface for running the simulator.

**Examples**
```bash
# Generate an example config
cargo run -p threebody-cli -- example-config --out config.json

# Run a simulation
cargo run -p threebody-cli -- simulate --config config.json --output traj.csv --steps 100 --dt 0.01

# Run in truth mode (adaptive RK45)
cargo run -p threebody-cli -- simulate --config config.json --output traj_truth.csv --mode truth

# Run discovery (top 3 equations)
cargo run -p threebody-cli -- discover --runs 50 --population 20 --out top_equations.json

# Show help
cargo run -p threebody-cli -- --help
```

**Flags**
- `--preset` selects initial conditions (`two-body`, `static`).
- `--integrator` overrides the integrator (`leapfrog`, `rk45`, `boris`, `implicit_midpoint`).
- `--no-em` and `--no-gravity` disable force components.
- `--dry-run` performs all steps without writing output files.
- `--mode truth` enables adaptive RK45 with strict tolerances.
- Default config uses adaptive RK45; `--mode standard` preserves user config without overrides.

**LLM**
Set `OPENAI_API_KEY` to enable LLM ranking in discovery (adds `grid_top3` ordering and interpretation):
```bash
export OPENAI_API_KEY="your_key_here"
cargo run -p threebody-cli -- discover --llm --model gpt-5
```
