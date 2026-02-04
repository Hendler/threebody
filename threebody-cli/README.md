# threebody-cli

Command-line interface for running the simulator.

**Examples**
```bash
# Generate an example config
cargo run -p threebody-cli -- example-config --out config.json

# Run a simulation
cargo run -p threebody-cli -- simulate --config config.json --output traj.csv --steps 100 --dt 0.01

# Show help
cargo run -p threebody-cli -- --help
```

**Flags**
- `--preset` selects initial conditions (`two-body`, `static`).
- `--integrator` overrides the integrator (`leapfrog`, `rk45`, `boris`, `implicit_midpoint`).
- `--no-em` and `--no-gravity` disable force components.
- `--dry-run` performs all steps without writing output files.
