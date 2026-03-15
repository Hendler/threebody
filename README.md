# threebody

`threebody` is a Rust workspace for simulating three-body systems and testing whether simple, interpretable models can predict their behavior.

The project has two main parts:
- A numerical simulator for three interacting bodies with Newtonian gravity and optional electromagnetic models (`quasistatic` by default, plus a low-velocity `darwin` finite-`c` correction).
- A discovery pipeline that learns sparse equations from simulator output and checks them by forward rollout.

This is research code, not a general-purpose physics engine. The current code is built around exactly three bodies, defaults to gravity-only runs, and treats EM as an approximation rather than full electrodynamics.

## Workspace

- `threebody-core`: simulation library, force models, integrators, diagnostics, CSV output, and JSON sidecars.
- `threebody-cli`: command-line interface for generating configs, running simulations, discovery, and experiment workflows.
- `threebody-discover`: sparse equation search and scoring tools, including STLS, LASSO, and a GA baseline.

## Typical workflow

1. Generate a config and initial conditions.
2. Run a simulation to produce `traj.csv` and `traj.json`.
3. Run discovery on that trajectory.
4. Optionally explore predictability or the LLM-assisted factory loop.

## Quick start

Requires Rust and Cargo.

```bash
mkdir -p results/manual

cargo run -p threebody-cli -- example-config --out results/manual/config.json
cargo run -p threebody-cli -- example-ic --preset three-body --out results/manual/ic.json

cargo run -p threebody-cli -- simulate \
  --config results/manual/config.json \
  --ic results/manual/ic.json \
  --output results/manual/traj.csv \
  --steps 200 \
  --dt 0.01

cargo run -p threebody-cli -- discover \
  --input results/manual/traj.csv \
  --sidecar results/manual/traj.json \
  --out results/manual/top_equations.json
```

Useful follow-ups:
- `cargo run -p threebody-cli -- --help`
- `cargo test`
- `just quickstart` for the full factory workflow when LLM access is configured
- `scripts/autoresearch_eval.sh` for the numeric-only quickstart profile used by external `autoresearch`

## What the simulator writes

- `traj.csv`: time series data for positions, velocities, accelerations, diagnostics, and optional field data.
- `traj.json`: metadata sidecar with config, warnings, timestep stats, and initial-state snapshot.

## Notes

- Truth-mode runs use adaptive RK45 for tighter error control.
- `em_model = "darwin"` adds a moving-source finite-`c` correction intended for low-velocity regimes; it is still not full retarded electrodynamics.
- Discovery works from simulator-provided accelerations instead of finite-differenced trajectories.
- Predictability tools and LLM-assisted workflows exist, but they are more experimental than the basic simulate-and-discover path.
- `quickstart --profile autoresearch` disables the internal LLM path, emits machine-readable score files, and is intended to be the fixed benchmark harness for `autoresearch`.

## More detail

- Previous long-form README: [README_OLD.md](README_OLD.md)
- Core crate notes: [threebody-core/README.md](threebody-core/README.md)
- CLI usage details: [threebody-cli/README.md](threebody-cli/README.md)
- Research writeup: [academic_paper.tex](academic_paper.tex)
