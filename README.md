# Three-Body Oracle + Predictor

This repo is a concise, implementation-focused research blueprint for a two-system approach to three-body dynamics, and I like how it balances rigorous simulation with interpretable model discovery.

**Summary**
This project proposes two coupled systems:
- A truth-grade numerical simulator (the "oracle") for three bodies under Newtonian gravity plus a quasi-static, nonrelativistic electromagnetic (EM) model.
- A sparse discovery engine (the "predictor") that learns compact, interpretable equations that approximate the oracle and are validated via forward rollout.

The implementation plan is explicitly TDD-first and emphasizes reproducibility, explicit assumptions, and regime-specific claims rather than universal closed-form solutions.

**Current Contents**
- `academic_paper.tex` describes the system architecture, equations, numerical methods, and modeling limitations.
- `todo.md` is the detailed, test-driven implementation plan for a Rust codebase.

**Core Ideas**
- Use a high-fidelity simulator as ground truth with explicit error control.
- Train discovery models on simulator-provided accelerations to avoid derivative noise in chaotic systems.
- Constrain discovery to structured feature libraries to keep equations interpretable.
- Validate discovered equations by rollout, not just pointwise acceleration fit.
- Be explicit about modeling limits and avoid overclaiming.

**Assumptions and Modeling Limits**
- Current implementation is fixed to three bodies (N=3) across the simulator, outputs, and discovery pipeline. General N-body support is future work.
- Units are SI by default; `G`, `k_e`, and `mu_0` are configurable.
- EM is quasi-static and nonrelativistic: no retardation, no radiation reaction, no self-fields.
- Energy and momentum conservation are not guaranteed once magnetic terms are enabled; energy is treated as a mechanical proxy.
- Close encounters are numerically stiff; softening is optional but changes physics.
- Long-horizon prediction is not meaningful in chaotic regimes; evaluation should emphasize short-horizon accuracy and stability.

**Simulator (Oracle) Overview**
State per body:
- Mass `m_i`, charge `q_i`
- Position `r_i(t)` and velocity `v_i(t)`

Forces and fields (ASCII form):
```text
Gravity:
  a_i(g) = sum_{j != i} G * m_j * (r_j - r_i) / |r_j - r_i|^3

Electrostatics:
  E_i = sum_{j != i} k_e * q_j * (r_i - r_j) / |r_i - r_j|^3

Magnetostatics (quasi-static Biot-Savart style):
  B_i = sum_{j != i} (mu_0 / 4pi) * q_j * (v_j x (r_i - r_j)) / |r_i - r_j|^3

Lorentz acceleration:
  a_i(em) = (q_i / m_i) * (E_i + v_i x B_i)

Total:
  a_i = a_i(g) + a_i(em)

Potentials:
  phi_i = sum_{j != i} k_e * q_j / |r_i - r_j|
  A_i   = sum_{j != i} (mu_0 / 4pi) * q_j * v_j / |r_i - r_j|
```

Numerical integration:
- Adaptive RK45 (Dormand-Prince) for truth-mode error control.
- Leapfrog (kick-drift-kick) for fast, long-horizon gravity runs.
- EM-suitable integrators are available (Boris and implicit midpoint) for experiments where EM is enabled.

Diagnostics:
- Linear momentum, angular momentum.
- Mechanical energy proxy (kinetic + gravitational potential when gravity is enabled + electric potential when EM is enabled). Magnetic field energy is not modeled.

Outputs:
- Wide CSV with positions, velocities, accelerations, fields, potentials, and diagnostics.
- Robust header-based parsing to avoid schema brittleness.
- JSON sidecar with config, header hash, warnings, last regime, dt stats (accepted/rejected/min/max/avg), and the initial state (masses/charges/pos/vel).

**Discovery Engine (Predictor) Overview**
- Learns sparse, interpretable models of acceleration from simulator output.
- Uses structured feature libraries (relative positions, distances, cross products, etc.).
- Fits sparse, interpretable equations on a fixed feature library using sparse regression (STLS or LASSO; STLS is the default). A simple GA baseline remains available.
- Evaluates models by rollout accuracy (position RMSE), divergence time, stability, and generalization within a regime.

**Accuracy / Truth Mode**
- Adaptive RK45 with step rejection and strict tolerances.
- Per-step `dt` recorded in output; close-encounter policy configurable.
- Intended for “truth-grade” runs where accuracy is prioritized over speed. The default config uses adaptive RK45.

**LLM Judge + Factory Loop (Optional)**
- The LLM is a supplemental judge, not a generator. It scores candidates with a fixed rubric (fidelity, parsimony, physical plausibility, regime consistency, stability risk) and outputs JSON only.
- The LLM can propose initial conditions within strict bounds to drive the `factory` loop.
- Prompts and responses are logged for reproducibility; numeric metrics (MSE, rollout RMSE, divergence time) remain the primary ranking.
- The rollout evaluator supports `euler` and `leapfrog`; the LLM can recommend which to use next, along with the discovery configuration (solver: `stls`/`lasso`/`ga` and key hyperparameters).
- The LLM does not directly generate equations or fit coefficients; it steers the loop (IC proposals, candidate ranking/interpretation, and next-step recommendations).
- `--llm-mode auto` uses OpenAI when a key is available, and otherwise falls back to a local mock judge/explainer. If the OpenAI call fails once (offline / network blocked / auth), it switches to mock for the remainder of the run.
- If `.openai_key` exists in the working directory, it is used for OpenAI mode and overrides `OPENAI_API_KEY`.
- You can pass `--openai-key-file` to use a specific key file.
- Create the key file with: `echo "sk-your-key" > .openai_key`.

**Status**
This repo now includes a working Rust workspace with `threebody-core` (library), `threebody-cli` (binary), and `threebody-discover` (discovery engine). The core simulator supports gravity + quasi-static EM, adaptive RK45 truth mode, and CSV + JSON sidecar output. The discovery engine supports STLS/LASSO sparse regression (default) plus an optional GA baseline, and can optionally use an LLM judge plus an end-to-end `factory` workflow.

**Quick Start**
The intended flow is:
1. Generate an example config with the CLI.
2. Run a simulation from that config and produce a CSV.
3. Inspect or visualize outputs, and then feed them into discovery.

Fastest path (uses `just`; requires a reachable LLM endpoint):
```bash
just llm-check
just quickstart
```
What it does:
- Creates a timestamped run directory under `results/quickstart_*/`.
- Runs **10** `factory` iterations with an LLM judge (`--llm-mode auto`) and writes evidence under `results/quickstart_*/factory/run_###/`.
- Writes a single novice-friendly results file to `results/quickstart_*/RESULTS.md` (and `RESULTS.pdf` when `pdflatex` is available).
- Updates the global “best so far” index at `results/best_results.md` (and `results/best_results.json`).

Tune simulation length (the only “knob” you should need):
```bash
just quickstart 400
```

Run 10 separate quickstarts (recommended when you want the LLM to steer across runs and build confidence):
```bash
just quickstart10 1000
```
What it does:
- Runs the full quickstart workflow 10 times (each quickstart has 10 factory iterations).
- Updates `results/best_results.md` after every run.
- Writes a held-out benchmark report at `results/<run>/factory/benchmark_eval.json` and includes it in `results/<run>/RESULTS.md` under “Held-out Benchmark”. This is a fixed IC suite intended for apples-to-apples comparisons across runs in the same `(steps, dt)` bucket.
- Requires a reachable LLM endpoint (OpenAI by default). Set `.openai_key` or `OPENAI_API_KEY`. To use a local OpenAI-compatible endpoint, set `OPENAI_BASE_URL` (e.g. `http://localhost:11434/v1`). If your endpoint only supports Chat Completions, set `OPENAI_API_STYLE=chat`. If your endpoint needs a different model name, set `THREEBODY_LLM_MODEL`.
- Writes an aggregated findings report to `results/findings.tex` (and `results/findings.pdf` when `pdflatex` is available). Each run also copies a timestamped PDF snapshot (e.g. `results/findings_<unix_seconds>.pdf`).

Notes on 3D learning:
- The factory loop forces a small deterministic `z` perturbation in initial conditions when the proposed ICs are planar. This prevents repeatedly rediscovering `az=0` from purely planar trajectories and makes `grav_z` identifiable in gravity-only runs.

How to confirm it really used an LLM (and didn’t silently fall back):
- `just quickstart` / `just quickstart10` run `llm-check` first and will fail fast if the LLM isn’t reachable.
- Runs created with `--require-llm` do not fall back to the mock; if the run completed, the LLM endpoint was reachable.
- Evidence is saved per iteration:
  - `results/<run>/factory/run_###/ic_prompt.txt`, `ic_response.txt`
  - `results/<run>/factory/run_###/judge_prompt.txt`, `judge_response.txt`
  - `results/<run>/factory/evaluation_prompt.txt`

EM identifiability benchmark (when EM keeps rediscovering gravity-only):
```bash
cargo run -p threebody-cli -- bench-em --steps 200 --dt 0.01 --max-cases 24 --heldout 0 --no-pdf
```
This writes `results/bench_em_*/RESULTS.md` plus per-case artifacts under `results/bench_em_*/factory/run_###/`, and logs a simple EM/gravity “signal ratio” to help diagnose when EM terms are detectable.

What to open first (novice-friendly):
- Global best: `results/best_results.md`
- Latest run: `results/quickstart_*/RESULTS.md`
- Aggregated paper: `results/findings.pdf` (if built) or `results/findings.tex`

Commands:
```bash
# Create an output directory (recommended: keep artifacts out of repo root)
out="results/manual_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$out"

# Verify LLM connectivity (recommended before long runs)
cargo run -p threebody-cli -- llm-check

# One-command quickstart (recommended; fails if LLM is unreachable)
cargo run -p threebody-cli -- quickstart --out-dir "$out" --steps 200 --require-llm

# Generate a starter config + initial conditions
cargo run -p threebody-cli -- example-config --out "$out/config.json"
cargo run -p threebody-cli -- example-ic --preset three-body --out "$out/ic.json"

# Run a simulation and emit CSV + sidecar JSON
cargo run -p threebody-cli -- simulate --config "$out/config.json" --ic "$out/ic.json" --output "$out/traj.csv" --steps 200 --dt 0.01

# Run a truth-mode simulation (adaptive RK45)
cargo run -p threebody-cli -- simulate --config "$out/config.json" --ic "$out/ic.json" --output "$out/traj_truth.csv" --mode truth

# Enable EM explicitly (EM is off by default)
cargo run -p threebody-cli -- simulate --config "$out/config.json" --ic "$out/ic.json" --output "$out/traj_em.csv" --em

# Run discovery (writes top 3 equations; default solver is STLS)
cargo run -p threebody-cli -- discover --solver stls --input "$out/traj.csv" --sidecar "$out/traj.json" --out "$out/top_equations.json"

# Try LASSO instead:
# cargo run -p threebody-cli -- discover --solver lasso --input "$out/traj.csv" --sidecar "$out/traj.json" --out "$out/top_equations.json"

# If your simulation output isn't named traj.csv/traj.json, pass explicit paths:
# cargo run -p threebody-cli -- discover --input run.csv --sidecar run.json --out top_equations.json

# Run the LLM-assisted factory loop (10 iterations) and generate an explainer at $out/factory/evaluation.md
cargo run -p threebody-cli -- factory --out-dir "$out/factory" --max-iters 10 --auto --config "$out/config.json" --steps 200 --dt 0.01 --llm-mode mock

# Open the high-school-friendly explainer
cat "$out/factory/evaluation.md"

# Regenerate the aggregated findings report (TeX always; PDF when pdflatex is available)
cargo run -p threebody-cli -- findings --results-dir results --out-tex results/findings.tex

# Use OpenAI when available, but fall back cleanly when offline
# cargo run -p threebody-cli -- factory --out-dir "$out/factory" --max-iters 10 --auto --config "$out/config.json" --steps 200 --dt 0.01 --llm-mode auto --model gpt-5.2

# Or use env var when no key file is present
# export OPENAI_API_KEY="your_key_here"
```

**Paper Build (LaTeX -> PDF)**
- `pdflatex` is a system dependency (TeX Live / BasicTeX). It is not installed via `uv` or `pip`.
- macOS (Homebrew): `brew install --cask basictex`, then restart your terminal or run `eval "$(/usr/libexec/path_helper)"`.
- Build the PDF: `pdflatex -interaction=nonstopmode academic_paper.tex`.
- Optional: create a local Python environment for tooling with `uv venv .venv`.

**Factory Outputs**
- Run root (`<out_dir>/`):
  - `evaluation.md`: single-file summary for novices + evidence (deterministic; numeric metrics are the source of truth).
  - `evaluation_llm.md`: optional narrative evaluation output (OpenAI or local fallback).
  - `evaluation_input.json`: structured summary of the run (for reproducibility).
  - `evaluation_history.json`: best-vs-prior-best comparison over all previous local attempts under `results/`.
  - `evaluation_prompt.txt`: the exact prompt used to generate `evaluation_llm.md` (when LLM mode is enabled).
  - `evaluation.tex`: reproducible LaTeX evaluation (exact equation text + ICs + metrics).
  - `evaluation.pdf`: built when `pdflatex` is available (best-effort).
  - `evaluation_pdf_error.txt`: present only if `pdflatex` is available but fails.
  - `evaluation_error.txt`: only present if the LLM call failed and a fallback was used.
- Per iteration (`<out_dir>/run_###/`):
  - `traj.csv` and `traj.json` sidecar.
  - `initial_conditions.json` and `ic_request.json`.
  - `discovery.json` with solver metadata plus `top3_x/top3_y/top3_z` and `vector_candidates`.
  - `rollout_trace.json` for the best vector candidate under the selected rollout integrator.
  - `judge_input.json` plus `judge_prompt.txt` and `judge_response.txt` (when LLM is enabled).
  - `report.json` and `report.md` summaries.

**How To Spot “Math Improvements” (No Math Required)**
Discovery produces equations, but you can judge progress using a few plain metrics:
- **`rollout_rmse`** (lower is better): average trajectory error during rollout vs the oracle.
- **`divergence_time`** (higher is better): how long the rollout stays “close enough” before drifting away.
- **`mse`** (lower is better): pointwise acceleration-fit error (useful, but not sufficient alone).
- **`complexity`** (lower is simpler): number of non-zero terms across x/y/z.

Where to look:
- Across all runs: start with `results/best_results.md` (single file that stays up to date).
- Single run: open `top_equations.json` and look at `top3[0].metrics` and `top3[0].equation_text`.
- Factory run: start with `<out_dir>/evaluation.md`, then open `<out_dir>/run_001/report.md` (human-readable) or `<out_dir>/run_001/report.json` (structured). Also check `<out_dir>/run_001/discovery.json` for `solver` metadata. The CLI default out-dir is `factory_out/`.

Practical checklist (for non-math users):
1. Prefer the model with **lower `rollout_rmse`** and **higher `divergence_time`** on the same regime/config.
2. If two models are within ~5% on `rollout_rmse`, prefer the **simpler** one (lower `complexity`).
3. Don’t trust “good `mse`” if rollout is bad—rollout is the real test.
4. Change only one thing at a time (solver, thresholds/alphas, rollout integrator, EM on/off).

**How To Communicate Improvements (For Non-Math / Non-Physics Stakeholders)**
Use a short “release-note” style summary plus artifacts:
- What changed: solver + key settings (from `solver` metadata).
- What got better: `rollout_rmse`, `divergence_time`, and `complexity` (before/after).
- Evidence (recommended): attach `results/best_results.md`, the run’s `RESULTS.md`, and the run’s `factory/evaluation_input.json`.

Copy/paste template:
```text
Goal/regime: gravity_only | em_quasistatic
Command: threebody-cli factory/discover (paste exact command)
Solver: stls/lasso/ga (include normalize + thresholds/alphas)
Best model (from report):
  rollout_rmse: <old> -> <new>
  divergence_time: <old> -> <new>
  complexity: <old> -> <new>
Equation (human-readable): <equation_text>
Artifacts: results/best_results.md + results/<run>/RESULTS.md + results/<run>/factory/evaluation_input.json
```

**Implementation Plan (TDD-first, condensed from `todo.md`)**
1. Create a Cargo workspace with crates `threebody-core` and `threebody-cli`. Tests: placeholder smoke test.
2. Establish code conventions in `threebody-core/src/lib.rs` and `threebody-core/src/prelude.rs` with re-exports. Tests: rustdoc examples compile.
3. Document exact physics equations in `threebody-core/src/physics.rs`. Tests: doc tests validate numeric examples.
4. Implement `Vec3` math (or document a dependency choice). Tests: add, sub, scalar mul, dot, cross, norm, normalize, approx.
5. Add `approx_eq` and `clamp` in `threebody-core/src/math/float.rs`. Tests: tolerance edge cases.
6. Define core data types in `threebody-core/src/state.rs` for bodies, state, and system. Tests: construction, clone, pair indices.
7. Implement barycentric initialization in `threebody-core/src/frames.rs`. Tests: COM position and momentum are zeroed.
8. Implement gravity in `threebody-core/src/forces/gravity.rs` with optional softening. Tests: two-body symmetry, zero masses.
9. Implement EM in `threebody-core/src/forces/em.rs` with softening. Tests: zero charges, zero velocities, zero `q_i`.
10. Implement potentials in `threebody-core/src/forces/potentials.rs`. Tests: zero charges, zero velocities.
11. Create force aggregator in `threebody-core/src/forces/mod.rs` with shared pairwise computations. Tests: matches individual modules, flags on/off deterministic.
12. Implement diagnostics in `threebody-core/src/diagnostics.rs`. Tests: bounded energy in gravity-only, EM proxy checks.
13. Define integrator trait in `threebody-core/src/integrators/mod.rs`. Tests: generic harness on harmonic oscillator.
14. Implement leapfrog in `threebody-core/src/integrators/leapfrog.rs`. Tests: bounded energy error for gravity-only.
15. Implement adaptive RK45 in `threebody-core/src/integrators/rk45.rs`. Tests: tolerance tracks error, short-horizon agreement.
16. Implement simulation driver in `threebody-core/src/sim.rs` with event hooks. Tests: deterministic output, encounter triggers.
17. Define config schema in `threebody-core/src/config.rs` using `serde`. Tests: JSON roundtrip, defaults, invalid configs.
18. Implement CSV output in `threebody-core/src/output/csv.rs` with header-based schema. Tests: header correctness, index map.
19. Implement header-based parser in `threebody-core/src/output/parse.rs`. Tests: extra columns ok, missing columns error.
20. Build CLI in `threebody-cli/src/main.rs` using `clap` with `example-config` and `simulate`. Tests: help text and valid JSON output.
21. Add public API facade in `threebody-core/src/lib.rs`. Tests: integration test runs tiny sim.
22. Add deterministic fixtures in `threebody-core/tests/fixtures/`. Tests: output matches within tolerance, no NaNs.
23. Add benchmarks in `threebody-core/benches/` using `criterion`. Tests: benches compile.
24. Add docs in `threebody-core/README.md` and `threebody-cli/README.md`. Tests: `cargo test --doc`.
25. Optional discovery crate `threebody-discover` with feature library, STLS/LASSO sparse regression, GA baseline, and rollout evaluator. Tests: recover synthetic coefficients in a controlled synthetic case; rollout checks.

**Mitigation Insertions (placed after referenced steps)**
1. After step 3: add a model validity matrix in `threebody-core/src/physics.rs` listing supported regimes and non-claims. Tests: doc test asserts regimes and non-claims are present.
2. After step 11: add `threebody-core/src/regime.rs` for regime diagnostics. Tests: deterministic, no NaNs.
3. After step 13: add integrator capability metadata table. Tests: includes all integrators.
4. After step 14: add an EM-suitable integrator (Boris or implicit midpoint). Tests: bounded energy and stable gyration under EM surrogate.
5. After step 16: add close-encounter policy in config (`soften`, `switch_integrator`, `stop_and_report`). Tests: deterministic triggers.
6. After step 17: add warning system for leapfrog + EM unless explicitly overridden. Tests: warning emitted.
7. After step 18: emit a JSON sidecar with config, header hash, warnings, last regime, and dt stats. Tests: sidecar schema roundtrip.
8. After step 22: add oracle-comparison tests (RK45 vs leapfrog vs EM integrator). Tests: divergence time finite and consistent.
9. After step 23: add accuracy-per-ms benchmarks with regression thresholds.

**Cross-Cutting Test Rules**
1. Every physics formula appears in code comments or module docs in the same file as its implementation, and each has a numeric test.
2. Invariants must be asserted by regime. Gravity-only and electrostatic-only can assert momentum conservation; EM with magnetic terms must not.
3. Tolerances live in `threebody-core/tests/tolerances.rs` and are reused everywhere.
4. Every public function has at least one failure-mode test (invalid inputs, division by zero, or epsilon edge cases).

**References (Selected)**
Simulation and integrators:
- Dormand, J. R., and Prince, P. J. (1980). A family of embedded Runge-Kutta formulae. Journal of Computational and Applied Mathematics, 6(1), 19-26. doi:10.1016/0771-050X(80)90013-3.
- Verlet, L. (1967). Computer "Experiments" on Classical Fluids. I. Thermodynamical Properties of Lennard-Jones Molecules. Physical Review, 159, 98-103. doi:10.1103/PhysRev.159.98.
- Ruth, R. D. (1983). A canonical integration technique. Proc. PAC'83, Santa Fe, NM, pp. 2669-2672.
- Forest, E., and Ruth, R. D. (1990). Fourth-order symplectic integration. Physica D, 43(1), 105-117. doi:10.1016/0167-2789(90)90019-L.
- Yoshida, H. (1990). Construction of higher order symplectic integrators. Physics Letters A, 150(5-7), 262-268. doi:10.1016/0375-9601(90)90092-3.
- Boris, J. P. (1970). Relativistic plasma simulation--optimization of a hybrid code. Proc. 4th Conf. on Numerical Simulation of Plasmas, Washington, DC, pp. 3-67.
- He, Y., Sun, Y., Liu, J., and Qin, H. (2015). Volume-preserving algorithms for charged particle dynamics. Journal of Computational Physics, 281, 135-147. doi:10.1016/j.jcp.2014.10.032.
- Higuera, A. V., and Cary, J. R. (2017). Structure-preserving second-order integration of relativistic charged particle trajectories in electromagnetic fields. Physics of Plasmas, 24, 052104. doi:10.1063/1.4979989.

EM modeling:
- Darwin, C. G. (1920). The dynamical motions of charged particles. Philosophical Magazine, 39(233), 537-551. doi:10.1080/14786440508636066.

Regularization and close encounters:
- Stiefel, E., and Kustaanheimo, P. (1965). Perturbation theory of Kepler motion based on spinor regularization. Journal fuer die reine und angewandte Mathematik, 218, 204-219. doi:10.1515/crll.1965.218.204.
- Mikkola, S., and Aarseth, S. J. (1993). An implementation of N-body chain regularization. Celestial Mechanics and Dynamical Astronomy, 57, 439-459. doi:10.1007/BF00695714.

Sparse discovery and equation learning:
- Bongard, J., and Lipson, H. (2007). Automated reverse engineering of nonlinear dynamical systems. PNAS, 104(24), 9943-9948. doi:10.1073/pnas.0609476104.
- Schmidt, M., and Lipson, H. (2009). Distilling Free-Form Natural Laws from Experimental Data. Science, 324(5923), 81-85. doi:10.1126/science.1165893.
- Brunton, S. L., Proctor, J. L., and Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. PNAS, 113(15), 3932-3937. doi:10.1073/pnas.1517384113.
- Rudy, S. H., Brunton, S. L., Proctor, J. L., and Kutz, J. N. (2017). Data-driven discovery of partial differential equations. Science Advances, 3(4), e1602614. doi:10.1126/sciadv.1602614.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. JRSS-B, 58(1), 267-288. doi:10.1111/j.2517-6161.1996.tb02080.x.

Chaos and predictability:
- Lorenz, E. N. (1963). Deterministic nonperiodic flow. Journal of the Atmospheric Sciences, 20(2), 130-141. doi:10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2.

**Where This Is Headed**
The near-term goal is a reproducible, testable simulator whose outputs can be used to discover compact, regime-specific predictive laws. The long-term goal is an "atlas of models" that balances interpretability with predictive power across regimes.
