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

Numerical integration (planned):
- Adaptive RK45 (Dormand-Prince) for truth-mode error control.
- Leapfrog (kick-drift-kick) for fast, long-horizon gravity runs.
- An EM-suitable integrator (Boris or implicit midpoint) is required when EM is enabled.

Diagnostics:
- Linear momentum, angular momentum.
- Mechanical energy proxy (kinetic + gravitational + electric potential).

Outputs:
- Wide CSV with positions, velocities, accelerations, fields, potentials, and diagnostics.
- Robust header-based parsing to avoid schema brittleness.
- JSON sidecar with config, header hash, warnings, last regime, and dt stats (accepted/rejected/min/max/avg).

**Discovery Engine (Predictor) Overview**
- Learns sparse, interpretable models of acceleration from simulator output.
- Uses structured feature libraries (relative positions, distances, cross products, etc.).
- Fits via sequential thresholded least squares (SINDy / STLS).
- Evaluates models by rollout accuracy, stability, and generalization within a regime.

**Accuracy / Truth Mode**
- Adaptive RK45 with step rejection and strict tolerances.
- Per-step `dt` recorded in output; close-encounter policy configurable.
- Intended for “truth-grade” runs where accuracy is prioritized over speed. The default config uses adaptive RK45.

**LLM Judge + Factory Loop (Optional)**
- The LLM is a supplemental judge, not a generator. It scores candidates with a fixed rubric (fidelity, parsimony, physical plausibility, regime consistency, stability risk) and outputs JSON only.
- The LLM can propose initial conditions within strict bounds to drive the `factory` loop.
- Prompts and responses are logged for reproducibility; numeric metrics remain the primary ranking.
- Requires an API key in `OPENAI_API_KEY` for `--llm-mode openai`. Offline and `--llm-mode mock` are supported.

**Status**
This repo now includes a working Rust workspace with `threebody-core` (library), `threebody-cli` (binary), and `threebody-discover` (discovery engine). The core simulator supports gravity + quasi-static EM, adaptive RK45 truth mode, and CSV + JSON sidecar output. The discovery engine runs a genetic search and can optionally use an LLM judge plus an end-to-end `factory` workflow.

**Quick Start**
The intended flow is:
1. Generate an example config with the CLI.
2. Run a simulation from that config and produce a CSV.
3. Inspect or visualize outputs, and then feed them into discovery.

Expected commands (exact flags will be finalized during implementation):
```bash
# Generate a starter config
cargo run -p threebody-cli -- example-config --out config.json

# Run a simulation and emit CSV output
cargo run -p threebody-cli -- simulate --config config.json --output run.csv

# Run a truth-mode simulation (adaptive RK45)
cargo run -p threebody-cli -- simulate --config config.json --output run_truth.csv --mode truth

# Run discovery (writes top 3 equations)
cargo run -p threebody-cli -- discover --runs 50 --population 20 --out top_equations.json

# Run one factory iteration with a mock LLM (no API key needed)
cargo run -p threebody-cli -- factory --max-iters 1 --auto --llm-mode mock

# Run one factory iteration with OpenAI (requires OPENAI_API_KEY)
export OPENAI_API_KEY="your_key_here"
cargo run -p threebody-cli -- factory --max-iters 1 --auto --llm-mode openai --model gpt-5
```

**Factory Outputs (Per Iteration)**
- `traj.csv` and `traj.json` sidecar.
- `initial_conditions.json` and `ic_request.json`.
- `discovery.json` with numeric top-3 candidates.
- `judge_input.json` plus `judge_prompt.txt` and `judge_response.txt` (when LLM is enabled).
- `report.json` and `report.md` summaries.

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
25. Optional discovery crate `threebody-discover` with feature library, STLS solver, and rollout evaluator. Tests: recover synthetic coefficients, rollout checks.

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
