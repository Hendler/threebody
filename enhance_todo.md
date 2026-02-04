# Enhancements TODO (Physics Correctness + Non-Technical UX)

This file captures the “physics researcher” code review findings and turns them into actionable engineering work items. Each item includes:
- what was wrong (and where),
- why it matters scientifically and for UX,
- how to fix it (implementation steps),
- what tests to add/update,
- how to verify locally.

Status note: most items below are **already implemented** in the current repo; they remain here as a durable record and as a checklist for future refactors.

## 1) Make `energy_proxy` Consistent With Force Toggles
- Status: **DONE**
- Verified issue (pre-fix): `threebody-core/src/diagnostics.rs` computed gravitational + electric potentials regardless of `enable_gravity/enable_em`. This made `--no-gravity` / `--no-em` runs show misleading energy summaries.
- Why it matters:
  - Scientifically: invariants/diagnostics must correspond to the equations being integrated, or drift plots become meaningless.
  - UX: non-technical users interpret “energy” as a conserved quantity; the tool must not silently report “energy” for forces that are disabled.
- Implementation (what we did / how to do it again):
  1. Change `compute_diagnostics` to take `&Config` instead of raw constants.
     - File: `threebody-core/src/diagnostics.rs`
  2. Make `energy_proxy(system, cfg)` include:
     - kinetic always
     - gravitational potential only if `cfg.enable_gravity`
     - electrostatic potential only if `cfg.enable_em`
     - explicitly exclude magnetic field energy (not modeled)
  3. Update call sites:
     - `threebody-core/src/sim.rs`
     - `threebody-core/src/output/csv.rs`
     - `threebody-core/src/integrators/leapfrog.rs` (test helper)
- Tests:
  - Add unit test ensuring toggles are respected.
    - File: `threebody-core/src/diagnostics.rs` (`energy_proxy_respects_force_toggles`)
  - Keep the manual static check (gravity+electric) as a numeric sanity test.
- Verify:
  - `cargo test -p threebody-core`
  - CLI smoke: `cargo run -p threebody-cli -- simulate --summary --no-gravity --no-em --steps 1`

## 2) Default EM Off (Non-Technical Safe Default) + Explicit `--em`
- Status: **DONE**
- Verified issue (pre-fix): `Config::default()` had `enable_em=true`, so casual users implicitly ran quasi-static EM with stronger caveats.
- Why it matters:
  - Scientifically: quasi-static EM is a model choice, not a default assumption; making it opt-in keeps gravity-only results interpretable.
  - UX: reduces “why is my energy drifting?” confusion from magnetic terms and non-modeled field energy.
- Implementation:
  1. Change `Config::default().enable_em = false`.
     - File: `threebody-core/src/config.rs`
  2. Add `--em` to CLI to turn EM on without editing JSON.
     - Files: `threebody-cli/src/main.rs`
  3. Guard against `--em` + `--no-em` conflict (error).
     - File: `threebody-cli/src/main.rs` (`build_config`)
  4. Update runner UX:
     - File: `justfile` (`simulate-em` now passes `--em`)
- Tests:
  - CLI integration coverage via existing tests + `just test`.
  - Optional extra test (nice-to-have): ensure `--em --no-em` fails with a clear message.
- Verify:
  - `cargo test`
  - `just simulate-em` produces nonzero EM-related outputs when charges/velocities are nonzero.

## 3) Make `threebody-cli discover` Use Real Simulation Output by Default
- Status: **DONE**
- Verified issue (pre-fix): `discover` ran on a toy dataset (`x`, `x^2`), which is surprising and misleading for experiments.
- Why it matters:
  - Scientifically: discovery must be tied to the oracle output.
  - UX: “simulate → discover” should work with no hidden prerequisites.
- Implementation:
  1. `discover` now accepts:
     - `--input` (defaults to `traj.csv`)
     - `--sidecar` (defaults to `<input>.json`)
     - optional `--body` (if omitted: all bodies)
     - `--rollout-integrator` (euler/leapfrog)
     - existing GA controls (`--runs`, `--population`, `--seed`) and optional LLM judge.
     - File: `threebody-cli/src/main.rs`
  2. Implement header-based CSV parsing using:
     - `threebody-core/src/output/parse.rs` (`parse_header`, `require_columns`)
     - Parse `t`, `dt`, and all `r*/v*` columns to reconstruct steps.
     - File: `threebody-cli/src/main.rs` (`load_steps_from_csv`)
  3. Require sidecar to obtain initial masses/charges.
     - Sidecar now contains `initial_state` (see item 4).
- Tests:
  - Update `threebody-cli/tests/cli.rs`:
    - run `simulate` in a temp dir to produce `traj.csv/traj.json`
    - then run `discover` with defaults and assert output JSON includes expected keys.
- Verify:
  - `cargo test -p threebody-cli`
  - Manual:
    - `cargo run -p threebody-cli -- simulate --steps 50`
    - `cargo run -p threebody-cli -- discover --runs 50 --population 20`

## 4) Preserve Direction + Physical Scaling in Discovery Features
- Status: **DONE (baseline structured features)**
- Verified issue (pre-fix): feature engineering used magnitudes (`norm`) like `|v×r|` and inverse-distance scalars, then attempted to fit vector acceleration components. This destroys sign/direction information and blocks rediscovery of vector laws.
- Why it matters:
  - Scientifically: the discovery target is a vector field; scalar magnitude features are not sufficient to represent directional laws.
  - UX: users want interpretable equations that resemble physical structure (`r/|r|^3`), not magnitude heuristics.
- Implementation:
  1. Replace magnitude-only features with direction-preserving component bases:
     - `grav_{x,y,z}`: Σ m_j (r_j - r_i)/|r|^3   (no G)
     - `elec_{x,y,z}`: (q_i/m_i) Σ q_j (r_i - r_j)/|r|^3   (no k_e)
     - `mag_{x,y,z}`: (q_i/m_i) (v_i × B_basis) where B_basis=(1/4π)Σ q_j (v_j×(r_i-r_j))/|r|^3   (no μ0)
     - Files:
       - `threebody-cli/src/main.rs` (`compute_feature_vector`)
       - `threebody-discover/src/library.rs` (feature list + descriptions)
  2. Gate feature computation by `cfg.enable_gravity/enable_em` so disabled physics doesn’t leak into discovery.
- Tests:
  - Current baseline tests ensure the vector dataset builder and rollout metrics remain finite:
    - File: `threebody-cli/src/main.rs` (`#[cfg(test)]`)
  - Recommended next tests (still valuable):
    - Analytic sign/direction test for `grav_x` on a 2-body line configuration.
    - “Rediscover gravity basis” test: run discovery on gravity-only data and assert top model heavily weights `grav_*`.
- Verify:
  - `cargo test`
  - Run `factory` once and inspect `factory_out/run_001/report.md` for vector-form equations containing component features.

## 5) Add IC File Workflow (Experiments Without Editing Rust)
- Status: **DONE**
- Verified issue (pre-fix): only `--preset two-body|static` existed for `simulate`, so non-technical users couldn’t change masses/charges/ICs.
- Why it matters:
  - Scientifically: meaningful experiments require systematic IC variation.
  - UX: editing Rust is a non-starter for most experimenters.
- Implementation:
  1. Add `threebody-cli example-ic`:
     - emits an `InitialConditionSpec` JSON for a preset (default `three-body`).
     - File: `threebody-cli/src/main.rs`
  2. Add `--ic <path>` to `simulate`/`run`:
     - parses IC JSON and validates min pair distance bounds.
     - File: `threebody-cli/src/main.rs` (`run_simulation`)
  3. Add a true 3-body preset:
     - `three-body`: Lagrange equilateral rotating solution in normalized units (charges 0).
     - File: `threebody-cli/src/main.rs` (`preset_system`)
- Tests:
  - CLI test: `example-ic` outputs valid JSON with `bodies`.
    - File: `threebody-cli/tests/cli.rs`
  - Optional: add a test that `simulate --ic ic.json` succeeds.
- Verify:
  - `cargo run -p threebody-cli -- example-ic --preset three-body --out ic.json`
  - `cargo run -p threebody-cli -- simulate --ic ic.json --steps 50 --output traj.csv`

## 6) Improve Non-Technical Naming: `factory` Alias `experiment`
- Status: **DONE**
- Verified issue: “factory” is not an intuitive name for a scientific workflow loop.
- Implementation:
  - Add clap alias: `factory` is callable as `experiment`.
  - File: `threebody-cli/src/main.rs`
- Tests:
  - Optional: CLI help snapshot contains `experiment` (or manually verify).
- Verify:
  - `cargo run -p threebody-cli -- experiment --max-iters 1 --auto --llm-mode mock`

## 7) Reproducibility: Store Initial State in Sidecar
- Status: **DONE**
- Verified issue (pre-fix): sidecar contained config + dt stats but not the initial masses/charges/pos/vel, blocking faithful “simulate → discover” from artifacts alone.
- Implementation:
  - Add `initial_state` to sidecar schema and populate it from the first step in `SimResult`.
  - File: `threebody-core/src/output/sidecar.rs`
- Tests:
  - Existing sidecar roundtrip test covers the new field.
    - File: `threebody-core/src/output/sidecar.rs`
- Verify:
  - `cargo run -p threebody-cli -- simulate --steps 1`
  - Inspect `traj.json` and confirm `initial_state` exists.

## Recommended Follow-Ups (Not Implemented Yet)
- Gate CSV columns by physics toggles:
  - If `enable_em=false`, default `output.include_fields/include_potentials` to false in CLI unless explicitly requested (reduces clutter).
- Add analytic unit tests for feature correctness:
  - gravity basis direction/sign tests, and EM basis sanity tests in simple configurations.
- Make discovery output schema more explicit for vector models:
  - Store `eq_x/eq_y/eq_z` explicitly rather than packing the vector model into `equation_text`.

