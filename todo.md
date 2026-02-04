# TODO: Rust Components for Three-Body Oracle + Predictor

**Assumptions (explicit so tests are honest)**
- Units are SI by default. Constants `G`, `k_e`, `mu_0` are configurable in code and config so tests can use normalized values.
- Electromagnetism is quasi-static and nonrelativistic. No retardation, no radiation reaction, no self-field effects. This means momentum and energy conservation are not guaranteed once magnetic terms are enabled, and tests must not assume them.
- Softening `epsilon` is optional and changes physics. Tests must separate `epsilon = 0` (physical) from `epsilon > 0` (numerical stability).

**Build Order (TDD-first, DRY-first)**
1. Create a Cargo workspace with two crates: `threebody-core` (library) and `threebody-cli` (binary). Put workspace in `Cargo.toml` at repo root, and crate manifests at `threebody-core/Cargo.toml` and `threebody-cli/Cargo.toml`. Tests: `cargo test` passes with a single placeholder test in `threebody-core/tests/smoke.rs` that asserts `2 + 2 == 4`.
2. Establish code conventions in `threebody-core/src/lib.rs` and `threebody-core/src/prelude.rs`. Include public re-exports for common types to keep DRY imports. Tests: `rustdoc` examples compile for prelude types.
3. Document physics requirements in `threebody-core/src/physics.rs` with `///` doc comments that state the exact equations used. Inline formulas must be ASCII and testable. Tests: doc tests in `threebody-core/src/physics.rs` that evaluate sample accelerations against known values for one simple configuration.
4. Add a minimal vector math type `Vec3` in `threebody-core/src/math/vec3.rs` unless we explicitly choose a dependency (decision recorded in `threebody-core/src/math/mod.rs`). Tests: `Vec3` addition, subtraction, scalar multiply, dot, cross, norm, and normalize edge cases, including zero vector behavior and `approx_eq`.
5. Add a numeric helper module `threebody-core/src/math/float.rs` with `approx_eq(a,b,rel,abs)` and `clamp`. Tests: relative/absolute tolerance edge cases.
6. Define core data types in `threebody-core/src/state.rs`:
   - `Body { mass, charge }`
   - `State { pos[3], vel[3] }` or `State { pos: [Vec3; 3], vel: [Vec3; 3] }`
   - `System { bodies: [Body; 3], state: State }`
   Tests: construction sanity, copy/clone behavior, and a helper that returns pair indices `(i,j)` with `i < j`.
7. Implement barycentric initialization in `threebody-core/src/frames.rs`:
   - Function `to_barycentric(system) -> system` that subtracts COM position and velocity from each body.
   Physics requirement: For gravity-only or electrostatic-only (no magnetic terms), total linear momentum should be zero after barycentric conversion and remain zero under internal forces.
   Tests: starting with arbitrary positions/velocities, COM position and momentum are zero after conversion; gravity-only integration keeps COM near zero.
8. Implement gravity force model in `threebody-core/src/forces/gravity.rs`:
   - Acceleration: `a_i = sum_{j!=i} G * m_j * (r_j - r_i) / |r_j - r_i|^3`
   - Optional softening: replace `|r|^3` with `(r^2 + epsilon^2)^(3/2)` when enabled.
   Tests: two-body symmetry with `epsilon = 0` gives equal-and-opposite accelerations scaled by masses; gravitational acceleration is zero if all masses are zero.
9. Implement electrostatic and magnetic force model in `threebody-core/src/forces/em.rs`:
   - Electric field at i: `E_i = sum_{j!=i} k_e * q_j * (r_i - r_j) / |r_i - r_j|^3`
   - Magnetic field at i: `B_i = sum_{j!=i} (mu_0 / 4pi) * q_j * (v_j x (r_i - r_j)) / |r_i - r_j|^3`
   - Lorentz acceleration: `a_i_em = (q_i / m_i) * (E_i + v_i x B_i)`
   - Same softening rule as gravity for denominators.
   Tests: with all charges = 0, EM acceleration is zero; with velocities = 0, magnetic terms are zero and Coulomb forces are equal-and-opposite; with `m_i` nonzero and `q_i = 0`, EM acceleration is zero.
10. Implement potentials in `threebody-core/src/forces/potentials.rs`:
    - Scalar potential at i: `phi_i = sum_{j!=i} k_e * q_j / |r_i - r_j|`
    - Vector potential at i: `A_i = sum_{j!=i} (mu_0 / 4pi) * q_j * v_j / |r_i - r_j|`
    Tests: with all charges = 0, both potentials are zero; with velocities = 0, vector potential is zero.
11. Create a unified force aggregator in `threebody-core/src/forces/mod.rs`:
    - DRY requirement: compute pairwise `r_ij`, `d_ij`, `inv_d3` once per step and pass to gravity/EM to avoid duplication.
    - Provide `compute_accel(system, config) -> [Vec3; 3]` and `compute_fields(system, config) -> {E,B,phi,A}`.
    Tests: aggregation with gravity-only matches gravity module; aggregation with EM-only matches EM module; turning flags on/off yields deterministic outputs.
12. Implement diagnostics in `threebody-core/src/diagnostics.rs`:
    - Linear momentum `P = sum m_i * v_i`
    - Angular momentum `L = sum r_i x (m_i * v_i)`
    - Energy proxy: kinetic + gravitational potential + electric potential
    Tests: for gravity-only two-body circular orbit (normalized), energy proxy stays bounded under small dt; EM energy proxy computed equals manual sum for a static configuration.
13. Define an integrator trait in `threebody-core/src/integrators/mod.rs`:
    - `step(system, dt, config) -> system`
    - DRY requirement: integrators must share the same force evaluation API.
    Tests: compile-time use via a generic test harness for a simple ODE `x'' = -x` (harmonic oscillator) to verify expected order.
14. Implement leapfrog (kick-drift-kick) in `threebody-core/src/integrators/leapfrog.rs`:
    - Physics requirement: for gravity-only, leapfrog should show bounded energy error over many steps, not monotonic drift.
    Tests: harmonic oscillator phase-space area stays bounded; gravity-only two-body orbit energy stays within tolerance over N steps.
15. Implement adaptive RK45 (Dormand-Prince) in `threebody-core/src/integrators/rk45.rs`:
    - Required outputs: next state, estimated error, suggested dt.
    - Physics requirement: local truncation error control should track tolerance for smooth problems.
    Tests: harmonic oscillator global error decreases when tolerance is tightened; gravity-only orbit matches leapfrog within expected tolerance for short horizons.
16. Implement a simulation driver in `threebody-core/src/sim.rs`:
    - Loop over steps, apply integrator, collect diagnostics, and optionally compute fields.
    - Provide event hooks for close-encounter detection with threshold `r_min`.
    Tests: deterministic output for fixed seed/config; close-encounter event triggers at known configurations.
17. Define config schema in `threebody-core/src/config.rs` using `serde`:
    - Fields: constants, integrator choice and tolerances, softening, enable_gravity, enable_em, output toggles.
    Tests: JSON roundtrip, defaults applied correctly, invalid configs rejected with clear errors.
18. Implement CSV output in `threebody-core/src/output/csv.rs`:
    - Header-based schema, stable ordering, optional columns for fields and diagnostics.
    - DRY requirement: a single header generator used by both writer and tests.
    Tests: CSV header includes all enabled columns; parsing header to index map matches expected indices.
19. Add a minimal parser in `threebody-core/src/output/parse.rs` to support future consumers:
    - Parse by header names, ignore unknown columns.
    Tests: parsing succeeds with extra columns; missing required columns return clear error.
20. Build CLI in `threebody-cli/src/main.rs` using `clap`:
    - Commands: `example-config`, `simulate`.
    - `simulate` reads config, runs sim, writes CSV.
    Tests: CLI help text includes command list; `example-config` produces valid JSON for `config.rs`.
21. Add a `threebody-core` public API facade in `threebody-core/src/lib.rs`:
    - Re-export `System`, `Config`, `simulate`, and integrators.
    Tests: external integration test in `threebody-core/tests/api.rs` that runs a tiny simulation end-to-end.
22. Add deterministic test fixtures in `threebody-core/tests/fixtures/`:
    - Static JSON configs and expected single-step outputs for gravity-only and electrostatic-only.
    Tests: compare outputs within tolerance; ensure no NaNs or infinities.
23. Implement a benchmarking harness in `threebody-core/benches/` using `criterion`:
    - Bench gravity-only and EM-enabled steps separately.
    Tests: benches compile in CI; no runtime assertions fail.
24. Add documentation examples in `threebody-core/README.md` and `threebody-cli/README.md`:
    - Include minimal code snippets and CLI invocations that compile/run.
    Tests: `cargo test --doc` passes.
25. Optional: Create a discovery crate `threebody-discover` (only if discovery is intended in Rust):
    - Feature library builder, STLS solver, and rollout evaluator.
    - Tests: STLS recovers coefficients on synthetic linear data; rollout on a simple known system reproduces trajectory within tolerance.

**Cross-cutting test rules (apply to every step)**
1. Every physics formula must appear in code comments or module docs in the same file as its implementation, and every formula must have at least one unit test that checks a numeric example.
2. For invariants, tests must specify the regime: gravity-only and electrostatic-only can assert momentum conservation; EM with magnetic terms must not assert strict conservation.
3. Tolerances must be explicit in tests, stored in `threebody-core/tests/tolerances.rs` to keep DRY.
4. Every public function must have at least one test that covers a failure mode (invalid inputs, division by zero, or `epsilon` edge cases).
