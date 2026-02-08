# Three-Body Discovery: Enhancement Roadmap

## Diagnosis: Why the System Has Hit a Ceiling

The discovery engine perfectly recovers Newton's law (`a_i = 1.0 * grav_i`) but every
attempt to go beyond it -- jerk, PN1, Yukawa, atlas gating, tidal invariants -- fails
the strict holdout gate. Median relative improvement is negative across all recent runs.

Root causes:

1. **Feature library echoes known physics.** Every feature is a rearrangement of the
   forces the oracle already computes. Sparse regression finds coefficient 1.0 because
   that *is* the answer within the current basis.
2. **Linear-in-features assumption.** The ansatz `a = Theta * xi` can only discover
   linear combinations of basis functions. Interesting three-body structure is nonlinear.
3. **Cartesian coordinates hide geometry.** Individual body positions `r_i` do not expose
   the natural symmetries and invariants of the three-body problem. The mathematical
   structure lives in relative coordinates, shape variables, and action-angle decompositions.

The system should stop asking *"what linear combination of force-like terms best predicts
acceleration?"* (answer: always Newtonian gravity) and instead ask *"what compact structure
exists in the three-body problem beyond the known forces?"*

That structure lives in:
- the **residual** after subtracting dominant two-body Keplerian motion (Jacobi decomposition),
- the **geometry** of the configuration (shape space, symmetric invariants),
- the **discrete dynamics** between encounters (return maps),
- the **approximate integrals** valid over many orbits (secular invariants like Kozai-Lidov).

---

## Ground Rules
- Keep numeric scoring primary; no LLM override.
- Preserve strict holdout policy (`highbar_v2_benchmark_first`).
- Report negative results explicitly.
- Use train/validation/holdout splits with no leakage.
- Keep all artifacts reproducible and versioned.

---

## Part I: Natural Mathematical Language (Feature Library Revolution)

These enhancements transform the feature library to speak the natural language of the
three-body problem. They are the highest-impact changes for discovering genuinely new math.

### Enhancement A: Jacobi Coordinates + Disturbing Function Decomposition

**Priority: Highest. Single most impactful change.**

Idea:
- Transform from body coordinates `(r1, r2, r3)` to Jacobi coordinates:
  - `rho1 = r2 - r1` (inner relative vector)
  - `rho2 = r3 - (m1*r1 + m2*r2) / (m1 + m2)` (outer relative vector)
- In these coordinates, the two-body problem (bodies 1-2) separates exactly.
- The acceleration decomposes as:
  - `a_Jacobi = a_Keplerian(rho1) + a_perturbation(rho1, rho2)`
- **Fit the residual** `a_actual - a_Keplerian` rather than the full acceleration.
  This dramatically changes the signal-to-noise ratio.

The perturbation has a known expansion in Legendre polynomials:
```
Phi_pert = -G * m3 * sum_n (rho1/rho2)^n * Pn(cos alpha) / rho2
```
where `alpha` is the angle between `rho1` and `rho2`.

New feature library atoms:
1. `cos_alpha` = angle between rho1 and rho2.
2. `rho_ratio` = |rho1| / |rho2| (hierarchy parameter).
3. `P2_cos_alpha`, `P3_cos_alpha` (Legendre polynomial terms).
4. `quadrupole_x/y/z` = leading tidal perturbation term from Legendre expansion.
5. `residual_x/y/z` = `a_oracle - a_Keplerian` (the target itself changes).

Implementation:
1. Add `jacobi.rs` in `threebody-core/src/math/` for coordinate transforms.
2. Add `keplerian_accel()` that computes the two-body part exactly.
3. Add Legendre polynomial evaluator up to order 4.
4. Extend `FeatureLibrary` with a `jacobi` variant containing the new atoms.
5. Modify discovery target to fit residual when `--target residual` is set.

TDD:
- Jacobi transform is invertible (round-trip within f64 tolerance).
- `a_Keplerian(rho1)` matches oracle in two-body limit (set m3 = 0).
- Residual is small relative to total acceleration for hierarchical triples.
- Legendre polynomials match known values at cos(alpha) = 0, 1, -1.
- Sparse regression on residual recovers quadrupole coefficient in synthetic test.

### Enhancement B: Shape-Space Features (Montgomery-Moeckel Decomposition)

**Priority: High.**

Idea:
- Decompose the three-body configuration into scale, shape, and rotation:
  - `R = sqrt(I / M)` where `I = sum_i m_i |r_i - r_COM|^2` (hyperradius)
  - `s = (r1, r2, r3) / R` (point on the shape sphere S^2)
- The shape sphere parameterizes all triangle shapes (equilateral at poles,
  collinear at equator).
- The dynamics decompose via the Lagrange-Jacobi identity:
  - `R_ddot = 2E/M + L^2/(M*R^3) + U(s)/(M*R^2)` (radial equation)
  - `s_dot = f(s, s_dot, R, R_dot)` (shape evolution)

New feature library atoms:
1. `hyperradius` R and its time derivatives `R_dot`, `R_ddot`.
2. `shape_potential` = U(s)/R (drives triangle deformation).
3. `virial_ratio` = 2T / |W| (kinetic vs potential energy ratio).
4. `triangle_area` = |rho1 x rho2| / 2 (scalar invariant of shape).
5. `eccentricity_inner` = eccentricity of inner Keplerian orbit (from rho1, rho1_dot).
6. Euler-Lagrange shape variables:
   - `xi = (d23^2 - d13^2) / (2*R^2)`
   - `eta = (d12^2 - (d23^2 + d13^2)/2) / (sqrt(3)*R^2)`

These are coordinate-invariant quantities that reveal structure invisible in Cartesian components.

Implementation:
1. Add `shape.rs` in `threebody-core/src/math/` for shape-space computations.
2. Compute moment of inertia, hyperradius, and shape coordinates each timestep.
3. Extend CSV output with shape-space columns (opt-in via `--shape-diagnostics`).
4. Add `shape` feature library variant with the above atoms.

TDD:
- Hyperradius is invariant under rigid rotation.
- Shape variables (xi, eta) map equilateral triangle to origin.
- Lagrange-Jacobi identity holds within integrator tolerance.
- Triangle area matches cross-product formula.

### Enhancement C: Poincare Return Map Discovery (Encounter-to-Encounter)

**Priority: High. Highest potential for genuinely novel compact laws.**

Idea:
- Instead of predicting the continuous trajectory r(t), predict the **discrete map**
  between successive close encounters. The existing `predictability extract` command
  already identifies encounters.
- Build a map: `(E_after, L_after, pair_after) = F(E_before, L_before, pair_before, impact_params)`
- This sidesteps chaos entirely: you predict *outcomes* of scattering events, not the
  chaotic trajectory between them.

A sparse model of the return map could be:
```
Delta_E / E = f(v_inf/v_orb, b/a, m3/m12, cos theta)
```
where `v_inf` is incoming velocity, `b` is impact parameter, `a` is semi-major axis,
`theta` is orientation angle.

Recent work (Stone-Leigh statistical theory, Boekholt-Portegies Zwart) suggests
three-body scattering outcomes have surprisingly compact statistical structure.

New feature atoms for the return map:
1. `v_inf_ratio` = v_infinity / v_orbital (normalized incoming speed).
2. `impact_param` = closest approach distance normalized by semi-major axis.
3. `mass_ratio` = m3 / (m1 + m2) (perturber mass fraction).
4. `cos_theta` = orientation of incoming trajectory relative to binary.
5. `energy_exchange` = Delta_E / E (target for the map).
6. `angular_momentum_exchange` = Delta_L / L (secondary target).

Implementation:
1. Extend `predictability extract` to compute orbital elements at each encounter.
2. Build encounter-pair dataset: (state_before, state_after) for each scattering event.
3. Add `return_map` discovery mode that fits sparse models on encounter-pair features.
4. Evaluate by predicting encounter outcomes on held-out encounters.

TDD:
- Encounter extraction is deterministic and captures all local minima of min_pair_dist.
- Orbital element computation matches known Keplerian test cases.
- Return map predictions conserve total energy within tolerance.
- Map accuracy on held-out encounters exceeds random baseline.

### Enhancement D: Symmetry-Adapted Invariant Features

**Priority: Medium. Low effort, good payoff for equal-mass systems.**

Idea:
- For equal-mass bodies, the configuration has S3 permutation symmetry.
- Build features from the **fundamental invariants** of this symmetry group:
  - `sigma1 = d12 + d23 + d13` (sum of distances)
  - `sigma2 = d12*d23 + d23*d13 + d13*d12` (sum of distance products)
  - `sigma3 = d12*d23*d13` (product of all distances)
- Any permutation-invariant function can be expressed in terms of (sigma1, sigma2, sigma3).
- For unequal masses, use mass-weighted versions.

Implementation:
1. Compute elementary symmetric polynomials of pairwise distances each timestep.
2. Add `symmetric` feature library variant with sigma1, sigma2, sigma3 and their time derivatives.
3. Include sigma-based gating: e.g., `sigma3 < threshold` detects near-collision.

TDD:
- Symmetric polynomials are invariant under body relabeling (permutation test).
- sigma1/sigma2/sigma3 distinguish equilateral from collinear from hierarchical configs.
- Feature values are finite and bounded away from NaN for non-degenerate configs.

### Enhancement E: Orbital Element Features (Secular Dynamics / Kozai-Lidov)

**Priority: Medium-High. Essential for hierarchical triples.**

Idea:
- For hierarchical triples (one tight pair + one distant body), the **Kozai-Lidov mechanism**
  produces oscillations between eccentricity and inclination:
  - `(1 - e^2) * cos^2(i) ~ const` (approximate integral of motion)
- The current system cannot discover this because it lacks orbital element features.

New feature atoms:
1. `e_inner` = eccentricity of inner binary (from rho1, rho1_dot).
2. `i_mutual` = mutual inclination between inner and outer orbit planes.
3. `omega_inner` = argument of pericenter of inner orbit.
4. `Omega_relative` = longitude of ascending node.
5. `kozai_invariant` = `(1 - e_inner^2) * cos^2(i_mutual)` (the KL constant).
6. `octupole_param` = `e_outer * (a_inner/a_outer)` (octupole-level KL parameter).

Implementation:
1. Add `orbital_elements.rs` in `threebody-core/src/math/` for (r, v) -> (a, e, i, omega, Omega, nu).
2. Handle edge cases: circular orbits (undefined omega), unbound orbits (hyperbolic e > 1).
3. Add `secular` feature library variant containing orbital element atoms.
4. Log KL invariant drift as a diagnostic for hierarchical configurations.

TDD:
- Orbital elements are correct for known Keplerian test orbit (circular, elliptical).
- KL invariant is approximately constant over many outer-orbit periods in hierarchical triple.
- Features handle near-circular and near-parabolic orbits without NaN.

### Enhancement F: Information-Geometric / Jacobi Metric Curvature

**Priority: Medium. Connects to sensitivity analysis already in the code.**

Idea:
- The Maupertuis principle says three-body orbits are geodesics of the Jacobi metric:
  - `ds^2 = (E - V(r)) * sum_i m_i dr_i^2`
- The Gaussian curvature of this metric determines local stability.
- A single scalar (Ricci curvature) encodes whether nearby trajectories converge or diverge.
- This is what the sensitivity analysis measures, but as a local geometric invariant.

New feature atoms:
1. `jacobi_curvature` = Ricci scalar of the Jacobi metric at current configuration.
2. `curvature_sign` = sign of curvature (positive = focusing, negative = defocusing).
3. `geodesic_deviation` = rate of separation of nearby geodesics.

Implementation:
1. Compute potential energy Hessian (second derivatives of V w.r.t. positions).
2. Compute Jacobi metric curvature from V, grad(V), and Hessian.
3. Add as optional diagnostic column and feature library atom.

TDD:
- Curvature is positive for bound two-body Keplerian orbit (known result).
- Curvature sign correlates with observed Lyapunov exponent sign.
- Computed curvature is invariant under rigid rotation.

---

## Part II: Chaos-Capture Enhancements (Math Outside Physics)

### Goal
Improve predictive power and scientific discovery for deterministic chaos by combining:
- local deterministic formulas,
- operator/statistical models after predictability horizon,
- strict holdout and falsification gates.

This section extends the factory pipeline with methods from dynamical systems,
information theory, and machine learning.

### Phase 0: Shared Infrastructure (First)
1. Add split manager for `train`, `val`, `holdout` trajectories and seeds.
2. Define horizons:
   - `short_horizon`: direct rollout RMSE target.
   - `shadow_horizon`: best-orbit matching target.
   - `stat_horizon`: distribution/invariant target.
3. Add common metrics report:
   - rollout RMSE,
   - divergence/shadowing time,
   - Wasserstein distance on occupancy,
   - entropy-rate error,
   - MDL score.
4. Add comparison runner that evaluates all candidate families on identical IC suites.

Acceptance gates:
- deterministic reruns reproduce metrics within tolerance,
- metric schema is stable and JSON serializable,
- no train/holdout contamination in tests.

### Enhancement 1: Takens Delay Embedding + Local Maps
Idea:
- Work in reconstructed state space:
  - `z_t = [x_t, x_{t-tau}, ..., x_{t-(m-1)tau}]`.
- Fit local models (kNN + ridge linear/rational) for one-step prediction.

Implementation:
1. Add embedding builder with parameters `(tau, m)`.
2. Add local model family:
   - local linear map,
   - local rational map with coefficient regularization.
3. Add hyperparameter search over `(tau, m, k, lambda)`.

TDD:
- embedding dimensions and indexing correctness,
- no future leakage,
- synthetic chaotic benchmark (Lorenz-like fixture) outperforms naive autoregression.

### Enhancement 2: Koopman / EDMD Operator Models
Idea:
- Lift to nonlinear observables `phi(z)` and fit linear operator:
  - `phi(z_{t+1}) ~ K phi(z_t)`.

Implementation:
1. Add dictionary library:
   - polynomial,
   - radial basis,
   - trig/mixed sparse features.
2. Fit `K` with ridge/Tikhonov regularization.
3. Multi-step forecast via repeated operator application.
4. Add spectral diagnostics (dominant eigenvalues/eigenfunctions).

TDD:
- stable matrix dimensions/eigendecomposition,
- one-step and multi-step error checks,
- regularization prevents blow-up on noisy fixtures.

### Enhancement 3: Ulam Transfer Operator (Probabilistic Evolution)
Idea:
- Partition embedded space into cells and estimate transition matrix `P`.
- Use `P` for long-horizon distribution forecasts when point forecasts fail.

Implementation:
1. Add partitioners:
   - k-means cells,
   - uniform grid fallback.
2. Estimate row-stochastic transition matrix with smoothing.
3. Forecast occupancy distribution and invariant measure estimate.

TDD:
- `P` rows sum to 1,
- stationary distribution exists for ergodic fixtures,
- distribution forecast beats persistence baseline at long horizon.

### Enhancement 4: Symbolic Dynamics + Entropy Models
Idea:
- Convert trajectories to symbol sequences and model transition grammar.

Implementation:
1. Symbolizer from partition IDs.
2. Fit variable-order Markov model (or fixed-order baseline first).
3. Compute entropy rate and sequence likelihood on holdout.
4. Add recurrence/periodicity diagnostics from symbol strings.

TDD:
- deterministic symbolization for fixed partition seed,
- entropy estimator sanity checks on known synthetic sequences,
- holdout likelihood improves over iid-symbol baseline.

### Enhancement 5: Topological Gating Features for Atlas Selection
Idea:
- Use recurrence and topology-inspired summaries to choose local chart/model.

Implementation:
1. Add recurrence quantification features:
   - recurrence rate,
   - determinism,
   - laminarity proxies.
2. Add sliding-window geometry features (local manifold curvature proxies).
3. Use these features to route between atlas charts.

TDD:
- feature invariance under simple reindexing/translation transforms where expected,
- gating model deterministic and bounded,
- routing improves local error vs single-chart baseline.

### Enhancement 6: MDL (Minimum Description Length) Ranking
Idea:
- Replace pure RMSE ranking with compression-aware objective:
  - `L = model_bits + residual_bits`.

Implementation:
1. Add model codelength approximation from term count/magnitude precision.
2. Add residual codelength via Gaussian/NLL proxy.
3. Integrate MDL into `candidate_sort_key` as alternative objective.

TDD:
- MDL monotonicity checks (more complex equally-accurate model should score worse),
- reproducible bit estimates across runs,
- no regression in strict holdout behavior.

### Enhancement 7: Shadowing-Based Evaluation
Idea:
- Chaotic systems diverge pointwise; measure whether prediction shadows true orbit.

Implementation:
1. Add finite-horizon shadowing metric:
   - minimal trajectory distance under local time re-alignment.
2. Report `shadowing_time` and `shadowing_error`.
3. Add gate requiring shadowing gains before improvement claims.

TDD:
- known near-identical trajectories get high shadowing time,
- random trajectories fail shadowing gate,
- metric stable under timestep refinement.

### Enhancement 8: Reservoir Baseline + Distillation to Sparse Charts
Idea:
- Use strong black-box short-horizon predictor, then distill to interpretable local equations.

Implementation:
1. Add simple Echo State Network baseline.
2. Use ESN predictions as teacher targets for sparse local chart fitting.
3. Keep teacher only as benchmark; claims remain tied to interpretable student.

TDD:
- ESN deterministic with fixed seed,
- ESN outperforms simple baselines short horizon,
- distilled sparse charts preserve most teacher gain on validation.

---

## Part III: Oracle and Infrastructure Improvements

### Enhancement 9: Electromagnetic Model Upgrades

The current quasi-static, nonrelativistic EM approximation omits retardation, radiation
reaction, and self-field effects, limiting validity to low-speed regimes.

#### 9a: Retarded Potentials (Lienard-Wiechert)
- Upgrade from instantaneous Coulomb/Biot-Savart to Lienard-Wiechert potentials that
  account for finite propagation speed.
- Requires solving the retarded-time equation `|r_i(t) - r_j(t_ret)| = c * (t - t_ret)`
  iteratively at each evaluation.
- Significant computational cost increase but essential for relativistic regimes.

#### 9b: Radiation Reaction (Abraham-Lorentz)
- Include the Abraham-Lorentz force `F_rad = (q^2 / 6*pi*epsilon_0*c^3) * a_dot`
  or its Lorentz-Dirac variant.
- Critical for charged particles in oscillatory or high-acceleration orbits.
- Introduces third-order ODE (jerk appears in equations of motion), requiring
  specialized integration or order reduction.

#### 9c: Full Field Solver
- Replace particle-based force summation with a grid-based Maxwell solver
  (e.g., finite-difference time-domain / FDTD) to capture wave propagation
  and field energy explicitly.
- Enables modeling of radiation, field energy conservation, and retardation effects
  without the Lienard-Wiechert iterative solve.
- Consider custom Rust FDTD implementation for performance.

#### 9d: Validation Against Full Maxwell-Lorentz
- Run benchmarks comparing quasi-static oracle against an established full-EM solver.
- Quantify approximation errors as a function of v/c and charge scale.
- Use symbolic verification (e.g., sympy) to validate force expressions analytically.

TDD:
- Lienard-Wiechert reduces to Coulomb in v -> 0 limit.
- Abraham-Lorentz force is zero for constant-velocity motion.
- Field solver energy conservation includes field energy (Poynting flux).
- Benchmark RMSE between quasi-static and full-EM quantified per regime.

### Enhancement 10: Numerical Integration Improvements

#### 10a: Structure-Preserving Methods for EM
- Implement the Boris pusher for charged particles, which conserves energy better
  in magnetic fields.
- For combined gravity-EM, explore variational integrators or symplectic methods
  tailored to non-Hamiltonian terms (He et al. 2015, Higuera-Cary 2017).
- The existing `boris.rs` and `implicit_midpoint.rs` should be promoted to
  first-class integrator choices in the factory loop.

#### 10b: Hybrid Integration Schemes
- Combine leapfrog for long-range gravity with substepping Boris for EM near
  close encounters.
- Use adaptive dt more aggressively, with user-tunable tolerances based on per-step
  energy drift.
- Support different integrators for different force components (operator splitting).

#### 10c: Higher Precision Arithmetic
- Switch to arbitrary-precision libraries for close encounters to mitigate
  floating-point errors in stiff regimes.
- Consider f128 or MPFR-based fallback when RK45 adaptive dt drops below a threshold.

#### 10d: Symplecticity Diagnostics
- Add diagnostics to verify long-term structure preservation:
  - monitor Poincare invariants,
  - track symplectic form deviation,
  - compare energy drift between symplectic and non-symplectic integrators.

TDD:
- Boris integrator conserves energy in uniform B-field gyration test.
- Hybrid scheme matches pure-RK45 within tolerance on gravity-only test.
- Higher precision mode reduces close-encounter error by measurable factor.
- Symplecticity diagnostic flags non-symplectic integrator correctly.

### Enhancement 11: Close Encounter and Stiffness Handling

#### 11a: KS / Chain Regularization
- Implement Kustaanheimo-Stiefel (KS) regularization for binary close approaches:
  transform to 4D spinor coordinates where the Kepler singularity is removed.
- Implement chain regularization (Mikkola-Aarseth) for multi-body encounters:
  chain the closest pairs and regularize each link.
- These are truth-grade alternatives to softening that preserve the actual physics.

#### 11b: Dynamic Softening Ramps
- Make epsilon adaptive based on both distance and velocity (not just distance):
  `epsilon = f(r_min, v_rel)` to minimize force discontinuities during fast encounters.
- Log all regularization events with before/after epsilon and step metadata.

#### 11c: Multi-Timescale Methods
- Use operator splitting with different timescales for fast EM vs slow gravity.
- Implement Sundman time transformation `dt/ds = r^alpha` to regularize the
  time variable near close encounters (alpha = 1 for Sundman, alpha = 2 for
  Levi-Civita).

#### 11d: Stiffness Benchmarking
- Expand the close-encounter policy to trigger high-precision modes automatically.
- Use predictability tooling to quantify post-encounter chaos amplification.
- Create a stiffness test suite with known near-collision trajectories.

TDD:
- KS transform is invertible (round-trip test in 4D spinor space).
- Chain regularization handles simultaneous close encounters (triple collision limit).
- Sundman-transformed integration matches RK45 truth mode within tolerance.
- Stiffness benchmark suite runs without NaN or dt collapse.

### Enhancement 12: Discovery Engine Upgrades

#### 12a: Expanded Feature Libraries
- Add post-Newtonian corrections up to PN2 (velocity-dependent gravity corrections).
- Add radiation damping atoms if oracle is upgraded to include Abraham-Lorentz.
- Add auto-generated basis functions from symbolic algebra (e.g., products of
  existing atoms up to degree 2).
- For EM, include retarded-time features if oracle supports Lienard-Wiechert.

#### 12b: Improved Identifiability
- Enhance active learning for initial conditions: prioritize EM-heavy regimes by
  maximizing the EM-to-gravity acceleration ratio diagnostic.
- Use ensembles of ICs to detect overfitting (equation fits one IC but not others).
- Add explicit identifiability score per feature: correlation with other features,
  contribution to residual variance.

#### 12c: Advanced Optimization
- Implement Bayesian sparse regression for uncertainty quantification on coefficients.
- Integrate GA more deeply with MCTS: use archive scores as fitness priors for GA.
- Add physics constraints to regression: symmetry enforcement (e.g., Newton's 3rd law),
  sign constraints (e.g., gravity is attractive), dimensional analysis checks.

#### 12d: Probabilistic Extensions
- For chaotic horizons, shift to ensemble/distribution predictions.
- Evaluate models by predicted distribution accuracy (Wasserstein, KL divergence)
  rather than point-prediction RMSE.
- Add a simple neural ODE baseline for comparison (keeping sparsity as the goal).

#### 12e: Atlas Refinements
- Learn multi-chart geometries dynamically on train/validation splits.
- Allow more than 2 regimes (close/far): cluster on (r_min, v_rel, energy) to find
  natural regime boundaries.
- Replace numerical Jacobians with analytic Jacobians for sensitivity diagnostics
  where the discovered equation is simple enough to differentiate symbolically.

TDD:
- PN2 features reduce to PN1 when higher-order terms are zeroed.
- Bayesian regression confidence intervals cover true coefficient in synthetic tests.
- Physics constraints reject equations violating Newton's 3rd law.
- Multi-chart atlas outperforms 2-chart on synthetic piecewise-linear test.

### Enhancement 13: System-Level and Reproducibility Improvements

#### 13a: N-Body Generalization
- Extend from fixed N=3 to arbitrary N.
- Requires: generalized force loops, dynamic CSV schema, N-body feature libraries.
- Enable scalability tests and comparison with established N-body codes.

#### 13b: Benchmarking and Claims
- Implement the planned `predict` and `score` commands with standardized benchmarks.
- Replace single-threshold claim gates with preregistered sequential testing.
- Add explicit multiple-comparison control across policy sweeps.
- Use bootstrap CI more rigorously across multi-seed suites (minimum 20+ seeds).

#### 13c: LLM Integration
- Refine judge rubric for ablation recommendations (auto-suggest library switches,
  coordinate transforms, hyperparameter changes).
- Ensure JSON outputs include uncertainty estimates on all scores.
- Add LLM-proposed feature atoms: let the judge suggest new basis functions in
  the project's feature language, scored by the same numeric evaluators.

#### 13d: Computational Efficiency
- Profile Rust code for bottlenecks (force computation, feature evaluation, sparse regression).
- Parallelize discovery across components (a_x, a_y, a_z are independent).
- Consider SIMD vectorization for force loops and feature matrix construction.
- Add incremental compilation support for faster iteration.

#### 13e: Validation Expansions
- Add holdout suites for relativistic edge cases (high v/c).
- Add hierarchical-triple test suites for secular dynamics validation.
- Add known periodic orbit test cases (e.g., figure-eight, Lagrange points) as
  regression fixtures with exact analytic solutions.

TDD:
- N-body mode produces correct 2-body Keplerian orbit.
- Sequential testing controls false positive rate below 5%.
- Parallelized discovery produces identical results to serial mode.
- Periodic orbit fixtures are stable within integrator tolerance over 100+ periods.

---

## Recommended Execution Order

### Tier 1: Break the Ceiling (do first)
1. **Enhancement A** (Jacobi coordinates + residual fitting) -- highest expected impact.
2. **Enhancement B** (shape-space features) -- complementary view, moderate effort.
3. **Enhancement D** (symmetry-adapted invariants) -- low effort, immediate payoff.

### Tier 2: Novel Discovery Modes
4. **Enhancement C** (Poincare return maps) -- highest novelty potential.
5. **Enhancement E** (orbital elements / Kozai-Lidov) -- essential for hierarchical triples.
6. Phase 0 infrastructure (splits, metrics, comparison runner).

### Tier 3: Chaos-Capture Layer
7. Enhancement 1 (Takens local maps).
8. Enhancement 2 (Koopman/EDMD).
9. Enhancement 7 (shadowing metrics).
10. Enhancement 6 (MDL ranking).

### Tier 4: Oracle Upgrades
11. Enhancement 11a (KS/chain regularization) -- truth-grade close encounters.
12. Enhancement 10a (Boris pusher promotion) -- EM integration quality.
13. Enhancement 9a (retarded potentials) -- EM model fidelity.

### Tier 5: Distribution and Long-Horizon
14. Enhancement 3 and 4 (transfer operator, symbolic dynamics).
15. Enhancement 5 (topological routing).
16. Enhancement 8 (reservoir + distillation).

### Tier 6: System-Level
17. Enhancement 12 (discovery engine upgrades).
18. Enhancement 13 (N-body, benchmarking, efficiency).

---

## Release Criteria For "Interesting New Direction"

Declare success only if all are true on strict holdout:
1. At least one new family improves short-horizon RMSE and shadowing time.
2. Benchmark non-regression passes.
3. Bootstrap CI on median relative improvement excludes zero in favorable direction.
4. Improvement replicates across at least two additional seed suites.
5. Report includes both wins and failure cases.

## Release Criteria For "Novel Mathematical Discovery"

Stronger gate for claiming genuinely new math:
1. All criteria above are met.
2. The discovered equation uses features from Part I (Jacobi, shape, orbital, or return map)
   that are not equivalent to a rearrangement of the Cartesian force basis.
3. The equation predicts structure not present in the Newtonian baseline (e.g., secular
   evolution, encounter outcomes, regime transitions).
4. The result is interpretable: a physicist can read the equation and understand
   what physical mechanism it captures.
5. The result generalizes to at least 3 distinct IC families (hierarchical, democratic,
   resonant) within its claimed regime of validity.
