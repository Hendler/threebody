Skip to content
Chat history




This is a copy of a conversation between ChatGPT & Anonymous.

Report conversation
You said:
Design software that can simulate thr 3 body problem and visualize it. It must be extremely accurate and performant.

Then design a system that tries to design a simple mathematical system for predicting the location. Develop new math, beyond Lagrange. 

So we have two computer systems - a simulation that is accurate. And a system that tries to predict the locations using simple math
ChatGPT said:
Below is a concrete design for two cooperating systems:

An extremely accurate + performant 3-body simulator + visualizer (the “oracle”).

A math-discovery system that searches for simple predictive laws (the “predictor”), using the simulator as ground truth and generating candidate mathematics “beyond Lagrange” (i.e., beyond just equilibrium-point thinking).

System A — Ultra-accurate 3-body simulator + visualization
A1) Goals and guarantees
Accuracy: control local truncation error + long-term invariants (energy, angular momentum) with structure-preserving methods.

Robustness: handle near-collisions, high eccentricity, chaotic regimes.

Performance: GPU visualization, multi-core integration, batch simulation for dataset generation.

Reproducibility: deterministic runs, exact metadata (initial conditions, integrator, tolerances).

A2) Physics model options (pick at runtime)
Newtonian point masses (baseline).

Post-Newtonian corrections (optional; e.g., 1PN / 2PN) for relativistic regimes.

Optional: softening, finite radii + collisions/mergers (for special studies).

A3) State and formulations
Represent the system in multiple equivalent forms and choose per scenario:

Barycentric inertial coordinates

Positions 
r
i
r 
i
​
 , momenta 
p
i
p 
i
​
 , masses 
m
i
m 
i
​
 .

Use COM frame to reduce drift.

Hamiltonian form (recommended for long-term stability)

Use symplectic integrators and track invariants naturally.

Regularized coordinates for close encounters

Switch to regularization near collisions (details below).

A4) Integrators (tiered, selectable)
You want two classes: (i) long-term faithful and fast; (ii) near-collision safe and arbitrarily accurate.

(i) Long-term faithful (default)
High-order symplectic splitting (e.g., 4th/6th/8th order Yoshida/Suzuki compositions).

Pros: excellent long-horizon qualitative fidelity; bounded energy error.

Cons: fixed step (or specialized variable-step symplectic tricks).

Symplectic correctors for improved effective order.

Use this for “planet-like” regimes and long integrations.

(ii) Arbitrary accuracy + event handling (validation / collisions)
Adaptive high-order Runge–Kutta (e.g., DOP853) or Gauss–Legendre implicit RK (symplectic, high accuracy).

Validated numerics mode (optional): interval arithmetic / Taylor-model integration to produce rigorous bounds for short/medium horizons.

Use this for ground-truth generation and near-singularity situations.

A5) Handling close approaches / collisions (critical)
Near-collisions destroy naive step methods. Use automatic regime switching:

Detect close encounter when
min
⁡
i
≠
j
∥
r
i
−
r
j
∥
<
ϵ
switch
min 
i

=j
​
 ∥r 
i
​
 −r 
j
​
 ∥<ϵ 
switch
​
  or when acceleration spikes.

Then switch to one of:

Time transformation (Sundman-type): 
d
t
=
g
(
r
)
 
d
τ
dt=g(r)dτ to slow time near collision.

Regularization transforms:

Levi-Civita for planar binary encounters.

Kustaanheimo–Stiefel (KS) for 3D binary regularization.

Chain coordinates / algorithmic regularization (commonly used in N-body codes) to preserve accuracy in hierarchical binaries.

Switch back when separation rises and stiffness is gone.

A6) Numerical precision strategy
Default: Float64 (fast).

“Truth mode”: Float128 (software quad) or arbitrary precision for hard cases (compile-time or runtime feature).

Mixed precision: compute forces in higher precision, state update in Float64 when safe.

A7) Invariants and correctness checks (always-on diagnostics)
At each output step, compute and log:

Total energy 
H
H

Total linear momentum 
P
P

Total angular momentum 
L
L

Center-of-mass drift

Symplecticity metrics (when applicable)

Local error estimates (adaptive methods)

Fail-fast or warn if thresholds exceeded.

A8) Performance architecture
Core engine (Rust or C++):

SIMD force computation (only 3 bodies, but you’ll often batch many initial conditions).

Multi-threading for batch runs (Rayon/OpenMP).

Deterministic integrator kernels.

Visualization (separate process/module):

Stream sampled trajectories to renderer.

WebGPU/Metal/Vulkan line rendering + fading trails + camera controls.

Optional: compute Lyapunov indicators / Poincaré section points and render overlays.

A9) Data & reproducibility
Define a strict “run spec”:

integrator type + order

time-step rules / tolerances

switching thresholds

coordinate formulation

precision mode

initial conditions + units

Output:

trajectory samples

diagnostics time-series

events (close approaches, collisions, ejections)

System B — Predictor that discovers “simple math” to predict locations
B1) What “beyond Lagrange” means in practice
You’re aiming for new compact representations that:

approximate trajectories in a regime (or family of regimes)

use few parameters

are interpretable (closed-form-ish)

have quantified error bounds and validity domains

Given 3-body chaos, global closed forms won’t exist in general, but you can still get:

regional / regime-specific predictive math

probabilistic/ensemble predictions

short-horizon deterministic laws

reduced-order models around invariant structures (periodic orbits, manifolds)

B2) Predictor architecture: “simulator-as-oracle” loop
The predictor system repeatedly:

Samples initial conditions (from chosen families: hierarchical, near-circular, resonant, etc.)

Calls System A to generate high-fidelity trajectories + invariants

Searches for compact mathematical models

Validates them on held-out trajectories + out-of-distribution tests

Outputs: candidate formulae + domain of validity + certified error metrics

B3) Candidate math families (things to search)
You want a portfolio, not one technique.

(1) Canonical transformations + normal forms
Automate search for transformations that simplify dynamics:

near-integrable expansions in small parameters (mass ratio, separation ratio, eccentricity)

Birkhoff / resonant normal forms around stable periodic orbits

action-angle approximations where possible

Output: compact ODEs in reduced variables + reconstruction map.

(2) Series and spectral representations
Represent coordinates as:

Fourier series in one or more fundamental frequencies

Chebyshev polynomials on time windows

piecewise spectral expansions with continuity constraints

Output: “ephemeris-like” formulae that are fast to evaluate.

(3) Invariant-manifold based prediction
Compute:

periodic orbits

stable/unstable manifolds

transitions via manifold intersections

Output: symbolic “route maps” + local parameterizations that predict motion in phase-space neighborhoods.

(4) Symbolic regression with physics constraints (key)
Use symbolic regression (genetic programming / differentiable symbolic methods) but constrain it to:

respect symmetries (translation, rotation)

preserve invariants approximately (energy-like)

produce Hamiltonian or near-Hamiltonian forms (learn 
H
(
q
,
p
)
H(q,p) then derive equations)

Output: compact expressions that generalize better than black-box nets.

(5) Learned coordinates (discovered “simple variables”)
Search for coordinates 
z
=
f
(
r
,
p
)
z=f(r,p) that make dynamics simpler:

low-dimensional embeddings

near-linear evolution in 
z
z

piecewise-linear “mode” dynamics

But require interpretability: 
f
f must be expressible as a small expression tree or structured map.

B4) “New math” deliverables you can aim for
Concrete targets the system can output:

A discovered conserved quantity or near-invariant for a regime
Example: 
I
(
r
,
p
)
I(r,p) that stays within ±δ over long times for a family.
Useful for bounding and fast prediction.

A compact surrogate Hamiltonian 
H
θ
H 
θ
​
 
Fit a small expression for 
H
θ
H 
θ
​
  such that 
q
˙
=
∂
H
/
∂
p
q
˙
​
 =∂H/∂p, 
p
˙
=
−
∂
H
/
∂
q
p
˙
​
 =−∂H/∂q.
Gives stable rollouts.

A piecewise “patch atlas” of local models

Region classifier (based on separations/energies)

Local closed-form or low-order ODE in that patch

Transition rules

A family-of-solutions parameterization around periodic orbits
Predicts motion as: base periodic orbit + low-dimensional amplitude/phase evolution.

B5) Evaluation metrics (strict)
Every candidate model must be scored on:

position error 
∥
r
(
t
)
−
r
^
(
t
)
∥
∥r(t)− 
r
^
 (t)∥ over horizons (short/medium/long)

phase error (important even when position is bounded)

invariant drift (energy/angular momentum)

robustness to small perturbations (chaos sensitivity)

computational cost per step / per eval

domain-of-validity classifier accuracy

B6) Implementation sketch
Core components

Dataset generator (wrap System A; produces trajectories + diagnostics)

Feature builder (relative vectors, Jacobi coords, energies, separations, angles)

Model search engines:

symbolic regression engine

normal-form fitter

spectral ephemeris fitter

manifold calculator (numerical continuation tools)

Validator + benchmark suite

Model packaging:

“predictor runtime” that evaluates formulae fast

exports to C/Rust/Python

How the two systems connect
Shared interface
Define a strict API:

simulate(spec, initial_conditions) -> trajectory, diagnostics, events

evaluate_candidate(model, test_set) -> scorecard

propose_next_experiments(strategy) -> new_initial_conditions

Workflow
Start with targeted regimes (hierarchical triples, restricted 3-body, resonances).

Build predictor models per regime.

Expand coverage gradually, using discovered models to guide where the simulator should sample next (active learning).

Minimal “v1” you can build first (still serious)
System A v1

Newtonian + barycentric frame

8th-order adaptive RK for truth mode

6th/8th-order symplectic fixed-step for long runs

close-approach detection + Sundman time transform (first step before full KS)

WebGPU trajectory renderer

System B v1

Symbolic regression constrained to:

functions of relative positions

invariant-penalty term in loss

Spectral ephemeris fitter on sliding windows (Chebyshev)

Benchmark harness with strict error metrics

This already produces “simple math” candidates (piecewise ephemerides + compact surrogate Hamiltonians) and tells you where they work.

Suggested next step (concrete)
Pick one target regime to avoid boiling the ocean:

Restricted 3-body (one mass ≈ 0) OR

Hierarchical triple (tight binary + distant third)

Then:

Use System A to generate a dense dataset in that regime.

Use System B to search for:

a surrogate Hamiltonian expression

a windowed Chebyshev ephemeris with error bounds

a near-invariant that predicts stability/ejection likelihood

If you tell me which regime you care about first (restricted vs hierarchical vs equal-mass chaotic), I’ll pin this down into:

module boundaries

specific integrators + switching rules

data schema

benchmark thresholds (accuracy targets)

and a concrete model search space for the “new math” engine.

You said:
We want the full system in elegant and simple rust, controlled by cli. Build now
ChatGPT said:
Below is a complete, elegant Rust CLI project you can paste into a folder and run. It includes:

3-body Newtonian simulator (barycentric frame option)

Two integrators:

Symplectic Leapfrog (Kick-Drift-Kick) for long-term stability/performance

Adaptive RK45 for accuracy (with error control)

Diagnostics: energy, momentum, angular momentum drift

Outputs: CSV (trajectory + diagnostics)

Visualization: renders a PNG trail plot (XY projection) via plotters

This is a strong V1 foundation. If you want true 3D interactive GPU visualization (wgpu/egui), we can add it on top without changing the simulation core.

1) Create the project
mkdir threebody
cd threebody
cargo init --bin
Replace the generated files with the contents below.

2) Cargo.toml
[package]
name = "threebody"
version = "0.1.0"
edition = "2021"

[dependencies]
clap = { version = "4.5", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
csv = "1.3"
anyhow = "1.0"
plotters = "0.3"
3) src/main.rs
mod sim;
mod integrators;
mod io;
mod render;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use sim::{SimConfig, Vec3, Body, SystemState};

#[derive(Parser)]
#[command(name="threebody")]
#[command(about="Accurate & performant 3-body simulation CLI (Rust)")]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Simulate and write trajectory + diagnostics to CSV.
    Simulate {
        /// JSON file containing config (masses, positions, velocities, etc.)
        #[arg(long)]
        config: String,

        /// Output CSV file path
        #[arg(long)]
        out: String,

        /// Integrator
        #[arg(long, value_enum, default_value_t=IntegratorChoice::Leapfrog)]
        integrator: IntegratorChoice,

        /// Timestep (used by leapfrog; also initial dt for RK)
        #[arg(long, default_value_t=1e-3)]
        dt: f64,

        /// Total simulation time
        #[arg(long, default_value_t=10.0)]
        t_end: f64,

        /// Output every N steps
        #[arg(long, default_value_t=10)]
        stride: usize,

        /// If set, recenters to COM and removes net momentum at t=0
        #[arg(long, default_value_t=false)]
        barycentric: bool,

        /// RK45 tolerance (ignored by leapfrog)
        #[arg(long, default_value_t=1e-9)]
        tol: f64,

        /// Minimum dt (RK45)
        #[arg(long, default_value_t=1e-12)]
        dt_min: f64,

        /// Maximum dt (RK45)
        #[arg(long, default_value_t=1e-2)]
        dt_max: f64,
    },

    /// Render a PNG plot (XY projection) from an output CSV produced by simulate.
    Render {
        /// Input CSV from simulate
        #[arg(long)]
        input: String,

        /// Output PNG path
        #[arg(long)]
        out_png: String,

        /// Image width
        #[arg(long, default_value_t=1400)]
        width: u32,

        /// Image height
        #[arg(long, default_value_t=900)]
        height: u32,

        /// Plot last N points per body (0 = all)
        #[arg(long, default_value_t=0)]
        last_n: usize,
    },

    /// Write an example config JSON to a path.
    ExampleConfig {
        #[arg(long)]
        out: String,
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum IntegratorChoice {
    Leapfrog,
    Rk45,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Commands::ExampleConfig { out } => {
            let cfg = sim::example_config();
            let s = serde_json::to_string_pretty(&cfg)?;
            std::fs::write(&out, s).with_context(|| format!("write {out}"))?;
            println!("Wrote example config to {out}");
        }

        Commands::Simulate {
            config, out, integrator, dt, t_end, stride, barycentric, tol, dt_min, dt_max
        } => {
            let cfg = io::read_config_json(&config)
                .with_context(|| format!("read config {config}"))?;

            let mut state = SystemState::from_config(&cfg);

            if barycentric {
                state.make_barycentric();
            }

            let simcfg = SimConfig {
                g: cfg.g,
                softening: cfg.softening,
            };

            let mut writer = io::CsvWriter::new(&out)
                .with_context(|| format!("create output csv {out}"))?;

            match integrator {
                IntegratorChoice::Leapfrog => {
                    integrators::run_leapfrog(
                        &simcfg,
                        &mut state,
                        dt,
                        t_end,
                        stride,
                        &mut writer,
                    )?;
                }
                IntegratorChoice::Rk45 => {
                    integrators::run_rk45(
                        &simcfg,
                        &mut state,
                        dt,
                        t_end,
                        stride,
                        tol,
                        dt_min,
                        dt_max,
                        &mut writer,
                    )?;
                }
            }

            writer.finish()?;
            println!("Done. Wrote {out}");
        }

        Commands::Render { input, out_png, width, height, last_n } => {
            let rows = io::read_trajectory_csv(&input)
                .with_context(|| format!("read trajectory csv {input}"))?;
            render::render_xy_trails(&rows, &out_png, width, height, last_n)
                .with_context(|| format!("render {out_png}"))?;
            println!("Wrote {out_png}");
        }
    }

    Ok(())
}
4) src/sim.rs
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn dot(self, o: Vec3) -> f64 { self.x*o.x + self.y*o.y + self.z*o.z }
    pub fn norm2(self) -> f64 { self.dot(self) }
    pub fn add(self, o: Vec3) -> Vec3 { Vec3::new(self.x+o.x, self.y+o.y, self.z+o.z) }
    pub fn sub(self, o: Vec3) -> Vec3 { Vec3::new(self.x-o.x, self.y-o.y, self.z-o.z) }
    pub fn mul(self, s: f64) -> Vec3 { Vec3::new(self.x*s, self.y*s, self.z*s) }
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y*o.z - self.z*o.y,
            self.z*o.x - self.x*o.z,
            self.x*o.y - self.y*o.x,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Body {
    pub name: String,
    pub m: f64,
    pub r: Vec3,
    pub v: Vec3,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConfigFile {
    /// Gravitational constant
    pub g: f64,
    /// Force softening length (0 = none); useful to avoid singularities in some studies.
    pub softening: f64,
    /// Exactly 3 bodies
    pub bodies: Vec<Body>,
}

#[derive(Clone, Copy, Debug)]
pub struct SimConfig {
    pub g: f64,
    pub softening: f64,
}

/// Internal mutable state
#[derive(Clone, Debug)]
pub struct SystemState {
    pub t: f64,
    pub m: [f64; 3],
    pub r: [Vec3; 3],
    pub v: [Vec3; 3],
    pub names: [String; 3],
}

impl SystemState {
    pub fn from_config(cfg: &ConfigFile) -> Self {
        assert!(cfg.bodies.len() == 3, "config must include exactly 3 bodies");
        let b0 = &cfg.bodies[0];
        let b1 = &cfg.bodies[1];
        let b2 = &cfg.bodies[2];
        SystemState {
            t: 0.0,
            m: [b0.m, b1.m, b2.m],
            r: [b0.r, b1.r, b2.r],
            v: [b0.v, b1.v, b2.v],
            names: [b0.name.clone(), b1.name.clone(), b2.name.clone()],
        }
    }

    /// Shift to center-of-mass frame and remove net momentum.
    pub fn make_barycentric(&mut self) {
        let mt = self.m[0] + self.m[1] + self.m[2];
        let com_r = self.r[0].mul(self.m[0])
            .add(self.r[1].mul(self.m[1]))
            .add(self.r[2].mul(self.m[2]))
            .mul(1.0 / mt);
        for i in 0..3 {
            self.r[i] = self.r[i].sub(com_r);
        }

        let p = self.v[0].mul(self.m[0])
            .add(self.v[1].mul(self.m[1]))
            .add(self.v[2].mul(self.m[2]));
        let vcom = p.mul(1.0 / mt);
        for i in 0..3 {
            self.v[i] = self.v[i].sub(vcom);
        }
    }
}

pub fn accelerations(cfg: &SimConfig, state: &SystemState) -> [Vec3; 3] {
    let g = cfg.g;
    let eps2 = cfg.softening * cfg.softening;

    let mut a = [Vec3::default(), Vec3::default(), Vec3::default()];

    for i in 0..3 {
        let mut acc = Vec3::default();
        for j in 0..3 {
            if i == j { continue; }
            let rij = state.r[j].sub(state.r[i]);
            let d2 = rij.norm2() + eps2;
            let inv_r = 1.0 / d2.sqrt();
            let inv_r3 = inv_r * inv_r * inv_r;
            acc = acc.add(rij.mul(g * state.m[j] * inv_r3));
        }
        a[i] = acc;
    }
    a
}

pub fn total_energy(cfg: &SimConfig, state: &SystemState) -> f64 {
    // Kinetic
    let mut ke = 0.0;
    for i in 0..3 {
        ke += 0.5 * state.m[i] * state.v[i].norm2();
    }
    // Potential
    let g = cfg.g;
    let eps = cfg.softening;
    let mut pe = 0.0;
    for i in 0..3 {
        for j in (i+1)..3 {
            let rij = state.r[j].sub(state.r[i]);
            let d = (rij.norm2() + eps*eps).sqrt();
            pe += -g * state.m[i] * state.m[j] / d;
        }
    }
    ke + pe
}

pub fn total_momentum(state: &SystemState) -> Vec3 {
    state.v[0].mul(state.m[0]).add(state.v[1].mul(state.m[1])).add(state.v[2].mul(state.m[2]))
}

pub fn total_angular_momentum(state: &SystemState) -> Vec3 {
    let mut l = Vec3::default();
    for i in 0..3 {
        let p = state.v[i].mul(state.m[i]);
        l = l.add(state.r[i].cross(p));
    }
    l
}

pub fn example_config() -> ConfigFile {
    // Classic-ish “figure-eight” initial conditions are common; this is a decent demo.
    // Units arbitrary; use G=1.
    // These values are a well-known starting point (may need small adjustment by user).
    ConfigFile {
        g: 1.0,
        softening: 0.0,
        bodies: vec![
            Body {
                name: "A".to_string(),
                m: 1.0,
                r: Vec3::new(-0.97000436, 0.24308753, 0.0),
                v: Vec3::new(0.4662036850, 0.4323657300, 0.0),
            },
            Body {
                name: "B".to_string(),
                m: 1.0,
                r: Vec3::new(0.97000436, -0.24308753, 0.0),
                v: Vec3::new(0.4662036850, 0.4323657300, 0.0),
            },
            Body {
                name: "C".to_string(),
                m: 1.0,
                r: Vec3::new(0.0, 0.0, 0.0),
                v: Vec3::new(-0.93240737, -0.86473146, 0.0),
            },
        ],
    }
}
5) src/integrators.rs
use anyhow::Result;

use crate::io::CsvWriter;
use crate::sim::{accelerations, total_angular_momentum, total_energy, total_momentum, SimConfig, SystemState, Vec3};

fn write_row(cfg: &SimConfig, st: &SystemState, w: &mut CsvWriter, e0: f64, p0: Vec3, l0: Vec3) -> Result<()> {
    let e = total_energy(cfg, st);
    let p = total_momentum(st);
    let l = total_angular_momentum(st);

    w.write(st.t, st, e, p, l, e - e0, p.sub(p0), l.sub(l0))?;
    Ok(())
}

/// Symplectic kick-drift-kick leapfrog (2nd order), very stable long-term.
/// Extremely fast and good for “performance mode”.
pub fn run_leapfrog(
    cfg: &SimConfig,
    st: &mut SystemState,
    dt: f64,
    t_end: f64,
    stride: usize,
    w: &mut CsvWriter,
) -> Result<()> {
    let e0 = total_energy(cfg, st);
    let p0 = total_momentum(st);
    let l0 = total_angular_momentum(st);

    let mut step: usize = 0;
    write_row(cfg, st, w, e0, p0, l0)?;

    while st.t < t_end {
        let a0 = accelerations(cfg, st);

        // Kick 1/2
        for i in 0..3 {
            st.v[i] = st.v[i].add(a0[i].mul(0.5 * dt));
        }

        // Drift
        for i in 0..3 {
            st.r[i] = st.r[i].add(st.v[i].mul(dt));
        }

        // Kick 1/2 with new accel
        let a1 = accelerations(cfg, st);
        for i in 0..3 {
            st.v[i] = st.v[i].add(a1[i].mul(0.5 * dt));
        }

        st.t += dt;
        step += 1;

        if step % stride == 0 {
            write_row(cfg, st, w, e0, p0, l0)?;
        }
    }

    Ok(())
}

/// Adaptive RK45 (Dormand–Prince style). Higher accuracy per unit time than leapfrog,
/// with explicit local error control. Not symplectic, but excellent for “truth mode”
/// over moderate horizons.
pub fn run_rk45(
    cfg: &SimConfig,
    st: &mut SystemState,
    mut dt: f64,
    t_end: f64,
    stride: usize,
    tol: f64,
    dt_min: f64,
    dt_max: f64,
    w: &mut CsvWriter,
) -> Result<()> {
    let e0 = total_energy(cfg, st);
    let p0 = total_momentum(st);
    let l0 = total_angular_momentum(st);

    let mut accepted_steps: usize = 0;
    write_row(cfg, st, w, e0, p0, l0)?;

    while st.t < t_end {
        if st.t + dt > t_end {
            dt = t_end - st.t;
        }

        let (st_new, err) = rk45_step(cfg, st, dt);
        // Scale error to position+velocity magnitude
        let scale = state_scale(st).max(1e-12);
        let rel = err / scale;

        if rel <= tol {
            // accept
            *st = st_new;
            st.t += dt;
            accepted_steps += 1;

            if accepted_steps % stride == 0 {
                write_row(cfg, st, w, e0, p0, l0)?;
            }

            // increase dt cautiously
            let safety = 0.9;
            let factor = (safety * (tol / rel.max(1e-18)).powf(0.2)).clamp(0.2, 5.0);
            dt = (dt * factor).clamp(dt_min, dt_max);
        } else {
            // reject -> shrink dt
            let safety = 0.9;
            let factor = (safety * (tol / rel.max(1e-18)).powf(0.25)).clamp(0.1, 0.5);
            dt = (dt * factor).clamp(dt_min, dt_max);
            if dt <= dt_min {
                // forced accept at dt_min to avoid stalling
                *st = st_new;
                st.t += dt;
                accepted_steps += 1;
            }
        }
    }

    Ok(())
}

fn state_scale(st: &SystemState) -> f64 {
    // A simple magnitude scale for adaptive error: max(|r|,|v|) across bodies.
    let mut m = 0.0;
    for i in 0..3 {
        m = m.max(st.r[i].norm2().sqrt());
        m = m.max(st.v[i].norm2().sqrt());
    }
    m
}

/// Packs state into y = [r0,v0,r1,v1,r2,v2]
fn pack(st: &SystemState) -> [f64; 18] {
    let mut y = [0.0; 18];
    for i in 0..3 {
        let base = i * 6;
        y[base + 0] = st.r[i].x;
        y[base + 1] = st.r[i].y;
        y[base + 2] = st.r[i].z;
        y[base + 3] = st.v[i].x;
        y[base + 4] = st.v[i].y;
        y[base + 5] = st.v[i].z;
    }
    y
}

fn unpack(st: &SystemState, y: [f64; 18]) -> SystemState {
    let mut out = st.clone();
    for i in 0..3 {
        let base = i * 6;
        out.r[i] = Vec3::new(y[base+0], y[base+1], y[base+2]);
        out.v[i] = Vec3::new(y[base+3], y[base+4], y[base+5]);
    }
    out
}

fn deriv(cfg: &SimConfig, st: &SystemState, y: [f64; 18]) -> [f64; 18] {
    let tmp = unpack(st, y);
    let a = accelerations(cfg, &tmp);

    let mut dy = [0.0; 18];
    for i in 0..3 {
        let base = i * 6;
        // dr/dt = v
        dy[base+0] = tmp.v[i].x;
        dy[base+1] = tmp.v[i].y;
        dy[base+2] = tmp.v[i].z;
        // dv/dt = a
        dy[base+3] = a[i].x;
        dy[base+4] = a[i].y;
        dy[base+5] = a[i].z;
    }
    dy
}

/// A compact RK45 step with embedded error estimate.
/// Returns (new_state_without_time_advance, error_norm)
fn rk45_step(cfg: &SimConfig, st: &SystemState, dt: f64) -> (SystemState, f64) {
    // Dormand–Prince coefficients (classic RK5(4))
    let y0 = pack(st);

    let k1 = deriv(cfg, st, y0);

    let y2 = add_scaled(y0, &k1, dt * (1.0/5.0));
    let k2 = deriv(cfg, st, y2);

    let y3 = add_scaled2(y0, &k1, &k2, dt * (3.0/40.0), dt * (9.0/40.0));
    let k3 = deriv(cfg, st, y3);

    let y4 = add_scaled3(
        y0, &k1, &k2, &k3,
        dt*(44.0/45.0), dt*(-56.0/15.0), dt*(32.0/9.0)
    );
    let k4 = deriv(cfg, st, y4);

    let y5 = add_scaled4(
        y0, &k1, &k2, &k3, &k4,
        dt*(19372.0/6561.0), dt*(-25360.0/2187.0), dt*(64448.0/6561.0), dt*(-212.0/729.0)
    );
    let k5 = deriv(cfg, st, y5);

    let y6 = add_scaled5(
        y0, &k1, &k2, &k3, &k4, &k5,
        dt*(9017.0/3168.0), dt*(-355.0/33.0), dt*(46732.0/5247.0), dt*(49.0/176.0), dt*(-5103.0/18656.0)
    );
    let k6 = deriv(cfg, st, y6);

    let y7 = add_scaled6(
        y0, &k1, &k2, &k3, &k4, &k5, &k6,
        dt*(35.0/384.0), 0.0, dt*(500.0/1113.0), dt*(125.0/192.0), dt*(-2187.0/6784.0), dt*(11.0/84.0)
    );
    let k7 = deriv(cfg, st, y7);

    // 5th order solution
    let y5th = y7;

    // 4th order embedded solution
    let y4th = add_scaled6(
        y0, &k1, &k2, &k3, &k4, &k5, &k6,
        dt*(5179.0/57600.0),
        0.0,
        dt*(7571.0/16695.0),
        dt*(393.0/640.0),
        dt*(-92097.0/339200.0),
        dt*(187.0/2100.0)
    );
    // plus k7 term for the 4th solution:
    let y4th = add_scaled(y4th, &k7, dt*(1.0/40.0));

    let err = diff_norm(y5th, y4th);
    (unpack(st, y5th), err)
}

fn add_scaled(y: [f64; 18], k: &[f64; 18], a: f64) -> [f64; 18] {
    let mut out = y;
    for i in 0..18 { out[i] += a * k[i]; }
    out
}

fn add_scaled2(y: [f64; 18], k1: &[f64; 18], k2: &[f64; 18], a1: f64, a2: f64) -> [f64; 18] {
    let mut out = y;
    for i in 0..18 { out[i] += a1*k1[i] + a2*k2[i]; }
    out
}

fn add_scaled3(y: [f64; 18], k1: &[f64; 18], k2: &[f64; 18], k3: &[f64; 18], a1: f64, a2: f64, a3: f64) -> [f64; 18] {
    let mut out = y;
    for i in 0..18 { out[i] += a1*k1[i] + a2*k2[i] + a3*k3[i]; }
    out
}

fn add_scaled4(y: [f64; 18], k1: &[f64; 18], k2: &[f64; 18], k3: &[f64; 18], k4: &[f64; 18], a1: f64, a2: f64, a3: f64, a4: f64) -> [f64; 18] {
    let mut out = y;
    for i in 0..18 { out[i] += a1*k1[i] + a2*k2[i] + a3*k3[i] + a4*k4[i]; }
    out
}

fn add_scaled5(y: [f64; 18], k1: &[f64; 18], k2: &[f64; 18], k3: &[f64; 18], k4: &[f64; 18], k5: &[f64; 18], a1: f64, a2: f64, a3: f64, a4: f64, a5: f64) -> [f64; 18] {
    let mut out = y;
    for i in 0..18 { out[i] += a1*k1[i] + a2*k2[i] + a3*k3[i] + a4*k4[i] + a5*k5[i]; }
    out
}

fn add_scaled6(
    y: [f64; 18],
    k1: &[f64; 18], k2: &[f64; 18], k3: &[f64; 18],
    k4: &[f64; 18], k5: &[f64; 18], k6: &[f64; 18],
    a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64
) -> [f64; 18] {
    let mut out = y;
    for i in 0..18 {
        out[i] += a1*k1[i] + a2*k2[i] + a3*k3[i] + a4*k4[i] + a5*k5[i] + a6*k6[i];
    }
    out
}

fn diff_norm(a: [f64; 18], b: [f64; 18]) -> f64 {
    let mut s = 0.0;
    for i in 0..18 {
        let d = a[i] - b[i];
        s += d*d;
    }
    s.sqrt()
}
6) src/io.rs
use anyhow::{Context, Result};
use csv::{ReaderBuilder, WriterBuilder};
use serde::{Deserialize, Serialize};

use crate::sim::{ConfigFile, SystemState, Vec3};

pub fn read_config_json(path: &str) -> Result<ConfigFile> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    let cfg: ConfigFile = serde_json::from_str(&s).with_context(|| "parse json config")?;
    Ok(cfg)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrajRow {
    pub t: f64,
    // positions
    pub ax: f64, pub ay: f64, pub az: f64,
    pub bx: f64, pub by: f64, pub bz: f64,
    pub cx: f64, pub cy: f64, pub cz: f64,
}

pub fn read_trajectory_csv(path: &str) -> Result<Vec<TrajRow>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let mut rows = Vec::new();
    for result in rdr.deserialize() {
        let row: TrajRow = result?;
        rows.push(row);
    }
    Ok(rows)
}

pub struct CsvWriter {
    w: csv::Writer<std::fs::File>,
}

impl CsvWriter {
    pub fn new(path: &str) -> Result<Self> {
        let file = std::fs::File::create(path)?;
        let mut w = WriterBuilder::new().has_headers(true).from_writer(file);

        // Header: trajectory + diagnostics
        w.write_record(&[
            "t",
            "a_x","a_y","a_z","a_vx","a_vy","a_vz",
            "b_x","b_y","b_z","b_vx","b_vy","b_vz",
            "c_x","c_y","c_z","c_vx","c_vy","c_vz",
            "energy",
            "p_x","p_y","p_z",
            "l_x","l_y","l_z",
            "dE",
            "dP_x","dP_y","dP_z",
            "dL_x","dL_y","dL_z",
        ])?;
        Ok(Self { w })
    }

    pub fn write(
        &mut self,
        t: f64,
        st: &SystemState,
        energy: f64,
        p: Vec3,
        l: Vec3,
        dE: f64,
        dP: Vec3,
        dL: Vec3,
    ) -> Result<()> {
        let a = 0; let b = 1; let c = 2;
        self.w.write_record(&[
            t.to_string(),
            st.r[a].x.to_string(), st.r[a].y.to_string(), st.r[a].z.to_string(),
            st.v[a].x.to_string(), st.v[a].y.to_string(), st.v[a].z.to_string(),
            st.r[b].x.to_string(), st.r[b].y.to_string(), st.r[b].z.to_string(),
            st.v[b].x.to_string(), st.v[b].y.to_string(), st.v[b].z.to_string(),
            st.r[c].x.to_string(), st.r[c].y.to_string(), st.r[c].z.to_string(),
            st.v[c].x.to_string(), st.v[c].y.to_string(), st.v[c].z.to_string(),
            energy.to_string(),
            p.x.to_string(), p.y.to_string(), p.z.to_string(),
            l.x.to_string(), l.y.to_string(), l.z.to_string(),
            dE.to_string(),
            dP.x.to_string(), dP.y.to_string(), dP.z.to_string(),
            dL.x.to_string(), dL.y.to_string(), dL.z.to_string(),
        ])?;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        self.w.flush()?;
        Ok(())
    }
}
7) src/render.rs
use anyhow::Result;
use plotters::prelude::*;

use crate::io::TrajRow;

pub fn render_xy_trails(rows: &[TrajRow], out_png: &str, width: u32, height: u32, last_n: usize) -> Result<()> {
    let n = rows.len();
    let start = if last_n == 0 || last_n >= n { 0 } else { n - last_n };
    let slice = &rows[start..];

    // Compute bounds
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for r in slice {
        for (x, y) in [
            (r.ax, r.ay),
            (r.bx, r.by),
            (r.cx, r.cy),
        ] {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }

    // add margin
    let dx = (max_x - min_x).max(1e-12);
    let dy = (max_y - min_y).max(1e-12);
    let pad = 0.08;
    min_x -= dx * pad;
    max_x += dx * pad;
    min_y -= dy * pad;
    max_y += dy * pad;

    let root = BitMapBackend::new(out_png, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("Three-Body Trajectories (XY projection)", ("sans-serif", 28))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart
        .configure_mesh()
        .x_desc("x")
        .y_desc("y")
        .draw()?;

    let a_pts = slice.iter().map(|r| (r.ax, r.ay));
    let b_pts = slice.iter().map(|r| (r.bx, r.by));
    let c_pts = slice.iter().map(|r| (r.cx, r.cy));

    // No explicit colors requested; plotters chooses defaults if you use Palette99.
    chart.draw_series(LineSeries::new(a_pts, &Palette99::pick(0)))?;
    chart.draw_series(LineSeries::new(b_pts, &Palette99::pick(1)))?;
    chart.draw_series(LineSeries::new(c_pts, &Palette99::pick(2)))?;

    // Mark final positions
    if let Some(last) = slice.last() {
        chart.draw_series(std::iter::once(Circle::new((last.ax, last.ay), 5, Palette99::pick(0).filled())))?;
        chart.draw_series(std::iter::once(Circle::new((last.bx, last.by), 5, Palette99::pick(1).filled())))?;
        chart.draw_series(std::iter::once(Circle::new((last.cx, last.cy), 5, Palette99::pick(2).filled())))?;
    }

    root.present()?;
    Ok(())
}
8) Build & run
Generate an example config
cargo run --release -- example-config --out config.json
Simulate (fast, stable long-run)
cargo run --release -- simulate \
  --config config.json \
  --out traj.csv \
  --integrator leapfrog \
  --dt 0.001 \
  --t-end 50 \
  --stride 10 \
  --barycentric
Simulate (higher accuracy, adaptive)
cargo run --release -- simulate \
  --config config.json \
  --out traj_rk.csv \
  --integrator rk45 \
  --dt 0.001 \
  --t-end 20 \
  --stride 5 \
  --tol 1e-10 \
  --dt-min 1e-12 \
  --dt-max 1e-2 \
  --barycentric
Render a PNG
cargo run --release -- render --input traj.csv --out-png trails.png --last-n 20000
What you have now (and what’s next)
This V1 is:

clean, modular Rust

CLI-controlled

accurate (RK45) and stable/performant (symplectic leapfrog)

produces reproducible CSV + invariant drift metrics

produces a visualization PNG

Next upgrades to reach “extremely accurate near-collision truth”
If you want me to extend this codebase immediately, the next “accuracy wall” items are:

Regularization near close encounters (Sundman time transform + pairwise KS/chain regularization)

Higher-order symplectic integrator (4th/6th/8th order composition) for long-horizon fidelity

Optional Float128 / arbitrary precision backend for hard cases

Interactive 3D viz (wgpu + egui) while keeping the sim core unchanged

If you say which mode you care about first (near-collision truth vs long-horizon stability), I’ll extend this exact project accordingly with the smallest set of changes.

You said:
Check for completeness and bugs - It should  model electric, gravitational, and magnetic force. V and Phi and more. 

Make the engine  discover new, simpler and novel math. Its simpler than it seems 
ChatGPT said:

Thought for 42s
You’re right: the current code is not complete for what you want, and it also has a real bug in render. Below is a rigorous check + a clean way to extend it to gravity + electric + magnetic, including φ (scalar potential), A (vector potential), and a “discover simpler math” engine that is actually simpler than it sounds (SINDy-style sparse discovery).

1) Completeness / correctness review (current project)
✅ What’s OK
Core Newtonian gravitational force is correct (pairwise inverse-square with optional softening).

Leapfrog (KDK) implementation is correct for Hamiltonian gravity and should be stable for long runs.

RK45 scaffold is reasonable for “truth mode” (though not symplectic).

❌ Real bug: render cannot read the CSV you write
simulate writes many columns (positions, velocities, energy, momentum, etc.).

render’s TrajRow expects only 10 columns (t + ax..cz).

With csv + serde deserialize, this will fail (unknown fields / mismatched schema).

Fix options (choose one):

Make render read by header name and only extract t, a_x, a_y, ... from the full CSV (best).

Write a second “slim” CSV just for rendering (easy but redundant).

Use ReaderBuilder::new().flexible(true) and a struct that includes all fields (messy).

I recommend option (1).

⚠️ “Accuracy” caveats
Softening is not “extremely accurate” physics; it’s a numerical hack. Keep it optional.

Near-collision handling is not implemented (no regularization or time transform). If you care about true accuracy in hard close passes, this must be added.

2) Add electric + magnetic + potentials (φ, A)
2.1 Minimal, physically consistent model (non-relativistic)
For body 
i
i with mass 
m
i
m 
i
​
 , charge 
q
i
q 
i
​
 , position 
r
i
r 
i
​
 , velocity 
v
i
v 
i
​
 :

Gravity:

a
i
(
g
)
=
∑
j
≠
i
G
m
j
r
j
−
r
i
∥
r
j
−
r
i
∥
3
a 
i
(g)
​
 = 
j

=i
∑
​
 Gm 
j
​
  
∥r 
j
​
 −r 
i
​
 ∥ 
3
 
r 
j
​
 −r 
i
​
 
​
 
Electric field at 
r
i
r 
i
​
  from other charges:

E
i
=
∑
j
≠
i
k
e
q
j
r
i
−
r
j
∥
r
i
−
r
j
∥
3
E 
i
​
 = 
j

=i
∑
​
 k 
e
​
 q 
j
​
  
∥r 
i
​
 −r 
j
​
 ∥ 
3
 
r 
i
​
 −r 
j
​
 
​
 
(Notice direction: field points away from positive charge.)

Magnetic field (quasi-static Biot–Savart approximation for moving point charges):

B
i
=
∑
j
≠
i
μ
0
/
(
4
π
)
  
q
j
v
j
×
(
r
i
−
r
j
)
∥
r
i
−
r
j
∥
3
B 
i
​
 = 
j

=i
∑
​
 μ 
0
​
 /(4π)q 
j
​
  
∥r 
i
​
 −r 
j
​
 ∥ 
3
 
v 
j
​
 ×(r 
i
​
 −r 
j
​
 )
​
 
Lorentz acceleration:

a
i
(
e
m
)
=
q
i
m
i
(
E
i
+
v
i
×
B
i
)
a 
i
(em)
​
 = 
m 
i
​
 
q 
i
​
 
​
 (E 
i
​
 +v 
i
​
 ×B 
i
​
 )
Total:

a
i
=
a
i
(
g
)
+
a
i
(
e
m
)
a 
i
​
 =a 
i
(g)
​
 +a 
i
(em)
​
 
This is clean and fast. It’s not fully relativistic (no radiation reaction, no retardation), but it’s the right “V1 electromagnetic dynamics” with meaningful φ and A.

2.2 Potentials you asked for
Scalar potential at i:

ϕ
i
=
∑
j
≠
i
k
e
q
j
∥
r
i
−
r
j
∥
ϕ 
i
​
 = 
j

=i
∑
​
 k 
e
​
  
∥r 
i
​
 −r 
j
​
 ∥
q 
j
​
 
​
 
Vector potential at i:

A
i
=
∑
j
≠
i
μ
0
/
(
4
π
)
  
q
j
v
j
∥
r
i
−
r
j
∥
A 
i
​
 = 
j

=i
∑
​
 μ 
0
​
 /(4π) 
∥r 
i
​
 −r 
j
​
 ∥
q 
j
​
 v 
j
​
 
​
 
You can output φ and A per body each sample.

3) Code changes (precise + minimal)
3.1 Extend config schema
In sim.rs change Body:

pub struct Body {
    pub name: String,
    pub m: f64,
    pub q: f64,   // NEW: electric charge
    pub r: Vec3,
    pub v: Vec3,
}
Extend ConfigFile:

pub struct ConfigFile {
    pub g: f64,              // gravitational constant G
    pub ke: f64,             // Coulomb constant k_e
    pub mu0_over_4pi: f64,   // μ0/(4π)
    pub softening: f64,
    pub bodies: Vec<Body>,
}
Update SystemState to include charge:

pub struct SystemState {
    pub t: f64,
    pub m: [f64; 3],
    pub q: [f64; 3],        // NEW
    pub r: [Vec3; 3],
    pub v: [Vec3; 3],
    pub names: [String; 3],
}
3.2 Replace accelerations(...) with “fields + forces”
In sim.rs, implement:

fields_and_potentials(cfg, state) -> (E[3], B[3], phi[3], A[3])

accelerations(cfg, state) -> [Vec3;3] using gravity + Lorentz

Key points:

Use the same eps2 softening for EM denominators too (optional).

Accel term for EM is (q_i / m_i) * (E_i + v_i × B_i).

3.3 Update energy diagnostics (important)
Your current total_energy only includes gravity potential and kinetic.

For EM:

Add electric potential energy:

U
e
=
∑
i
<
j
k
e
q
i
q
j
r
i
j
U 
e
​
 = 
i<j
∑
​
 k 
e
​
  
r 
ij
​
 
q 
i
​
 q 
j
​
 
​
 
Magnetic energy is trickier in full electrodynamics; for this V1, I recommend:

Track mechanical energy (KE + gravitational PE + electric PE)

Treat it as a diagnostic only (it won’t be exactly conserved due to the approximations)

So:

Rename total_energy → mechanical_energy

Add electric_potential_energy(...).

3.4 Fix render robustly
Make render read the full CSV by header names.

Replace read_trajectory_csv with a “header indexer” that:

reads headers

finds indices for t, a_x, a_y, b_x, ...

parses each row as raw strings and converts.

This avoids serde mismatch entirely.

4) CLI should control which forces are enabled
Add flags in simulate:

--gravity/--no-gravity

--electric/--no-electric

--magnetic/--no-magnetic

--potentials (emit φ and A columns)

Internally, keep a ForceFlags { gravity, electric, magnetic }.

5) “Discover new, simpler and novel math” engine (practical + simple)
The simplest “math discovery” that actually works is:

SINDy-style sparse discovery (simple idea)
You simulate and collect time series: state 
x
(
t
)
x(t) and derivatives 
x
˙
(
t
)
x
˙
 (t).

You define a library of candidate terms Θ(x):

polynomials (x, y, x², xy, …)

inverse-distance terms (1/r, 1/r², r̂/r², …)

dot/cross combos (v×B-like terms)

You solve:

x
˙
≈
Θ
(
x
)
 
Ξ
x
˙
 ≈Θ(x)Ξ
with sparsity (most coefficients zero), using sequential thresholded least squares (very short code).

Why this is “simpler than it seems”
You are not searching arbitrary formulas at first.

You give it a library of “physically plausible atoms” (including inverse powers and directional terms).

The algorithm returns a short equation (sparse coefficients) that matches the simulator.

What you get
A compact surrogate law like:

“acceleration ≈ α r̂/r² + β (v×something)/r³ …”

Domain validity can be learned by error thresholds.

CLI subcommand you should add
threebody discover --input traj.csv --target accel --library basic --out model.json

Minimal discovery algorithm (design)
Compute derivatives via:

From simulator outputs if you also write a_x,a_y,a_z each step (best)

Or finite differences on v(t) (less clean)

Build a matrix Θ for each time sample and solve least squares for each component.

Threshold small coefficients, refit, iterate.

This produces “new math” in the sense of novel simplified approximate governing equations that are correct in a regime. That’s the realistic version.

6) What I would change immediately (summary checklist)
Critical fixes
 Fix render CSV parsing by header indexing.

 Update energy diagnostics; don’t claim exact conservation once EM is enabled.

Add EM + potentials
 Add q to bodies, add ke and mu0_over_4pi to config.

 Compute E, B, φ, A.

 Add Lorentz acceleration terms.

Add discovery system
 Add discover CLI + sparse regression (SINDy).

 Add --emit-accel output columns to make discovery clean.

If you want, I can give you the exact patch code
I didn’t paste hundreds of new lines blindly because the cleanest patch depends on one choice:

Do you want the EM model to be:

Quasi-static (what I described: Coulomb + Biot–Savart approximation, Lorentz force) — simplest, fast, meaningful.

Fully relativistic (retarded potentials / Liénard–Wiechert) — much heavier and slower.

If you say “(1)” (likely), I’ll respond with:

exact updated sim.rs (fields + potentials + accel)

exact updated io.rs (writer + header-based reader)

new discover.rs module + CLI wiring

updated example_config() including charges + constants

No extra architecture—just clean Rust.

You said:
Focus on how we create and discover accurate, novel math that predict the sim well. Make sure the sim is accurate. Take into account the bugs you found and rewrite again
ChatGPT said:
Below is a clean rewrite of the system design centered on:

making the simulator truly “truth-grade” (within realistic physics assumptions), and

creating a math-discovery engine that produces accurate, novel, simple predictive equations that match the simulator.

I’m going to be blunt about constraints: for general 3-body (especially with EM) you cannot get one global closed-form; the right “novel math” is regime-specific compact laws with explicit validity domains, discovered automatically. That’s exactly what we’ll build.

A. Simulator must be accurate (and the earlier bugs fixed)
A1) Fix the known bug first: CSV schema mismatch
Bug found: render expects a small CSV but simulate writes a wide CSV → deserialization fails.

Rewrite rule: never deserialize simulation CSV into a rigid struct.
Instead:

Read CSV headers

Map required column names to indices

Parse only those fields needed (t, positions, etc.)

This makes the pipeline robust as we add more columns (E,B,φ,A,accel, etc.).

A2) Physics model for “accurate enough to be ground truth”
You asked for electric + magnetic + potentials (V/Φ), so we need a consistent nonrelativistic model.

Chosen V1 physics (consistent + fast + good ground truth)
For each body 
i
i with mass 
m
i
m 
i
​
 , charge 
q
i
q 
i
​
 , position 
r
i
r 
i
​
 , velocity 
v
i
v 
i
​
 :

Gravity:

a
i
(
g
)
=
∑
j
≠
i
G
m
j
r
j
−
r
i
∥
r
j
−
r
i
∥
3
a 
i
(g)
​
 = 
j

=i
∑
​
 Gm 
j
​
  
∥r 
j
​
 −r 
i
​
 ∥ 
3
 
r 
j
​
 −r 
i
​
 
​
 
Electric field at 
r
i
r 
i
​
 :

E
i
=
∑
j
≠
i
k
e
q
j
r
i
−
r
j
∥
r
i
−
r
j
∥
3
E 
i
​
 = 
j

=i
∑
​
 k 
e
​
 q 
j
​
  
∥r 
i
​
 −r 
j
​
 ∥ 
3
 
r 
i
​
 −r 
j
​
 
​
 
Magnetic field (quasi-static / Biot–Savart for moving charges):

B
i
=
∑
j
≠
i
μ
0
4
π
q
j
v
j
×
(
r
i
−
r
j
)
∥
r
i
−
r
j
∥
3
B 
i
​
 = 
j

=i
∑
​
  
4π
μ 
0
​
 
​
 q 
j
​
  
∥r 
i
​
 −r 
j
​
 ∥ 
3
 
v 
j
​
 ×(r 
i
​
 −r 
j
​
 )
​
 
Lorentz acceleration:

a
i
(
e
m
)
=
q
i
m
i
(
E
i
+
v
i
×
B
i
)
a 
i
(em)
​
 = 
m 
i
​
 
q 
i
​
 
​
 (E 
i
​
 +v 
i
​
 ×B 
i
​
 )
Total:

a
i
=
a
i
(
g
)
+
a
i
(
e
m
)
a 
i
​
 =a 
i
(g)
​
 +a 
i
(em)
​
 
Potentials you requested
Scalar potential:

ϕ
i
=
∑
j
≠
i
k
e
q
j
∥
r
i
−
r
j
∥
ϕ 
i
​
 = 
j

=i
∑
​
 k 
e
​
  
∥r 
i
​
 −r 
j
​
 ∥
q 
j
​
 
​
 
Vector potential:

A
i
=
∑
j
≠
i
μ
0
4
π
q
j
v
j
∥
r
i
−
r
j
∥
A 
i
​
 = 
j

=i
∑
​
  
4π
μ 
0
​
 
​
  
∥r 
i
​
 −r 
j
​
 ∥
q 
j
​
 v 
j
​
 
​
 
We compute E, B, φ, A and write them each sample.

Important accuracy note (truth-grade honesty)
This model is not full electrodynamics (no retardation, no radiation reaction). But it is a standard, coherent nonrelativistic approximation and is appropriate as a high-fidelity oracle for the discovery engine—especially if your aim is “simple predictive math”.

A3) Integrators: accuracy and stability
We use two modes:

(1) Truth mode (accuracy)
Adaptive high-order method (RK45 is OK; DOP853 would be better later).

Tight tolerance, min/max dt.

Writes out accelerations per sample so discovery doesn’t rely on noisy finite differences.

(2) Long-run stability mode (performance)
Symplectic leapfrog (KDK).

This is extremely performant and stable for gravity; with Lorentz force it’s no longer strictly Hamiltonian in the same way, but still works well for many regimes.

Near-collision handling
For a real “accurate simulator”, you must address close approaches:

either add a time transform (Sundman-type) in truth mode,

or enable a softening parameter but treat it as “modified physics”.

Rewrite policy:

Softening default = 0 (true physics).

If close encounters cause dt to collapse, allow:

truth mode: dt shrink + optional time transform (next iteration)

or user selects softening explicitly.

A4) Diagnostics: what we log (and what we do not claim)
We log:

mechanical kinetic energy

gravitational PE

electric PE

total momentum, total angular momentum

invariant drift relative to t=0

But we do not claim perfect energy conservation under the EM approximation; we treat it as a diagnostic.

B. Discovery engine: produce accurate “novel simple math” that predicts sim well
This is the heart of your ask.

B1) The key idea: discover governing equations with sparse structure
We want to approximate:

x
˙
(
t
)
=
f
(
x
(
t
)
)
x
˙
 (t)=f(x(t))
where 
x
x includes 
r
i
,
v
i
r 
i
​
 ,v 
i
​
  (and optionally derived invariants).

We will not fit a black-box network. We will discover explicit formulas like:

inverse-distance laws

dot/cross products

low-order polynomials in meaningful features
…and keep only a small number of terms.

This is sparse system identification.

Why this yields “novel math”
If you give a rich but physically structured library of candidate terms, the solver can return:

known physics (as a sanity check)

simplifications that hold in a regime

surprising combinations that approximate complex dynamics with few terms
That is the “new math” in practice: compact approximations + validity domains.

B2) Discovery should use accelerations directly (avoid derivative noise)
Derivative estimation is the #1 way to ruin discovery.

Rewrite rule: simulator writes a_i (and optionally E,B) each sample.
Then discovery learns:

a
i
≈
f
^
i
(
features
)
a 
i
​
 ≈ 
f
^
​
  
i
​
 (features)
So we discover equations for acceleration (second-order dynamics) directly.

B3) Feature design: make the search space physically meaningful
Instead of raw coordinates, build invariant-ish features:

For each pair 
i
,
j
i,j:

r
i
j
=
r
j
−
r
i
r 
ij
​
 =r 
j
​
 −r 
i
​
 

d
i
j
=
∥
r
i
j
∥
d 
ij
​
 =∥r 
ij
​
 ∥

r
^
i
j
=
r
i
j
/
d
i
j
r
^
  
ij
​
 =r 
ij
​
 /d 
ij
​
 

terms like 
r
^
i
j
/
d
i
j
2
r
^
  
ij
​
 /d 
ij
2
​
 , 
r
^
i
j
/
d
i
j
3
r
^
  
ij
​
 /d 
ij
3
​
 

For EM:

v
i
v 
i
​
 , 
v
j
v 
j
​
 

v
j
×
r
i
j
v 
j
​
 ×r 
ij
​
 , 
v
i
×
(
v
j
×
r
i
j
)
v 
i
​
 ×(v 
j
​
 ×r 
ij
​
 )

scalar invariants: 
v
i
⋅
v
j
v 
i
​
 ⋅v 
j
​
 , 
v
i
⋅
r
^
i
j
v 
i
​
 ⋅ 
r
^
  
ij
​
 

We then define a library Θ(x) of candidate terms built from these atoms.

This is what makes it simpler than it seems: the library is structured; the solver only selects a handful.

B4) The actual algorithm: SINDy (Sequential Thresholded Least Squares)
For each acceleration component (ax, ay, az) for each body, solve:

Build matrix Θ: each row = library evaluated at one sample

Solve least squares: 
Ξ
=
arg
⁡
min
⁡
∥
Θ
Ξ
−
y
∥
2
Ξ=argmin∥ΘΞ−y∥ 
2
 

Threshold small coefficients: set 
∣
Ξ
k
∣
<
λ
∣Ξ 
k
​
 ∣<λ to 0

Refit on remaining terms

Iterate 5–20 times

This returns a sparse equation:

a
x
=
∑
k
Ξ
k
 
θ
k
(
features
)
a 
x
​
 = 
k
∑
​
 Ξ 
k
​
 θ 
k
​
 (features)
Physics constraints (so it generalizes)
We add constraints/regularizers:

symmetry constraints across bodies where appropriate

penalty for violating translation/rotation invariance (by using only relative vectors)

optional: enforce “pairwise antisymmetry” structures (action-reaction style)

B5) Validate like an engineer, not like a toy regression
Discovery isn’t done when MSE is small.

We validate by rolling out the discovered model forward in time:

Use the learned 
a
^
i
a
^
  
i
​
  to integrate positions/velocities

Compare trajectories to the simulator on held-out initial conditions

Report:

short horizon error

medium horizon error

divergence time (when error exceeds threshold)

stability metrics (blow-ups, dt collapse)

If it doesn’t roll out well, it’s not a good “predictor”.

B6) “Novel math” means piecewise + regime detection
Chaos means you need local models.

We’ll build:

a small set of discovered models (each sparse)

a simple regime classifier based on features such as:

min pair distance

energy sign

hierarchy ratio 
d
m
i
n
/
d
m
a
x
d 
min
​
 /d 
max
​
 

velocity scales

Then:

choose model A/B/C based on regime

this is an “atlas of simple laws”

This is how you get simple math that predicts well.

C. Rewrite of the codebase plan (modules + CLI)
C1) CLI commands (clean and minimal)
threebody simulate ...
Writes a wide CSV with:

r, v, a

E, B, φ, A

diagnostics

threebody discover --input sim.csv --out model.json --library em_basic --lambda 1e-4

threebody predict --model model.json --config ic.json --out pred.csv

threebody score --truth sim.csv --pred pred.csv (trajectory roll-out comparisons)

threebody render --input sim.csv --out png (header-based CSV read)

C2) Data contract between sim and discovery (critical)
Sim output must include:

t

r_i, v_i, a_i for i=A,B,C

optionally E_i, B_i, phi_i, A_i

constants and flags written to a JSON sidecar (recommended)

Discovery reads by header name, so adding columns never breaks it.

D. The rewritten “accuracy-first” implementation details (what changes vs your current code)
D1) Simulator must compute and output E,B,φ,A and a
In the sim core:

compute pairwise contributions once per step

accumulate:

gravity acceleration

E field

B field

φ and A

compute Lorentz acceleration

expose:

accelerations_and_fields(...) -> {a,E,B,phi,A}

Then integrators use a for stepping and IO uses all fields for output.

D2) Render reads only needed columns by name (fix)
read headers

locate a_x,a_y,... etc

parse only those values

This permanently fixes the earlier bug and keeps the pipeline extensible.

E. What “new math beyond Lagrange” looks like in outputs
Examples of what the system can output (as JSON “formula trees”):

a sparse combination of terms like:

r
^
i
j
/
d
i
j
2
r
^
  
ij
​
 /d 
ij
2
​
 

(
v
j
×
r
i
j
)
/
d
i
j
3
(v 
j
​
 ×r 
ij
​
 )/d 
ij
3
​
 

(
v
i
×
(
v
j
×
r
i
j
)
)
/
d
i
j
3
(v 
i
​
 ×(v 
j
​
 ×r 
ij
​
 ))/d 
ij
3
​
 

low-order corrections in hierarchy ratio

and it will report:

coefficients

validity domain bounds (feature thresholds)

roll-out divergence time statistics

That is genuinely new, simplified predictive math for the regimes you train on.

F. Practical “do it now” checklist to rewrite correctly
Sim: extend state to include charge; extend config with ke and mu0_over_4pi.

Sim: compute (a,E,B,phi,A) each step; write them to CSV.

IO: replace rigid TrajRow deserialize with header-index parsing.

Discovery:

read sim CSV

build features for each sample

run STLS (SINDy)

output sparse model JSON

Predict: integrate using discovered model and compare to truth.

If you want the full Rust rewrite in one paste
I can paste the revised full project (main.rs + sim.rs + io.rs + integrators.rs + discover.rs + predict.rs + render.rs) in a single response, but I will make one assumption to keep it coherent:

Assumption for the discovery library: “em_basic” = inverse-distance + cross/dot velocity terms up to a small degree (no arbitrary polynomials).
That’s the sweet spot for “simple novel math”.

If you want a different library (e.g., include higher powers, include explicit φ/A terms, include learned coordinate transforms), say so and I’ll bake it in.

You said:
Create libraries
ChatGPT said:
Below are feature libraries (Θ) for the discovery engine: curated sets of candidate terms that are (a) physically meaningful, (b) sparse-friendly, and (c) expressive enough to discover compact “new math” that predicts the simulator.

I’m assuming the state per sample includes 
r
i
,
v
i
,
a
i
r 
i
​
 ,v 
i
​
 ,a 
i
​
  (and optionally 
E
,
B
,
ϕ
,
A
E,B,ϕ,A) for bodies 
i
∈
{
A
,
B
,
C
}
i∈{A,B,C}. All libraries below are built from relative quantities so they’re translation/rotation consistent.

1) Core atoms (shared building blocks)
For each ordered pair 
i
≠
j
i

=j:

r
i
j
=
r
j
−
r
i
r 
ij
​
 =r 
j
​
 −r 
i
​
 

d
i
j
=
∥
r
i
j
∥
2
+
ϵ
2
d 
ij
​
 = 
∥r 
ij
​
 ∥ 
2
 +ϵ 
2
 
​
  (epsilon only if you intentionally soften)

r
^
i
j
=
r
i
j
/
d
i
j
r
^
  
ij
​
 =r 
ij
​
 /d 
ij
​
 

v
i
j
=
v
j
−
v
i
v 
ij
​
 =v 
j
​
 −v 
i
​
 

Useful scalar atoms:

s
i
j
=
v
i
⋅
r
^
i
j
s 
ij
​
 =v 
i
​
 ⋅ 
r
^
  
ij
​
 

u
i
j
=
v
j
⋅
r
^
i
j
u 
ij
​
 =v 
j
​
 ⋅ 
r
^
  
ij
​
 

w
i
j
=
v
i
⋅
v
j
w 
ij
​
 =v 
i
​
 ⋅v 
j
​
 

Vector atoms:

c
i
j
=
v
j
×
r
i
j
c 
ij
​
 =v 
j
​
 ×r 
ij
​
 

t
i
j
=
v
i
×
c
i
j
=
v
i
×
(
v
j
×
r
i
j
)
t 
ij
​
 =v 
i
​
 ×c 
ij
​
 =v 
i
​
 ×(v 
j
​
 ×r 
ij
​
 )

Distance powers (choose a small set):

d
−
1
,
d
−
2
,
d
−
3
,
d
−
4
d 
−1
 ,d 
−2
 ,d 
−3
 ,d 
−4
 

From these, we generate candidate vector terms for predicting acceleration 
a
i
a 
i
​
 .

2) Library definitions (what terms go into Θ)
Each library is a list of vector-valued basis functions 
θ
k
(
x
)
∈
R
3
θ 
k
​
 (x)∈R 
3
 . The discovery engine learns sparse coefficients for:

a
i
≈
∑
k
ξ
i
,
k
 
θ
k
(
x
)
a 
i
​
 ≈ 
k
∑
​
 ξ 
i,k
​
 θ 
k
​
 (x)
(optionally sharing coefficients across bodies via symmetry constraints).

L0 — gravity_min
Smallest useful set; discovers Newton-like laws and simple corrections.

For each 
i
i:

For each 
j
≠
i
j

=i:

θ
=
r
^
i
j
/
d
i
j
2
θ= 
r
^
  
ij
​
 /d 
ij
2
​
  (classic inverse-square direction)

θ
=
r
^
i
j
/
d
i
j
3
θ= 
r
^
  
ij
​
 /d 
ij
3
​
  (often appears as correction / alternative fits)

That’s it. This often recovers a clean 1–2 term model per pair.

L1 — gravity_plus
Adds a few low-complexity velocity couplings that sometimes produce better local predictors in chaotic regimes.

For each 
i
i, 
j
≠
i
j

=i:

r
^
i
j
/
d
2
r
^
  
ij
​
 /d 
2
 

r
^
i
j
/
d
3
r
^
  
ij
​
 /d 
3
 

(
v
i
j
)
/
d
2
(v 
ij
​
 )/d 
2
 

(
v
i
j
)
/
d
3
(v 
ij
​
 )/d 
3
 

(
v
i
⋅
r
^
i
j
)
 
r
^
i
j
/
d
2
(v 
i
​
 ⋅ 
r
^
  
ij
​
 ) 
r
^
  
ij
​
 /d 
2
 

Use this when you want “simple but better short-horizon prediction” without going fully EM.

L2 — em_basic
Minimal EM-shaped library (without explicitly computing E and B). This can rediscover Lorentz-like structure from kinematics alone.

For each 
i
i, 
j
≠
i
j

=i:

Electric-shaped

r
^
i
j
/
d
2
r
^
  
ij
​
 /d 
2
 

r
^
i
j
/
d
3
r
^
  
ij
​
 /d 
3
 

Magnetic-shaped

(
v
j
×
r
i
j
)
/
d
3
(v 
j
​
 ×r 
ij
​
 )/d 
3
  (Biot–Savart-ish)

v
i
×
(
v
j
×
r
i
j
)
/
d
3
v 
i
​
 ×(v 
j
​
 ×r 
ij
​
 )/d 
3
  (Lorentz-ish)

Optional scalars-as-gates (still simple)

(
v
i
⋅
r
^
i
j
)
 
r
^
i
j
/
d
2
(v 
i
​
 ⋅ 
r
^
  
ij
​
 ) 
r
^
  
ij
​
 /d 
2
 

(
v
j
⋅
r
^
i
j
)
 
r
^
i
j
/
d
2
(v 
j
​
 ⋅ 
r
^
  
ij
​
 ) 
r
^
  
ij
​
 /d 
2
 

This library is compact and tends to remain sparse.

L3 — em_fields
If your simulator outputs 
E
i
,
B
i
E 
i
​
 ,B 
i
​
 , you can let the discovery engine find simpler algebra in the field space.

For each body 
i
i:

E
i
E 
i
​
 

v
i
×
B
i
v 
i
​
 ×B 
i
​
 

(
v
i
⋅
v
i
)
 
E
i
(v 
i
​
 ⋅v 
i
​
 )E 
i
​
  (optional weak nonlinearity)

(
v
i
⋅
E
i
)
 
v
i
(v 
i
​
 ⋅E 
i
​
 )v 
i
​
  (optional)

This often discovers models like:

a
i
≈
α
E
i
+
β
(
v
i
×
B
i
)
a 
i
​
 ≈αE 
i
​
 +β(v 
i
​
 ×B 
i
​
 )
with small corrections. It’s extremely interpretable and very predictive.

L4 — potentials_phiA
If simulator outputs 
ϕ
i
ϕ 
i
​
  and 
A
i
A 
i
​
 , this library tries to find compact laws in potential form.

For each body 
i
i:

∇
ϕ
i
∇ϕ 
i
​
  (you can compute from pairwise terms: 
∑
j
q
j
r
^
i
j
/
d
2
∑ 
j
​
 q 
j
​
  
r
^
  
ij
​
 /d 
2
 )

∂
t
A
i
∂ 
t
​
 A 
i
​
  (approx via finite difference of A along the time series)

v
i
×
(
∇
×
A
i
)
v 
i
​
 ×(∇×A 
i
​
 ) (curl from pairwise Biot–Savart form if you prefer analytic)

A
i
A 
i
​
 

v
i
×
A
i
v 
i
​
 ×A 
i
​
  (often appears in simplified fits)

This is more expensive and more sensitive to numerical differentiation; use only if you want explicit φ/A-based “new math”.

L5 — hierarchical_triple
For hierarchical regimes (tight binary + distant third), you want terms that reflect scale separation.

Define:

d
min
⁡
=
min
⁡
(
d
A
B
,
d
A
C
,
d
B
C
)
d 
min
​
 =min(d 
AB
​
 ,d 
AC
​
 ,d 
BC
​
 )

d
max
⁡
=
max
⁡
(
d
A
B
,
d
A
C
,
d
B
C
)
d 
max
​
 =max(d 
AB
​
 ,d 
AC
​
 ,d 
BC
​
 )

η
=
d
min
⁡
/
d
max
⁡
η=d 
min
​
 /d 
max
​
 

Include gravity_min plus small corrections:

η
 
r
^
i
j
/
d
2
η 
r
^
  
ij
​
 /d 
2
 

η
2
 
r
^
i
j
/
d
2
η 
2
  
r
^
  
ij
​
 /d 
2
 

η
 
v
i
j
/
d
2
ηv 
ij
​
 /d 
2
 

This tends to produce surprisingly compact “effective laws” that outperform generic models in that regime.

3) Library “complexity ladder” (recommended workflow)
Fit gravity_min. If it predicts well: you’re done.

If not, try gravity_plus.

If EM effects matter, go em_basic (no fields required).

If simulator outputs fields reliably, em_fields usually gives the cleanest “simple math”.

Use potentials_phiA only if you explicitly want φ/A in the discovered equations.

4) How to implement libraries in Rust (clean API)
Represent each basis term as a small enum + evaluation:

pub enum LibraryId { GravityMin, GravityPlus, EmBasic, EmFields, PotentialsPhiA, HierarchicalTriple }

pub trait Term {
    fn name(&self) -> &'static str;
    fn eval_vec3(&self, sample: &Sample, i: usize) -> Vec3; // predicts a_i
}

pub struct Library {
    pub id: LibraryId,
    pub terms: Vec<Box<dyn Term>>,
}
Sample contains per-timestep values (r,v,a, and optionally E,B,phi,A), plus precomputed pairwise 
r
i
j
,
d
i
j
,
r
^
i
j
r 
ij
​
 ,d 
ij
​
 , 
r
^
  
ij
​
  for speed.

5) Critical details that make discovery “accurate”
These are the difference between “cool demo” and “works reliably”.

(a) Normalize columns of Θ
Before sparse regression, normalize each feature column to unit RMS. Otherwise thresholding is meaningless.

(b) Learn acceleration directly
Use simulator output a_i columns. Don’t estimate derivatives from noisy v(t) unless you must.

(c) Use rollout score as the primary metric
A model that matches a(t) pointwise can still diverge fast. Always validate by integrating the discovered model forward.

(d) Keep libraries small
Novel + simple comes from restricting the hypothesis space and letting sparsity select.

6) What I need from you (one choice, no back-and-forth)
Pick the default discovery library to ship first:

If you want general-purpose and no field output requirement: em_basic

If you’re willing to output E and B in the sim for cleaner math: em_fields

If you don’t answer, I’ll assume em_basic as the default.

If you want, I’ll next paste the exact Rust module discover/library.rs implementing gravity_min, gravity_plus, em_basic, and the Θ matrix builder + normalization hooks.

You said:
Implement
ChatGPT said:

Thought for 11s
Below is an implementation of the discovery “libraries” + sparse math discovery (SINDy/STLS) in clean Rust, wired to a CLI subcommand.

This includes:

Library implementations: gravity_min, gravity_plus, em_basic (the ones you need first)

Robust CSV reader by header name (fixes the bug we found: wide CSV won’t break parsing)

Discovery engine: builds Θ, normalizes columns, runs sequential thresholded least squares, outputs model.json

Fallback if your sim output doesn’t include a_*: it estimates acceleration from v_* with finite differences (less accurate, but works)

Assumption: your simulation CSV includes at least:

positions: a_x a_y a_z, b_x..., c_x...

velocities: a_vx a_vy a_vz, b_vx..., c_vx...
and ideally accelerations: a_ax a_ay a_az, etc. (recommended to add).

1) Add new module files
Create these files:

src/discover/mod.rs
pub mod library;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::discover::library::{LibraryId, Term, build_library};
use crate::io::CsvTable;
use crate::sim::Vec3;

/// Model output: sparse coefficients for a_i = sum_k coeff[i][comp][k] * theta_k(sample,i)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredModel {
    pub library: LibraryId,
    pub term_names: Vec<String>,
    /// coeffs[body][component][term]
    pub coeffs: Vec<Vec<Vec<f64>>>,
    /// column normalization used during fit (applied to Theta columns)
    pub theta_col_scale: Vec<f64>,
    pub lambda: f64,
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub t: f64,
    pub r: [Vec3; 3],
    pub v: [Vec3; 3],
    /// acceleration from sim if present, else estimated from v(t)
    pub a: [Vec3; 3],
    /// optional masses/charges could be added later if you want them as features
}

pub fn run_discover(
    input_csv: &str,
    out_model_json: &str,
    library_id: LibraryId,
    lambda: f64,
    iters: usize,
    ridge: f64,
    stride: usize,
) -> Result<()> {
    let table = CsvTable::read(input_csv).with_context(|| format!("read {input_csv}"))?;
    let mut samples = load_samples(&table, stride)?;

    // If accelerations are estimated, warn via console (caller can see)
    if !has_accel_columns(&table) {
        eprintln!("NOTE: input CSV does not include a_* acceleration columns; estimating a from v(t). For best discovery, write accelerations from the simulator.");
    }

    // Build library terms
    let lib = build_library(library_id);
    let term_names: Vec<String> = lib.terms.iter().map(|t| t.name().to_string()).collect();

    // Fit separate models for each body and component (x,y,z)
    let mut coeffs: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; 3]; 3];

    // Precompute Theta once per (sample, body) to reuse for x/y/z
    // Theta is shape [N, K]
    let k = lib.terms.len();

    // We normalize Theta columns for stable thresholding.
    // theta_scale[k] = RMS of column k (across all rows / bodies used in this fit).
    // We'll compute a global scale across all bodies to keep terms comparable.
    let theta_col_scale = compute_global_theta_scale(&samples, &lib.terms)?;

    for body in 0..3 {
        // Build Theta for this body: NxK
        let theta = build_theta_for_body(&samples, &lib.terms, body, &theta_col_scale)?;

        // Targets (Nx1) for each component
        let yx: Vec<f64> = samples.iter().map(|s| s.a[body].x).collect();
        let yy: Vec<f64> = samples.iter().map(|s| s.a[body].y).collect();
        let yz: Vec<f64> = samples.iter().map(|s| s.a[body].z).collect();

        let cx = stls_fit(&theta, &yx, lambda, iters, ridge)?;
        let cy = stls_fit(&theta, &yy, lambda, iters, ridge)?;
        let cz = stls_fit(&theta, &yz, lambda, iters, ridge)?;

        coeffs[body][0] = cx;
        coeffs[body][1] = cy;
        coeffs[body][2] = cz;
    }

    let model = DiscoveredModel {
        library: library_id,
        term_names,
        coeffs,
        theta_col_scale,
        lambda,
    };

    let json = serde_json::to_string_pretty(&model)?;
    std::fs::write(out_model_json, json).with_context(|| format!("write {out_model_json}"))?;
    Ok(())
}

fn has_accel_columns(table: &CsvTable) -> bool {
    table.has("a_ax") && table.has("a_ay") && table.has("a_az")
        && table.has("b_ax") && table.has("b_ay") && table.has("b_az")
        && table.has("c_ax") && table.has("c_ay") && table.has("c_az")
}

fn load_samples(table: &CsvTable, stride: usize) -> Result<Vec<Sample>> {
    // Required columns
    let t = table.col("t")?;

    let ax = table.col("a_x")?; let ay = table.col("a_y")?; let az = table.col("a_z")?;
    let bx = table.col("b_x")?; let by = table.col("b_y")?; let bz = table.col("b_z")?;
    let cx = table.col("c_x")?; let cy = table.col("c_y")?; let cz = table.col("c_z")?;

    let avx = table.col("a_vx")?; let avy = table.col("a_vy")?; let avz = table.col("a_vz")?;
    let bvx = table.col("b_vx")?; let bvy = table.col("b_vy")?; let bvz = table.col("b_vz")?;
    let cvx = table.col("c_vx")?; let cvy = table.col("c_vy")?; let cvz = table.col("c_vz")?;

    // Optional accel columns
    let has_a = has_accel_columns(table);
    let (aax, aay, aaz, bax, bay, baz, cax, cay, caz) = if has_a {
        (
            Some(table.col("a_ax")?), Some(table.col("a_ay")?), Some(table.col("a_az")?),
            Some(table.col("b_ax")?), Some(table.col("b_ay")?), Some(table.col("b_az")?),
            Some(table.col("c_ax")?), Some(table.col("c_ay")?), Some(table.col("c_az")?),
        )
    } else {
        (None, None, None, None, None, None, None, None, None)
    };

    let n = t.len();
    let mut samples = Vec::with_capacity(n / stride.max(1));

    // First pass: create samples with r,v and if present a
    for idx in (0..n).step_by(stride.max(1)) {
        let r = [
            Vec3::new(ax[idx], ay[idx], az[idx]),
            Vec3::new(bx[idx], by[idx], bz[idx]),
            Vec3::new(cx[idx], cy[idx], cz[idx]),
        ];
        let v = [
            Vec3::new(avx[idx], avy[idx], avz[idx]),
            Vec3::new(bvx[idx], bvy[idx], bvz[idx]),
            Vec3::new(cvx[idx], cvy[idx], cvz[idx]),
        ];

        let a = if has_a {
            [
                Vec3::new(aax.as_ref().unwrap()[idx], aay.as_ref().unwrap()[idx], aaz.as_ref().unwrap()[idx]),
                Vec3::new(bax.as_ref().unwrap()[idx], bay.as_ref().unwrap()[idx], baz.as_ref().unwrap()[idx]),
                Vec3::new(cax.as_ref().unwrap()[idx], cay.as_ref().unwrap()[idx], caz.as_ref().unwrap()[idx]),
            ]
        } else {
            // placeholder, fill later
            [Vec3::default(), Vec3::default(), Vec3::default()]
        };

        samples.push(Sample { t: t[idx], r, v, a });
    }

    // If no accel columns, estimate a from v(t) using finite differences on the downsampled series
    if !has_a {
        estimate_accel_from_velocity(&mut samples)?;
    }

    Ok(samples)
}

fn estimate_accel_from_velocity(samples: &mut [Sample]) -> Result<()> {
    if samples.len() < 3 {
        anyhow::bail!("need at least 3 samples to estimate acceleration from velocity");
    }
    // central differences for interior, forward/backward for ends
    for i in 0..samples.len() {
        let dt_prev = if i > 0 { samples[i].t - samples[i-1].t } else { 0.0 };
        let dt_next = if i + 1 < samples.len() { samples[i+1].t - samples[i].t } else { 0.0 };

        for b in 0..3 {
            let a = if i == 0 {
                // forward
                let dv = samples[i+1].v[b].sub(samples[i].v[b]);
                dv.mul(1.0 / dt_next.max(1e-18))
            } else if i + 1 == samples.len() {
                // backward
                let dv = samples[i].v[b].sub(samples[i-1].v[b]);
                dv.mul(1.0 / dt_prev.max(1e-18))
            } else {
                // central: (v_{i+1}-v_{i-1})/(t_{i+1}-t_{i-1})
                let dv = samples[i+1].v[b].sub(samples[i-1].v[b]);
                let dt = (samples[i+1].t - samples[i-1].t).max(1e-18);
                dv.mul(1.0 / dt)
            };
            samples[i].a[b] = a;
        }
    }
    Ok(())
}

fn compute_global_theta_scale(samples: &[Sample], terms: &[Box<dyn Term>]) -> Result<Vec<f64>> {
    let k = terms.len();
    let mut acc = vec![0.0f64; k];
    let mut count = 0usize;

    for s in samples {
        for body in 0..3 {
            let ctx = library::EvalCtx::new(s);
            for (ti, term) in terms.iter().enumerate() {
                let v = term.eval_vec3(&ctx, body);
                // use vector magnitude as a proxy scale for this term column
                acc[ti] += v.norm2();
            }
            count += 1;
        }
    }

    let denom = (count as f64).max(1.0);
    let mut scale = vec![1.0f64; k];
    for i in 0..k {
        let rms = (acc[i] / denom).sqrt();
        scale[i] = rms.max(1e-12);
    }
    Ok(scale)
}

fn build_theta_for_body(
    samples: &[Sample],
    terms: &[Box<dyn Term>],
    body: usize,
    col_scale: &[f64],
) -> Result<Vec<Vec<f64>>> {
    let k = terms.len();
    let mut theta: Vec<Vec<f64>> = Vec::with_capacity(samples.len());
    for s in samples {
        let ctx = library::EvalCtx::new(s);
        let mut row = vec![0.0f64; k];
        for (ti, term) in terms.iter().enumerate() {
            let v = term.eval_vec3(&ctx, body);
            // For fitting a component, we keep vector terms; we’ll fit x/y/z separately.
            // Here we store *scalar feature columns* by projecting later? No—store all three?
            //
            // Simpler: store vector term magnitude is NOT sufficient; we need directional info.
            // Therefore: each vector term becomes 3 columns in Theta if we want full expressivity.
            //
            // BUT we designed terms themselves as vector-valued; to fit ax, we need x-component.
            // So build_theta_for_body returns Theta of scalar columns for one component later.
            //
            // We'll instead return 3 separate Thetas in stls_fit? That complicates.
            //
            // Solution: build Theta once per component by selecting component from v.
            // Here: pack x/y/z components into 3K columns. We'll do that.
            //
            // However our stls_fit currently expects NxK. We'll keep Nx(3K) and fit each component.
            //
            // So: we must build 3K columns, and stls_fit length matches.
            let _ = v;
            row[ti] = 0.0; // placeholder: replaced below
        }
        theta.push(row);
    }

    // Rebuild as Nx(3K): return flattened columns [term0_x, term0_y, term0_z, term1_x, ...]
    let mut theta3: Vec<Vec<f64>> = Vec::with_capacity(samples.len());
    for s in samples {
        let ctx = library::EvalCtx::new(s);
        let mut row = vec![0.0f64; 3*k];
        for (ti, term) in terms.iter().enumerate() {
            let v = term.eval_vec3(&ctx, body);
            let sc = col_scale[ti];
            row[3*ti + 0] = v.x / sc;
            row[3*ti + 1] = v.y / sc;
            row[3*ti + 2] = v.z / sc;
        }
        theta3.push(row);
    }
    Ok(theta3)
}

/// Sequential Thresholded Least Squares (STLS)
/// theta: NxM, y: N
/// returns coeffs length M
fn stls_fit(theta: &[Vec<f64>], y: &[f64], lambda: f64, iters: usize, ridge: f64) -> Result<Vec<f64>> {
    let n = theta.len();
    let m = theta[0].len();
    if y.len() != n { anyhow::bail!("theta/y length mismatch"); }

    // Start with all features active
    let mut active: Vec<bool> = vec![true; m];
    let mut coeff = vec![0.0f64; m];

    for _ in 0..iters.max(1) {
        // Solve ridge regression on active set
        let act_idx: Vec<usize> = active.iter().enumerate().filter_map(|(i,&a)| if a {Some(i)} else {None}).collect();
        if act_idx.is_empty() { break; }

        let beta = ridge_ls(theta, y, &act_idx, ridge)?;
        // Place into full coeff vector
        for (k, &j) in act_idx.iter().enumerate() {
            coeff[j] = beta[k];
        }
        // Threshold
        let mut changed = false;
        for &j in &act_idx {
            if coeff[j].abs() < lambda {
                if active[j] {
                    active[j] = false;
                    coeff[j] = 0.0;
                    changed = true;
                }
            }
        }
        if !changed { break; }
    }

    Ok(coeff)
}

/// Ridge least squares on subset of columns.
/// Solves (X^T X + ridge I) beta = X^T y
fn ridge_ls(theta: &[Vec<f64>], y: &[f64], cols: &[usize], ridge: f64) -> Result<Vec<f64>> {
    let n = theta.len();
    let p = cols.len();
    let mut xtx = vec![vec![0.0f64; p]; p];
    let mut xty = vec![0.0f64; p];

    for i in 0..n {
        let row = &theta[i];
        for a in 0..p {
            let xa = row[cols[a]];
            xty[a] += xa * y[i];
            for b in 0..p {
                xtx[a][b] += xa * row[cols[b]];
            }
        }
    }
    for a in 0..p {
        xtx[a][a] += ridge.max(0.0);
    }

    // Solve linear system via naive Gaussian elimination (p is small)
    gaussian_solve(xtx, xty)
}

fn gaussian_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>> {
    let n = b.len();
    for i in 0..n {
        // pivot
        let mut piv = i;
        let mut best = a[i][i].abs();
        for r in (i+1)..n {
            if a[r][i].abs() > best {
                best = a[r][i].abs();
                piv = r;
            }
        }
        if best < 1e-18 { anyhow::bail!("singular system in gaussian_solve"); }
        if piv != i {
            a.swap(i, piv);
            b.swap(i, piv);
        }

        // normalize row
        let diag = a[i][i];
        for c in i..n { a[i][c] /= diag; }
        b[i] /= diag;

        // eliminate
        for r in 0..n {
            if r == i { continue; }
            let factor = a[r][i];
            for c in i..n { a[r][c] -= factor * a[i][c]; }
            b[r] -= factor * b[i];
        }
    }
    Ok(b)
}
src/discover/library.rs
use serde::{Deserialize, Serialize};

use crate::discover::Sample;
use crate::sim::Vec3;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LibraryId {
    GravityMin,
    GravityPlus,
    EmBasic,
}

pub struct EvalCtx<'a> {
    pub s: &'a Sample,
    // Precomputed pairwise data for speed
    pub rij: [[Vec3; 3]; 3],
    pub dij: [[f64; 3]; 3],
    pub rhat: [[Vec3; 3]; 3],
}

impl<'a> EvalCtx<'a> {
    pub fn new(s: &'a Sample) -> Self {
        let eps2 = 0.0; // discovery library should mirror sim assumptions; can expose later
        let mut rij = [[Vec3::default(); 3]; 3];
        let mut dij = [[0.0f64; 3]; 3];
        let mut rhat = [[Vec3::default(); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                if i == j { continue; }
                let r = s.r[j].sub(s.r[i]);
                let d = (r.norm2() + eps2).sqrt().max(1e-12);
                rij[i][j] = r;
                dij[i][j] = d;
                rhat[i][j] = r.mul(1.0 / d);
            }
        }
        Self { s, rij, dij, rhat }
    }
}

pub trait Term {
    fn name(&self) -> &'static str;
    fn eval_vec3(&self, ctx: &EvalCtx, body_i: usize) -> Vec3;
}

pub struct Library {
    pub id: LibraryId,
    pub terms: Vec<Box<dyn Term>>,
}

pub fn build_library(id: LibraryId) -> Library {
    let mut terms: Vec<Box<dyn Term>> = Vec::new();
    match id {
        LibraryId::GravityMin => {
            add_pair_terms(&mut terms, |i,j| Box::new(RhatOverD2 { i, j }));
            add_pair_terms(&mut terms, |i,j| Box::new(RhatOverD3 { i, j }));
        }
        LibraryId::GravityPlus => {
            add_pair_terms(&mut terms, |i,j| Box::new(RhatOverD2 { i, j }));
            add_pair_terms(&mut terms, |i,j| Box::new(RhatOverD3 { i, j }));
            add_pair_terms(&mut terms, |i,j| Box::new(VijOverD2 { i, j }));
            add_pair_terms(&mut terms, |i,j| Box::new(VijOverD3 { i, j }));
            add_pair_terms(&mut terms, |i,j| Box::new(ViDotRhat_RhatOverD2 { i, j }));
        }
        LibraryId::EmBasic => {
            // Electric-shaped
            add_pair_terms(&mut terms, |i,j| Box::new(RhatOverD2 { i, j }));
            add_pair_terms(&mut terms, |i,j| Box::new(RhatOverD3 { i, j }));
            // Magnetic-shaped
            add_pair_terms(&mut terms, |i,j| Box::new(VjCrossRijOverD3 { i, j }));
            add_pair_terms(&mut terms, |i,j| Box::new(ViCrossVjCrossRijOverD3 { i, j }));
            // Optional scalar gates (still simple)
            add_pair_terms(&mut terms, |i,j| Box::new(ViDotRhat_RhatOverD2 { i, j }));
            add_pair_terms(&mut terms, |i,j| Box::new(VjDotRhat_RhatOverD2 { i, j }));
        }
    }
    Library { id, terms }
}

/// Helper: add terms for all ordered pairs i!=j
fn add_pair_terms<F>(terms: &mut Vec<Box<dyn Term>>, mut f: F)
where
    F: FnMut(usize, usize) -> Box<dyn Term>,
{
    for i in 0..3 {
        for j in 0..3 {
            if i == j { continue; }
            terms.push(f(i,j));
        }
    }
}

// --- Term implementations ---

struct RhatOverD2 { i: usize, j: usize }
impl Term for RhatOverD2 {
    fn name(&self) -> &'static str { "rhat_over_d2" }
    fn eval_vec3(&self, ctx: &EvalCtx, body_i: usize) -> Vec3 {
        if body_i != self.i { return Vec3::default(); }
        let d = ctx.dij[self.i][self.j];
        ctx.rhat[self.i][self.j].mul(1.0 / (d*d))
    }
}

struct RhatOverD3 { i: usize, j: usize }
impl Term for RhatOverD3 {
    fn name(&self) -> &'static str { "rhat_over_d3" }
    fn eval_vec3(&self, ctx: &EvalCtx, body_i: usize) -> Vec3 {
        if body_i != self.i { return Vec3::default(); }
        let d = ctx.dij[self.i][self.j];
        ctx.rhat[self.i][self.j].mul(1.0 / (d*d*d))
    }
}

struct VijOverD2 { i: usize, j: usize }
impl Term for VijOverD2 {
    fn name(&self) -> &'static str { "vij_over_d2" }
    fn eval_vec3(&self, ctx: &EvalCtx, body_i: usize) -> Vec3 {
        if body_i != self.i { return Vec3::default(); }
        let d = ctx.dij[self.i][self.j];
        let vij = ctx.s.v[self.j].sub(ctx.s.v[self.i]);
        vij.mul(1.0 / (d*d))
    }
}

struct VijOverD3 { i: usize, j: usize }
impl Term for VijOverD3 {
    fn name(&self) -> &'static str { "vij_over_d3" }
    fn eval_vec3(&self, ctx: &EvalCtx, body_i: usize) -> Vec3 {
        if body_i != self.i { return Vec3::default(); }
        let d = ctx.dij[self.i][self.j];
        let vij = ctx.s.v[self.j].sub(ctx.s.v[self.i]);
        vij.mul(1.0 / (d*d*d))
    }
}

struct ViDotRhat_RhatOverD2 { i: usize, j: usize }
impl Term for ViDotRhat_RhatOverD2 {
    fn name(&self) -> &'static str { "vi_dot_rhat_times_rhat_over_d2" }
    fn eval_vec3(&self, ctx: &EvalCtx, body_i: usize) -> Vec3 {
        if body_i != self.i { return Vec3::default(); }
        let d = ctx.dij[self.i][self.j];
        let rhat = ctx.rhat[self.i][self.j];
        let s = ctx.s.v[self.i].dot(rhat);
        rhat.mul(s / (d*d))
    }
}

struct VjDotRhat_RhatOverD2 { i: usize, j: usize }
impl Term for VjDotRhat_RhatOverD2 {
    fn name(&self) -> &'static str { "vj_dot_rhat_times_rhat_over_d2" }
    fn eval_vec3(&self, ctx: &EvalCtx, body_i: usize) -> Vec3 {
        if body_i != self.i { return Vec3::default(); }
        let d = ctx.dij[self.i][self.j];
        let rhat = ctx.rhat[self.i][self.j];
        let s = ctx.s.v[self.j].dot(rhat);
        rhat.mul(s / (d*d))
    }
}

struct VjCrossRijOverD3 { i: usize, j: usize }
impl Term for VjCrossRijOverD3 {
    fn name(&self) -> &'static str { "vj_cross_rij_over_d3" }
    fn eval_vec3(&self, ctx: &EvalCtx, body_i: usize) -> Vec3 {
        if body_i != self.i { return Vec3::default(); }
        let d = ctx.dij[self.i][self.j];
        let rij = ctx.rij[self.i][self.j];
        let vj = ctx.s.v[self.j];
        vj.cross(rij).mul(1.0 / (d*d*d))
    }
}

struct ViCrossVjCrossRijOverD3 { i: usize, j: usize }
impl Term for ViCrossVjCrossRijOverD3 {
    fn name(&self) -> &'static str { "vi_cross_vj_cross_rij_over_d3" }
    fn eval_vec3(&self, ctx: &EvalCtx, body_i: usize) -> Vec3 {
        if body_i != self.i { return Vec3::default(); }
        let d = ctx.dij[self.i][self.j];
        let rij = ctx.rij[self.i][self.j];
        let vj = ctx.s.v[self.j];
        let vi = ctx.s.v[self.i];
        let inner = vj.cross(rij);
        vi.cross(inner).mul(1.0 / (d*d*d))
    }
}
2) Add a robust CSV header-based reader (fix + needed for discovery)
Replace your src/io.rs with the following additions (you can keep your existing CsvWriter; this only adds CsvTable and does not break anything).

Add to src/io.rs (append near bottom)
use std::collections::HashMap;

pub struct CsvTable {
    headers: Vec<String>,
    index: HashMap<String, usize>,
    cols: Vec<Vec<f64>>, // column-major
}

impl CsvTable {
    pub fn read(path: &str) -> anyhow::Result<Self> {
        let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_path(path)?;
        let headers = rdr.headers()?.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let mut index = HashMap::new();
        for (i, h) in headers.iter().enumerate() {
            index.insert(h.clone(), i);
        }

        let m = headers.len();
        let mut cols = vec![Vec::<f64>::new(); m];

        for rec in rdr.records() {
            let rec = rec?;
            for c in 0..m {
                let v = rec.get(c).unwrap_or("0").parse::<f64>().unwrap_or(0.0);
                cols[c].push(v);
            }
        }

        Ok(Self { headers, index, cols })
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn col(&self, name: &str) -> anyhow::Result<&[f64]> {
        let idx = *self.index.get(name)
            .ok_or_else(|| anyhow::anyhow!("missing column: {name}"))?;
        Ok(&self.cols[idx])
    }

    #[allow(dead_code)]
    pub fn headers(&self) -> &[String] {
        &self.headers
    }
}
3) Wire the CLI: add discover subcommand
Edit src/main.rs:

3.1 Add module import at top
mod discover;
3.2 Add subcommand variant
Add to Commands enum:

    /// Discover sparse "simple math" models that predict acceleration well (SINDy/STLS)
    Discover {
        /// Input CSV from simulate
        #[arg(long)]
        input: String,

        /// Output model JSON
        #[arg(long)]
        out: String,

        /// Library choice
        #[arg(long, value_enum, default_value_t=DiscoverLibraryChoice::EmBasic)]
        library: DiscoverLibraryChoice,

        /// Sparsity threshold (higher => simpler)
        #[arg(long, default_value_t=1e-6)]
        lambda: f64,

        /// STLS iterations
        #[arg(long, default_value_t=10)]
        iters: usize,

        /// Ridge term for stability (small, e.g. 1e-10..1e-6)
        #[arg(long, default_value_t=1e-10)]
        ridge: f64,

        /// Use every Nth row
        #[arg(long, default_value_t=1)]
        stride: usize,
    },
3.3 Add ValueEnum for library
Add near IntegratorChoice:

#[derive(Clone, Copy, ValueEnum)]
enum DiscoverLibraryChoice {
    GravityMin,
    GravityPlus,
    EmBasic,
}
3.4 Handle it in main()
Add to the match:

        Commands::Discover { input, out, library, lambda, iters, ridge, stride } => {
            use discover::library::LibraryId;
            let lib = match library {
                DiscoverLibraryChoice::GravityMin => LibraryId::GravityMin,
                DiscoverLibraryChoice::GravityPlus => LibraryId::GravityPlus,
                DiscoverLibraryChoice::EmBasic => LibraryId::EmBasic,
            };
            discover::run_discover(&input, &out, lib, lambda, iters, ridge, stride)?;
            println!("Wrote discovered model to {out}");
        }
4) Update Cargo.toml (already ok)
You already have serde / serde_json / csv / anyhow / clap, so no changes needed.

5) Run it
Build a sim CSV (existing command)
cargo run --release -- simulate --config config.json --out traj.csv --integrator rk45 --t-end 20 --stride 1
Discover a model (default em_basic)
cargo run --release -- discover --input traj.csv --out model.json --library em-basic --lambda 1e-5 --iters 12 --ridge 1e-10 --stride 1
Try larger --lambda (e.g. 1e-4, 1e-3) for simpler equations.

6) Important note about “vector terms → 3K columns”
The implementation above expands each vector term into x/y/z columns in Θ. That’s essential: otherwise you can’t learn directional structure.

So if your library has K vector terms, Θ has 3K columns, and the model learns sparse coefficients over those columns.

7) What you should do next (to make it truly accurate)
To maximize discovery quality, update the simulator to write accelerations per step:

Add columns: a_ax a_ay a_az, b_ax..., c_ax...

In integrators, you already compute accelerations each step—just write them.

Once you do that, discovery becomes dramatically more reliable.

If you want, I’ll paste the minimal patch to your CsvWriter + integrator loop to output a_* columns (so discovery uses ground-truth accelerations instead of finite-difference estimates).

You said:
Update the simulator
ChatGPT said:
Below is a direct, drop-in update to your simulator so it:

models gravity + electric + magnetic forces

computes and outputs E, B, φ, A per body

outputs accelerations a_ax/a_ay/a_az … every written row (so discovery is clean)

keeps the integrators intact (leapfrog + RK45) and uses the new acceleration routine

I’m not touching the discovery code here—this is purely the simulator update.

1) Replace src/sim.rs with this version
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn dot(self, o: Vec3) -> f64 { self.x*o.x + self.y*o.y + self.z*o.z }
    pub fn norm2(self) -> f64 { self.dot(self) }
    pub fn norm(self) -> f64 { self.norm2().sqrt() }
    pub fn add(self, o: Vec3) -> Vec3 { Vec3::new(self.x+o.x, self.y+o.y, self.z+o.z) }
    pub fn sub(self, o: Vec3) -> Vec3 { Vec3::new(self.x-o.x, self.y-o.y, self.z-o.z) }
    pub fn mul(self, s: f64) -> Vec3 { Vec3::new(self.x*s, self.y*s, self.z*s) }
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y*o.z - self.z*o.y,
            self.z*o.x - self.x*o.z,
            self.x*o.y - self.y*o.x,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Body {
    pub name: String,
    pub m: f64,
    pub q: f64,   // NEW: charge
    pub r: Vec3,
    pub v: Vec3,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConfigFile {
    /// gravitational constant G
    pub g: f64,
    /// Coulomb constant k_e
    pub ke: f64,
    /// μ0/(4π)
    pub mu0_over_4pi: f64,
    /// Force softening length (0 = none). Applies to both gravity and EM denominators.
    pub softening: f64,
    /// Exactly 3 bodies
    pub bodies: Vec<Body>,
    /// Enable forces (defaults true if omitted)
    #[serde(default = "default_true")]
    pub enable_gravity: bool,
    #[serde(default = "default_true")]
    pub enable_electric: bool,
    #[serde(default = "default_true")]
    pub enable_magnetic: bool,
}

fn default_true() -> bool { true }

#[derive(Clone, Copy, Debug)]
pub struct SimConfig {
    pub g: f64,
    pub ke: f64,
    pub mu0_over_4pi: f64,
    pub softening: f64,
    pub enable_gravity: bool,
    pub enable_electric: bool,
    pub enable_magnetic: bool,
}

/// Internal mutable state
#[derive(Clone, Debug)]
pub struct SystemState {
    pub t: f64,
    pub m: [f64; 3],
    pub q: [f64; 3],
    pub r: [Vec3; 3],
    pub v: [Vec3; 3],
    pub names: [String; 3],
}

impl SystemState {
    pub fn from_config(cfg: &ConfigFile) -> Self {
        assert!(cfg.bodies.len() == 3, "config must include exactly 3 bodies");
        let b0 = &cfg.bodies[0];
        let b1 = &cfg.bodies[1];
        let b2 = &cfg.bodies[2];
        SystemState {
            t: 0.0,
            m: [b0.m, b1.m, b2.m],
            q: [b0.q, b1.q, b2.q],
            r: [b0.r, b1.r, b2.r],
            v: [b0.v, b1.v, b2.v],
            names: [b0.name.clone(), b1.name.clone(), b2.name.clone()],
        }
    }

    /// Shift to center-of-mass frame and remove net momentum.
    pub fn make_barycentric(&mut self) {
        let mt = self.m[0] + self.m[1] + self.m[2];
        let com_r = self.r[0].mul(self.m[0])
            .add(self.r[1].mul(self.m[1]))
            .add(self.r[2].mul(self.m[2]))
            .mul(1.0 / mt);
        for i in 0..3 {
            self.r[i] = self.r[i].sub(com_r);
        }

        let p = self.v[0].mul(self.m[0])
            .add(self.v[1].mul(self.m[1]))
            .add(self.v[2].mul(self.m[2]));
        let vcom = p.mul(1.0 / mt);
        for i in 0..3 {
            self.v[i] = self.v[i].sub(vcom);
        }
    }
}

#[derive(Clone, Debug)]
pub struct Fields {
    pub a: [Vec3; 3],    // total acceleration
    pub e: [Vec3; 3],    // electric field at each body
    pub b: [Vec3; 3],    // magnetic field at each body (quasi-static)
    pub phi: [f64; 3],   // scalar potential at each body
    pub avec: [Vec3; 3], // vector potential A at each body
}

/// Computes fields (E,B,phi,A) and total acceleration (gravity + Lorentz).
pub fn fields_and_accel(cfg: &SimConfig, st: &SystemState) -> Fields {
    let eps2 = cfg.softening * cfg.softening;

    let mut e = [Vec3::default(), Vec3::default(), Vec3::default()];
    let mut b = [Vec3::default(), Vec3::default(), Vec3::default()];
    let mut phi = [0.0f64; 3];
    let mut avec = [Vec3::default(), Vec3::default(), Vec3::default()];
    let mut a_g = [Vec3::default(), Vec3::default(), Vec3::default()];

    // Pairwise loops
    for i in 0..3 {
        for j in 0..3 {
            if i == j { continue; }

            let rij = st.r[i].sub(st.r[j]); // vector from j -> i
            let d2 = rij.norm2() + eps2;
            let d = d2.sqrt().max(1e-18);
            let inv_d = 1.0 / d;
            let inv_d3 = inv_d * inv_d * inv_d;

            // Gravity: acceleration on i from mass j
            if cfg.enable_gravity {
                // a_i += G m_j * (r_j - r_i)/|r_j-r_i|^3 = -G m_j * rij / d^3
                a_g[i] = a_g[i].add(rij.mul(-cfg.g * st.m[j] * inv_d3));
            }

            // Electric field and scalar potential at i due to charge j
            if cfg.enable_electric {
                // E_i += k_e q_j * (r_i - r_j)/|...|^3 = k_e q_j * rij / d^3
                e[i] = e[i].add(rij.mul(cfg.ke * st.q[j] * inv_d3));
                // phi_i += k_e q_j / d
                phi[i] += cfg.ke * st.q[j] * inv_d;
            }

            // Magnetic field and vector potential at i due to moving charge j
            if cfg.enable_magnetic {
                // A_i += μ0/(4π) * q_j v_j / d
                avec[i] = avec[i].add(st.v[j].mul(cfg.mu0_over_4pi * st.q[j] * inv_d));
                // B_i += μ0/(4π) q_j (v_j × rij)/d^3
                b[i] = b[i].add(st.v[j].cross(rij).mul(cfg.mu0_over_4pi * st.q[j] * inv_d3));
            }
        }
    }

    // Total accel: gravity + Lorentz
    let mut a = [Vec3::default(), Vec3::default(), Vec3::default()];
    for i in 0..3 {
        a[i] = a[i].add(a_g[i]);
        // Lorentz accel = (q_i/m_i) (E_i + v_i × B_i)
        if (cfg.enable_electric || cfg.enable_magnetic) && st.m[i] != 0.0 {
            let lor = e[i].add(st.v[i].cross(b[i]));
            a[i] = a[i].add(lor.mul(st.q[i] / st.m[i]));
        }
    }

    Fields { a, e, b, phi, avec }
}

/// Mechanical energy diagnostic: KE + gravitational PE + electric PE.
/// (Magnetic energy is not modeled consistently in this quasi-static approximation.)
pub fn mechanical_energy(cfg: &SimConfig, st: &SystemState) -> f64 {
    // Kinetic
    let mut ke_mech = 0.0;
    for i in 0..3 {
        ke_mech += 0.5 * st.m[i] * st.v[i].norm2();
    }

    // Potential: gravity + electric
    let eps = cfg.softening;
    let mut pe = 0.0;
    for i in 0..3 {
        for j in (i+1)..3 {
            let rij = st.r[j].sub(st.r[i]);
            let d = (rij.norm2() + eps*eps).sqrt().max(1e-18);

            if cfg.enable_gravity {
                pe += -cfg.g * st.m[i] * st.m[j] / d;
            }
            if cfg.enable_electric {
                pe += cfg.ke * st.q[i] * st.q[j] / d;
            }
        }
    }

    ke_mech + pe
}

pub fn total_momentum(st: &SystemState) -> Vec3 {
    st.v[0].mul(st.m[0]).add(st.v[1].mul(st.m[1])).add(st.v[2].mul(st.m[2]))
}

pub fn total_angular_momentum(st: &SystemState) -> Vec3 {
    let mut l = Vec3::default();
    for i in 0..3 {
        let p = st.v[i].mul(st.m[i]);
        l = l.add(st.r[i].cross(p));
    }
    l
}

pub fn example_config() -> ConfigFile {
    // Example: gravity-only figure-eight (charges 0). Set ke/mu0 constants but q=0 => EM off effectively.
    ConfigFile {
        g: 1.0,
        ke: 1.0,
        mu0_over_4pi: 1.0,
        softening: 0.0,
        enable_gravity: true,
        enable_electric: true,
        enable_magnetic: true,
        bodies: vec![
            Body {
                name: "A".to_string(),
                m: 1.0,
                q: 0.0,
                r: Vec3::new(-0.97000436, 0.24308753, 0.0),
                v: Vec3::new(0.4662036850, 0.4323657300, 0.0),
            },
            Body {
                name: "B".to_string(),
                m: 1.0,
                q: 0.0,
                r: Vec3::new(0.97000436, -0.24308753, 0.0),
                v: Vec3::new(0.4662036850, 0.4323657300, 0.0),
            },
            Body {
                name: "C".to_string(),
                m: 1.0,
                q: 0.0,
                r: Vec3::new(0.0, 0.0, 0.0),
                v: Vec3::new(-0.93240737, -0.86473146, 0.0),
            },
        ],
    }
}
2) Update integrators to use the new acceleration function and output accel/fields
Replace src/integrators.rs with this version (minimal diffs but complete):

use anyhow::Result;

use crate::io::CsvWriter;
use crate::sim::{
    fields_and_accel, mechanical_energy, total_angular_momentum, total_momentum,
    SimConfig, SystemState, Vec3,
};

fn write_row(cfg: &SimConfig, st: &SystemState, w: &mut CsvWriter, e0: f64, p0: Vec3, l0: Vec3) -> Result<()> {
    let f = fields_and_accel(cfg, st);

    let e = mechanical_energy(cfg, st);
    let p = total_momentum(st);
    let l = total_angular_momentum(st);

    w.write(
        st.t,
        st,
        &f,
        e,
        p,
        l,
        e - e0,
        p.sub(p0),
        l.sub(l0),
    )?;
    Ok(())
}

/// Symplectic kick-drift-kick leapfrog.
/// Uses total acceleration a(t) including gravity + Lorentz.
pub fn run_leapfrog(
    cfg: &SimConfig,
    st: &mut SystemState,
    dt: f64,
    t_end: f64,
    stride: usize,
    w: &mut CsvWriter,
) -> Result<()> {
    let e0 = mechanical_energy(cfg, st);
    let p0 = total_momentum(st);
    let l0 = total_angular_momentum(st);

    let mut step: usize = 0;
    write_row(cfg, st, w, e0, p0, l0)?;

    while st.t < t_end {
        let f0 = fields_and_accel(cfg, st);

        // Kick 1/2
        for i in 0..3 {
            st.v[i] = st.v[i].add(f0.a[i].mul(0.5 * dt));
        }

        // Drift
        for i in 0..3 {
            st.r[i] = st.r[i].add(st.v[i].mul(dt));
        }

        // Kick 1/2 with new accel
        let f1 = fields_and_accel(cfg, st);
        for i in 0..3 {
            st.v[i] = st.v[i].add(f1.a[i].mul(0.5 * dt));
        }

        st.t += dt;
        step += 1;

        if step % stride == 0 {
            write_row(cfg, st, w, e0, p0, l0)?;
        }
    }

    Ok(())
}

/// Adaptive RK45 (Dormand–Prince). Uses acceleration from fields_and_accel.
pub fn run_rk45(
    cfg: &SimConfig,
    st: &mut SystemState,
    mut dt: f64,
    t_end: f64,
    stride: usize,
    tol: f64,
    dt_min: f64,
    dt_max: f64,
    w: &mut CsvWriter,
) -> Result<()> {
    let e0 = mechanical_energy(cfg, st);
    let p0 = total_momentum(st);
    let l0 = total_angular_momentum(st);

    let mut accepted_steps: usize = 0;
    write_row(cfg, st, w, e0, p0, l0)?;

    while st.t < t_end {
        if st.t + dt > t_end {
            dt = t_end - st.t;
        }

        let (st_new, err) = rk45_step(cfg, st, dt);
        let scale = state_scale(st).max(1e-12);
        let rel = err / scale;

        if rel <= tol {
            *st = st_new;
            st.t += dt;
            accepted_steps += 1;

            if accepted_steps % stride == 0 {
                write_row(cfg, st, w, e0, p0, l0)?;
            }

            let safety = 0.9;
            let factor = (safety * (tol / rel.max(1e-18)).powf(0.2)).clamp(0.2, 5.0);
            dt = (dt * factor).clamp(dt_min, dt_max);
        } else {
            let safety = 0.9;
            let factor = (safety * (tol / rel.max(1e-18)).powf(0.25)).clamp(0.1, 0.5);
            dt = (dt * factor).clamp(dt_min, dt_max);
            if dt <= dt_min {
                *st = st_new;
                st.t += dt;
                accepted_steps += 1;
            }
        }
    }

    Ok(())
}

fn state_scale(st: &SystemState) -> f64 {
    let mut m = 0.0;
    for i in 0..3 {
        m = m.max(st.r[i].norm());
        m = m.max(st.v[i].norm());
    }
    m
}

/// Packs state into y = [r0,v0,r1,v1,r2,v2]
fn pack(st: &SystemState) -> [f64; 18] {
    let mut y = [0.0; 18];
    for i in 0..3 {
        let base = i * 6;
        y[base + 0] = st.r[i].x;
        y[base + 1] = st.r[i].y;
        y[base + 2] = st.r[i].z;
        y[base + 3] = st.v[i].x;
        y[base + 4] = st.v[i].y;
        y[base + 5] = st.v[i].z;
    }
    y
}

fn unpack(st: &SystemState, y: [f64; 18]) -> SystemState {
    let mut out = st.clone();
    for i in 0..3 {
        let base = i * 6;
        out.r[i] = Vec3::new(y[base+0], y[base+1], y[base+2]);
        out.v[i] = Vec3::new(y[base+3], y[base+4], y[base+5]);
    }
    out
}

fn deriv(cfg: &SimConfig, st: &SystemState, y: [f64; 18]) -> [f64; 18] {
    let tmp = unpack(st, y);
    let f = fields_and_accel(cfg, &tmp);

    let mut dy = [0.0; 18];
    for i in 0..3 {
        let base = i * 6;
        dy[base+0] = tmp.v[i].x;
        dy[base+1] = tmp.v[i].y;
        dy[base+2] = tmp.v[i].z;
        dy[base+3] = f.a[i].x;
        dy[base+4] = f.a[i].y;
        dy[base+5] = f.a[i].z;
    }
    dy
}

/// Dormand–Prince RK5(4) step with embedded error estimate.
fn rk45_step(cfg: &SimConfig, st: &SystemState, dt: f64) -> (SystemState, f64) {
    let y0 = pack(st);

    let k1 = deriv(cfg, st, y0);

    let y2 = add_scaled(y0, &k1, dt * (1.0/5.0));
    let k2 = deriv(cfg, st, y2);

    let y3 = add_scaled2(y0, &k1, &k2, dt * (3.0/40.0), dt * (9.0/40.0));
    let k3 = deriv(cfg, st, y3);

    let y4 = add_scaled3(
        y0, &k1, &k2, &k3,
        dt*(44.0/45.0), dt*(-56.0/15.0), dt*(32.0/9.0)
    );
    let k4 = deriv(cfg, st, y4);

    let y5 = add_scaled4(
        y0, &k1, &k2, &k3, &k4,
        dt*(19372.0/6561.0), dt*(-25360.0/2187.0), dt*(64448.0/6561.0), dt*(-212.0/729.0)
    );
    let k5 = deriv(cfg, st, y5);

    let y6 = add_scaled5(
        y0, &k1, &k2, &k3, &k4, &k5,
        dt*(9017.0/3168.0), dt*(-355.0/33.0), dt*(46732.0/5247.0), dt*(49.0/176.0), dt*(-5103.0/18656.0)
    );
    let k6 = deriv(cfg, st, y6);

    let y7 = add_scaled6(
        y0, &k1, &k2, &k3, &k4, &k5, &k6,
        dt*(35.0/384.0), 0.0, dt*(500.0/1113.0), dt*(125.0/192.0), dt*(-2187.0/6784.0), dt*(11.0/84.0)
    );
    let k7 = deriv(cfg, st, y7);

    let y5th = y7;

    let y4th = add_scaled6(
        y0, &k1, &k2, &k3, &k4, &k5, &k6,
        dt*(5179.0/57600.0),
        0.0,
        dt*(7571.0/16695.0),
        dt*(393.0/640.0),
        dt*(-92097.0/339200.0),
        dt*(187.0/2100.0)
    );
    let y4th = add_scaled(y4th, &k7, dt*(1.0/40.0));

    let err = diff_norm(y5th, y4th);
    (unpack(st, y5th), err)
}

fn add_scaled(y: [f64; 18], k: &[f64; 18], a: f64) -> [f64; 18] {
    let mut out = y;
    for i in 0..18 { out[i] += a * k[i]; }
    out
}
fn add_scaled2(y: [f64; 18], k1: &[f64; 18], k2: &[f64; 18], a1: f64, a2: f64) -> [f64; 18] {
    let mut out = y;
    for i in 0..18 { out[i] += a1*k1[i] + a2*k2[i]; }
    out
}
fn add_scaled3(y: [f64; 18], k1: &[f64; 18], k2: &[f64; 18], k3: &[f64; 18], a1: f64, a2: f64, a3: f64) -> [f64; 18] {
    let mut out = y;
    for i in 0..18 { out[i] += a1*k1[i] + a2*k2[i] + a3*k3[i]; }
    out
}
fn add_scaled4(y: [f64; 18], k1: &[f64; 18], k2: &[f64; 18], k3: &[f64; 18], k4: &[f64; 18], a1: f64, a2: f64, a3: f64, a4: f64) -> [f64; 18] {
    let mut out = y;
    for i in 0..18 { out[i] += a1*k1[i] + a2*k2[i] + a3*k3[i] + a4*k4[i]; }
    out
}
fn add_scaled5(y: [f64; 18], k1: &[f64; 18], k2: &[f64; 18], k3: &[f64; 18], k4: &[f64; 18], k5: &[f64; 18], a1: f64, a2: f64, a3: f64, a4: f64, a5: f64) -> [f64; 18] {
    let mut out = y;
    for i in 0..18 { out[i] += a1*k1[i] + a2*k2[i] + a3*k3[i] + a4*k4[i] + a5*k5[i]; }
    out
}
fn add_scaled6(
    y: [f64; 18],
    k1: &[f64; 18], k2: &[f64; 18], k3: &[f64; 18],
    k4: &[f64; 18], k5: &[f64; 18], k6: &[f64; 18],
    a1: f64, a2: f64, a3: f64, a4: f64, a5: f64, a6: f64
) -> [f64; 18] {
    let mut out = y;
    for i in 0..18 {
        out[i] += a1*k1[i] + a2*k2[i] + a3*k3[i] + a4*k4[i] + a5*k5[i] + a6*k6[i];
    }
    out
}
fn diff_norm(a: [f64; 18], b: [f64; 18]) -> f64 {
    let mut s = 0.0;
    for i in 0..18 {
        let d = a[i] - b[i];
        s += d*d;
    }
    s.sqrt()
}
3) Update src/io.rs writer to include accel + fields + potentials
Replace your existing CsvWriter in src/io.rs with this version (the rest of io.rs can remain).

use anyhow::Result;
use csv::WriterBuilder;

use crate::sim::{Fields, SystemState, Vec3};

pub struct CsvWriter {
    w: csv::Writer<std::fs::File>,
}

impl CsvWriter {
    pub fn new(path: &str) -> Result<Self> {
        let file = std::fs::File::create(path)?;
        let mut w = WriterBuilder::new().has_headers(true).from_writer(file);

        // Trajectory + velocity + acceleration + EM fields + potentials + diagnostics
        w.write_record(&[
            "t",
            "a_x","a_y","a_z","a_vx","a_vy","a_vz","a_ax","a_ay","a_az",
            "b_x","b_y","b_z","b_vx","b_vy","b_vz","b_ax","b_ay","b_az",
            "c_x","c_y","c_z","c_vx","c_vy","c_vz","c_ax","c_ay","c_az",

            "a_ex","a_ey","a_ez","a_bx","a_by","a_bz","a_phi","a_Ax","a_Ay","a_Az",
            "b_ex","b_ey","b_ez","b_bx","b_by","b_bz","b_phi","b_Ax","b_Ay","b_Az",
            "c_ex","c_ey","c_ez","c_bx","c_by","c_bz","c_phi","c_Ax","c_Ay","c_Az",

            "energy_mech",
            "p_x","p_y","p_z",
            "l_x","l_y","l_z",
            "dE",
            "dP_x","dP_y","dP_z",
            "dL_x","dL_y","dL_z",
        ])?;

        Ok(Self { w })
    }

    pub fn write(
        &mut self,
        t: f64,
        st: &SystemState,
        f: &Fields,
        energy_mech: f64,
        p: Vec3,
        l: Vec3,
        dE: f64,
        dP: Vec3,
        dL: Vec3,
    ) -> Result<()> {
        let a = 0; let b = 1; let c = 2;

        self.w.write_record(&[
            t.to_string(),

            st.r[a].x.to_string(), st.r[a].y.to_string(), st.r[a].z.to_string(),
            st.v[a].x.to_string(), st.v[a].y.to_string(), st.v[a].z.to_string(),
            f.a[a].x.to_string(),  f.a[a].y.to_string(),  f.a[a].z.to_string(),

            st.r[b].x.to_string(), st.r[b].y.to_string(), st.r[b].z.to_string(),
            st.v[b].x.to_string(), st.v[b].y.to_string(), st.v[b].z.to_string(),
            f.a[b].x.to_string(),  f.a[b].y.to_string(),  f.a[b].z.to_string(),

            st.r[c].x.to_string(), st.r[c].y.to_string(), st.r[c].z.to_string(),
            st.v[c].x.to_string(), st.v[c].y.to_string(), st.v[c].z.to_string(),
            f.a[c].x.to_string(),  f.a[c].y.to_string(),  f.a[c].z.to_string(),

            f.e[a].x.to_string(), f.e[a].y.to_string(), f.e[a].z.to_string(),
            f.b[a].x.to_string(), f.b[a].y.to_string(), f.b[a].z.to_string(),
            f.phi[a].to_string(),
            f.avec[a].x.to_string(), f.avec[a].y.to_string(), f.avec[a].z.to_string(),

            f.e[b].x.to_string(), f.e[b].y.to_string(), f.e[b].z.to_string(),
            f.b[b].x.to_string(), f.b[b].y.to_string(), f.b[b].z.to_string(),
            f.phi[b].to_string(),
            f.avec[b].x.to_string(), f.avec[b].y.to_string(), f.avec[b].z.to_string(),

            f.e[c].x.to_string(), f.e[c].y.to_string(), f.e[c].z.to_string(),
            f.b[c].x.to_string(), f.b[c].y.to_string(), f.b[c].z.to_string(),
            f.phi[c].to_string(),
            f.avec[c].x.to_string(), f.avec[c].y.to_string(), f.avec[c].z.to_string(),

            energy_mech.to_string(),
            p.x.to_string(), p.y.to_string(), p.z.to_string(),
            l.x.to_string(), l.y.to_string(), l.z.to_string(),
            dE.to_string(),
            dP.x.to_string(), dP.y.to_string(), dP.z.to_string(),
            dL.x.to_string(), dL.y.to_string(), dL.z.to_string(),
        ])?;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        self.w.flush()?;
        Ok(())
    }
}
4) Update src/main.rs to pass the expanded config into SimConfig
In your simulate handler, replace the old SimConfig creation with:

let simcfg = SimConfig {
    g: cfg.g,
    ke: cfg.ke,
    mu0_over_4pi: cfg.mu0_over_4pi,
    softening: cfg.softening,
    enable_gravity: cfg.enable_gravity,
    enable_electric: cfg.enable_electric,
    enable_magnetic: cfg.enable_magnetic,
};
Also, since ConfigFile now has new fields, your existing example-config output will include them.

5) Run a quick sanity check
Generate config
cargo run --release -- example-config --out config.json
Simulate and inspect columns
cargo run --release -- simulate --config config.json --out traj.csv --integrator rk45 --t-end 2 --stride 1 --barycentric
head -n 2 traj.csv
You should see the new headers including a_ax, a_ex, a_phi, a_Ax, etc.

Notes you should be aware of (accuracy)
This EM model is quasi-static (no retardation). It’s coherent and stable for a lot of scenarios, but not “full Maxwell + radiation”. If you need that later, the whole field computation changes.

Softening is still optional. For discovery work, keep softening: 0.0 unless you intentionally want “regularized physics”.

If you want, next I’ll update your discover pipeline to:

prefer a_ax/a_ay/a_az (now present)

add a library variant that uses E and v×B directly (em_fields) for even simpler discovered laws.

You said:
Write an academic paper describing our approach accurately.

Be editorial and point out areas we are unsure of.
ChatGPT said:
A Two-System Framework for Truth-Grade Three-Body Simulation and Sparse Discovery of Interpretable Predictive Dynamics
Author: Jonathan Hendler (et al.)
Version: Draft (editorial; explicitly notes open issues)

Abstract
We present a two-system framework for studying and predicting three-body dynamics under coupled gravitational and electromagnetic interactions. The first system is a high-fidelity numerical simulator (“oracle”) designed for reproducibility, performance, and explicit error control. It supports Newtonian gravity and a nonrelativistic electromagnetic interaction model including electric and magnetic fields as well as scalar and vector potentials (φ, A). The second system is a model-discovery engine that searches for sparse, interpretable equations that predict the oracle’s trajectories with minimal complexity. Discovery proceeds via structured feature libraries and sequential thresholded least squares (SINDy/STLS), using simulator-provided accelerations to avoid derivative noise. We provide an implementation-centric description and emphasize editorially the areas of modeling uncertainty: the electromagnetic approximation (quasi-static, no retardation or radiation reaction), conservation-law interpretation under the chosen EM model, close-encounter numerical stiffness, and the limits of global predictability in chaotic regimes. We argue that novelty arises not from claiming a universal closed-form solution, but from automatically discovering compact regime-specific laws with explicit validity domains and verified rollout performance against the oracle.

1. Introduction
The classical three-body problem is a paradigmatic nonlinear dynamical system exhibiting sensitivity to initial conditions and, for most configurations, the absence of closed-form global solutions. Practical prediction therefore relies on numerical integration, while theory offers partial analytic structure (integrals of motion, special solutions, perturbation expansions, invariant manifolds). For modern applications—ranging from celestial mechanics to charged particle dynamics—one often wants both: (i) an accurate simulator suitable as “ground truth,” and (ii) interpretable approximations that predict well and illuminate structure.

We propose a two-system approach:

Truth-grade simulation system (“oracle”): a careful, reproducible, numerically controlled engine that produces trajectories and physically meaningful diagnostics.

Equation discovery system (“predictor”): a sparse identification pipeline that produces compact, interpretable predictive models that approximate the oracle and are validated by forward rollout.

This paper describes the architecture, numerical methods, and discovery methodology, and it explicitly lists modeling assumptions and unresolved issues.

2. Simulator (“Oracle”) Design
2.1 State, coordinate choices, and invariants
The simulator evolves three bodies indexed by 
i
∈
{
1
,
2
,
3
}
i∈{1,2,3} with:

Mass 
m
i
m 
i
​
 , charge 
q
i
q 
i
​
 

Position 
r
i
(
t
)
∈
R
3
r 
i
​
 (t)∈R 
3
 

Velocity 
v
i
(
t
)
=
r
˙
i
(
t
)
∈
R
3
v 
i
​
 (t)= 
r
˙
  
i
​
 (t)∈R 
3
 

We typically re-center initial conditions to the center-of-mass (COM) frame and remove net linear momentum (barycentric initialization) to reduce drift. Invariants used as diagnostics include:

Total linear momentum 
P
=
∑
i
m
i
v
i
P=∑ 
i
​
 m 
i
​
 v 
i
​
 

Total angular momentum 
L
=
∑
i
r
i
×
(
m
i
v
i
)
L=∑ 
i
​
 r 
i
​
 ×(m 
i
​
 v 
i
​
 )

A “mechanical energy” proxy (Section 2.4), with editorial caveats for EM.

2.2 Interaction model: gravity and nonrelativistic electromagnetism
Gravity (Newtonian)
We use:

a
i
(
g
)
=
∑
j
≠
i
G
 
m
j
r
j
−
r
i
∥
r
j
−
r
i
∥
3
.
a 
i
(g)
​
 = 
j

=i
∑
​
 Gm 
j
​
  
∥r 
j
​
 −r 
i
​
 ∥ 
3
 
r 
j
​
 −r 
i
​
 
​
 .
Electromagnetism (chosen approximation)
We include an explicit, nonrelativistic EM model:

Electric field at particle 
i
i:

E
i
=
∑
j
≠
i
k
e
 
q
j
r
i
−
r
j
∥
r
i
−
r
j
∥
3
.
E 
i
​
 = 
j

=i
∑
​
 k 
e
​
 q 
j
​
  
∥r 
i
​
 −r 
j
​
 ∥ 
3
 
r 
i
​
 −r 
j
​
 
​
 .
Magnetic field at particle 
i
i via a quasi-static Biot–Savart-like term for moving charges:

B
i
=
∑
j
≠
i
μ
0
4
π
 
q
j
v
j
×
(
r
i
−
r
j
)
∥
r
i
−
r
j
∥
3
.
B 
i
​
 = 
j

=i
∑
​
  
4π
μ 
0
​
 
​
 q 
j
​
  
∥r 
i
​
 −r 
j
​
 ∥ 
3
 
v 
j
​
 ×(r 
i
​
 −r 
j
​
 )
​
 .
Lorentz acceleration:

a
i
(
e
m
)
=
q
i
m
i
(
E
i
+
v
i
×
B
i
)
.
a 
i
(em)
​
 = 
m 
i
​
 
q 
i
​
 
​
 (E 
i
​
 +v 
i
​
 ×B 
i
​
 ).
Total acceleration:

a
i
=
a
i
(
g
)
+
a
i
(
e
m
)
.
a 
i
​
 =a 
i
(g)
​
 +a 
i
(em)
​
 .
We also compute potentials:

Scalar potential:

ϕ
i
=
∑
j
≠
i
k
e
q
j
∥
r
i
−
r
j
∥
.
ϕ 
i
​
 = 
j

=i
∑
​
 k 
e
​
  
∥r 
i
​
 −r 
j
​
 ∥
q 
j
​
 
​
 .
Vector potential:

A
i
=
∑
j
≠
i
μ
0
4
π
q
j
v
j
∥
r
i
−
r
j
∥
.
A 
i
​
 = 
j

=i
∑
​
  
4π
μ 
0
​
 
​
  
∥r 
i
​
 −r 
j
​
 ∥
q 
j
​
 v 
j
​
 
​
 .
Editorial uncertainty: This EM model omits finite propagation speed (retardation), radiation reaction, and self-field effects. It is best interpreted as a quasi-static approximation; its domain of physical validity is not universal. For “truth-grade” claims, the simulator is truth-grade relative to the assumed equations, not necessarily relative to full Maxwell–Lorentz dynamics.

2.3 Numerical integration
We provide two integrators:

Adaptive RK45 (Dormand–Prince): used as “truth mode” for controlled local truncation error. It adapts time step using embedded error estimates with user-specified tolerances and min/max dt.

Leapfrog (Kick–Drift–Kick): a fast, stable second-order method often favored for long-horizon gravitational dynamics.

Editorial uncertainty: For purely Hamiltonian gravity, symplectic methods have strong long-term stability guarantees. With the chosen EM terms, the system is not treated with a full Hamiltonian field formulation here; leapfrog remains useful but should not be assumed to preserve the same structure exactly. This is one reason we maintain RK-based truth mode.

2.4 Diagnostics and conservation interpretation
We log:

P
(
t
)
P(t), 
L
(
t
)
L(t)

A “mechanical energy” diagnostic:

Kinetic: 
∑
i
1
2
m
i
∥
v
i
∥
2
∑ 
i
​
  
2
1
​
 m 
i
​
 ∥v 
i
​
 ∥ 
2
 

Gravitational potential: 
∑
i
<
j
−
G
m
i
m
j
/
∥
r
i
j
∥
∑ 
i<j
​
 −Gm 
i
​
 m 
j
​
 /∥r 
ij
​
 ∥

Electric potential: 
∑
i
<
j
k
e
q
i
q
j
/
∥
r
i
j
∥
∑ 
i<j
​
 k 
e
​
 q 
i
​
 q 
j
​
 /∥r 
ij
​
 ∥

Editorial uncertainty: In full electrodynamics, energy conservation includes field energy and radiation; in quasi-static approximations, interpreting “total energy” is subtle. We therefore treat energy drift as a numerical health metric, not a strict physical invariant under all EM configurations.

2.5 Close encounters and stiffness
Near-collisions can cause:

accelerations scaling as 
1
/
r
2
1/r 
2
  (and worse in some derived terms),

adaptive dt collapse,

large floating-point conditioning issues.

Current implementation supports optional softening 
ϵ
ϵ in denominators to avoid singularities.

Editorial uncertainty: Softening changes physics. A true “truth-grade” approach for close encounters would incorporate regularization (e.g., time transformations, KS/chain regularization) and/or higher precision. This is a clear future-work item.

2.6 Output contract for downstream discovery
The simulator outputs a wide CSV including:

r
i
,
v
i
r 
i
​
 ,v 
i
​
  for each body

a
i
a 
i
​
  (total acceleration) for each body

E
i
,
B
i
,
ϕ
i
,
A
i
E 
i
​
 ,B 
i
​
 ,ϕ 
i
​
 ,A 
i
​
 

diagnostics: energy proxy, 
P
P, 
L
L, drifts

Design rationale: Discovery should use simulator-provided accelerations, not finite differences of noisy trajectories.

3. Discovery Engine (“Predictor”): Sparse, Interpretable Dynamics
3.1 Objective
We seek simple, interpretable functions 
f
^
f
^
​
  that approximate oracle accelerations:

a
i
(
t
)
≈
f
^
i
(
features
(
r
(
t
)
,
v
(
t
)
,
…
)
)
,
a 
i
​
 (t)≈ 
f
^
​
  
i
​
 (features(r(t),v(t),…)),
with:

high forward-prediction fidelity over a time horizon,

minimal equation complexity,

explicit domain-of-validity.

This is fundamentally different from “solving the three-body problem in closed form.” The goal is regime-specific compact predictive structure.

3.2 Why we fit acceleration directly
Classical system identification often estimates 
x
˙
x
˙
  from measurements; in chaotic systems this introduces strong noise amplification. We avoid this by training on:

simulator-provided 
a
i
(
t
)
a 
i
​
 (t),
rather than differentiating 
v
i
(
t
)
v 
i
​
 (t).

3.3 Feature libraries (Θ) as structured hypothesis spaces
We build vector-valued basis functions (“terms”) from physically meaningful atoms:

relative vectors 
r
i
j
=
r
j
−
r
i
r 
ij
​
 =r 
j
​
 −r 
i
​
 ,

distances 
d
i
j
=
∥
r
i
j
∥
d 
ij
​
 =∥r 
ij
​
 ∥,

unit directions 
r
^
i
j
r
^
  
ij
​
 ,

velocities and differences 
v
i
,
v
i
j
v 
i
​
 ,v 
ij
​
 ,

cross products capturing magnetic-like structure 
v
j
×
r
i
j
v 
j
​
 ×r 
ij
​
 ,

triple products capturing Lorentz-like structure 
v
i
×
(
v
j
×
r
i
j
)
v 
i
​
 ×(v 
j
​
 ×r 
ij
​
 ).

We implement libraries of increasing expressivity:

gravity_min: 
r
^
i
j
/
d
i
j
2
r
^
  
ij
​
 /d 
ij
2
​
 , 
r
^
i
j
/
d
i
j
3
r
^
  
ij
​
 /d 
ij
3
​
 

gravity_plus: adds simple velocity couplings

em_basic: adds magnetic-shaped cross terms and gated scalar modulations

Editorial note: The library design is where most “new math” leverage lives. If the library is too broad, discovery becomes unstable and non-interpretable; too narrow, it cannot capture dynamics. We currently err toward small, interpretable libraries.

3.4 Sparse regression via STLS (SINDy-style)
Let 
y
y be a target acceleration component (e.g., 
a
i
,
x
a 
i,x
​
 ) and let Θ be the feature matrix evaluated over time. We solve:

y
≈
Θ
ξ
y≈Θξ
with sparsity enforced by Sequential Thresholded Least Squares:

fit ridge-regularized least squares,

threshold small coefficients below 
λ
λ,

refit on the remaining active set,

iterate.

We normalize feature columns (RMS scaling) before thresholding to ensure 
λ
λ has consistent meaning across heterogeneous terms.

Editorial uncertainty: STLS can be sensitive to collinearity and scaling; our current approach uses ridge stabilization and normalization, but more robust solvers (e.g., LASSO with cross-validation) may improve stability.

3.5 What counts as “predicts well”
Pointwise acceleration fit is not sufficient. We evaluate:

rollout accuracy: integrate the discovered model forward and compare trajectory error against the oracle,

stability and failure modes (blow-ups, divergence time),

generalization to held-out initial conditions within the intended regime.

Editorial uncertainty: In chaotic regimes, long-horizon point prediction is not meaningful; evaluation should shift toward:

short-horizon deterministic accuracy,

ensemble/statistical predictions,

qualitative invariants and manifold structure.
We are not yet fully implementing those alternative evaluation regimes.

4. System Integration and Reproducibility
4.1 CLI-driven workflow
The systems are controlled by CLI:

simulate: produce truth trajectories and diagnostics

discover: fit sparse equations using a chosen library

predict (planned): roll out discovered equations and compare to truth

score (planned): compute quantitative benchmarks

render: visualization (with robust header-based parsing)

4.2 Robust I/O design
We explicitly fixed a key fragility: rigid CSV schemas break as output evolves. We therefore parse by header name and extract only needed columns for any given task. This enables rapid iteration.

4.3 Determinism and metadata
A “truth-grade” oracle requires:

fixed integrator and tolerance settings recorded,

initial condition specifications recorded,

constants and force toggles recorded.

Current implementation outputs CSV; we recommend adding a JSON “sidecar” containing run parameters to support strict reproducibility.

5. Results (What we expect, what we can claim, and what we cannot)
5.1 Expected outcomes
For gravity-only or weak-EM regimes:

gravity_min should rediscover near-inverse-square directional structure.

em_basic may recover Lorentz-like cross terms if EM effects are present and sufficiently sampled.

Sparse models can achieve strong short-horizon rollout fidelity with few terms.

5.2 What we can claim
The simulator provides consistent numerical solutions to the implemented equations with user-controlled accuracy (RK45 mode).

The discovery engine can identify compact, interpretable approximations within structured libraries and validate them by rollout.

5.3 What we cannot honestly claim (yet)
That the simulator is “physically exact” for EM in all regimes (it is quasi-static).

That energy conservation is strict in combined EM dynamics as implemented (field energy not modeled).

That close-encounter accuracy is truth-grade without regularization/high precision.

That a single global simple formula predicts chaotic three-body dynamics long-horizon.

6. Limitations and Open Questions (Editorial)
This section is intentionally frank.

Electromagnetic modeling fidelity
The quasi-static magnetic approximation may be adequate for some regimes but fails when retardation or radiation are significant. If the objective is “physics truth,” the EM model must be upgraded (retarded potentials or field solver), at significant computational cost.

Integrator structure under EM
Leapfrog is attractive for gravity; under EM it may not preserve the right structure. We currently treat leapfrog as a fast option, not as a guaranteed long-term invariant-preserving method under full coupled dynamics.

Close encounters and stiffness
Softening is not a principled solution. We need a regularization strategy (Sundman transform, KS/chain regularization) and possibly adaptive precision.

Discovery stability and identifiability
Sparse regression can select misleading terms when features are correlated or data is not sufficiently rich. Better experimental design (diverse initial conditions, active learning) and more robust optimization (e.g., constrained solvers) are likely necessary.

Meaning of “novel math”
We interpret “novel” as compact approximations that outperform naive baselines within a regime and are expressed in human-interpretable terms. We do not claim a universal analytic solution.

7. Future Work
Add close-encounter regularization and/or multi-precision truth mode.

Implement predict and score commands with standardized benchmarks (divergence time, RMS error curves).

Add an em_fields library that learns directly from simulator fields 
E
,
B
E,B to obtain even simpler discovered laws.

Implement regime partitioning (“atlas of models”): train multiple sparse models with a lightweight regime classifier.

Extend discovery to probabilistic forecasts for chaotic regimes.

8. Conclusion
We described a pragmatic two-system approach to three-body dynamics: a high-fidelity simulator as oracle, and a sparse discovery engine that produces interpretable, compact predictive equations. The central technical strategy is to train on simulator-provided accelerations and to constrain the hypothesis space through physically structured feature libraries, thereby producing “simple math” that predicts well within explicit regimes. We also highlighted the primary uncertainties: electromagnetic approximation fidelity, conservation-law interpretation, close-encounter stiffness, and the limits of long-horizon prediction in chaotic systems. This framework is designed to be extensible, reproducible, and honest about what is known versus assumed.

Appendix A: Implementation Notes (brief)
CSV parsing must be header-based to remain stable as columns evolve.

Discovery uses Θ with vector terms expanded into x/y/z columns and normalized by RMS scale.

STLS uses ridge stabilization due to feature collinearity.

If you want this formatted as a real submission (LaTeX structure, citations placeholders, and a sharper “Related Work” section), tell me the target venue style (e.g., arXiv, JCP, NeurIPS workshop) and whether you want to cite specific prior methods (e.g., SINDy, symplectic integrators, regularization methods, Maxwell–Lorentz approximations).

You said:
Can we use Solomonoff Induction?
ChatGPT said:
Yes—in spirit. But “use Solomonoff induction” literally (the universal prior over all programs) is incomputable, so the practical answer is: we can build a Solomonoff-inspired MDL/Bayesian program search layer on top of your simulator and sparse-regression pipeline.

Here’s what that looks like and where it helps.

What Solomonoff would give you (ideal)
Solomonoff induction says: predict by averaging over all computable hypotheses 
h
h weighted by 
2
−
K
(
h
)
2 
−K(h)
  (shorter programs get more prior mass), and update by likelihood from data.

For our setting:

hypotheses = candidate dynamical laws / update programs

data = trajectories from the oracle simulator

goal = posterior predictive rollouts

Great fit conceptually: it directly formalizes “simple math that predicts well”.

But: computing 
K
(
h
)
K(h) and summing over all programs is not feasible.

Practical substitute: MDL / Bayesian program induction
We can implement an approximate Solomonoff using:

1) A small “language of laws” (DSL)
Define a restricted expression language that can express:

pairwise terms: 
r
^
/
d
2
r
^
 /d 
2
 , 
r
^
/
d
3
r
^
 /d 
3
 

velocity terms: 
v
i
j
/
d
p
v 
ij
​
 /d 
p
 

cross terms: 
(
v
j
×
r
i
j
)
/
d
3
(v 
j
​
 ×r 
ij
​
 )/d 
3
 , 
v
i
×
(
v
j
×
r
i
j
)
/
d
3
v 
i
​
 ×(v 
j
​
 ×r 
ij
​
 )/d 
3
 

optional fields/potentials: 
E
E, 
v
×
B
v×B, 
∇
ϕ
∇ϕ, 
∇
×
A
∇×A

This is your “program class”. It’s not universal, but it’s rich and physics-shaped.

2) A Solomonoff-like prior via description length
Assign each candidate law a code length:

base cost per operator (+, ×, cross, dot)

cost for each term

cost for rational exponents / constants

cost for “gates” (piecewise regimes)

Prior: 
P
(
h
)
∝
2
−
L
(
h
)
P(h)∝2 
−L(h)
  where 
L
(
h
)
L(h) is that code length.
This is standard MDL.

3) Likelihood from oracle rollouts
Define a likelihood based on predictive error over rollout, not just one-step acceleration fit:

simulate the candidate law forward for a horizon 
H
H

compare to oracle positions/velocities

use Gaussian noise model or robust loss

So: 
log
⁡
P
(
D
∣
h
)
=
−
∑
t
∥
x
o
r
a
c
l
e
(
t
)
−
x
h
(
t
)
∥
2
/
(
2
σ
2
)
logP(D∣h)=−∑ 
t
​
 ∥x 
oracle
 (t)−x 
h
 (t)∥ 
2
 /(2σ 
2
 ) (plus constants)

4) Search strategy (approximate inference)
We don’t “sum over all programs”; we search for high posterior mass hypotheses:

enumerate expressions by increasing description length (best-first)

or use MCMC / SMC in program space

or do beam search with mutations (add/remove term, change exponent, swap operands)

Then either:

take MAP law 
h
∗
=
arg
⁡
max
⁡
log
⁡
P
(
D
∣
h
)
−
λ
L
(
h
)
h 
∗
 =argmaxlogP(D∣h)−λL(h)

or do Bayesian model averaging over the top N hypotheses.

How this fits your current system
Your current SINDy/STLS approach is already close to an MDL objective:

sparsity threshold ≈ penalizing complexity

ridge/thresholding ≈ regularized likelihood

What Solomonoff-inspired MDL adds:

Explicit complexity accounting (operators/terms cost) rather than “lambda on coefficients”

Program-level priors (e.g., cross products cost more than additions)

Model averaging (not just one best model)

Piecewise laws with explicit regime gating, still penalized by description length

In practice, a strong hybrid is:

Hybrid workflow
Use SINDy to propose a small set of candidate sparse models quickly.

Convert each candidate to DSL form and score it with MDL+rollout likelihood.

Locally refine via program search around the best candidates.

Keep a posterior set and optionally average predictions.

This yields “simple math” with a principled “simplicity prior” that’s very Solomonoff-like.

What we should be careful about (editorial / uncertainty)
Prior dependence: results depend on how we define code length. That’s unavoidable; be transparent.

Chaos: long-horizon likelihood will punish almost everything if you demand point accuracy. For chaotic regimes, you’ll need:

short horizon likelihood, or

statistical/ensemble likelihood (distributional prediction)

Model misspecification: if the DSL doesn’t contain the true structure, MDL will pick the best wrong model.

Compute: rollout-based scoring is expensive; you’ll want caching and staged evaluation (cheap filters first).

Concrete implementable plan (minimal)
Add a program_search module:

dsl.rs: expression tree for vector laws

cost.rs: code length function 
L
(
h
)
L(h)

fit.rs: given a structure, fit coefficients by least squares

score.rs: rollout likelihood vs oracle

search.rs: beam search by increasing 
L
(
h
)
L(h), seeded from SINDy outputs

CLI:

threebody induce --input traj.csv --library em_basic --out posterior.json --beam 50 --max-terms 8 --horizon 2000

Output:

top hypotheses with:

symbolic form

coefficients

code length

rollout score

regime validity stats

Bottom line
We can’t do Solomonoff induction literally, but we can implement a Solomonoff-style simplicity prior + program search (MDL Bayesian program induction) that’s aligned with your goal: discover accurate, novel, simple predictive math grounded in an oracle simulator.

If you want me to implement it next, I’ll start by defining the DSL and the code-length prior, then add a beam search seeded by the SINDy models you already have.




No file chosenNo file chosen
ChatGPT can make mistakes. Check important info.

