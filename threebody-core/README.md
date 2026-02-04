# threebody-core

Core library for three-body simulation with gravity and quasi-static EM.

**Quick example**
```rust,no_run
use threebody_core::config::Config;
use threebody_core::integrators::leapfrog::Leapfrog;
use threebody_core::math::vec3::Vec3;
use threebody_core::sim::{simulate, SimOptions};
use threebody_core::state::{Body, State, System};

let bodies = [Body::new(1.0, 0.0), Body::new(1.0, 0.0), Body::new(0.0, 0.0)];
let pos = [Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0), Vec3::zero()];
let vel = [Vec3::zero(); 3];
let system = System::new(bodies, State::new(pos, vel));
let cfg = Config::default();
let integrator = Leapfrog;
let options = SimOptions { steps: 10, dt: 0.01 };

let result = simulate(system, &cfg, &integrator, None, options);
println!("steps={}", result.steps.len());
```

**Key modules**
- `config`: model parameters and integrator choices.
- `forces`: gravity and EM models.
- `integrators`: leapfrog, RK45, Boris.
- `sim`: simulation driver, adaptive RK45 truth mode, and encounter policy.
- `output`: CSV writer (includes per-step `dt`) and sidecar metadata.
