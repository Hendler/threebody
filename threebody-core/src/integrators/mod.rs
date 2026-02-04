use crate::config::Config;
use crate::state::System;

pub mod leapfrog;
pub mod boris;
pub mod rk45;
pub mod implicit_midpoint;

/// Core integrator interface.
pub trait Integrator {
    fn step(&self, system: &System, dt: f64, cfg: &Config) -> System;
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IntegratorMeta {
    pub name: &'static str,
    pub supports_em: bool,
    pub symplectic: bool,
    pub adaptive_dt: bool,
}

pub const INTEGRATOR_METADATA: &[IntegratorMeta] = &[
    IntegratorMeta {
        name: "leapfrog",
        supports_em: false,
        symplectic: true,
        adaptive_dt: false,
    },
    IntegratorMeta {
        name: "rk45",
        supports_em: true,
        symplectic: false,
        adaptive_dt: true,
    },
    IntegratorMeta {
        name: "boris",
        supports_em: true,
        symplectic: true,
        adaptive_dt: false,
    },
    IntegratorMeta {
        name: "implicit_midpoint",
        supports_em: true,
        symplectic: true,
        adaptive_dt: false,
    },
];

pub fn integrator_metadata() -> &'static [IntegratorMeta] {
    INTEGRATOR_METADATA
}

#[cfg(test)]
mod tests {
    use super::Integrator;
    use crate::config::Config;
    use crate::math::vec3::Vec3;
    use crate::state::{Body, State, System};

    /// Simple explicit Euler integrator for x'' = -x on body 0's x-axis.
    struct EulerOscillator;

    impl Integrator for EulerOscillator {
        fn step(&self, system: &System, dt: f64, _cfg: &Config) -> System {
            let mut next = *system;
            let x = system.state.pos[0].x;
            let v = system.state.vel[0].x;
            let a = -x;
            next.state.pos[0].x = x + dt * v;
            next.state.vel[0].x = v + dt * a;
            next
        }
    }

    fn simulate(integrator: &impl Integrator, dt: f64, steps: usize) -> f64 {
        let bodies = [Body::new(1.0, 0.0), Body::new(0.0, 0.0), Body::new(0.0, 0.0)];
        let pos = [Vec3::new(1.0, 0.0, 0.0), Vec3::zero(), Vec3::zero()];
        let vel = [Vec3::new(0.0, 0.0, 0.0), Vec3::zero(), Vec3::zero()];
        let mut system = System::new(bodies, State::new(pos, vel));
        let cfg = Config::default();
        for _ in 0..steps {
            system = integrator.step(&system, dt, &cfg);
        }
        // True solution at t = steps * dt for x'' = -x with x(0)=1, v(0)=0 is cos(t).
        let t = dt * steps as f64;
        let x_true = t.cos();
        (system.state.pos[0].x - x_true).abs()
    }

    #[test]
    fn error_decreases_with_smaller_dt() {
        let integrator = EulerOscillator;
        let err_dt = simulate(&integrator, 0.1, 100);
        let err_dt2 = simulate(&integrator, 0.05, 200);
        assert!(err_dt2 < err_dt);
    }

    #[test]
    fn metadata_includes_all_integrators() {
        let names: Vec<&str> = super::integrator_metadata()
            .iter()
            .map(|meta| meta.name)
            .collect();
        for required in ["leapfrog", "rk45", "boris", "implicit_midpoint"] {
            assert!(names.contains(&required));
        }
    }
}
