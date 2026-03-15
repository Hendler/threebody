//! Force models and aggregate computation.

pub mod em;
pub mod gravity;
pub mod potentials;

use crate::config::{Config, EmModel};
use crate::math::vec3::Vec3;
use crate::state::System;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ForceConfig {
    pub g: f64,
    pub k_e: f64,
    pub mu_0: f64,
    pub c: f64,
    pub epsilon: f64,
    pub enable_gravity: bool,
    pub enable_em: bool,
    pub em_model: EmModel,
}

impl ForceConfig {
    pub fn from_config(cfg: &Config, epsilon: f64) -> Self {
        Self {
            g: cfg.constants.g,
            k_e: cfg.constants.k_e,
            mu_0: cfg.constants.mu_0,
            c: cfg.constants.c,
            epsilon,
            enable_gravity: cfg.enable_gravity,
            enable_em: cfg.enable_em,
            em_model: cfg.em_model,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FieldOutput {
    pub e: [Vec3; 3],
    pub b: [Vec3; 3],
    pub phi: [f64; 3],
    pub a: [Vec3; 3],
}

/// Compute total acceleration for all bodies.
pub fn compute_accel(system: &System, cfg: &ForceConfig) -> [Vec3; 3] {
    let mut acc = [Vec3::zero(); 3];
    if cfg.enable_gravity {
        acc = gravity::gravity_accel(&system.bodies, &system.state.pos, cfg.g, cfg.epsilon);
    }

    if cfg.enable_em {
        let em = em::em_accel(
            &system.bodies,
            &system.state.pos,
            &system.state.vel,
            cfg.k_e,
            cfg.mu_0,
            cfg.epsilon,
            cfg.em_model,
            cfg.c,
        );
        for i in 0..3 {
            acc[i] = acc[i] + em[i];
        }
    }

    acc
}

/// Compute fields and potentials.
pub fn compute_fields(system: &System, cfg: &ForceConfig) -> FieldOutput {
    let (e, b) = if cfg.enable_em {
        let fields = em::em_fields(
            &system.bodies,
            &system.state.pos,
            &system.state.vel,
            cfg.k_e,
            cfg.mu_0,
            cfg.epsilon,
            cfg.em_model,
            cfg.c,
        );
        (fields.e, fields.b)
    } else {
        ([Vec3::zero(); 3], [Vec3::zero(); 3])
    };
    let (phi, a) = if cfg.enable_em {
        let p = potentials::potentials(
            &system.bodies,
            &system.state.pos,
            &system.state.vel,
            cfg.k_e,
            cfg.mu_0,
            cfg.epsilon,
            cfg.em_model,
            cfg.c,
        );
        (p.phi, p.a)
    } else {
        ([0.0; 3], [Vec3::zero(); 3])
    };

    FieldOutput { e, b, phi, a }
}

#[cfg(test)]
mod tests {
    use super::{ForceConfig, compute_accel, compute_fields};
    use crate::config::EmModel;
    use crate::forces::{em::em_accel, gravity::gravity_accel};
    use crate::math::vec3::Vec3;
    use crate::state::{Body, State, System};

    fn base_system() -> System {
        let bodies = [
            Body::new(2.0, 1.0),
            Body::new(3.0, -1.0),
            Body::new(1.0, 2.0),
        ];
        let pos = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let vel = [
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.0, 0.2, 0.0),
            Vec3::zero(),
        ];
        System::new(bodies, State::new(pos, vel))
    }

    #[test]
    fn gravity_only_matches_module() {
        let system = base_system();
        let cfg = ForceConfig {
            g: 1.0,
            k_e: 0.0,
            mu_0: 0.0,
            c: 10.0,
            epsilon: 0.0,
            enable_gravity: true,
            enable_em: false,
            em_model: EmModel::Quasistatic,
        };
        let agg = compute_accel(&system, &cfg);
        let direct = gravity_accel(&system.bodies, &system.state.pos, cfg.g, cfg.epsilon);
        assert_eq!(agg, direct);
    }

    #[test]
    fn em_only_matches_module() {
        let system = base_system();
        let cfg = ForceConfig {
            g: 0.0,
            k_e: 1.0,
            mu_0: 1.0,
            c: 10.0,
            epsilon: 0.0,
            enable_gravity: false,
            enable_em: true,
            em_model: EmModel::Quasistatic,
        };
        let agg = compute_accel(&system, &cfg);
        let direct = em_accel(
            &system.bodies,
            &system.state.pos,
            &system.state.vel,
            cfg.k_e,
            cfg.mu_0,
            cfg.epsilon,
            cfg.em_model,
            cfg.c,
        );
        assert_eq!(agg, direct);
    }

    #[test]
    fn gravity_and_em_matches_sum_of_modules() {
        let system = base_system();
        let cfg = ForceConfig {
            g: 1.0,
            k_e: 2.0,
            mu_0: 0.7,
            c: 10.0,
            epsilon: 0.0,
            enable_gravity: true,
            enable_em: true,
            em_model: EmModel::Quasistatic,
        };
        let agg = compute_accel(&system, &cfg);
        let grav = gravity_accel(&system.bodies, &system.state.pos, cfg.g, cfg.epsilon);
        let em = em_accel(
            &system.bodies,
            &system.state.pos,
            &system.state.vel,
            cfg.k_e,
            cfg.mu_0,
            cfg.epsilon,
            cfg.em_model,
            cfg.c,
        );
        for i in 0..3 {
            let expected = grav[i] + em[i];
            assert!(
                agg[i].approx_eq(expected, 1e-12, 1e-12),
                "body {i}: agg={:?} expected={:?}",
                agg[i],
                expected
            );
        }
    }

    #[test]
    fn flags_toggle_deterministic_outputs() {
        let system = base_system();
        let cfg = ForceConfig {
            g: 1.0,
            k_e: 1.0,
            mu_0: 1.0,
            c: 10.0,
            epsilon: 0.0,
            enable_gravity: true,
            enable_em: true,
            em_model: EmModel::Quasistatic,
        };
        let a1 = compute_accel(&system, &cfg);
        let a2 = compute_accel(&system, &cfg);
        assert_eq!(a1, a2);

        let fields = compute_fields(&system, &cfg);
        let fields2 = compute_fields(&system, &cfg);
        assert_eq!(fields, fields2);
    }

    #[test]
    fn darwin_mode_is_deterministic() {
        let system = base_system();
        let cfg = ForceConfig {
            g: 0.0,
            k_e: 1.0,
            mu_0: 1.0,
            c: 5.0,
            epsilon: 0.0,
            enable_gravity: false,
            enable_em: true,
            em_model: EmModel::Darwin,
        };
        let a1 = compute_accel(&system, &cfg);
        let a2 = compute_accel(&system, &cfg);
        let f1 = compute_fields(&system, &cfg);
        let f2 = compute_fields(&system, &cfg);
        assert_eq!(a1, a2);
        assert_eq!(f1, f2);
    }
}
