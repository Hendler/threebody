use crate::config::Config;
use crate::forces::{ForceConfig, compute_accel};
use crate::integrators::Integrator;
use crate::state::System;

pub struct Leapfrog;

impl Integrator for Leapfrog {
    fn step(&self, system: &System, dt: f64, cfg: &Config) -> System {
        let force_cfg = ForceConfig {
            g: cfg.constants.g,
            k_e: cfg.constants.k_e,
            mu_0: cfg.constants.mu_0,
            epsilon: cfg.softening,
            enable_gravity: cfg.enable_gravity,
            enable_em: cfg.enable_em,
        };
        let acc = compute_accel(system, &force_cfg);
        let mut next = system.clone();
        let mut v_half = system.state.vel.clone();
        let n = system.n();
        for i in 0..n {
            v_half[i] = v_half[i] + acc[i] * (0.5 * dt);
            next.state.pos[i] = next.state.pos[i] + v_half[i] * dt;
        }

        let temp = System::new(
            system.bodies.clone(),
            crate::state::State::new(next.state.pos.clone(), v_half.clone()),
        );
        let acc_new = compute_accel(&temp, &force_cfg);
        for i in 0..n {
            next.state.vel[i] = v_half[i] + acc_new[i] * (0.5 * dt);
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use super::Leapfrog;
    use crate::config::Config;
    use crate::diagnostics::energy_proxy;
    use crate::integrators::Integrator;
    use crate::math::vec3::Vec3;
    use crate::state::{Body, State, System};

    #[test]
    fn gravity_energy_proxy_is_bounded() {
        let bodies = [
            Body::new(1.0, 0.0),
            Body::new(1.0, 0.0),
            Body::new(0.0, 0.0),
        ];
        let pos = [
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::zero(),
        ];
        let v = (0.5_f64).sqrt();
        let vel = [
            Vec3::new(0.0, v, 0.0),
            Vec3::new(0.0, -v, 0.0),
            Vec3::zero(),
        ];
        let mut system = System::new(bodies, State::new(pos, vel));

        let mut cfg = Config::default();
        cfg.enable_em = false;
        cfg.constants.g = 1.0;
        cfg.softening = 0.0;

        let integrator = Leapfrog;
        let dt = 0.01;
        let e0 = energy_proxy(&system, &cfg);
        for _ in 0..1000 {
            system = integrator.step(&system, dt, &cfg);
        }
        let e1 = energy_proxy(&system, &cfg);
        assert!((e1 - e0).abs() < 1e-2);
    }
}
