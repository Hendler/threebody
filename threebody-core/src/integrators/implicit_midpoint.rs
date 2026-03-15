use crate::config::Config;
use crate::forces::{ForceConfig, compute_accel};
use crate::integrators::Integrator;
use crate::math::vec3::Vec3;
use crate::state::{State, System};

pub struct ImplicitMidpoint;

impl Integrator for ImplicitMidpoint {
    fn step(&self, system: &System, dt: f64, cfg: &Config) -> System {
        let force_cfg = ForceConfig::from_config(cfg, cfg.softening);

        // Explicit Euler as initial guess.
        let acc0 = compute_accel(system, &force_cfg);
        let mut guess_pos = system.state.pos.clone();
        let mut guess_vel = system.state.vel.clone();
        let n = system.n();
        for i in 0..n {
            guess_pos[i] = guess_pos[i] + system.state.vel[i] * dt;
            guess_vel[i] = guess_vel[i] + acc0[i] * dt;
        }

        let mut next = System::new(system.bodies.clone(), State::new(guess_pos, guess_vel));
        let tol = cfg.integrator.implicit_tol;
        let max_iters = cfg.integrator.implicit_max_iters;

        for _ in 0..max_iters {
            let mut mid_pos = vec![Vec3::zero(); n];
            let mut mid_vel = vec![Vec3::zero(); n];
            for i in 0..n {
                mid_pos[i] = (system.state.pos[i] + next.state.pos[i]) * 0.5;
                mid_vel[i] = (system.state.vel[i] + next.state.vel[i]) * 0.5;
            }
            let mid_system =
                System::new(system.bodies.clone(), State::new(mid_pos, mid_vel.clone()));
            let acc_mid = compute_accel(&mid_system, &force_cfg);

            let mut new_pos = vec![Vec3::zero(); n];
            let mut new_vel = vec![Vec3::zero(); n];
            for i in 0..n {
                new_pos[i] = system.state.pos[i] + mid_vel[i] * dt;
                new_vel[i] = system.state.vel[i] + acc_mid[i] * dt;
            }

            let mut max_diff: f64 = 0.0;
            for i in 0..n {
                let dp = (new_pos[i] - next.state.pos[i]).norm();
                let dv = (new_vel[i] - next.state.vel[i]).norm();
                max_diff = max_diff.max(dp).max(dv);
            }
            next = System::new(system.bodies.clone(), State::new(new_pos, new_vel));
            if max_diff <= tol {
                break;
            }
        }

        next
    }
}

#[cfg(test)]
mod tests {
    use super::ImplicitMidpoint;
    use crate::config::Config;
    use crate::integrators::Integrator;
    use crate::math::vec3::Vec3;
    use crate::state::{Body, State, System};

    #[test]
    fn zero_forces_keeps_velocity_constant() {
        let bodies = [
            Body::new(0.0, 0.0),
            Body::new(0.0, 0.0),
            Body::new(0.0, 0.0),
        ];
        let pos = [Vec3::new(1.0, 2.0, 3.0), Vec3::zero(), Vec3::zero()];
        let vel = [Vec3::new(0.1, -0.2, 0.3), Vec3::zero(), Vec3::zero()];
        let system = System::new(bodies, State::new(pos, vel));
        let mut cfg = Config::default();
        cfg.enable_gravity = false;
        cfg.enable_em = false;
        let integrator = ImplicitMidpoint;
        let dt = 0.5;
        let next = integrator.step(&system, dt, &cfg);
        assert!(next.state.vel[0].approx_eq(vel[0], 1e-12, 1e-12));
        let expected_pos = pos[0] + vel[0] * dt;
        assert!(next.state.pos[0].approx_eq(expected_pos, 1e-12, 1e-12));
    }

    #[test]
    fn implicit_midpoint_close_to_leapfrog_small_dt() {
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
        let system = System::new(bodies, State::new(pos, vel));
        let mut cfg = Config::default();
        cfg.enable_em = false;
        let dt = 0.001;
        let imp = ImplicitMidpoint;
        let lf = crate::integrators::leapfrog::Leapfrog;
        let a = imp.step(&system, dt, &cfg);
        let b = lf.step(&system, dt, &cfg);
        let diff = (a.state.pos[0] - b.state.pos[0]).norm();
        assert!(diff < 1e-3);
    }
}
