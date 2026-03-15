use crate::config::Config;
use crate::forces::{ForceConfig, compute_fields, gravity::gravity_accel};
use crate::integrators::Integrator;
use crate::math::vec3::Vec3;
use crate::state::System;

pub struct Boris;

impl Integrator for Boris {
    fn step(&self, system: &System, dt: f64, cfg: &Config) -> System {
        let force_cfg = ForceConfig::from_config(cfg, cfg.softening);

        let mut next = system.clone();
        let n = system.n();
        // Gravity half-kick (if enabled).
        if cfg.enable_gravity {
            let g_acc = gravity_accel(
                &system.bodies,
                &system.state.pos,
                cfg.constants.g,
                cfg.softening,
            );
            for i in 0..n {
                next.state.vel[i] = next.state.vel[i] + g_acc[i] * (0.5 * dt);
            }
        }

        // EM Boris push for velocities.
        if cfg.enable_em {
            let fields = compute_fields(system, &force_cfg);
            for i in 0..n {
                let qi = system.bodies[i].charge;
                let mi = system.bodies[i].mass;
                if qi == 0.0 || mi == 0.0 {
                    continue;
                }
                let q_over_m = qi / mi;
                next.state.vel[i] =
                    boris_push(next.state.vel[i], q_over_m, fields.e[i], fields.b[i], dt);
            }
        }

        // Drift positions.
        for i in 0..n {
            next.state.pos[i] = next.state.pos[i] + next.state.vel[i] * dt;
        }

        // Gravity half-kick at new positions (if enabled).
        if cfg.enable_gravity {
            let g_acc = gravity_accel(
                &system.bodies,
                &next.state.pos,
                cfg.constants.g,
                cfg.softening,
            );
            for i in 0..n {
                next.state.vel[i] = next.state.vel[i] + g_acc[i] * (0.5 * dt);
            }
        }

        next
    }
}

fn boris_push(v: Vec3, q_over_m: f64, e: Vec3, b: Vec3, dt: f64) -> Vec3 {
    let half = 0.5 * dt * q_over_m;
    let v_minus = v + e * half;
    let t = b * half;
    let t_mag2 = t.norm_sq();
    let s = t * (2.0 / (1.0 + t_mag2));
    let v_prime = v_minus + v_minus.cross(t);
    let v_plus = v_minus + v_prime.cross(s);
    v_plus + e * half
}

#[cfg(test)]
mod tests {
    use super::boris_push;
    use crate::math::vec3::Vec3;

    #[test]
    fn boris_preserves_speed_in_uniform_b() {
        let v0 = Vec3::new(1.0, 0.0, 0.0);
        let e = Vec3::zero();
        let b = Vec3::new(0.0, 0.0, 1.0);
        let q_over_m = 1.0;
        let dt = 0.1;
        let mut v = v0;
        for _ in 0..100 {
            v = boris_push(v, q_over_m, e, b, dt);
        }
        let speed0 = v0.norm();
        let speed = v.norm();
        assert!((speed - speed0).abs() < 1e-6);
    }

    #[test]
    fn boris_handles_zero_q_over_m() {
        let v0 = Vec3::new(1.0, 2.0, 3.0);
        let v = boris_push(v0, 0.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zero(), 0.1);
        assert!(v.approx_eq(v0, 1e-12, 1e-12));
    }
}
