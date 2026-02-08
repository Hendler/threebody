use crate::config::Config;
use crate::math::vec3::Vec3;
use crate::state::System;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Diagnostics {
    pub linear_momentum: Vec3,
    pub angular_momentum: Vec3,
    pub energy_proxy: f64,
}

pub fn compute_diagnostics(system: &System, cfg: &Config) -> Diagnostics {
    Diagnostics {
        linear_momentum: linear_momentum(system),
        angular_momentum: angular_momentum(system),
        energy_proxy: energy_proxy(system, cfg),
    }
}

pub fn linear_momentum(system: &System) -> Vec3 {
    let mut p = Vec3::zero();
    for i in 0..3 {
        p = p + system.state.vel[i] * system.bodies[i].mass;
    }
    p
}

pub fn angular_momentum(system: &System) -> Vec3 {
    let mut l = Vec3::zero();
    for i in 0..3 {
        let mv = system.state.vel[i] * system.bodies[i].mass;
        l = l + system.state.pos[i].cross(mv);
    }
    l
}

/// Mechanical energy proxy consistent with the enabled force model.
///
/// Notes:
/// - Always includes kinetic energy.
/// - Includes gravitational potential only if `cfg.enable_gravity`.
/// - Includes electrostatic potential only if `cfg.enable_em`.
/// - Does not include magnetic field energy (not modeled).
pub fn energy_proxy(system: &System, cfg: &Config) -> f64 {
    let mut kinetic = 0.0;
    for i in 0..3 {
        let v2 = system.state.vel[i].norm_sq();
        kinetic += 0.5 * system.bodies[i].mass * v2;
    }

    let mut grav_pot = 0.0;
    let mut elec_pot = 0.0;
    for i in 0..3 {
        for j in (i + 1)..3 {
            let r = system.state.pos[j] - system.state.pos[i];
            let r2 = r.norm_sq();
            let inv_r = softened_inv_r(r2, cfg.softening);
            if inv_r == 0.0 {
                continue;
            }
            if cfg.enable_gravity {
                grav_pot +=
                    -cfg.constants.g * system.bodies[i].mass * system.bodies[j].mass * inv_r;
            }
            if cfg.enable_em {
                elec_pot +=
                    cfg.constants.k_e * system.bodies[i].charge * system.bodies[j].charge * inv_r;
            }
        }
    }

    kinetic + grav_pot + elec_pot
}

fn softened_inv_r(r2: f64, epsilon: f64) -> f64 {
    if r2 == 0.0 {
        return 0.0;
    }
    let soft2 = if epsilon == 0.0 {
        r2
    } else {
        r2 + epsilon * epsilon
    };
    1.0 / soft2.sqrt()
}

#[cfg(test)]
mod tests {
    use super::{energy_proxy, linear_momentum};
    use crate::config::Config;
    use crate::math::vec3::Vec3;
    use crate::state::{Body, State, System};

    #[test]
    fn energy_proxy_matches_manual_static() {
        let bodies = [
            Body::new(2.0, 1.0),
            Body::new(3.0, -2.0),
            Body::new(0.0, 0.0),
        ];
        let pos = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::zero(),
        ];
        let vel = [Vec3::zero(); 3];
        let system = System::new(bodies, State::new(pos, vel));
        let mut cfg = Config::default();
        cfg.enable_gravity = true;
        cfg.enable_em = true;
        cfg.constants.g = 1.0;
        cfg.constants.k_e = 1.0;
        cfg.softening = 0.0;
        let e = energy_proxy(&system, &cfg);
        // Only pair (0,1) contributes with r=2.
        let grav = -cfg.constants.g * bodies[0].mass * bodies[1].mass / 2.0;
        let elec = cfg.constants.k_e * bodies[0].charge * bodies[1].charge / 2.0;
        let manual = grav + elec;
        assert!((e - manual).abs() < 1e-12);
    }

    #[test]
    fn linear_momentum_zero_for_balanced_velocities() {
        let bodies = [
            Body::new(1.0, 0.0),
            Body::new(1.0, 0.0),
            Body::new(0.0, 0.0),
        ];
        let pos = [Vec3::zero(); 3];
        let vel = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::zero(),
        ];
        let system = System::new(bodies, State::new(pos, vel));
        let p = linear_momentum(&system);
        assert!(p.approx_eq(Vec3::zero(), 1e-12, 1e-12));
    }

    #[test]
    fn energy_proxy_respects_force_toggles() {
        let bodies = [
            Body::new(2.0, 1.0),
            Body::new(3.0, -2.0),
            Body::new(0.0, 0.0),
        ];
        let pos = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::zero(),
        ];
        let vel = [Vec3::zero(); 3];
        let system = System::new(bodies, State::new(pos, vel));

        let mut cfg = Config::default();
        cfg.constants.g = 1.0;
        cfg.constants.k_e = 1.0;
        cfg.softening = 0.0;

        cfg.enable_gravity = false;
        cfg.enable_em = false;
        let e_none = energy_proxy(&system, &cfg);
        assert!((e_none - 0.0).abs() < 1e-12);

        cfg.enable_gravity = true;
        cfg.enable_em = false;
        let e_grav = energy_proxy(&system, &cfg);
        assert!(e_grav < 0.0);

        cfg.enable_gravity = false;
        cfg.enable_em = true;
        let e_elec = energy_proxy(&system, &cfg);
        assert!(e_elec.is_finite());
        assert!(e_elec != e_grav);
    }
}
