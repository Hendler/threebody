//! Force models and aggregate computation.

pub mod gravity;
pub mod em;
pub mod potentials;

use crate::math::vec3::Vec3;
use crate::state::System;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ForceConfig {
    pub g: f64,
    pub k_e: f64,
    pub mu_0: f64,
    pub epsilon: f64,
    pub enable_gravity: bool,
    pub enable_em: bool,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FieldOutput {
    pub e: [Vec3; 3],
    pub b: [Vec3; 3],
    pub phi: [f64; 3],
    pub a: [Vec3; 3],
}

/// Compute total acceleration for all bodies using shared pairwise caches.
pub fn compute_accel(system: &System, cfg: &ForceConfig) -> [Vec3; 3] {
    let cache = PairCache::new(system, cfg.epsilon);
    let mut acc = [Vec3::zero(); 3];
    if cfg.enable_gravity {
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    continue;
                }
                let r_ij = cache.r_ij[i][j];
                let inv_r3 = cache.inv_r3[i][j];
                acc[i] = acc[i] + r_ij * (cfg.g * system.bodies[j].mass * inv_r3);
            }
        }
    }

    if cfg.enable_em {
        let mu0_over_4pi = cfg.mu_0 / (4.0 * std::f64::consts::PI);
        let mut e = [Vec3::zero(); 3];
        let mut b = [Vec3::zero(); 3];
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    continue;
                }
                let r_ij = cache.r_ij[i][j];
                let inv_r3 = cache.inv_r3[i][j];
                let qj = system.bodies[j].charge;
                // E_i uses (r_i - r_j) = -r_ij.
                e[i] = e[i] + (-r_ij) * (cfg.k_e * qj * inv_r3);
                // B_i uses v_j x (r_i - r_j) = v_j x (-r_ij).
                let vj_cross_r = system.state.vel[j].cross(-r_ij);
                b[i] = b[i] + vj_cross_r * (mu0_over_4pi * qj * inv_r3);
            }
        }
        for i in 0..3 {
            let qi = system.bodies[i].charge;
            let mi = system.bodies[i].mass;
            if qi == 0.0 || mi == 0.0 {
                continue;
            }
            acc[i] = acc[i] + (e[i] + system.state.vel[i].cross(b[i])) * (qi / mi);
        }
    }

    acc
}

/// Compute fields and potentials using shared pairwise caches.
pub fn compute_fields(system: &System, cfg: &ForceConfig) -> FieldOutput {
    let cache = PairCache::new(system, cfg.epsilon);
    let mut e = [Vec3::zero(); 3];
    let mut b = [Vec3::zero(); 3];
    let mut phi = [0.0; 3];
    let mut a = [Vec3::zero(); 3];
    let mu0_over_4pi = cfg.mu_0 / (4.0 * std::f64::consts::PI);

    if cfg.enable_em {
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    continue;
                }
                let r_ij = cache.r_ij[i][j];
                let inv_r = cache.inv_r[i][j];
                let inv_r3 = cache.inv_r3[i][j];
                let qj = system.bodies[j].charge;
                e[i] = e[i] + (-r_ij) * (cfg.k_e * qj * inv_r3);
                let vj_cross_r = system.state.vel[j].cross(-r_ij);
                b[i] = b[i] + vj_cross_r * (mu0_over_4pi * qj * inv_r3);
                phi[i] += cfg.k_e * qj * inv_r;
                a[i] = a[i] + system.state.vel[j] * (mu0_over_4pi * qj * inv_r);
            }
        }
    }

    FieldOutput { e, b, phi, a }
}

#[derive(Copy, Clone, Debug)]
struct PairCache {
    r_ij: [[Vec3; 3]; 3],
    inv_r: [[f64; 3]; 3],
    inv_r3: [[f64; 3]; 3],
}

impl PairCache {
    fn new(system: &System, epsilon: f64) -> Self {
        let mut r_ij = [[Vec3::zero(); 3]; 3];
        let mut inv_r = [[0.0; 3]; 3];
        let mut inv_r3 = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    continue;
                }
                let r = system.state.pos[j] - system.state.pos[i];
                let r2 = r.norm_sq();
                let (ir, ir3) = softened_inv_r_and_r3(r2, epsilon);
                r_ij[i][j] = r;
                inv_r[i][j] = ir;
                inv_r3[i][j] = ir3;
            }
        }
        Self { r_ij, inv_r, inv_r3 }
    }
}

fn softened_inv_r_and_r3(r2: f64, epsilon: f64) -> (f64, f64) {
    if r2 == 0.0 {
        return (0.0, 0.0);
    }
    let soft2 = if epsilon == 0.0 { r2 } else { r2 + epsilon * epsilon };
    let r = soft2.sqrt();
    let inv_r = 1.0 / r;
    let inv_r3 = inv_r * inv_r * inv_r;
    (inv_r, inv_r3)
}

#[cfg(test)]
mod tests {
    use super::{compute_accel, compute_fields, ForceConfig};
    use crate::forces::{em::em_accel, gravity::gravity_accel};
    use crate::math::vec3::Vec3;
    use crate::state::{Body, State, System};

    fn base_system() -> System {
        let bodies = [Body::new(2.0, 1.0), Body::new(3.0, -1.0), Body::new(1.0, 2.0)];
        let pos = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let vel = [Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.0, 0.2, 0.0), Vec3::zero()];
        System::new(bodies, State::new(pos, vel))
    }

    #[test]
    fn gravity_only_matches_module() {
        let system = base_system();
        let cfg = ForceConfig {
            g: 1.0,
            k_e: 0.0,
            mu_0: 0.0,
            epsilon: 0.0,
            enable_gravity: true,
            enable_em: false,
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
            epsilon: 0.0,
            enable_gravity: false,
            enable_em: true,
        };
        let agg = compute_accel(&system, &cfg);
        let direct = em_accel(
            &system.bodies,
            &system.state.pos,
            &system.state.vel,
            cfg.k_e,
            cfg.mu_0,
            cfg.epsilon,
        );
        assert_eq!(agg, direct);
    }

    #[test]
    fn flags_toggle_deterministic_outputs() {
        let system = base_system();
        let cfg = ForceConfig {
            g: 1.0,
            k_e: 1.0,
            mu_0: 1.0,
            epsilon: 0.0,
            enable_gravity: true,
            enable_em: true,
        };
        let a1 = compute_accel(&system, &cfg);
        let a2 = compute_accel(&system, &cfg);
        assert_eq!(a1, a2);

        let fields = compute_fields(&system, &cfg);
        let fields2 = compute_fields(&system, &cfg);
        assert_eq!(fields, fields2);
    }
}
