//! Analysis utilities for predictability and encounter modeling.
//!
//! This module intentionally focuses on *reduced* (pairwise / hierarchical) quantities that
//! are useful for reasoning about three-body chaos:
//! - pairwise relative states `(r_ij, v_ij)`
//! - effective 1/r central coefficients for gravity + electrostatics
//! - osculating (approximate) orbital elements under the 1/r component
//!
//! Note: The simulator also includes quasi-static magnetic forces which are not central 1/r.
//! The "osculating elements" here are therefore an approximation that is still useful as a
//! stable summary statistic, especially in regimes where magnetic terms are small or when
//! we only need a coarse hierarchy classifier.

use crate::config::Config;
use crate::math::vec3::Vec3;
use crate::state::{Body, System};

/// The three unique body pairs for N=3, listed with i<j.
pub const PAIRS_3: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairCentralCoefficients {
    /// `K = G*m_i*m_j - k_e*q_i*q_j` in potential units (so that U = -K/r).
    pub k: f64,
    /// Reduced mass `μ = m_i*m_j/(m_i+m_j)`.
    pub reduced_mass: f64,
    /// `κ = K/μ`, the coefficient in the relative equation `r¨ = -κ r / |r|^3` for the 1/r component.
    pub kappa: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OsculatingElements {
    pub kappa: f64,
    pub specific_energy: f64,
    pub h: Vec3,
    pub e_vec: Vec3,
    pub e: f64,
    pub a: Option<f64>,
}

pub fn reduced_mass(mi: f64, mj: f64) -> f64 {
    if mi <= 0.0 || mj <= 0.0 {
        return 0.0;
    }
    let denom = mi + mj;
    if denom == 0.0 {
        return 0.0;
    }
    (mi * mj) / denom
}

pub fn pair_central_coefficients(cfg: &Config, bi: &Body, bj: &Body) -> PairCentralCoefficients {
    let mi = bi.mass;
    let mj = bj.mass;
    let qi = bi.charge;
    let qj = bj.charge;
    let k = cfg.constants.g * mi * mj - cfg.constants.k_e * qi * qj;
    let mu = reduced_mass(mi, mj);
    let kappa = if mu == 0.0 { 0.0 } else { k / mu };
    PairCentralCoefficients {
        k,
        reduced_mass: mu,
        kappa,
    }
}

/// Relative position and velocity for pair (i,j), with r_ij = r_j - r_i and v_ij = v_j - v_i.
pub fn relative_state(system: &System, i: usize, j: usize) -> (Vec3, Vec3) {
    let r = system.state.pos[j] - system.state.pos[i];
    let v = system.state.vel[j] - system.state.vel[i];
    (r, v)
}

pub fn softened_distance(r: Vec3, softening: f64) -> f64 {
    let r2 = r.norm_sq();
    if r2 == 0.0 {
        return 0.0;
    }
    let soft2 = if softening == 0.0 {
        r2
    } else {
        r2 + softening * softening
    };
    soft2.sqrt()
}

/// Specific orbital energy (per unit reduced mass) for the 1/r component:
/// `ε = 0.5 |v|^2 - κ / r_soft`.
pub fn specific_energy_1overr(r: Vec3, v: Vec3, kappa: f64, softening: f64) -> f64 {
    let r_soft = softened_distance(r, softening);
    if r_soft == 0.0 || !r_soft.is_finite() {
        return 0.5 * v.norm_sq();
    }
    0.5 * v.norm_sq() - kappa / r_soft
}

/// Compute osculating elements for the central 1/r component (gravity + electrostatics).
///
/// Returns `None` if the inputs are degenerate (e.g., r=0 or kappa=0).
pub fn osculating_elements_1overr(
    r: Vec3,
    v: Vec3,
    kappa: f64,
    softening: f64,
) -> Option<OsculatingElements> {
    let r_soft = softened_distance(r, softening);
    if r_soft == 0.0 || !r_soft.is_finite() || kappa == 0.0 || !kappa.is_finite() {
        return None;
    }

    let specific_energy = specific_energy_1overr(r, v, kappa, softening);
    let h = r.cross(v);
    let r_hat = r / r_soft;
    let e_vec = v.cross(h) / kappa - r_hat;
    let e = e_vec.norm();

    let a = if specific_energy.is_finite() && specific_energy != 0.0 {
        let val = -kappa / (2.0 * specific_energy);
        (val.is_finite() && val > 0.0).then_some(val)
    } else {
        None
    };

    Some(OsculatingElements {
        kappa,
        specific_energy,
        h,
        e_vec,
        e,
        a,
    })
}

pub fn pair_osculating_elements(
    cfg: &Config,
    system: &System,
    i: usize,
    j: usize,
) -> Option<OsculatingElements> {
    let (r, v) = relative_state(system, i, j);
    let coeffs = pair_central_coefficients(cfg, &system.bodies[i], &system.bodies[j]);
    osculating_elements_1overr(r, v, coeffs.kappa, cfg.softening)
}

pub fn min_pair_distance(pos: &[Vec3]) -> (f64, (usize, usize)) {
    let mut best = (f64::INFINITY, (0, 1));
    for (i, j) in PAIRS_3 {
        let d = (pos[j] - pos[i]).norm();
        if d < best.0 {
            best = (d, (i, j));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::{
        PAIRS_3, min_pair_distance, osculating_elements_1overr, pair_central_coefficients,
        reduced_mass,
    };
    use crate::config::Config;
    use crate::math::vec3::Vec3;
    use crate::state::{Body, State, System};

    #[test]
    fn reduced_mass_matches_known_identity() {
        // μ = m1*m2/(m1+m2)
        let mu = reduced_mass(2.0, 3.0);
        assert!((mu - (6.0 / 5.0)).abs() < 1e-12);
        assert_eq!(reduced_mass(0.0, 1.0), 0.0);
    }

    #[test]
    fn pair_coefficients_reduce_to_g_times_sum_masses_for_gravity() {
        let mut cfg = Config::default();
        cfg.constants.g = 2.0;
        cfg.constants.k_e = 0.0;
        let bi = Body::new(1.0, 0.0);
        let bj = Body::new(3.0, 0.0);
        let c = pair_central_coefficients(&cfg, &bi, &bj);
        // kappa = G*(m1+m2) for gravity-only.
        assert!((c.kappa - cfg.constants.g * (bi.mass + bj.mass)).abs() < 1e-12);
    }

    #[test]
    fn osculating_elements_match_circular_two_body_case() {
        // For gravity-only, r=1, kappa=2 => circular v = sqrt(kappa/r)=sqrt(2), e≈0, a≈1.
        let r = Vec3::new(1.0, 0.0, 0.0);
        let v = Vec3::new(0.0, 2.0_f64.sqrt(), 0.0);
        let els = osculating_elements_1overr(r, v, 2.0, 0.0).expect("elements");
        assert!(els.e.is_finite());
        assert!(els.e < 1e-10, "e too large: {}", els.e);
        let a = els.a.expect("a");
        assert!((a - 1.0).abs() < 1e-10, "a wrong: {}", a);
        assert!(els.specific_energy < 0.0);
    }

    #[test]
    fn min_pair_distance_returns_correct_pair() {
        let pos = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
        ];
        let (d, (i, j)) = min_pair_distance(&pos);
        assert!((d - 0.5).abs() < 1e-12);
        assert_eq!((i, j), (0, 2));

        // Make sure PAIRS_3 lists all unique pairs.
        assert_eq!(PAIRS_3.len(), 3);
    }

    #[test]
    fn pair_coeffs_work_inside_system() {
        let cfg = Config::default();
        let bodies = [
            Body::new(2.0, 1.0),
            Body::new(3.0, -1.0),
            Body::new(1.0, 0.0),
        ];
        let pos = [
            Vec3::zero(),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let vel = [Vec3::zero(); 3];
        let system = System::new(bodies, State::new(pos, vel));

        let c01 = pair_central_coefficients(&cfg, &system.bodies[0], &system.bodies[1]);
        assert!(c01.k.is_finite());
        assert!(c01.kappa.is_finite());
    }
}
