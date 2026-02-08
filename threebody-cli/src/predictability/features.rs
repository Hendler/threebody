use serde::{Deserialize, Serialize};
use threebody_core::analysis;
use threebody_core::config::Config;
use threebody_core::state::System;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub(crate) struct PairId {
    pub i: usize,
    pub j: usize,
}

impl PairId {
    pub(crate) fn new(i: usize, j: usize) -> Self {
        if i <= j {
            Self { i, j }
        } else {
            Self { i: j, j: i }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct PairMetrics {
    pub pair: PairId,
    pub r: f64,
    pub speed: f64,
    pub cos_approach: f64,
    pub specific_energy: f64,
    pub h: f64,
    pub e: Option<f64>,
    pub a: Option<f64>,
}

pub(crate) fn min_pair(system: &System) -> (f64, PairId) {
    let (d, (i, j)) = analysis::min_pair_distance(&system.state.pos);
    (d, PairId::new(i, j))
}

pub(crate) fn pair_metrics(cfg: &Config, system: &System, i: usize, j: usize) -> PairMetrics {
    let (r_vec, v_vec) = analysis::relative_state(system, i, j);
    let r = r_vec.norm();
    let speed = v_vec.norm();
    let cos_approach = if r > 0.0 && speed > 0.0 {
        (r_vec.dot(v_vec) / (r * speed)).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let coeffs = analysis::pair_central_coefficients(cfg, &system.bodies[i], &system.bodies[j]);
    let specific_energy =
        analysis::specific_energy_1overr(r_vec, v_vec, coeffs.kappa, cfg.softening);
    let h = r_vec.cross(v_vec).norm();
    let els = analysis::osculating_elements_1overr(r_vec, v_vec, coeffs.kappa, cfg.softening);
    let (e, a) = match els {
        Some(els) => (Some(els.e), els.a),
        None => (None, None),
    };
    PairMetrics {
        pair: PairId::new(i, j),
        r,
        speed,
        cos_approach,
        specific_energy,
        h,
        e,
        a,
    }
}

pub(crate) fn energy_pair(cfg: &Config, system: &System) -> (f64, PairId) {
    let mut best_pair = PairId::new(0, 1);
    let mut best_energy = f64::INFINITY;
    for (i, j) in analysis::PAIRS_3 {
        let m = pair_metrics(cfg, system, i, j);
        if m.specific_energy < best_energy {
            best_energy = m.specific_energy;
            best_pair = m.pair;
        }
    }
    (best_energy, best_pair)
}

pub(crate) fn all_pair_metrics(cfg: &Config, system: &System) -> Vec<PairMetrics> {
    analysis::PAIRS_3
        .iter()
        .map(|(i, j)| pair_metrics(cfg, system, *i, *j))
        .collect()
}
