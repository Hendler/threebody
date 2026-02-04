use crate::math::vec3::Vec3;
use crate::state::System;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RegimeDiagnostics {
    pub min_pair_dist: f64,
    pub max_speed: f64,
    pub max_accel: f64,
    pub dt_ratio: f64,
}

/// Compute basic regime diagnostics.
/// dt_ratio is defined as dt * max_speed / min_pair_dist (CFL-like); 0 if min_pair_dist is 0.
pub fn compute_regime(system: &System, acc: &[Vec3; 3], dt: f64) -> RegimeDiagnostics {
    let mut min_pair_dist = f64::INFINITY;
    for i in 0..3 {
        for j in (i + 1)..3 {
            let r = system.state.pos[j] - system.state.pos[i];
            let d = r.norm();
            if d < min_pair_dist {
                min_pair_dist = d;
            }
        }
    }

    let mut max_speed: f64 = 0.0;
    let mut max_accel: f64 = 0.0;
    for i in 0..3 {
        max_speed = max_speed.max(system.state.vel[i].norm());
        max_accel = max_accel.max(acc[i].norm());
    }

    let dt_ratio = if min_pair_dist == 0.0 || !min_pair_dist.is_finite() {
        0.0
    } else {
        dt * max_speed / min_pair_dist
    };

    RegimeDiagnostics {
        min_pair_dist,
        max_speed,
        max_accel,
        dt_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::compute_regime;
    use crate::math::vec3::Vec3;
    use crate::state::{Body, State, System};

    #[test]
    fn static_configuration_is_deterministic() {
        let bodies = [Body::new(1.0, 0.0), Body::new(1.0, 0.0), Body::new(1.0, 0.0)];
        let pos = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.0, 3.0, 0.0)];
        let vel = [Vec3::zero(); 3];
        let acc = [Vec3::zero(); 3];
        let system = System::new(bodies, State::new(pos, vel));
        let diag = compute_regime(&system, &acc, 0.1);
        assert!(diag.min_pair_dist.is_finite());
        assert_eq!(diag.max_speed, 0.0);
        assert_eq!(diag.max_accel, 0.0);
        assert_eq!(diag.dt_ratio, 0.0);
    }
}
