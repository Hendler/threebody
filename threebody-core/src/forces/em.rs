use crate::config::EmModel;
use crate::math::vec3::Vec3;
use crate::state::Body;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EmFields {
    pub e: [Vec3; 3],
    pub b: [Vec3; 3],
}

/// Compute electric and magnetic fields at each body.
/// E_i = sum_{j != i} k_e * q_j * (r_i - r_j) / |r_i - r_j|^3
/// B_i = sum_{j != i} (mu_0 / 4pi) * q_j * (v_j x (r_i - r_j)) / |r_i - r_j|^3
pub fn em_fields(
    bodies: &[Body],
    pos: &[Vec3],
    vel: &[Vec3],
    k_e: f64,
    mu_0: f64,
    epsilon: f64,
    model: EmModel,
    c: f64,
) -> EmFields {
    debug_assert_eq!(bodies.len(), 3);
    debug_assert_eq!(pos.len(), 3);
    debug_assert_eq!(vel.len(), 3);
    let mut e = [Vec3::zero(); 3];
    let mut b = [Vec3::zero(); 3];
    let mu0_over_4pi = mu_0 / (4.0 * std::f64::consts::PI);

    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                continue;
            }
            let r_ij = pos[i] - pos[j];
            let inv_r3 = moving_source_inv_r3(r_ij, vel[j], epsilon, model, c);
            if inv_r3 == 0.0 {
                continue;
            }
            let qj = bodies[j].charge;
            e[i] = e[i] + r_ij * (k_e * qj * inv_r3);
            let vj_cross_r = vel[j].cross(r_ij);
            b[i] = b[i] + vj_cross_r * (mu0_over_4pi * qj * inv_r3);
        }
    }

    EmFields { e, b }
}

/// Compute Lorentz acceleration for all bodies.
/// a_i(em) = (q_i / m_i) * (E_i + v_i x B_i)
pub fn em_accel(
    bodies: &[Body],
    pos: &[Vec3],
    vel: &[Vec3],
    k_e: f64,
    mu_0: f64,
    epsilon: f64,
    model: EmModel,
    c: f64,
) -> [Vec3; 3] {
    let fields = em_fields(bodies, pos, vel, k_e, mu_0, epsilon, model, c);
    let mut acc = [Vec3::zero(); 3];
    for i in 0..3 {
        let qi = bodies[i].charge;
        let mi = bodies[i].mass;
        if qi == 0.0 || mi == 0.0 {
            acc[i] = Vec3::zero();
            continue;
        }
        acc[i] = (fields.e[i] + vel[i].cross(fields.b[i])) * (qi / mi);
    }
    acc
}

pub(crate) fn moving_source_inv_r(r_ij: Vec3, v_source: Vec3, epsilon: f64, model: EmModel, c: f64) -> f64 {
    let (r, _, beta_perp_sq) = moving_source_geometry(r_ij, v_source, epsilon, model, c);
    if r == 0.0 {
        return 0.0;
    }
    let denom = (1.0 - beta_perp_sq).max(1e-12).sqrt();
    1.0 / (r * denom)
}

pub(crate) fn moving_source_inv_r3(
    r_ij: Vec3,
    v_source: Vec3,
    epsilon: f64,
    model: EmModel,
    c: f64,
) -> f64 {
    let (r, beta_sq, beta_perp_sq) = moving_source_geometry(r_ij, v_source, epsilon, model, c);
    if r == 0.0 {
        return 0.0;
    }
    match model {
        EmModel::Quasistatic => 1.0 / (r * r * r),
        EmModel::Darwin => {
            let numer = (1.0 - beta_sq).max(1e-12);
            let denom = (1.0 - beta_perp_sq).max(1e-12).powf(1.5);
            numer / (r * r * r * denom)
        }
    }
}

fn moving_source_geometry(
    r_ij: Vec3,
    v_source: Vec3,
    epsilon: f64,
    model: EmModel,
    c: f64,
) -> (f64, f64, f64) {
    let r2 = r_ij.norm_sq();
    if r2 == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let soft2 = if epsilon == 0.0 {
        r2
    } else {
        r2 + epsilon * epsilon
    };
    let r = soft2.sqrt();
    if !r.is_finite() || r == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    if matches!(model, EmModel::Quasistatic) {
        return (r, 0.0, 0.0);
    }
    if !c.is_finite() || c <= 0.0 {
        return (r, 0.0, 0.0);
    }
    let mut beta = v_source / c;
    let beta_sq_raw = beta.norm_sq();
    if beta_sq_raw >= 0.999_999 {
        beta = beta * (0.999_999 / beta_sq_raw).sqrt();
    }
    let beta_sq = beta.norm_sq();
    let n = r_ij / r;
    let beta_n = beta.dot(n);
    let beta_perp_sq = (beta_sq - beta_n * beta_n).clamp(0.0, 0.999_999);
    (r, beta_sq, beta_perp_sq)
}

#[cfg(test)]
mod tests {
    use crate::config::EmModel;
    use super::{em_accel, em_fields};
    use crate::math::vec3::Vec3;
    use crate::state::Body;

    #[test]
    fn zero_charges_give_zero_accel() {
        let bodies = [
            Body::new(1.0, 0.0),
            Body::new(2.0, 0.0),
            Body::new(3.0, 0.0),
        ];
        let pos = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::zero(),
        ];
        let vel = [Vec3::zero(); 3];
        let acc = em_accel(&bodies, &pos, &vel, 1.0, 1.0, 0.0, EmModel::Quasistatic, 10.0);
        assert_eq!(acc, [Vec3::zero(); 3]);
    }

    #[test]
    fn zero_velocities_remove_magnetic_terms() {
        let bodies = [
            Body::new(1.0, 1.0),
            Body::new(1.0, -1.0),
            Body::new(0.0, 0.0),
        ];
        let pos = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::zero(),
        ];
        let vel = [Vec3::zero(); 3];
        let fields = em_fields(
            &bodies,
            &pos,
            &vel,
            1.0,
            1.0,
            0.0,
            EmModel::Quasistatic,
            10.0,
        );
        assert_eq!(fields.b, [Vec3::zero(); 3]);

        let acc = em_accel(
            &bodies,
            &pos,
            &vel,
            1.0,
            1.0,
            0.0,
            EmModel::Quasistatic,
            10.0,
        );
        let m0_a0 = acc[0] * bodies[0].mass;
        let m1_a1 = acc[1] * bodies[1].mass;
        assert!(m0_a0.approx_eq(-m1_a1, 1e-12, 1e-12));
    }

    #[test]
    fn zero_qi_gives_zero_accel_for_that_body() {
        let bodies = [
            Body::new(1.0, 0.0),
            Body::new(1.0, 1.0),
            Body::new(1.0, -1.0),
        ];
        let pos = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::zero(),
        ];
        let vel = [Vec3::new(0.1, 0.0, 0.0); 3];
        let acc = em_accel(
            &bodies,
            &pos,
            &vel,
            1.0,
            1.0,
            0.0,
            EmModel::Quasistatic,
            10.0,
        );
        assert!(acc[0].approx_eq(Vec3::zero(), 1e-12, 1e-12));
    }

    #[test]
    fn darwin_reduces_to_quasistatic_for_static_sources() {
        let bodies = [
            Body::new(1.0, 1.0),
            Body::new(1.0, 0.0),
            Body::new(1.0, 0.0),
        ];
        let pos = [
            Vec3::zero(),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let vel = [Vec3::zero(); 3];
        let quasi = em_fields(
            &bodies,
            &pos,
            &vel,
            1.0,
            1.0,
            0.0,
            EmModel::Quasistatic,
            10.0,
        );
        let darwin = em_fields(
            &bodies,
            &pos,
            &vel,
            1.0,
            1.0,
            0.0,
            EmModel::Darwin,
            10.0,
        );
        assert!(quasi.e[1].approx_eq(darwin.e[1], 1e-12, 1e-12));
        assert!(quasi.e[2].approx_eq(darwin.e[2], 1e-12, 1e-12));
        assert!(quasi.b[1].approx_eq(darwin.b[1], 1e-12, 1e-12));
        assert!(quasi.b[2].approx_eq(darwin.b[2], 1e-12, 1e-12));
    }

    #[test]
    fn darwin_compresses_parallel_and_enhances_transverse_field() {
        let bodies = [
            Body::new(1.0, 1.0),
            Body::new(1.0, 0.0),
            Body::new(1.0, 0.0),
        ];
        let pos = [
            Vec3::zero(),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let vel = [
            Vec3::new(0.8, 0.0, 0.0),
            Vec3::zero(),
            Vec3::zero(),
        ];
        let quasi = em_fields(
            &bodies,
            &pos,
            &vel,
            1.0,
            1.0,
            0.0,
            EmModel::Quasistatic,
            1.0,
        );
        let darwin = em_fields(
            &bodies,
            &pos,
            &vel,
            1.0,
            1.0,
            0.0,
            EmModel::Darwin,
            1.0,
        );
        assert!(darwin.e[1].norm() < quasi.e[1].norm());
        assert!(darwin.e[2].norm() > quasi.e[2].norm());
    }
}
