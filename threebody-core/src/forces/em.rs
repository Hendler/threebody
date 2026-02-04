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
    bodies: &[Body; 3],
    pos: &[Vec3; 3],
    vel: &[Vec3; 3],
    k_e: f64,
    mu_0: f64,
    epsilon: f64,
) -> EmFields {
    let mut e = [Vec3::zero(); 3];
    let mut b = [Vec3::zero(); 3];
    let mu0_over_4pi = mu_0 / (4.0 * std::f64::consts::PI);

    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                continue;
            }
            let r_ij = pos[i] - pos[j];
            let r2 = r_ij.norm_sq();
            let inv_r3 = softened_inv_r3(r2, epsilon);
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
    bodies: &[Body; 3],
    pos: &[Vec3; 3],
    vel: &[Vec3; 3],
    k_e: f64,
    mu_0: f64,
    epsilon: f64,
) -> [Vec3; 3] {
    let fields = em_fields(bodies, pos, vel, k_e, mu_0, epsilon);
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

fn softened_inv_r3(r2: f64, epsilon: f64) -> f64 {
    if r2 == 0.0 {
        return 0.0;
    }
    let soft2 = if epsilon == 0.0 { r2 } else { r2 + epsilon * epsilon };
    let r = soft2.sqrt();
    1.0 / (r * r * r)
}

#[cfg(test)]
mod tests {
    use super::{em_accel, em_fields};
    use crate::math::vec3::Vec3;
    use crate::state::Body;

    #[test]
    fn zero_charges_give_zero_accel() {
        let bodies = [Body::new(1.0, 0.0), Body::new(2.0, 0.0), Body::new(3.0, 0.0)];
        let pos = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::zero()];
        let vel = [Vec3::zero(); 3];
        let acc = em_accel(&bodies, &pos, &vel, 1.0, 1.0, 0.0);
        assert_eq!(acc, [Vec3::zero(); 3]);
    }

    #[test]
    fn zero_velocities_remove_magnetic_terms() {
        let bodies = [Body::new(1.0, 1.0), Body::new(1.0, -1.0), Body::new(0.0, 0.0)];
        let pos = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::zero()];
        let vel = [Vec3::zero(); 3];
        let fields = em_fields(&bodies, &pos, &vel, 1.0, 1.0, 0.0);
        assert_eq!(fields.b, [Vec3::zero(); 3]);

        let acc = em_accel(&bodies, &pos, &vel, 1.0, 1.0, 0.0);
        let m0_a0 = acc[0] * bodies[0].mass;
        let m1_a1 = acc[1] * bodies[1].mass;
        assert!(m0_a0.approx_eq(-m1_a1, 1e-12, 1e-12));
    }

    #[test]
    fn zero_qi_gives_zero_accel_for_that_body() {
        let bodies = [Body::new(1.0, 0.0), Body::new(1.0, 1.0), Body::new(1.0, -1.0)];
        let pos = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::zero()];
        let vel = [Vec3::new(0.1, 0.0, 0.0); 3];
        let acc = em_accel(&bodies, &pos, &vel, 1.0, 1.0, 0.0);
        assert!(acc[0].approx_eq(Vec3::zero(), 1e-12, 1e-12));
    }
}
