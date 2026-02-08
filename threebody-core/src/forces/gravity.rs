use crate::math::vec3::Vec3;
use crate::state::Body;

/// Compute gravitational accelerations for all bodies.
/// Formula: a_i = sum_{j != i} G * m_j * (r_j - r_i) / |r_j - r_i|^3.
pub fn gravity_accel(bodies: &[Body], pos: &[Vec3], g: f64, epsilon: f64) -> [Vec3; 3] {
    debug_assert_eq!(bodies.len(), 3);
    debug_assert_eq!(pos.len(), 3);
    let mut acc = [Vec3::zero(); 3];
    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                continue;
            }
            let r = pos[j] - pos[i];
            let r2 = r.norm_sq();
            let inv_r3 = softened_inv_r3(r2, epsilon);
            acc[i] = acc[i] + r * (g * bodies[j].mass * inv_r3);
        }
    }
    acc
}

fn softened_inv_r3(r2: f64, epsilon: f64) -> f64 {
    if r2 == 0.0 {
        return 0.0;
    }
    let soft2 = if epsilon == 0.0 {
        r2
    } else {
        r2 + epsilon * epsilon
    };
    let r = soft2.sqrt();
    1.0 / (r * r * r)
}

#[cfg(test)]
mod tests {
    use super::gravity_accel;
    use crate::math::vec3::Vec3;
    use crate::state::Body;

    #[test]
    fn two_body_symmetry_scaled_by_masses() {
        let bodies = [
            Body::new(2.0, 0.0),
            Body::new(3.0, 0.0),
            Body::new(0.0, 0.0),
        ];
        let pos = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::zero(),
        ];
        let acc = gravity_accel(&bodies, &pos, 1.0, 0.0);
        let m0_a0 = acc[0] * bodies[0].mass;
        let m1_a1 = acc[1] * bodies[1].mass;
        assert!(m0_a0.approx_eq(-m1_a1, 1e-12, 1e-12));
    }

    #[test]
    fn zero_masses_produce_zero_accel() {
        let bodies = [
            Body::new(0.0, 0.0),
            Body::new(0.0, 0.0),
            Body::new(0.0, 0.0),
        ];
        let pos = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::zero(),
        ];
        let acc = gravity_accel(&bodies, &pos, 1.0, 0.0);
        assert_eq!(acc, [Vec3::zero(); 3]);
    }
}
