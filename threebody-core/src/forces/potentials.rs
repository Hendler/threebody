use crate::config::EmModel;
use crate::forces::em::moving_source_inv_r;
use crate::math::vec3::Vec3;
use crate::state::Body;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Potentials {
    pub phi: [f64; 3],
    pub a: [Vec3; 3],
}

/// Compute scalar and vector potentials for each body.
/// phi_i = sum_{j != i} k_e * q_j / |r_i - r_j|
/// A_i   = sum_{j != i} (mu_0 / 4pi) * q_j * v_j / |r_i - r_j|
pub fn potentials(
    bodies: &[Body],
    pos: &[Vec3],
    vel: &[Vec3],
    k_e: f64,
    mu_0: f64,
    epsilon: f64,
    model: EmModel,
    c: f64,
) -> Potentials {
    debug_assert_eq!(bodies.len(), 3);
    debug_assert_eq!(pos.len(), 3);
    debug_assert_eq!(vel.len(), 3);
    let mut phi = [0.0; 3];
    let mut a = [Vec3::zero(); 3];
    let mu0_over_4pi = mu_0 / (4.0 * std::f64::consts::PI);

    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                continue;
            }
            let r_ij = pos[i] - pos[j];
            let inv_r = moving_source_inv_r(r_ij, vel[j], epsilon, model, c);
            if inv_r == 0.0 {
                continue;
            }
            let qj = bodies[j].charge;
            phi[i] += k_e * qj * inv_r;
            a[i] = a[i] + vel[j] * (mu0_over_4pi * qj * inv_r);
        }
    }

    Potentials { phi, a }
}
#[cfg(test)]
mod tests {
    use crate::config::EmModel;
    use super::potentials;
    use crate::math::vec3::Vec3;
    use crate::state::Body;

    #[test]
    fn zero_charges_give_zero_potentials() {
        let bodies = [
            Body::new(1.0, 0.0),
            Body::new(1.0, 0.0),
            Body::new(1.0, 0.0),
        ];
        let pos = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::zero(),
        ];
        let vel = [Vec3::new(0.1, 0.2, 0.3); 3];
        let p = potentials(
            &bodies,
            &pos,
            &vel,
            1.0,
            1.0,
            0.0,
            EmModel::Quasistatic,
            10.0,
        );
        assert_eq!(p.phi, [0.0; 3]);
        assert_eq!(p.a, [Vec3::zero(); 3]);
    }

    #[test]
    fn zero_velocities_give_zero_vector_potential() {
        let bodies = [
            Body::new(1.0, 1.0),
            Body::new(1.0, -1.0),
            Body::new(1.0, 2.0),
        ];
        let pos = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::zero(),
        ];
        let vel = [Vec3::zero(); 3];
        let p = potentials(
            &bodies,
            &pos,
            &vel,
            1.0,
            1.0,
            0.0,
            EmModel::Quasistatic,
            10.0,
        );
        assert_eq!(p.a, [Vec3::zero(); 3]);
        assert!(p.phi[0] != 0.0);
    }
}
