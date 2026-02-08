use crate::math::vec3::Vec3;
use crate::state::System;

/// Convert a system to the barycentric (center-of-mass) frame by removing COM
/// position and velocity from all bodies.
pub fn to_barycentric(system: System) -> System {
    let total_mass = system.bodies.iter().map(|b| b.mass).sum::<f64>();
    if total_mass == 0.0 {
        return system;
    }

    let mut com_pos = Vec3::zero();
    let mut com_vel = Vec3::zero();
    for i in 0..3 {
        let m = system.bodies[i].mass;
        com_pos = com_pos + system.state.pos[i] * m;
        com_vel = com_vel + system.state.vel[i] * m;
    }
    com_pos = com_pos / total_mass;
    com_vel = com_vel / total_mass;

    let mut out = system;
    for i in 0..3 {
        out.state.pos[i] = out.state.pos[i] - com_pos;
        out.state.vel[i] = out.state.vel[i] - com_vel;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::to_barycentric;
    use crate::math::vec3::Vec3;
    use crate::state::{Body, State, System};

    #[test]
    fn com_position_and_momentum_zeroed() {
        let bodies = [
            Body::new(1.0, 0.0),
            Body::new(2.0, 0.0),
            Body::new(3.0, 0.0),
        ];
        let pos = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 3.0),
        ];
        let vel = [
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, 0.0, 0.25),
        ];
        let system = System::new(bodies, State::new(pos, vel));
        let bary = to_barycentric(system);

        let total_mass = bary.bodies.iter().map(|b| b.mass).sum::<f64>();
        let mut com_pos = Vec3::zero();
        let mut momentum = Vec3::zero();
        for i in 0..3 {
            let m = bary.bodies[i].mass;
            com_pos = com_pos + bary.state.pos[i] * m;
            momentum = momentum + bary.state.vel[i] * m;
        }
        com_pos = com_pos / total_mass;

        assert!(com_pos.approx_eq(Vec3::zero(), 1e-12, 1e-12));
        assert!(momentum.approx_eq(Vec3::zero(), 1e-12, 1e-12));
    }

    #[test]
    fn zero_total_mass_is_noop() {
        let bodies = [
            Body::new(0.0, 0.0),
            Body::new(0.0, 0.0),
            Body::new(0.0, 0.0),
        ];
        let pos = [Vec3::new(1.0, 2.0, 3.0); 3];
        let vel = [Vec3::new(0.1, 0.2, 0.3); 3];
        let system = System::new(bodies, State::new(pos, vel));
        let bary = to_barycentric(system.clone());
        assert_eq!(bary, system);
    }
}
