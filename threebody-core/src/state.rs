use crate::math::vec3::Vec3;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Body {
    pub mass: f64,
    pub charge: f64,
}

impl Body {
    pub const fn new(mass: f64, charge: f64) -> Self {
        Self { mass, charge }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct State {
    pub pos: [Vec3; 3],
    pub vel: [Vec3; 3],
}

impl State {
    pub const fn new(pos: [Vec3; 3], vel: [Vec3; 3]) -> Self {
        Self { pos, vel }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct System {
    pub bodies: [Body; 3],
    pub state: State,
}

impl System {
    pub const fn new(bodies: [Body; 3], state: State) -> Self {
        Self { bodies, state }
    }
}

pub const PAIR_INDICES: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

pub const fn pair_indices() -> [(usize, usize); 3] {
    PAIR_INDICES
}

#[cfg(test)]
mod tests {
    use super::{pair_indices, Body, State, System, PAIR_INDICES};
    use crate::math::vec3::Vec3;

    #[test]
    fn construction_and_clone() {
        let bodies = [Body::new(1.0, 0.0), Body::new(2.0, 1.0), Body::new(3.0, -1.0)];
        let state = State::new([Vec3::zero(); 3], [Vec3::zero(); 3]);
        let system = System::new(bodies, state);
        let system_copy = system;
        assert_eq!(system, system_copy);
    }

    #[test]
    fn pair_indices_are_sorted() {
        assert_eq!(pair_indices(), PAIR_INDICES);
        for (i, j) in pair_indices().iter().copied() {
            assert!(i < j);
        }
    }
}
