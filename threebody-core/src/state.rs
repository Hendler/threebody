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

#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub pos: Vec<Vec3>,
    pub vel: Vec<Vec3>,
}

impl State {
    pub fn try_new<P, V>(pos: P, vel: V) -> Result<Self, String>
    where
        P: Into<Vec<Vec3>>,
        V: Into<Vec<Vec3>>,
    {
        let pos = pos.into();
        let vel = vel.into();
        if pos.len() != vel.len() {
            return Err(format!(
                "state pos/vel length mismatch: pos={} vel={}",
                pos.len(),
                vel.len()
            ));
        }
        Ok(Self { pos, vel })
    }

    pub fn new<P, V>(pos: P, vel: V) -> Self
    where
        P: Into<Vec<Vec3>>,
        V: Into<Vec<Vec3>>,
    {
        Self::try_new(pos, vel).expect("invalid state: pos/vel length mismatch")
    }

    pub fn len(&self) -> usize {
        self.pos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct System {
    pub bodies: Vec<Body>,
    pub state: State,
}

impl System {
    pub fn try_new<B>(bodies: B, state: State) -> Result<Self, String>
    where
        B: Into<Vec<Body>>,
    {
        let bodies = bodies.into();
        if bodies.len() != state.len() {
            return Err(format!(
                "system bodies/state length mismatch: bodies={} state={}",
                bodies.len(),
                state.len()
            ));
        }
        Ok(Self { bodies, state })
    }

    pub fn new<B>(bodies: B, state: State) -> Self
    where
        B: Into<Vec<Body>>,
    {
        Self::try_new(bodies, state).expect("invalid system: bodies/state length mismatch")
    }

    pub fn n(&self) -> usize {
        self.bodies.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{Body, State, System};
    use crate::math::vec3::Vec3;

    #[test]
    fn construction_and_clone() {
        let bodies = vec![
            Body::new(1.0, 0.0),
            Body::new(2.0, 1.0),
            Body::new(3.0, -1.0),
        ];
        let state = State::new(vec![Vec3::zero(); 3], vec![Vec3::zero(); 3]);
        let system = System::new(bodies, state);
        let system_copy = system.clone();
        assert_eq!(system, system_copy);
    }

    #[test]
    fn try_new_rejects_mismatched_lengths() {
        let err = State::try_new(vec![Vec3::zero(); 2], vec![Vec3::zero(); 3]).unwrap_err();
        assert!(err.contains("length mismatch"));
    }
}
