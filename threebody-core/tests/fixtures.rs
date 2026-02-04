mod tolerances;

use std::fs;

use threebody_core::config::{Config, IntegratorKind};
use threebody_core::integrators::leapfrog::Leapfrog;
use threebody_core::integrators::Integrator;
use threebody_core::math::vec3::Vec3;
use threebody_core::state::{Body, State, System};

#[derive(serde::Deserialize)]
struct ExpectedState {
    pos: [[f64; 3]; 3],
    vel: [[f64; 3]; 3],
}

#[test]
fn gravity_only_fixture_matches_expected() {
    let cfg_json = fs::read_to_string("tests/fixtures/gravity_only_config.json").unwrap();
    let cfg: Config = serde_json::from_str(&cfg_json).unwrap();
    assert_eq!(cfg.integrator.kind, IntegratorKind::Leapfrog);

    let bodies = [Body::new(1.0, 0.0), Body::new(1.0, 0.0), Body::new(0.0, 0.0)];
    let pos = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::zero()];
    let vel = [Vec3::zero(); 3];
    let system = System::new(bodies, State::new(pos, vel));

    let integrator = Leapfrog;
    let next = integrator.step(&system, cfg.integrator.dt, &cfg);

    let expected_json = fs::read_to_string("tests/fixtures/gravity_only_expected.json").unwrap();
    let expected: ExpectedState = serde_json::from_str(&expected_json).unwrap();

    for i in 0..3 {
        assert_with_tol(next.state.pos[i].x, expected.pos[i][0]);
        assert_with_tol(next.state.pos[i].y, expected.pos[i][1]);
        assert_with_tol(next.state.pos[i].z, expected.pos[i][2]);
        assert_with_tol(next.state.vel[i].x, expected.vel[i][0]);
        assert_with_tol(next.state.vel[i].y, expected.vel[i][1]);
        assert_with_tol(next.state.vel[i].z, expected.vel[i][2]);
    }
}

fn assert_with_tol(actual: f64, expected: f64) {
    let diff = (actual - expected).abs();
    let tol = tolerances::ABS + tolerances::REL * expected.abs();
    assert!(diff <= tol, "diff {diff} exceeded tol {tol}");
}
