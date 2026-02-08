use threebody_core::config::Config;
use threebody_core::integrators::leapfrog::Leapfrog;
use threebody_core::math::vec3::Vec3;
use threebody_core::sim::{SimOptions, simulate};
use threebody_core::state::{Body, State, System};

#[test]
fn tiny_simulation_runs() {
    let bodies = [
        Body::new(1.0, 0.0),
        Body::new(1.0, 0.0),
        Body::new(0.0, 0.0),
    ];
    let pos = [
        Vec3::new(-0.5, 0.0, 0.0),
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::zero(),
    ];
    let vel = [
        Vec3::new(0.0, 0.7, 0.0),
        Vec3::new(0.0, -0.7, 0.0),
        Vec3::zero(),
    ];
    let system = System::new(bodies, State::new(pos, vel));
    let cfg = Config::default();
    let options = SimOptions { steps: 2, dt: 0.01 };
    let integrator = Leapfrog;
    let result = simulate(system, &cfg, &integrator, None, options);
    assert_eq!(result.steps.len(), 3);
}
