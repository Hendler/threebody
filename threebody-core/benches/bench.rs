use criterion::{criterion_group, criterion_main, Criterion};
use threebody_core::config::Config;
use threebody_core::integrators::leapfrog::Leapfrog;
use threebody_core::integrators::Integrator;
use threebody_core::math::vec3::Vec3;
use threebody_core::state::{Body, State, System};

fn bench_leapfrog(c: &mut Criterion) {
    let bodies = [Body::new(1.0, 0.0), Body::new(1.0, 0.0), Body::new(0.0, 0.0)];
    let pos = [Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0), Vec3::zero()];
    let v = (0.5_f64).sqrt();
    let vel = [Vec3::new(0.0, v, 0.0), Vec3::new(0.0, -v, 0.0), Vec3::zero()];
    let system = System::new(bodies, State::new(pos, vel));
    let cfg = Config::default();
    let integrator = Leapfrog;
    let dt = 0.01;

    c.bench_function("leapfrog_step", |b| {
        b.iter(|| {
            let _ = integrator.step(&system, dt, &cfg);
        })
    });
}

criterion_group!(benches, bench_leapfrog);
criterion_main!(benches);
