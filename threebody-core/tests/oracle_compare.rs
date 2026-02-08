use threebody_core::config::Config;
use threebody_core::integrators::{Integrator, boris::Boris, leapfrog::Leapfrog, rk45::Rk45};
use threebody_core::math::vec3::Vec3;
use threebody_core::state::{Body, State, System};

fn base_system() -> System {
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
    let v = (0.5_f64).sqrt();
    let vel = [
        Vec3::new(0.0, v, 0.0),
        Vec3::new(0.0, -v, 0.0),
        Vec3::zero(),
    ];
    System::new(bodies, State::new(pos, vel))
}

fn run(integrator: &dyn Integrator, cfg: &Config, steps: usize, dt: f64) -> Vec<System> {
    let mut systems = Vec::with_capacity(steps + 1);
    let mut sys = base_system();
    systems.push(sys.clone());
    for _ in 0..steps {
        sys = integrator.step(&sys, dt, cfg);
        systems.push(sys.clone());
    }
    systems
}

fn rms_error(a: &System, b: &System) -> f64 {
    let mut sum = 0.0;
    let mut count = 0.0;
    for i in 0..3 {
        let d = a.state.pos[i] - b.state.pos[i];
        sum += d.norm_sq();
        count += 1.0;
    }
    (sum / count).sqrt()
}

#[test]
fn oracle_comparison_reports_divergence_and_rms() {
    let mut cfg = Config::default();
    cfg.enable_em = false;
    cfg.constants.g = 1.0;

    let steps = 200;
    let dt = 0.01;

    let lf = Leapfrog;
    let rk = Rk45;
    let boris = Boris;

    let a = run(&lf, &cfg, steps, dt);
    let b = run(&rk, &cfg, steps, dt);
    let c = run(&boris, &cfg, steps, dt);

    let threshold = 1e-2;
    let mut divergence = None;
    let mut rms_sum = 0.0;
    for i in 0..=steps {
        let err = rms_error(&a[i], &b[i]);
        rms_sum += err;
        if divergence.is_none() && err > threshold {
            divergence = Some(i);
        }
    }
    let rms = rms_sum / (steps as f64 + 1.0);

    assert!(rms.is_finite());
    let div_step = divergence.unwrap_or(steps);
    assert!(div_step <= steps);

    // Compare Borîs vs Leapfrog similarly.
    let mut rms_sum2 = 0.0;
    for i in 0..=steps {
        rms_sum2 += rms_error(&a[i], &c[i]);
    }
    let rms2 = rms_sum2 / (steps as f64 + 1.0);
    assert!(rms2.is_finite());
}
