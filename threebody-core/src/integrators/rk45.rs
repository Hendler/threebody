use crate::config::Config;
use crate::forces::{ForceConfig, compute_accel};
use crate::integrators::Integrator;
use crate::math::vec3::Vec3;
use crate::state::{State, System};

pub struct Rk45;

impl Integrator for Rk45 {
    fn step(&self, system: &System, dt: f64, cfg: &Config) -> System {
        let (next, _err, _dt_suggested) = step_with_error(system, dt, cfg);
        next
    }
}

pub fn step_with_error(system: &System, dt: f64, cfg: &Config) -> (System, f64, f64) {
    let force_cfg = ForceConfig::from_config(cfg, cfg.softening);

    let (k1p, k1v) = deriv(system, &force_cfg);
    let (k2p, k2v) = deriv(
        &add_scaled(system, &k1p, &k1v, dt * (1.0 / 5.0)),
        &force_cfg,
    );
    let (k3p, k3v) = deriv(
        &add_scaled(
            system,
            &lincomb(&[(&k1p, 3.0 / 40.0), (&k2p, 9.0 / 40.0)]),
            &lincomb(&[(&k1v, 3.0 / 40.0), (&k2v, 9.0 / 40.0)]),
            dt,
        ),
        &force_cfg,
    );
    let (k4p, k4v) = deriv(
        &add_scaled(
            system,
            &lincomb(&[
                (&k1p, 44.0 / 45.0),
                (&k2p, -56.0 / 15.0),
                (&k3p, 32.0 / 9.0),
            ]),
            &lincomb(&[
                (&k1v, 44.0 / 45.0),
                (&k2v, -56.0 / 15.0),
                (&k3v, 32.0 / 9.0),
            ]),
            dt,
        ),
        &force_cfg,
    );
    let (k5p, k5v) = deriv(
        &add_scaled(
            system,
            &lincomb(&[
                (&k1p, 19372.0 / 6561.0),
                (&k2p, -25360.0 / 2187.0),
                (&k3p, 64448.0 / 6561.0),
                (&k4p, -212.0 / 729.0),
            ]),
            &lincomb(&[
                (&k1v, 19372.0 / 6561.0),
                (&k2v, -25360.0 / 2187.0),
                (&k3v, 64448.0 / 6561.0),
                (&k4v, -212.0 / 729.0),
            ]),
            dt,
        ),
        &force_cfg,
    );
    let (k6p, k6v) = deriv(
        &add_scaled(
            system,
            &lincomb(&[
                (&k1p, 9017.0 / 3168.0),
                (&k2p, -355.0 / 33.0),
                (&k3p, 46732.0 / 5247.0),
                (&k4p, 49.0 / 176.0),
                (&k5p, -5103.0 / 18656.0),
            ]),
            &lincomb(&[
                (&k1v, 9017.0 / 3168.0),
                (&k2v, -355.0 / 33.0),
                (&k3v, 46732.0 / 5247.0),
                (&k4v, 49.0 / 176.0),
                (&k5v, -5103.0 / 18656.0),
            ]),
            dt,
        ),
        &force_cfg,
    );
    let (k7p, k7v) = deriv(
        &add_scaled(
            system,
            &lincomb(&[
                (&k1p, 35.0 / 384.0),
                (&k3p, 500.0 / 1113.0),
                (&k4p, 125.0 / 192.0),
                (&k5p, -2187.0 / 6784.0),
                (&k6p, 11.0 / 84.0),
            ]),
            &lincomb(&[
                (&k1v, 35.0 / 384.0),
                (&k3v, 500.0 / 1113.0),
                (&k4v, 125.0 / 192.0),
                (&k5v, -2187.0 / 6784.0),
                (&k6v, 11.0 / 84.0),
            ]),
            dt,
        ),
        &force_cfg,
    );

    let y5 = add_scaled(
        system,
        &lincomb(&[
            (&k1p, 35.0 / 384.0),
            (&k3p, 500.0 / 1113.0),
            (&k4p, 125.0 / 192.0),
            (&k5p, -2187.0 / 6784.0),
            (&k6p, 11.0 / 84.0),
        ]),
        &lincomb(&[
            (&k1v, 35.0 / 384.0),
            (&k3v, 500.0 / 1113.0),
            (&k4v, 125.0 / 192.0),
            (&k5v, -2187.0 / 6784.0),
            (&k6v, 11.0 / 84.0),
        ]),
        dt,
    );

    let y4 = add_scaled(
        system,
        &lincomb(&[
            (&k1p, 5179.0 / 57600.0),
            (&k3p, 7571.0 / 16695.0),
            (&k4p, 393.0 / 640.0),
            (&k5p, -92097.0 / 339200.0),
            (&k6p, 187.0 / 2100.0),
            (&k7p, 1.0 / 40.0),
        ]),
        &lincomb(&[
            (&k1v, 5179.0 / 57600.0),
            (&k3v, 7571.0 / 16695.0),
            (&k4v, 393.0 / 640.0),
            (&k5v, -92097.0 / 339200.0),
            (&k6v, 187.0 / 2100.0),
            (&k7v, 1.0 / 40.0),
        ]),
        dt,
    );

    let err_norm = error_norm(system, &y5, &y4, cfg.integrator.rtol, cfg.integrator.atol);
    let dt_suggested = if err_norm == 0.0 {
        dt * 2.0
    } else {
        let safety = cfg.integrator.safety.clamp(1e-6, 1.0);
        let factor = safety * err_norm.powf(-0.2);
        dt * factor.clamp(0.2, 5.0)
    };

    (y5, err_norm, dt_suggested)
}

fn deriv(system: &System, cfg: &ForceConfig) -> ([Vec3; 3], [Vec3; 3]) {
    let dpos: [Vec3; 3] = system
        .state
        .vel
        .clone()
        .try_into()
        .expect("rk45 requires exactly 3 velocity vectors");
    let dvel = compute_accel(system, cfg);
    (dpos, dvel)
}

fn add_scaled(system: &System, dpos: &[Vec3; 3], dvel: &[Vec3; 3], scale: f64) -> System {
    let mut pos = system.state.pos.clone();
    let mut vel = system.state.vel.clone();
    for i in 0..3 {
        pos[i] = pos[i] + dpos[i] * scale;
        vel[i] = vel[i] + dvel[i] * scale;
    }
    System::new(system.bodies.clone(), State::new(pos, vel))
}

fn lincomb(terms: &[(&[Vec3; 3], f64)]) -> [Vec3; 3] {
    let mut out = [Vec3::zero(); 3];
    for (arr, scale) in terms {
        for i in 0..3 {
            out[i] = out[i] + arr[i] * *scale;
        }
    }
    out
}

fn error_norm(prev: &System, y5: &System, y4: &System, rtol: f64, atol: f64) -> f64 {
    let mut max_err = 0.0;
    for i in 0..3 {
        for (a, b, c) in [
            (prev.state.pos[i], y5.state.pos[i], y4.state.pos[i]),
            (prev.state.vel[i], y5.state.vel[i], y4.state.vel[i]),
        ] {
            for (av, bv, cv) in [(a.x, b.x, c.x), (a.y, b.y, c.y), (a.z, b.z, c.z)] {
                let denom = atol + rtol * av.abs().max(bv.abs());
                let err = (bv - cv).abs() / denom;
                if err > max_err {
                    max_err = err;
                }
            }
        }
    }
    max_err
}

#[cfg(test)]
mod tests {
    use super::{Rk45, step_with_error};
    use crate::config::Config;
    use crate::integrators::{Integrator, leapfrog::Leapfrog};
    use crate::math::vec3::Vec3;
    use crate::state::{Body, State, System};

    fn two_body_system() -> System {
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

    #[test]
    fn error_estimate_decreases_with_smaller_dt() {
        let mut cfg = Config::default();
        cfg.enable_em = false;
        cfg.constants.g = 1.0;
        let system = two_body_system();
        let (_y1, err1, _dt1) = step_with_error(&system, 0.05, &cfg);
        let (_y2, err2, _dt2) = step_with_error(&system, 0.025, &cfg);
        assert!(err2 < err1);
    }

    #[test]
    fn short_horizon_matches_leapfrog() {
        let mut cfg = Config::default();
        cfg.enable_em = false;
        cfg.constants.g = 1.0;
        let mut s1 = two_body_system();
        let mut s2 = s1.clone();
        let rk = Rk45;
        let lf = Leapfrog;
        let dt = 0.01;
        for _ in 0..100 {
            s1 = rk.step(&s1, dt, &cfg);
            s2 = lf.step(&s2, dt, &cfg);
        }
        let diff = (s1.state.pos[0] - s2.state.pos[0]).norm();
        assert!(diff < 1e-2);
    }

    #[test]
    fn safety_factor_changes_dt_suggestion() {
        let mut cfg_lo = Config::default();
        cfg_lo.enable_em = false;
        cfg_lo.constants.g = 1.0;
        cfg_lo.integrator.safety = 0.5;

        let mut cfg_hi = cfg_lo;
        cfg_hi.integrator.safety = 0.95;

        let system = two_body_system();
        let (_y_lo, err_lo, dt_lo) = step_with_error(&system, 0.05, &cfg_lo);
        let (_y_hi, err_hi, dt_hi) = step_with_error(&system, 0.05, &cfg_hi);

        assert!((err_lo - err_hi).abs() < 1e-12);
        assert!(dt_hi > dt_lo);
    }
}
