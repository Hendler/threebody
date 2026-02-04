use crate::config::Config;
use crate::diagnostics::{compute_diagnostics, Diagnostics};
use crate::forces::{compute_accel, ForceConfig};
use crate::integrators::Integrator;
use crate::regime::{compute_regime, RegimeDiagnostics};
use crate::state::System;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncounterEvent {
    pub step: usize,
    pub min_pair_dist: f64,
}

#[derive(Debug, Clone)]
pub struct SimStep {
    pub system: System,
    pub diagnostics: Diagnostics,
    pub regime: RegimeDiagnostics,
}

#[derive(Debug, Clone)]
pub struct SimResult {
    pub steps: Vec<SimStep>,
    pub encounter: Option<EncounterEvent>,
}

#[derive(Debug, Clone, Copy)]
pub struct SimOptions {
    pub steps: usize,
    pub dt: f64,
    pub r_min: f64,
    pub stop_on_encounter: bool,
}

pub fn simulate(
    mut system: System,
    cfg: &Config,
    integrator: &impl Integrator,
    options: SimOptions,
) -> SimResult {
    let force_cfg = ForceConfig {
        g: cfg.constants.g,
        k_e: cfg.constants.k_e,
        mu_0: cfg.constants.mu_0,
        epsilon: cfg.softening,
        enable_gravity: cfg.enable_gravity,
        enable_em: cfg.enable_em,
    };

    let mut steps = Vec::with_capacity(options.steps + 1);
    let mut encounter = None;

    for step in 0..=options.steps {
        let acc = compute_accel(&system, &force_cfg);
        let regime = compute_regime(&system, &acc, options.dt);
        let diagnostics = compute_diagnostics(&system, cfg.constants.g, cfg.constants.k_e, cfg.softening);
        if encounter.is_none() && regime.min_pair_dist < options.r_min {
            encounter = Some(EncounterEvent {
                step,
                min_pair_dist: regime.min_pair_dist,
            });
            if options.stop_on_encounter {
                steps.push(SimStep {
                    system,
                    diagnostics,
                    regime,
                });
                break;
            }
        }
        steps.push(SimStep {
            system,
            diagnostics,
            regime,
        });

        if step == options.steps {
            break;
        }
        system = integrator.step(&system, options.dt, cfg);
    }

    SimResult { steps, encounter }
}

#[cfg(test)]
mod tests {
    use super::{simulate, SimOptions};
    use crate::config::Config;
    use crate::integrators::leapfrog::Leapfrog;
    use crate::math::vec3::Vec3;
    use crate::state::{Body, State, System};

    #[test]
    fn deterministic_output_for_fixed_inputs() {
        let bodies = [Body::new(1.0, 0.0), Body::new(1.0, 0.0), Body::new(0.0, 0.0)];
        let pos = [Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0), Vec3::zero()];
        let vel = [Vec3::new(0.0, 0.7, 0.0), Vec3::new(0.0, -0.7, 0.0), Vec3::zero()];
        let system = System::new(bodies, State::new(pos, vel));
        let cfg = Config::default();
        let options = SimOptions {
            steps: 5,
            dt: 0.01,
            r_min: 0.1,
            stop_on_encounter: false,
        };
        let integrator = Leapfrog;
        let a = simulate(system, &cfg, &integrator, options);
        let b = simulate(system, &cfg, &integrator, options);
        assert_eq!(a.steps.len(), b.steps.len());
        assert_eq!(a.steps[0].system, b.steps[0].system);
        assert_eq!(a.steps[5].system, b.steps[5].system);
    }

    #[test]
    fn close_encounter_triggers_event() {
        let bodies = [Body::new(1.0, 0.0), Body::new(1.0, 0.0), Body::new(0.0, 0.0)];
        let pos = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.05, 0.0, 0.0), Vec3::zero()];
        let vel = [Vec3::zero(); 3];
        let system = System::new(bodies, State::new(pos, vel));
        let cfg = Config::default();
        let options = SimOptions {
            steps: 1,
            dt: 0.01,
            r_min: 0.1,
            stop_on_encounter: true,
        };
        let integrator = Leapfrog;
        let result = simulate(system, &cfg, &integrator, options);
        assert!(result.encounter.is_some());
        assert_eq!(result.encounter.unwrap().step, 0);
    }
}
