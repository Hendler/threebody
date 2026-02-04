use crate::config::{CloseEncounterAction, Config};
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
    pub encounter_action: Option<CloseEncounterAction>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct SimOptions {
    pub steps: usize,
    pub dt: f64,
}

pub fn simulate(
    mut system: System,
    cfg: &Config,
    integrator: &dyn Integrator,
    encounter_integrator: Option<&dyn Integrator>,
    options: SimOptions,
) -> SimResult {
    let mut local_cfg = *cfg;
    let mut active_integrator = integrator;
    let mut steps = Vec::with_capacity(options.steps + 1);
    let mut encounter = None;
    let mut encounter_action = None;
    let warnings = cfg.warnings();

    for step in 0..=options.steps {
        let force_cfg = ForceConfig {
            g: local_cfg.constants.g,
            k_e: local_cfg.constants.k_e,
            mu_0: local_cfg.constants.mu_0,
            epsilon: local_cfg.softening,
            enable_gravity: local_cfg.enable_gravity,
            enable_em: local_cfg.enable_em,
        };
        let acc = compute_accel(&system, &force_cfg);
        let regime = compute_regime(&system, &acc, options.dt);
        let diagnostics =
            compute_diagnostics(&system, local_cfg.constants.g, local_cfg.constants.k_e, local_cfg.softening);
        if encounter.is_none() && regime.min_pair_dist < cfg.close_encounter.r_min {
            encounter = Some(EncounterEvent {
                step,
                min_pair_dist: regime.min_pair_dist,
            });
            encounter_action = Some(cfg.close_encounter.action);
            match cfg.close_encounter.action {
                CloseEncounterAction::StopAndReport => {
                    steps.push(SimStep {
                        system,
                        diagnostics,
                        regime,
                    });
                    break;
                }
                CloseEncounterAction::Soften => {
                    if cfg.close_encounter.softening > local_cfg.softening {
                        local_cfg.softening = cfg.close_encounter.softening;
                    }
                }
                CloseEncounterAction::SwitchIntegrator => {
                    if let Some(alt) = encounter_integrator {
                        active_integrator = alt;
                    }
                }
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
        system = active_integrator.step(&system, options.dt, &local_cfg);
    }

    SimResult {
        steps,
        encounter,
        encounter_action,
        warnings,
    }
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
        let options = SimOptions { steps: 5, dt: 0.01 };
        let integrator = Leapfrog;
        let a = simulate(system, &cfg, &integrator, None, options);
        let b = simulate(system, &cfg, &integrator, None, options);
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
        let mut cfg = Config::default();
        cfg.close_encounter.r_min = 0.1;
        cfg.close_encounter.action = crate::config::CloseEncounterAction::StopAndReport;
        let options = SimOptions { steps: 1, dt: 0.01 };
        let integrator = Leapfrog;
        let result = simulate(system, &cfg, &integrator, None, options);
        assert!(result.encounter.is_some());
        assert_eq!(result.encounter.unwrap().step, 0);
        assert_eq!(
            result.encounter_action,
            Some(crate::config::CloseEncounterAction::StopAndReport)
        );
    }
}
