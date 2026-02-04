use crate::config::{CloseEncounterAction, Config};
use crate::diagnostics::{compute_diagnostics, Diagnostics};
use crate::forces::{compute_accel, ForceConfig};
use crate::integrators::Integrator;
use crate::regime::{compute_regime, RegimeDiagnostics};
use crate::state::System;
use serde::{Deserialize, Serialize};

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
    pub t: f64,
    pub dt: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SimStats {
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub dt_min: Option<f64>,
    pub dt_max: Option<f64>,
    pub dt_avg: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct SimResult {
    pub steps: Vec<SimStep>,
    pub encounter: Option<EncounterEvent>,
    pub encounter_action: Option<CloseEncounterAction>,
    pub warnings: Vec<String>,
    pub terminated_early: bool,
    pub termination_reason: Option<String>,
    pub stats: SimStats,
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
    let mut terminated_early = false;
    let mut termination_reason = None;
    let mut t = 0.0;
    let mut dt = options.dt;
    let mut accepted_steps = 0usize;
    let mut rejected_steps = 0usize;
    let mut dt_sum = 0.0;
    let mut dt_min: Option<f64> = None;
    let mut dt_max: Option<f64> = None;

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
        let regime = compute_regime(&system, &acc, dt);
        let diagnostics = compute_diagnostics(&system, &local_cfg);
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
                        t,
                        dt,
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
            t,
            dt,
        });

        if step == options.steps {
            break;
        }
        if local_cfg.integrator.adaptive
            && matches!(local_cfg.integrator.kind, crate::config::IntegratorKind::Rk45)
        {
            let mut rejects = 0;
            loop {
                let dt_used = dt;
                let (next, err_norm, dt_suggested) =
                    crate::integrators::rk45::step_with_error(&system, dt_used, &local_cfg);
                if err_norm <= 1.0 {
                    system = next;
                    t += dt_used;
                    accepted_steps += 1;
                    dt_sum += dt_used;
                    dt_min = Some(match dt_min {
                        Some(current) => current.min(dt_used),
                        None => dt_used,
                    });
                    dt_max = Some(match dt_max {
                        Some(current) => current.max(dt_used),
                        None => dt_used,
                    });
                    dt = dt_suggested
                        .clamp(local_cfg.integrator.dt_min, local_cfg.integrator.dt_max);
                    break;
                }
                dt = dt_suggested.clamp(local_cfg.integrator.dt_min, local_cfg.integrator.dt_max);
                rejects += 1;
                rejected_steps += 1;
                if rejects >= local_cfg.integrator.max_rejects {
                    terminated_early = true;
                    termination_reason = Some("max_rejects_exceeded".to_string());
                    break;
                }
            }
            if terminated_early {
                break;
            }
        } else {
            system = active_integrator.step(&system, dt, &local_cfg);
            t += dt;
            accepted_steps += 1;
            dt_sum += dt;
            dt_min = Some(match dt_min {
                Some(current) => current.min(dt),
                None => dt,
            });
            dt_max = Some(match dt_max {
                Some(current) => current.max(dt),
                None => dt,
            });
        }
    }

    let dt_avg = if accepted_steps > 0 {
        Some(dt_sum / accepted_steps as f64)
    } else {
        None
    };

    SimResult {
        steps,
        encounter,
        encounter_action,
        warnings,
        terminated_early,
        termination_reason,
        stats: SimStats {
            accepted_steps,
            rejected_steps,
            dt_min,
            dt_max,
            dt_avg,
        },
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

    #[test]
    fn adaptive_rk45_adjusts_dt() {
        let bodies = [Body::new(1.0, 0.0), Body::new(1.0, 0.0), Body::new(0.0, 0.0)];
        let pos = [Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0), Vec3::zero()];
        let vel = [Vec3::new(0.0, 0.7, 0.0), Vec3::new(0.0, -0.7, 0.0), Vec3::zero()];
        let system = System::new(bodies, State::new(pos, vel));

        let mut cfg = Config::default();
        cfg.integrator.kind = crate::config::IntegratorKind::Rk45;
        cfg.integrator.adaptive = true;
        cfg.integrator.dt_max = 0.1;
        cfg.integrator.dt_min = 1e-6;
        let options = SimOptions { steps: 2, dt: 0.1 };
        let integrator = crate::integrators::rk45::Rk45;

        let result = simulate(system, &cfg, &integrator, None, options);
        assert!(!result.terminated_early);
        assert!(result.steps.len() >= 2);
        let dt0 = result.steps[0].dt;
        let dt1 = result.steps[1].dt;
        assert!(dt0 > 0.0);
        assert!(dt1 >= cfg.integrator.dt_min);
        assert!(dt1 <= cfg.integrator.dt_max);
    }

    #[test]
    fn stats_track_fixed_dt() {
        let bodies = [Body::new(1.0, 0.0), Body::new(1.0, 0.0), Body::new(0.0, 0.0)];
        let pos = [Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0), Vec3::zero()];
        let vel = [Vec3::zero(); 3];
        let system = System::new(bodies, State::new(pos, vel));
        let mut cfg = Config::default();
        cfg.integrator.kind = crate::config::IntegratorKind::Leapfrog;
        cfg.integrator.adaptive = false;
        let options = SimOptions { steps: 3, dt: 0.02 };
        let integrator = Leapfrog;

        let result = simulate(system, &cfg, &integrator, None, options);
        assert_eq!(result.stats.accepted_steps, 3);
        assert_eq!(result.stats.rejected_steps, 0);
        assert_eq!(result.stats.dt_min, Some(0.02));
        assert_eq!(result.stats.dt_max, Some(0.02));
        assert_eq!(result.stats.dt_avg, Some(0.02));
    }
}
