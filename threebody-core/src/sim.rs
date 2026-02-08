use crate::config::{CloseEncounterAction, Config};
use crate::diagnostics::{Diagnostics, compute_diagnostics};
use crate::forces::{ForceConfig, compute_accel};
use crate::integrators::Integrator;
use crate::regime::{RegimeDiagnostics, compute_regime};
use crate::state::System;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncounterEvent {
    pub step: usize,
    pub min_pair_dist: f64,
    pub epsilon_before: Option<f64>,
    pub epsilon_after: Option<f64>,
    pub substeps_used: Option<usize>,
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

fn min_pair_distance(system: &System) -> f64 {
    let mut min_pair_dist = f64::INFINITY;
    for i in 0..3 {
        for j in (i + 1)..3 {
            let r = system.state.pos[j] - system.state.pos[i];
            let d = r.norm();
            if d < min_pair_dist {
                min_pair_dist = d;
            }
        }
    }
    min_pair_dist
}

fn smooth_softening(epsilon_base: f64, epsilon_target: f64, min_pair_dist: f64, r_min: f64) -> f64 {
    if !(epsilon_base.is_finite()
        && epsilon_target.is_finite()
        && min_pair_dist.is_finite()
        && r_min.is_finite())
    {
        return epsilon_base;
    }
    if epsilon_target <= epsilon_base || r_min <= 0.0 {
        return epsilon_base;
    }
    if min_pair_dist >= r_min {
        return epsilon_base;
    }
    let u = (1.0 - (min_pair_dist / r_min)).clamp(0.0, 1.0);
    // Smoothstep: 3u^2 - 2u^3
    let s = u * u * (3.0 - 2.0 * u);
    epsilon_base + s * (epsilon_target - epsilon_base)
}

fn desired_substeps(cfg: &Config, regime: &RegimeDiagnostics) -> usize {
    let max = cfg.close_encounter.substeps_max.max(1);
    if max == 1 {
        return 1;
    }

    let mut n = 1usize;
    let r_min = if cfg.close_encounter.substep_r_min > 0.0 {
        cfg.close_encounter.substep_r_min
    } else {
        cfg.close_encounter.r_min
    };
    if r_min > 0.0
        && regime.min_pair_dist.is_finite()
        && regime.min_pair_dist > 0.0
        && regime.min_pair_dist < r_min
    {
        let ratio = r_min / regime.min_pair_dist;
        if ratio.is_finite() && ratio > 1.0 {
            n = n.max(ratio.ceil() as usize);
        }
    }

    let dt_ratio_max = cfg.close_encounter.substep_dt_ratio_max;
    if dt_ratio_max > 0.0 && regime.dt_ratio.is_finite() && regime.dt_ratio > dt_ratio_max {
        let ratio = regime.dt_ratio / dt_ratio_max;
        if ratio.is_finite() && ratio > 1.0 {
            n = n.max(ratio.ceil() as usize);
        }
    }

    n.clamp(1, max)
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
        let epsilon_before = local_cfg.softening;
        if cfg.close_encounter.action == CloseEncounterAction::Soften
            && cfg.close_encounter.softening_smooth
            && cfg.close_encounter.r_min > 0.0
            && cfg.close_encounter.softening > local_cfg.softening
        {
            let min_dist = min_pair_distance(&system);
            let eps_eff = smooth_softening(
                local_cfg.softening,
                cfg.close_encounter.softening,
                min_dist,
                cfg.close_encounter.r_min,
            );
            if eps_eff > local_cfg.softening {
                local_cfg.softening = eps_eff;
            }
        }

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
        let substeps_next = desired_substeps(cfg, &regime);
        if encounter.is_none() && regime.min_pair_dist < cfg.close_encounter.r_min {
            encounter_action = Some(cfg.close_encounter.action);
            match cfg.close_encounter.action {
                CloseEncounterAction::StopAndReport => {
                    encounter = Some(EncounterEvent {
                        step,
                        min_pair_dist: regime.min_pair_dist,
                        epsilon_before: Some(epsilon_before),
                        epsilon_after: Some(local_cfg.softening),
                        substeps_used: None,
                    });
                    steps.push(SimStep {
                        system: system.clone(),
                        diagnostics,
                        regime,
                        t,
                        dt,
                    });
                    break;
                }
                CloseEncounterAction::Soften => {
                    if !cfg.close_encounter.softening_smooth
                        && cfg.close_encounter.softening > local_cfg.softening
                    {
                        local_cfg.softening = cfg.close_encounter.softening;
                    }
                }
                CloseEncounterAction::SwitchIntegrator => {
                    if let Some(alt) = encounter_integrator {
                        active_integrator = alt;
                    }
                }
            }
            encounter = Some(EncounterEvent {
                step,
                min_pair_dist: regime.min_pair_dist,
                epsilon_before: Some(epsilon_before),
                epsilon_after: Some(local_cfg.softening),
                substeps_used: Some(substeps_next),
            });
        }
        steps.push(SimStep {
            system: system.clone(),
            diagnostics,
            regime,
            t,
            dt,
        });

        if step == options.steps {
            break;
        }
        if local_cfg.integrator.adaptive
            && matches!(
                local_cfg.integrator.kind,
                crate::config::IntegratorKind::Rk45
            )
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
            let dt_used = dt / substeps_next.max(1) as f64;
            for _ in 0..substeps_next.max(1) {
                system = active_integrator.step(&system, dt_used, &local_cfg);
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
            }
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
    use super::{SimOptions, simulate};
    use crate::config::Config;
    use crate::integrators::leapfrog::Leapfrog;
    use crate::math::vec3::Vec3;
    use crate::state::{Body, State, System};

    #[test]
    fn deterministic_output_for_fixed_inputs() {
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
        let options = SimOptions { steps: 5, dt: 0.01 };
        let integrator = Leapfrog;
        let a = simulate(system.clone(), &cfg, &integrator, None, options);
        let b = simulate(system, &cfg, &integrator, None, options);
        assert_eq!(a.steps.len(), b.steps.len());
        assert_eq!(a.steps[0].system, b.steps[0].system);
        assert_eq!(a.steps[5].system, b.steps[5].system);
    }

    #[test]
    fn close_encounter_triggers_event() {
        let bodies = [
            Body::new(1.0, 0.0),
            Body::new(1.0, 0.0),
            Body::new(0.0, 0.0),
        ];
        let pos = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.05, 0.0, 0.0),
            Vec3::zero(),
        ];
        let vel = [Vec3::zero(); 3];
        let system = System::new(bodies, State::new(pos, vel));
        let mut cfg = Config::default();
        cfg.close_encounter.r_min = 0.1;
        cfg.close_encounter.action = crate::config::CloseEncounterAction::StopAndReport;
        let options = SimOptions { steps: 1, dt: 0.01 };
        let integrator = Leapfrog;
        let result = simulate(system, &cfg, &integrator, None, options);
        let enc = result.encounter.expect("expected encounter");
        assert_eq!(enc.step, 0);
        assert_eq!(enc.epsilon_before, Some(cfg.softening));
        assert_eq!(enc.epsilon_after, Some(cfg.softening));
        assert!(enc.substeps_used.is_none());
        assert_eq!(
            result.encounter_action,
            Some(crate::config::CloseEncounterAction::StopAndReport)
        );
    }

    #[test]
    fn smooth_softening_is_continuous_at_threshold() {
        let eps0 = 0.01;
        let eps1 = 0.5;
        let r_min = 1.0;
        let at = super::smooth_softening(eps0, eps1, r_min, r_min);
        assert_eq!(at, eps0);
        let just_below = super::smooth_softening(eps0, eps1, r_min * (1.0 - 1e-6), r_min);
        assert!(just_below >= eps0);
        assert!((just_below - eps0).abs() < 1e-9);
        let deeper = super::smooth_softening(eps0, eps1, 0.25, r_min);
        assert!(deeper > just_below);
        let deepest = super::smooth_softening(eps0, eps1, 0.0, r_min);
        assert!((deepest - eps1).abs() < 1e-12);
    }

    #[test]
    fn smooth_softening_does_not_jump_to_target_immediately() {
        let bodies = [Body::new(1.0, 0.0); 3];
        let pos = [
            Vec3::zero(),
            Vec3::new(0.099, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let vel = [Vec3::zero(); 3];
        let system = System::new(bodies, State::new(pos, vel));
        let mut cfg = Config::default();
        cfg.close_encounter.r_min = 0.1;
        cfg.close_encounter.action = crate::config::CloseEncounterAction::Soften;
        cfg.close_encounter.softening = 0.1;
        cfg.close_encounter.softening_smooth = true;
        let options = SimOptions { steps: 1, dt: 0.01 };
        let result = simulate(system, &cfg, &Leapfrog, None, options);
        let enc = result.encounter.expect("encounter expected");
        let eps_after = enc.epsilon_after.expect("epsilon_after expected");
        assert!(eps_after > 0.0);
        assert!(eps_after < cfg.close_encounter.softening);
    }

    #[test]
    fn close_encounter_substeps_reduce_dt_and_increase_accepted_steps() {
        let bodies = [Body::new(1.0, 0.0); 3];
        let pos = [
            Vec3::zero(),
            Vec3::new(0.025, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let vel = [Vec3::zero(); 3];
        let system = System::new(bodies, State::new(pos, vel));
        let mut cfg = Config::default();
        cfg.integrator.kind = crate::config::IntegratorKind::Leapfrog;
        cfg.integrator.adaptive = false;
        cfg.close_encounter.r_min = 0.1;
        cfg.close_encounter.action = crate::config::CloseEncounterAction::Soften;
        cfg.close_encounter.substeps_max = 4;
        let options = SimOptions { steps: 1, dt: 0.04 };
        let result = simulate(system, &cfg, &Leapfrog, None, options);
        assert_eq!(result.steps.len(), 2);
        assert_eq!(result.stats.accepted_steps, 4);
        assert_eq!(result.stats.rejected_steps, 0);
        assert_eq!(result.stats.dt_min, Some(0.01));
        assert_eq!(result.stats.dt_max, Some(0.01));
        assert_eq!(result.stats.dt_avg, Some(0.01));
        let enc = result.encounter.expect("encounter expected");
        assert_eq!(enc.substeps_used, Some(4));
    }

    #[test]
    fn adaptive_rk45_adjusts_dt() {
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
