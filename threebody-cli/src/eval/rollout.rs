use threebody_core::config::Config;
use threebody_core::math::vec3::Vec3;
use threebody_core::state::{Body, State, System};
use threebody_discover::Dataset;

use crate::RolloutIntegrator;

#[derive(Clone)]
pub(crate) struct VectorModel {
    pub(crate) eq_x: threebody_discover::Equation,
    pub(crate) eq_y: threebody_discover::Equation,
    pub(crate) eq_z: threebody_discover::Equation,
}

pub(crate) fn format_vector_model(model: &VectorModel) -> String {
    format!(
        "ax={} ; ay={} ; az={}",
        model.eq_x.format(),
        model.eq_y.format(),
        model.eq_z.format()
    )
}

pub(crate) fn rollout_metrics(
    model: &VectorModel,
    feature_names: &[String],
    result: &threebody_core::sim::SimResult,
    cfg: &Config,
    rollout_integrator: RolloutIntegrator,
) -> (f64, Option<f64>) {
    let mut system = result
        .steps
        .first()
        .map(|s| s.system.clone())
        .unwrap_or_else(|| {
            let bodies = [
                Body::new(1.0, 0.0),
                Body::new(1.0, 0.0),
                Body::new(1.0, 0.0),
            ];
            let pos = [Vec3::zero(); 3];
            let vel = [Vec3::zero(); 3];
            System::new(bodies, State::new(pos, vel))
        });
    let feature_dataset = Dataset::new(feature_names.to_vec(), vec![], vec![]);
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    let mut t = 0.0;
    let threshold = result.steps.first().map_or(0.1, |s| {
        (0.5 * crate::min_pair_distance(&s.system.state.pos)).max(0.1)
    });
    let mut divergence_time = None;
    for i in 0..(result.steps.len().saturating_sub(1)) {
        let dt = result.steps[i].dt;
        system = rollout_step(
            &system,
            &feature_dataset,
            model,
            cfg,
            dt,
            rollout_integrator,
        );
        t += dt;
        let err = rms_pos_error(&system, &result.steps[i + 1].system);
        sum_sq += err * err;
        count += 1;
        if divergence_time.is_none() && err > threshold {
            divergence_time = Some(t);
        }
    }
    let rmse = if count > 0 {
        (sum_sq / count as f64).sqrt()
    } else {
        0.0
    };
    (rmse, divergence_time)
}

pub(crate) fn rollout_trace(
    model: &VectorModel,
    feature_names: &[String],
    result: &threebody_core::sim::SimResult,
    cfg: &Config,
    rollout_integrator: RolloutIntegrator,
) -> Vec<RolloutTraceStep> {
    let mut system = result
        .steps
        .first()
        .map(|s| s.system.clone())
        .unwrap_or_else(|| {
            let bodies = [
                Body::new(1.0, 0.0),
                Body::new(1.0, 0.0),
                Body::new(1.0, 0.0),
            ];
            let pos = [Vec3::zero(); 3];
            let vel = [Vec3::zero(); 3];
            System::new(bodies, State::new(pos, vel))
        });
    let feature_dataset = Dataset::new(feature_names.to_vec(), vec![], vec![]);
    let mut trace = Vec::new();
    let mut t = 0.0;
    for i in 0..(result.steps.len().saturating_sub(1)) {
        let dt = result.steps[i].dt;
        system = rollout_step(
            &system,
            &feature_dataset,
            model,
            cfg,
            dt,
            rollout_integrator,
        );
        t += dt;
        let err = rms_pos_error(&system, &result.steps[i + 1].system);
        trace.push(RolloutTraceStep {
            t,
            pos: system.state.pos.iter().map(|p| [p.x, p.y, p.z]).collect(),
            rmse_pos: err,
        });
    }
    trace
}

#[derive(serde::Serialize)]
pub(crate) struct RolloutTraceStep {
    t: f64,
    pos: Vec<[f64; 3]>,
    rmse_pos: f64,
}

#[derive(Clone, serde::Serialize)]
pub(crate) struct SensitivityTraceStep {
    pub(crate) t: f64,
    pub(crate) observed_norm: f64,
    pub(crate) linearized_norm: f64,
    pub(crate) relative_norm_error: f64,
}

#[derive(Clone, serde::Serialize)]
pub(crate) struct SensitivityEval {
    pub(crate) version: String,
    pub(crate) perturbation_scale: f64,
    pub(crate) jacobian_eps: f64,
    pub(crate) horizon_t: f64,
    pub(crate) steps: usize,
    pub(crate) ftle_observed: Option<f64>,
    pub(crate) ftle_linearized: Option<f64>,
    pub(crate) relative_error_median: Option<f64>,
    pub(crate) relative_error_p90: Option<f64>,
    pub(crate) relative_error_max: Option<f64>,
    pub(crate) final_observed_norm: f64,
    pub(crate) final_linearized_norm: f64,
    pub(crate) trace: Vec<SensitivityTraceStep>,
}

fn predict_accel(
    system: &System,
    feature_dataset: &Dataset,
    model: &VectorModel,
    cfg: &Config,
) -> [Vec3; 3] {
    let mut acc = [Vec3::zero(); 3];
    for body in 0..3 {
        let features =
            crate::compute_feature_vector(system, body, cfg, &feature_dataset.feature_names);
        let ax = model.eq_x.predict(feature_dataset, &features);
        let ay = model.eq_y.predict(feature_dataset, &features);
        let az = model.eq_z.predict(feature_dataset, &features);
        acc[body] = Vec3::new(ax, ay, az);
    }
    acc
}

fn rollout_step(
    system: &System,
    feature_dataset: &Dataset,
    model: &VectorModel,
    cfg: &Config,
    dt: f64,
    integrator: RolloutIntegrator,
) -> System {
    let acc = predict_accel(system, feature_dataset, model, cfg);
    match integrator {
        RolloutIntegrator::Euler => {
            let mut new_pos = system.state.pos.clone();
            let mut new_vel = system.state.vel.clone();
            for b in 0..3 {
                new_vel[b] = new_vel[b] + acc[b] * dt;
                new_pos[b] = new_pos[b] + new_vel[b] * dt;
            }
            System::new(system.bodies.clone(), State::new(new_pos, new_vel))
        }
        RolloutIntegrator::Leapfrog => {
            let mut v_half = system.state.vel.clone();
            for b in 0..3 {
                v_half[b] = v_half[b] + acc[b] * (0.5 * dt);
            }
            let mut new_pos = system.state.pos.clone();
            for b in 0..3 {
                new_pos[b] = new_pos[b] + v_half[b] * dt;
            }
            let interim = System::new(system.bodies.clone(), State::new(new_pos, v_half.clone()));
            let acc_new = predict_accel(&interim, feature_dataset, model, cfg);
            let mut new_vel = v_half;
            for b in 0..3 {
                new_vel[b] = new_vel[b] + acc_new[b] * (0.5 * dt);
            }
            System::new(
                system.bodies.clone(),
                State::new(interim.state.pos.clone(), new_vel),
            )
        }
    }
}

fn system_to_state_vec(system: &System) -> [f64; 18] {
    let mut out = [0.0f64; 18];
    for b in 0..3 {
        out[b * 3] = system.state.pos[b].x;
        out[b * 3 + 1] = system.state.pos[b].y;
        out[b * 3 + 2] = system.state.pos[b].z;
        out[9 + b * 3] = system.state.vel[b].x;
        out[9 + b * 3 + 1] = system.state.vel[b].y;
        out[9 + b * 3 + 2] = system.state.vel[b].z;
    }
    out
}

fn system_from_state_vec(template: &System, state: &[f64; 18]) -> System {
    let mut pos = [Vec3::zero(); 3];
    let mut vel = [Vec3::zero(); 3];
    for b in 0..3 {
        pos[b] = Vec3::new(state[b * 3], state[b * 3 + 1], state[b * 3 + 2]);
        vel[b] = Vec3::new(state[9 + b * 3], state[9 + b * 3 + 1], state[9 + b * 3 + 2]);
    }
    System::new(template.bodies.clone(), State::new(pos, vel))
}

fn rhs_state_vec(
    system: &System,
    feature_dataset: &Dataset,
    model: &VectorModel,
    cfg: &Config,
) -> [f64; 18] {
    let mut rhs = [0.0f64; 18];
    let acc = predict_accel(system, feature_dataset, model, cfg);
    for b in 0..3 {
        let v = system.state.vel[b];
        rhs[b * 3] = v.x;
        rhs[b * 3 + 1] = v.y;
        rhs[b * 3 + 2] = v.z;
        rhs[9 + b * 3] = acc[b].x;
        rhs[9 + b * 3 + 1] = acc[b].y;
        rhs[9 + b * 3 + 2] = acc[b].z;
    }
    rhs
}

fn numerical_jacobian(
    system: &System,
    feature_dataset: &Dataset,
    model: &VectorModel,
    cfg: &Config,
    jacobian_eps: f64,
) -> [[f64; 18]; 18] {
    let base_state = system_to_state_vec(system);
    let base_rhs = rhs_state_vec(system, feature_dataset, model, cfg);
    let eps_base = jacobian_eps.abs().max(1e-9);
    let mut jac = [[0.0f64; 18]; 18];

    for col in 0..18 {
        let mut pert = base_state;
        let h = eps_base * base_state[col].abs().max(1.0);
        pert[col] += h;
        let pert_system = system_from_state_vec(system, &pert);
        let rhs_pert = rhs_state_vec(&pert_system, feature_dataset, model, cfg);
        for row in 0..18 {
            jac[row][col] = (rhs_pert[row] - base_rhs[row]) / h;
        }
    }

    jac
}

fn mat_vec_mul(mat: &[[f64; 18]; 18], v: &[f64; 18]) -> [f64; 18] {
    let mut out = [0.0f64; 18];
    for row in 0..18 {
        let mut sum = 0.0;
        for col in 0..18 {
            sum += mat[row][col] * v[col];
        }
        out[row] = sum;
    }
    out
}

fn norm18(v: &[f64; 18]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn pctl(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() - 1) as f64 * p.clamp(0.0, 1.0)).round() as usize;
    sorted.get(idx).copied()
}

pub(crate) fn sensitivity_eval(
    model: &VectorModel,
    feature_names: &[String],
    result: &threebody_core::sim::SimResult,
    cfg: &Config,
    rollout_integrator: RolloutIntegrator,
    perturbation_scale: f64,
    jacobian_eps: f64,
) -> SensitivityEval {
    let Some(init_system) = result.steps.first().map(|s| s.system.clone()) else {
        return SensitivityEval {
            version: "v1".to_string(),
            perturbation_scale,
            jacobian_eps,
            horizon_t: 0.0,
            steps: 0,
            ftle_observed: None,
            ftle_linearized: None,
            relative_error_median: None,
            relative_error_p90: None,
            relative_error_max: None,
            final_observed_norm: 0.0,
            final_linearized_norm: 0.0,
            trace: Vec::new(),
        };
    };

    let mut base = init_system.clone();
    let mut pert = init_system;
    let eps = perturbation_scale.abs().max(1e-9);
    pert.state.pos[0].x += eps;
    let feature_dataset = Dataset::new(feature_names.to_vec(), vec![], vec![]);

    let mut delta = [0.0f64; 18];
    delta[0] = eps;
    let init_norm = norm18(&delta).max(1e-12);

    let mut t = 0.0;
    let mut rel_errors: Vec<f64> = Vec::new();
    let mut trace: Vec<SensitivityTraceStep> = Vec::new();
    let mut final_observed_norm = init_norm;
    let mut final_linearized_norm = init_norm;

    for i in 0..(result.steps.len().saturating_sub(1)) {
        let dt = result.steps[i].dt;

        let jac = numerical_jacobian(&base, &feature_dataset, model, cfg, jacobian_eps);
        let j_delta = mat_vec_mul(&jac, &delta);
        for k in 0..18 {
            delta[k] += dt * j_delta[k];
        }
        final_linearized_norm = norm18(&delta).max(1e-15);

        base = rollout_step(&base, &feature_dataset, model, cfg, dt, rollout_integrator);
        pert = rollout_step(&pert, &feature_dataset, model, cfg, dt, rollout_integrator);
        t += dt;

        let base_vec = system_to_state_vec(&base);
        let pert_vec = system_to_state_vec(&pert);
        let mut obs = [0.0f64; 18];
        for k in 0..18 {
            obs[k] = pert_vec[k] - base_vec[k];
        }
        final_observed_norm = norm18(&obs).max(1e-15);
        let rel =
            ((final_linearized_norm - final_observed_norm).abs()) / final_observed_norm.max(1e-12);
        rel_errors.push(rel);
        trace.push(SensitivityTraceStep {
            t,
            observed_norm: final_observed_norm,
            linearized_norm: final_linearized_norm,
            relative_norm_error: rel,
        });
    }

    let ftle_observed = (t > 0.0).then(|| (final_observed_norm / init_norm).ln() / t);
    let ftle_linearized = (t > 0.0).then(|| (final_linearized_norm / init_norm).ln() / t);

    SensitivityEval {
        version: "v1".to_string(),
        perturbation_scale: eps,
        jacobian_eps: jacobian_eps.abs().max(1e-9),
        horizon_t: t,
        steps: trace.len(),
        ftle_observed,
        ftle_linearized,
        relative_error_median: pctl(&rel_errors, 0.5),
        relative_error_p90: pctl(&rel_errors, 0.9),
        relative_error_max: rel_errors.iter().copied().reduce(f64::max),
        final_observed_norm,
        final_linearized_norm,
        trace,
    }
}

fn rms_pos_error(pred: &System, truth: &System) -> f64 {
    let mut sum = 0.0;
    for b in 0..3 {
        let diff = pred.state.pos[b] - truth.state.pos[b];
        sum += diff.norm_sq();
    }
    (sum / 3.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::{VectorModel, rollout_metrics, rollout_trace, sensitivity_eval};
    use crate::RolloutIntegrator;
    use threebody_core::config::Config;
    use threebody_core::diagnostics::Diagnostics;
    use threebody_core::math::vec3::Vec3;
    use threebody_core::regime::RegimeDiagnostics;
    use threebody_core::sim::{SimResult, SimStats, SimStep};
    use threebody_core::state::{Body, State, System};

    fn dummy_step(system: System, t: f64, dt: f64) -> SimStep {
        SimStep {
            system,
            diagnostics: Diagnostics {
                linear_momentum: Vec3::zero(),
                angular_momentum: Vec3::zero(),
                energy_proxy: 0.0,
            },
            regime: RegimeDiagnostics {
                min_pair_dist: 1.0,
                max_speed: 0.0,
                max_accel: 0.0,
                dt_ratio: 0.0,
            },
            t,
            dt,
        }
    }

    fn zero_model() -> VectorModel {
        VectorModel {
            eq_x: threebody_discover::Equation { terms: vec![] },
            eq_y: threebody_discover::Equation { terms: vec![] },
            eq_z: threebody_discover::Equation { terms: vec![] },
        }
    }

    #[test]
    fn rollout_trace_has_expected_length_and_zero_error_for_static_truth() {
        let cfg = Config::default();
        let model = zero_model();
        let feature_names: Vec<String> = Vec::new();

        let bodies = [Body::new(1.0, 0.0); 3];
        let pos = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let vel = [Vec3::zero(); 3];
        let system = System::new(bodies, State::new(pos, vel));

        let dt = 0.1;
        let steps = vec![
            dummy_step(system.clone(), 0.0, dt),
            dummy_step(system.clone(), dt, dt),
            dummy_step(system, 2.0 * dt, dt),
        ];
        let result = SimResult {
            steps,
            encounter: None,
            encounter_action: None,
            warnings: vec![],
            terminated_early: false,
            termination_reason: None,
            stats: SimStats {
                accepted_steps: 2,
                rejected_steps: 0,
                dt_min: Some(dt),
                dt_max: Some(dt),
                dt_avg: Some(dt),
            },
        };

        let trace = rollout_trace(
            &model,
            &feature_names,
            &result,
            &cfg,
            RolloutIntegrator::Euler,
        );
        assert_eq!(trace.len(), result.steps.len() - 1);

        let (rmse, div) = rollout_metrics(
            &model,
            &feature_names,
            &result,
            &cfg,
            RolloutIntegrator::Euler,
        );
        assert!(rmse.abs() < 1e-12);
        assert!(div.is_none());
    }

    #[test]
    fn rollout_metrics_records_divergence_time_when_error_crosses_threshold() {
        let cfg = Config::default();
        let model = zero_model();
        let feature_names: Vec<String> = Vec::new();

        let bodies = [Body::new(1.0, 0.0); 3];
        let pos0 = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let vel = [Vec3::zero(); 3];
        let sys0 = System::new(bodies, State::new(pos0, vel));

        // Move body 0 far enough that RMSE exceeds threshold at t=dt.
        let mut pos1 = pos0;
        pos1[0].x += 2.0;
        let sys1 = System::new(bodies, State::new(pos1, vel));

        let dt = 0.1;
        let steps = vec![dummy_step(sys0, 0.0, dt), dummy_step(sys1, dt, dt)];
        let result = SimResult {
            steps,
            encounter: None,
            encounter_action: None,
            warnings: vec![],
            terminated_early: false,
            termination_reason: None,
            stats: SimStats {
                accepted_steps: 1,
                rejected_steps: 0,
                dt_min: Some(dt),
                dt_max: Some(dt),
                dt_avg: Some(dt),
            },
        };

        let (_rmse, div) = rollout_metrics(
            &model,
            &feature_names,
            &result,
            &cfg,
            RolloutIntegrator::Euler,
        );
        assert_eq!(div, Some(dt));
    }

    #[test]
    fn sensitivity_eval_returns_finite_summary_for_static_case() {
        let cfg = Config::default();
        let model = zero_model();
        let feature_names: Vec<String> = Vec::new();

        let bodies = [Body::new(1.0, 0.0); 3];
        let pos = [
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let vel = [Vec3::zero(); 3];
        let system = System::new(bodies, State::new(pos, vel));

        let dt = 0.1;
        let steps = vec![
            dummy_step(system.clone(), 0.0, dt),
            dummy_step(system.clone(), dt, dt),
            dummy_step(system, 2.0 * dt, dt),
        ];
        let result = SimResult {
            steps,
            encounter: None,
            encounter_action: None,
            warnings: vec![],
            terminated_early: false,
            termination_reason: None,
            stats: SimStats {
                accepted_steps: 2,
                rejected_steps: 0,
                dt_min: Some(dt),
                dt_max: Some(dt),
                dt_avg: Some(dt),
            },
        };

        let sens = sensitivity_eval(
            &model,
            &feature_names,
            &result,
            &cfg,
            RolloutIntegrator::Euler,
            1e-6,
            1e-6,
        );
        assert_eq!(sens.steps, result.steps.len() - 1);
        assert!(sens.final_observed_norm.is_finite());
        assert!(sens.final_linearized_norm.is_finite());
        assert!(sens.relative_error_median.unwrap_or(0.0).is_finite());
    }
}
