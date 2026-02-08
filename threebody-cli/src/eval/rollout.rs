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
    use super::{VectorModel, rollout_metrics, rollout_trace};
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
}
