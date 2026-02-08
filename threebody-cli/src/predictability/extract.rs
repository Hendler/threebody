use std::fs;
use std::io::Write;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use threebody_core::config::Config;
use threebody_core::output::sidecar::Sidecar;
use threebody_core::sim::SimStep;
use threebody_core::state::Body;

use crate::predictability::PREDICTABILITY_VERSION;

use super::features::{PairId, PairMetrics, all_pair_metrics, energy_pair, min_pair};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct Snapshot {
    pub(crate) step: usize,
    pub(crate) t: f64,
    pub(crate) dt: f64,
    pub(crate) min_pair: PairId,
    pub(crate) min_pair_dist: f64,
    pub(crate) energy_pair: PairId,
    pub(crate) energy_pair_energy: f64,
    pub(crate) pairs: Vec<PairMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct EncounterRecord {
    pub(crate) version: String,
    pub(crate) event_step: usize,
    pub(crate) event_t: f64,
    pub(crate) event_pair: PairId,
    pub(crate) event_min_pair_dist: f64,
    pub(crate) pre: Snapshot,
    pub(crate) post: Snapshot,
    pub(crate) labels: EncounterLabels,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct EncounterLabels {
    pub(crate) binary_pair_swapped: bool,
    pub(crate) pre_energy_pair: PairId,
    pub(crate) post_energy_pair: PairId,
    pub(crate) event_pair_equals_pre_energy_pair: bool,
    pub(crate) event_pair_equals_post_energy_pair: bool,
    pub(crate) delta_event_pair_specific_energy: f64,
    pub(crate) delta_event_pair_h: f64,
}

#[derive(Debug, Clone, Serialize)]
struct EncounterSummary {
    version: String,
    input_csv: String,
    sidecar_json: String,
    out_jsonl: String,
    events_written: usize,
    pre_window: f64,
    post_window: f64,
    min_event_dist_max: f64,
    min_event_dist: Option<f64>,
    mean_event_dist: Option<f64>,
}

pub(crate) fn run_extract(
    input: PathBuf,
    sidecar: Option<PathBuf>,
    out: PathBuf,
    summary_out: PathBuf,
    min_event_dist_max: f64,
    pre_window: f64,
    post_window: f64,
) -> anyhow::Result<()> {
    if !input.exists() {
        anyhow::bail!("input CSV not found: {}", input.display());
    }
    let sidecar_path = sidecar.unwrap_or_else(|| input.with_extension("json"));
    if !sidecar_path.exists() {
        anyhow::bail!(
            "sidecar JSON not found: {} (expected alongside CSV)",
            sidecar_path.display()
        );
    }
    let sidecar_json = fs::read_to_string(&sidecar_path)?;
    let sidecar: Sidecar = serde_json::from_str(&sidecar_json)?;
    let cfg = sidecar.config;
    cfg.validate().map_err(anyhow::Error::msg)?;

    let init = sidecar.initial_state.ok_or_else(|| {
        anyhow::anyhow!("sidecar missing initial_state; rerun simulate to regenerate")
    })?;
    let mut bodies = [Body::new(0.0, 0.0); 3];
    for i in 0..3 {
        bodies[i] = Body::new(init.mass[i], init.charge[i]);
    }
    let steps = crate::load_steps_from_csv(&input, bodies, &cfg)?;
    let encounters = extract_encounters(&steps, &cfg, pre_window, post_window, min_event_dist_max);

    let mut out_file = fs::File::create(&out)?;
    for rec in &encounters {
        writeln!(out_file, "{}", serde_json::to_string(rec)?)?;
    }

    let (min_event_dist, mean_event_dist) = summarize_event_distances(&encounters);
    let summary = EncounterSummary {
        version: PREDICTABILITY_VERSION.to_string(),
        input_csv: input.display().to_string(),
        sidecar_json: sidecar_path.display().to_string(),
        out_jsonl: out.display().to_string(),
        events_written: encounters.len(),
        pre_window,
        post_window,
        min_event_dist_max,
        min_event_dist,
        mean_event_dist,
    };
    fs::write(&summary_out, serde_json::to_string_pretty(&summary)?)?;

    eprintln!("wrote {} events to {}", encounters.len(), out.display());
    eprintln!("summary: {}", summary_out.display());
    Ok(())
}

fn summarize_event_distances(encounters: &[EncounterRecord]) -> (Option<f64>, Option<f64>) {
    if encounters.is_empty() {
        return (None, None);
    }
    let mut min = f64::INFINITY;
    let mut sum = 0.0;
    let mut n = 0usize;
    for e in encounters {
        let d = e.event_min_pair_dist;
        if d.is_finite() {
            min = min.min(d);
            sum += d;
            n += 1;
        }
    }
    if n == 0 {
        return (None, None);
    }
    (Some(min), Some(sum / n as f64))
}

fn extract_encounters(
    steps: &[SimStep],
    cfg: &Config,
    pre_window: f64,
    post_window: f64,
    min_event_dist_max: f64,
) -> Vec<EncounterRecord> {
    if steps.len() < 3 {
        return Vec::new();
    }
    let min_series: Vec<f64> = steps.iter().map(|s| s.regime.min_pair_dist).collect();
    let event_indices = local_minima_indices(&min_series);
    let times: Vec<f64> = steps.iter().map(|s| s.t).collect();

    let mut records = Vec::new();
    for &event_idx in &event_indices {
        let d_event = min_series[event_idx];
        if min_event_dist_max > 0.0 && d_event.is_finite() && d_event > min_event_dist_max {
            continue;
        }
        let (pre_idx, post_idx) = match window_indices(&times, event_idx, pre_window, post_window) {
            Some(v) => v,
            None => continue,
        };

        let event_step = event_idx;
        let event_t = steps[event_idx].t;
        let (event_min_dist, event_pair) = min_pair(&steps[event_idx].system);
        let pre = snapshot_at(steps, cfg, pre_idx);
        let post = snapshot_at(steps, cfg, post_idx);

        let pre_energy_pair = pre.energy_pair;
        let post_energy_pair = post.energy_pair;
        let binary_pair_swapped = pre_energy_pair != post_energy_pair;
        let event_pair_equals_pre_energy_pair = event_pair == pre_energy_pair;
        let event_pair_equals_post_energy_pair = event_pair == post_energy_pair;

        let pre_event_pair = pre.pairs.iter().find(|p| p.pair == event_pair).cloned();
        let post_event_pair = post.pairs.iter().find(|p| p.pair == event_pair).cloned();
        let delta_event_pair_specific_energy =
            match (pre_event_pair.as_ref(), post_event_pair.as_ref()) {
                (Some(a), Some(b)) => b.specific_energy - a.specific_energy,
                _ => 0.0,
            };
        let delta_event_pair_h = match (pre_event_pair.as_ref(), post_event_pair.as_ref()) {
            (Some(a), Some(b)) => b.h - a.h,
            _ => 0.0,
        };

        records.push(EncounterRecord {
            version: PREDICTABILITY_VERSION.to_string(),
            event_step,
            event_t,
            event_pair,
            event_min_pair_dist: event_min_dist,
            pre,
            post,
            labels: EncounterLabels {
                binary_pair_swapped,
                pre_energy_pair,
                post_energy_pair,
                event_pair_equals_pre_energy_pair,
                event_pair_equals_post_energy_pair,
                delta_event_pair_specific_energy,
                delta_event_pair_h,
            },
        });
    }
    records
}

fn snapshot_at(steps: &[SimStep], cfg: &Config, idx: usize) -> Snapshot {
    let step = &steps[idx];
    let (min_dist, min_pair) = min_pair(&step.system);
    let pairs = all_pair_metrics(cfg, &step.system);
    let (best_energy, best_energy_pair) = energy_pair(cfg, &step.system);
    Snapshot {
        step: idx,
        t: step.t,
        dt: step.dt,
        min_pair,
        min_pair_dist: min_dist,
        energy_pair: best_energy_pair,
        energy_pair_energy: best_energy,
        pairs,
    }
}

fn local_minima_indices(values: &[f64]) -> Vec<usize> {
    let mut indices = Vec::new();
    if values.len() < 3 {
        return indices;
    }
    for i in 1..(values.len() - 1) {
        let a = values[i - 1];
        let b = values[i];
        let c = values[i + 1];
        if !a.is_finite() || !b.is_finite() || !c.is_finite() {
            continue;
        }
        if b < a && b < c {
            indices.push(i);
        }
    }
    indices
}

fn window_indices(
    times: &[f64],
    center_idx: usize,
    pre_window: f64,
    post_window: f64,
) -> Option<(usize, usize)> {
    if times.is_empty() || center_idx >= times.len() {
        return None;
    }
    if pre_window < 0.0 || post_window < 0.0 {
        return None;
    }
    let t0 = times[center_idx];
    if !t0.is_finite() {
        return None;
    }
    let t_pre = t0 - pre_window;
    let t_post = t0 + post_window;
    if !t_pre.is_finite() || !t_post.is_finite() {
        return None;
    }

    let pre_insert = times.partition_point(|t| *t < t_pre);
    if pre_insert == 0 {
        return None;
    }
    let pre_idx = pre_insert - 1;
    if pre_idx >= center_idx {
        return None;
    }

    let post_idx = times.partition_point(|t| *t < t_post);
    if post_idx >= times.len() {
        return None;
    }
    if post_idx <= center_idx {
        return None;
    }

    Some((pre_idx, post_idx))
}

#[cfg(test)]
mod tests {
    use super::super::features::PairId;
    use super::{extract_encounters, local_minima_indices, window_indices};
    use threebody_core::config::Config;
    use threebody_core::diagnostics::Diagnostics;
    use threebody_core::math::vec3::Vec3;
    use threebody_core::regime::RegimeDiagnostics;
    use threebody_core::sim::SimStep;
    use threebody_core::state::{Body, State, System};

    fn dummy_step(t: f64, dt: f64, system: System, min_pair_dist: f64) -> SimStep {
        SimStep {
            system,
            diagnostics: Diagnostics {
                linear_momentum: Vec3::zero(),
                angular_momentum: Vec3::zero(),
                energy_proxy: 0.0,
            },
            regime: RegimeDiagnostics {
                min_pair_dist,
                max_speed: 0.0,
                max_accel: 0.0,
                dt_ratio: 0.0,
            },
            t,
            dt,
        }
    }

    #[test]
    fn local_minima_detects_strict_minimum() {
        let v = vec![3.0, 2.0, 1.0, 2.0, 3.0];
        assert_eq!(local_minima_indices(&v), vec![2]);

        let plateau = vec![3.0, 2.0, 1.0, 1.0, 2.0];
        assert!(local_minima_indices(&plateau).is_empty());
    }

    #[test]
    fn window_indices_requires_full_pre_and_post_window() {
        let times = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        // Center at t=0.3; pre=0.15 -> pre time=0.15, last <0.15 is t=0.1 idx=1.
        // post=0.15 -> post time=0.45, first >=0.45 is t=0.5 idx=5.
        let (pre, post) = window_indices(&times, 3, 0.15, 0.15).expect("window");
        assert_eq!(pre, 1);
        assert_eq!(post, 5);

        // Not enough pre-window.
        assert!(window_indices(&times, 1, 0.5, 0.1).is_none());
        // Not enough post-window.
        assert!(window_indices(&times, 4, 0.1, 0.5).is_none());
    }

    #[test]
    fn extract_encounters_finds_single_event_and_features() {
        let cfg = Config::default();
        let bodies = [
            Body::new(1.0, 0.0),
            Body::new(1.0, 0.0),
            Body::new(1.0, 0.0),
        ];

        // Construct a trajectory where pair (0,1) distance dips at step 5.
        let d: Vec<f64> = vec![2.0, 1.5, 1.0, 0.6, 0.4, 0.3, 0.4, 0.6, 1.0, 1.5, 2.0];
        let mut steps = Vec::new();
        for (i, &dist) in d.iter().enumerate() {
            let pos = [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(dist, 0.0, 0.0),
                Vec3::new(10.0, 0.0, 0.0),
            ];
            let vel = [Vec3::zero(); 3];
            let system = System::new(bodies, State::new(pos, vel));
            steps.push(dummy_step(i as f64 * 0.1, 0.1, system, dist));
        }

        let events = extract_encounters(&steps, &cfg, 0.2, 0.2, 0.0);
        assert_eq!(events.len(), 1);
        let ev = &events[0];
        assert_eq!(ev.event_step, 5);
        assert_eq!(ev.event_pair, PairId { i: 0, j: 1 });
        assert!((ev.event_min_pair_dist - 0.3).abs() < 1e-12);

        // Pre/post windows should be satisfied.
        assert!(ev.pre.step < ev.event_step);
        assert!(ev.post.step > ev.event_step);
        assert_eq!(ev.labels.pre_energy_pair, PairId { i: 0, j: 1 });
        assert_eq!(ev.labels.post_energy_pair, PairId { i: 0, j: 1 });
        assert!(!ev.labels.binary_pair_swapped);
    }
}
