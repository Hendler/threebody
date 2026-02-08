use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use serde::Serialize;
use threebody_core::output::sidecar::Sidecar;
use threebody_core::sim::SimStep;
use threebody_core::state::Body;

use super::ensemble::EnsembleManifest;
use super::features::{PairId, pair_metrics};
use crate::predictability::PREDICTABILITY_VERSION;

#[derive(Debug, Clone, Serialize)]
struct LockPointRecord {
    step: usize,
    t_mean: f64,
    t_min: f64,
    t_max: f64,
    mode_pair: PairId,
    mode_frac: f64,
    final_mode_pair: PairId,
    final_mode_frac: f64,
    committed_frac: f64,
    n_members: usize,
    binary_specific_energy_mean: f64,
    binary_specific_energy_std: f64,
    binary_e_mean: Option<f64>,
    binary_e_std: Option<f64>,
    binary_a_mean: Option<f64>,
    binary_a_std: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct DetectSummary {
    version: String,
    ensemble_dir: String,
    n_members: usize,
    steps_common: usize,
    min_mode_frac: f64,
    window: usize,
    final_mode_pair: PairId,
    final_mode_frac: f64,
    lock: Option<LockFound>,
}

#[derive(Debug, Clone, Serialize)]
struct LockFound {
    start_step: usize,
    start_t_mean: f64,
    mode_pair: PairId,
    committed_frac_at_detection: f64,
}

pub(crate) fn run_detect(
    ensemble_dir: PathBuf,
    out: PathBuf,
    summary_out: PathBuf,
    min_mode_frac: f64,
    window: usize,
) -> anyhow::Result<()> {
    if window == 0 {
        anyhow::bail!("--window must be >= 1");
    }
    if !(0.0..=1.0).contains(&min_mode_frac) {
        anyhow::bail!("--min-mode-frac must be in [0,1]");
    }
    if !ensemble_dir.exists() {
        anyhow::bail!("ensemble dir not found: {}", ensemble_dir.display());
    }
    let manifest_path = ensemble_dir.join("ensemble_manifest.json");
    if !manifest_path.exists() {
        anyhow::bail!(
            "manifest not found: {} (expected from `predictability ensemble`)",
            manifest_path.display()
        );
    }
    let raw = fs::read_to_string(&manifest_path)?;
    let manifest: EnsembleManifest = serde_json::from_str(&raw)?;
    if manifest.members.is_empty() {
        anyhow::bail!("manifest has no members");
    }

    let cfg = manifest.config;
    let mut member_steps: Vec<Vec<SimStep>> = Vec::with_capacity(manifest.members.len());

    for m in &manifest.members {
        let sidecar_path = ensemble_dir.join(&m.sidecar_json);
        let csv_path = ensemble_dir.join(&m.traj_csv);
        let sidecar_raw = fs::read_to_string(&sidecar_path)?;
        let sidecar: Sidecar = serde_json::from_str(&sidecar_raw)?;
        let init = sidecar.initial_state.ok_or_else(|| {
            anyhow::anyhow!(
                "sidecar missing initial_state: {} (rerun ensemble to regenerate)",
                sidecar_path.display()
            )
        })?;
        let mut bodies = [Body::new(0.0, 0.0); 3];
        for i in 0..3 {
            bodies[i] = Body::new(init.mass[i], init.charge[i]);
        }
        let steps = crate::load_steps_from_csv(&csv_path, bodies, &cfg)?;
        member_steps.push(steps);
    }

    let steps_common = member_steps.iter().map(|s| s.len()).min().unwrap_or(0);
    if steps_common == 0 {
        anyhow::bail!("no steps loaded");
    }

    let mut out_file = fs::File::create(&out)?;

    let last_idx = steps_common - 1;

    // Precompute the energy-minimizing pair per member per step (cheap outcome proxy).
    let mut member_pairs: Vec<Vec<PairId>> = Vec::with_capacity(member_steps.len());
    for s in &member_steps {
        let mut pairs = Vec::with_capacity(steps_common);
        for step_idx in 0..steps_common {
            pairs.push(energy_pair_for_step(&cfg, &s[step_idx]).0);
        }
        member_pairs.push(pairs);
    }

    // "Final outcome" proxy: energy-minimizing pair at the final common step.
    let final_pairs: Vec<PairId> = member_pairs.iter().map(|p| p[last_idx]).collect();
    let (final_mode_pair, final_mode_frac) = mode_pair_and_frac(&final_pairs);

    // Settling step per member: earliest step after which the final pair never changes again.
    let settle_steps: Vec<usize> = member_pairs
        .iter()
        .zip(final_pairs.iter())
        .map(|(series, final_pair)| settle_step(series, *final_pair))
        .collect();

    let mut lock: Option<LockFound> = None;
    let mut run_len = 0usize;

    for step_idx in 0..steps_common {
        let mut times = Vec::with_capacity(member_steps.len());
        let mut pairs = Vec::with_capacity(member_steps.len());
        for (mi, s) in member_steps.iter().enumerate() {
            times.push(s[step_idx].t);
            pairs.push(member_pairs[mi][step_idx]);
        }
        let (t_mean, t_min, t_max) = time_stats(&times);
        let (mode_pair, mode_frac) = mode_pair_and_frac(&pairs);

        let (e_mean, e_std, a_mean, a_std, eps_mean, eps_std) =
            binary_metric_stats_for_mode(&cfg, &member_steps, step_idx, mode_pair);

        let committed_frac =
            committed_fraction(final_mode_pair, &final_pairs, &settle_steps, step_idx);

        let rec = LockPointRecord {
            step: step_idx,
            t_mean,
            t_min,
            t_max,
            mode_pair,
            mode_frac,
            final_mode_pair,
            final_mode_frac,
            committed_frac,
            n_members: member_steps.len(),
            binary_specific_energy_mean: eps_mean,
            binary_specific_energy_std: eps_std,
            binary_e_mean: e_mean,
            binary_e_std: e_std,
            binary_a_mean: a_mean,
            binary_a_std: a_std,
        };
        writeln!(out_file, "{}", serde_json::to_string(&rec)?)?;

        if lock.is_none() {
            if committed_frac >= min_mode_frac {
                run_len += 1;
                if run_len >= window {
                    let start_step = step_idx + 1 - window;
                    let (start_t_mean, _, _) = time_stats(
                        &member_steps
                            .iter()
                            .map(|s| s[start_step].t)
                            .collect::<Vec<_>>(),
                    );
                    lock = Some(LockFound {
                        start_step,
                        start_t_mean,
                        mode_pair: final_mode_pair,
                        committed_frac_at_detection: committed_frac,
                    });
                }
            } else {
                run_len = 0;
            }
        }
    }

    let summary = DetectSummary {
        version: PREDICTABILITY_VERSION.to_string(),
        ensemble_dir: ensemble_dir.display().to_string(),
        n_members: member_steps.len(),
        steps_common,
        min_mode_frac,
        window,
        final_mode_pair,
        final_mode_frac,
        lock,
    };
    fs::write(&summary_out, serde_json::to_string_pretty(&summary)?)?;
    eprintln!("wrote: {}", out.display());
    eprintln!("summary: {}", summary_out.display());
    Ok(())
}

fn energy_pair_for_step(cfg: &threebody_core::config::Config, step: &SimStep) -> (PairId, f64) {
    let mut best_pair = PairId::new(0, 1);
    let mut best_energy = f64::INFINITY;
    for (i, j) in threebody_core::analysis::PAIRS_3 {
        let m = pair_metrics(cfg, &step.system, i, j);
        if m.specific_energy < best_energy {
            best_energy = m.specific_energy;
            best_pair = m.pair;
        }
    }
    (best_pair, best_energy)
}

fn settle_step(series: &[PairId], final_pair: PairId) -> usize {
    // Earliest index s.t. series[s..] are all `final_pair`.
    if series.is_empty() {
        return 0;
    }
    for i in (0..series.len()).rev() {
        if series[i] != final_pair {
            return i + 1;
        }
    }
    0
}

fn committed_fraction(
    final_mode: PairId,
    final_pairs: &[PairId],
    settle_steps: &[usize],
    step_idx: usize,
) -> f64 {
    let n = final_pairs.len().max(1);
    let mut committed = 0usize;
    for mi in 0..final_pairs.len() {
        if final_pairs[mi] == final_mode && step_idx >= settle_steps[mi] {
            committed += 1;
        }
    }
    committed as f64 / n as f64
}

fn mode_pair_and_frac(pairs: &[PairId]) -> (PairId, f64) {
    let n = pairs.len().max(1);
    let mut counts: HashMap<PairId, usize> = HashMap::new();
    for p in pairs {
        *counts.entry(*p).or_insert(0) += 1;
    }
    let mut best = PairId::new(0, 1);
    let mut best_count = 0usize;
    for (p, c) in counts {
        if c > best_count {
            best = p;
            best_count = c;
        }
    }
    (best, best_count as f64 / n as f64)
}

fn time_stats(times: &[f64]) -> (f64, f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut n = 0usize;
    for &t in times {
        if !t.is_finite() {
            continue;
        }
        min = min.min(t);
        max = max.max(t);
        sum += t;
        n += 1;
    }
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    (sum / n as f64, min, max)
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut n = 0usize;
    for &v in values {
        if !v.is_finite() {
            continue;
        }
        sum += v;
        sum_sq += v * v;
        n += 1;
    }
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = sum / n as f64;
    let var = (sum_sq / n as f64) - mean * mean;
    (mean, var.max(0.0).sqrt())
}

fn mean_std_opt(values: &[Option<f64>]) -> (Option<f64>, Option<f64>) {
    let mut buf = Vec::with_capacity(values.len());
    for v in values {
        if let Some(x) = *v {
            if x.is_finite() {
                buf.push(x);
            }
        }
    }
    if buf.is_empty() {
        return (None, None);
    }
    let (m, s) = mean_std(&buf);
    (Some(m), Some(s))
}

fn binary_metric_stats_for_mode(
    cfg: &threebody_core::config::Config,
    member_steps: &[Vec<SimStep>],
    step_idx: usize,
    mode_pair: PairId,
) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>, f64, f64) {
    let mut e_vals: Vec<Option<f64>> = Vec::with_capacity(member_steps.len());
    let mut a_vals: Vec<Option<f64>> = Vec::with_capacity(member_steps.len());
    let mut eps_vals: Vec<f64> = Vec::with_capacity(member_steps.len());
    for s in member_steps {
        let step = &s[step_idx];
        let m = pair_metrics(cfg, &step.system, mode_pair.i, mode_pair.j);
        e_vals.push(m.e);
        a_vals.push(m.a);
        eps_vals.push(m.specific_energy);
    }
    let (e_mean, e_std) = mean_std_opt(&e_vals);
    let (a_mean, a_std) = mean_std_opt(&a_vals);
    let (eps_mean, eps_std) = mean_std(&eps_vals);
    (e_mean, e_std, a_mean, a_std, eps_mean, eps_std)
}

#[cfg(test)]
fn first_lock_from_series(
    series: &[(PairId, f64)],
    min_mode_frac: f64,
    window: usize,
) -> Option<(usize, PairId)> {
    if window == 0 {
        return None;
    }
    let mut run_len = 0usize;
    let mut current: Option<PairId> = None;
    for (i, (p, frac)) in series.iter().enumerate() {
        if *frac >= min_mode_frac {
            if current == Some(*p) {
                run_len += 1;
            } else {
                current = Some(*p);
                run_len = 1;
            }
            if run_len >= window {
                let start = i + 1 - window;
                return Some((start, *p));
            }
        } else {
            current = None;
            run_len = 0;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{first_lock_from_series, mode_pair_and_frac, settle_step};
    use crate::predictability::features::PairId;

    #[test]
    fn mode_pair_and_frac_counts_correctly() {
        let p01 = PairId::new(0, 1);
        let p12 = PairId::new(1, 2);
        let (mode, frac) = mode_pair_and_frac(&[p01, p01, p12, p01]);
        assert_eq!(mode, p01);
        assert!((frac - 0.75).abs() < 1e-12);
    }

    #[test]
    fn first_lock_requires_sustained_window() {
        let p01 = PairId::new(0, 1);
        let p02 = PairId::new(0, 2);
        let series = vec![
            (p01, 0.5),
            (p01, 0.91),
            (p01, 0.92),
            (p01, 0.93),
            (p01, 0.94),
        ];
        assert_eq!(first_lock_from_series(&series, 0.9, 3), Some((1, p01)));

        // Pair changes inside the window => no lock.
        let series2 = vec![(p01, 0.95), (p02, 0.95), (p01, 0.95)];
        assert!(first_lock_from_series(&series2, 0.9, 3).is_none());
    }

    #[test]
    fn settle_step_finds_last_change_point() {
        let p01 = PairId::new(0, 1);
        let p02 = PairId::new(0, 2);
        let series = vec![p01, p01, p02, p02, p02];
        assert_eq!(settle_step(&series, p02), 2);
        assert_eq!(settle_step(&series, p01), 5); // p01 never becomes permanent again
        assert_eq!(settle_step(&[p01, p01], p01), 0);
    }
}
