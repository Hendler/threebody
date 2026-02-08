use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::takens;
use crate::predictability::PREDICTABILITY_VERSION;

#[derive(Debug, Clone, Serialize)]
struct ContextThresholds {
    improvement_threshold: f64,
    max_sensitivity_median: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ContextPoint {
    m: usize,
    tau: usize,
    n_sensors: usize,
    context_time_window: usize,
    context_volume: usize,
    model: String,
    holdout_mse: f64,
    holdout_baseline_mse: f64,
    holdout_relative_improvement: f64,
    holdout_sensitivity_median: f64,
    effective: bool,
    takens_report_path: String,
}

#[derive(Debug, Clone, Serialize)]
struct ContextSummary {
    n_points: usize,
    n_effective: usize,
    effective_rate: f64,
    min_effective_m: Option<usize>,
    min_effective_context_time_window: Option<usize>,
    min_effective_context_volume: Option<usize>,
    best_relative_improvement: f64,
    best_m: usize,
    best_context_volume: usize,
    threshold_law_supported: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ContextWindowReport {
    version: String,
    input_csv: String,
    column: String,
    sensors_raw: String,
    m_grid: Vec<usize>,
    tau: usize,
    thresholds: ContextThresholds,
    points: Vec<ContextPoint>,
    summary: ContextSummary,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct InputSensitivity {
    #[serde(default)]
    median_rel_error: f64,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct InputBestConfig {
    #[serde(default)]
    m: usize,
    #[serde(default)]
    tau: usize,
    #[serde(default)]
    model: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct InputBest {
    #[serde(default)]
    config: InputBestConfig,
    #[serde(default)]
    holdout_mse: f64,
    #[serde(default)]
    holdout_baseline_mse: f64,
    #[serde(default)]
    holdout_relative_improvement: f64,
    #[serde(default)]
    holdout_sensitivity: InputSensitivity,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct InputTakensReport {
    #[serde(default)]
    sensors: Vec<String>,
    #[serde(default)]
    best: InputBest,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_context_window(
    input: PathBuf,
    column: String,
    sensors_raw: String,
    out: PathBuf,
    markdown_out: Option<PathBuf>,
    tau: usize,
    m_raw: String,
    k_raw: String,
    lambda_raw: String,
    model_raw: String,
    split_mode_raw: String,
    seed: u64,
    train_frac: f64,
    val_frac: f64,
    sensitivity_weight: f64,
    improvement_threshold: f64,
    max_sensitivity_median: f64,
    reports_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
    if !input.exists() {
        anyhow::bail!("input CSV not found: {}", input.display());
    }
    if tau == 0 {
        anyhow::bail!("tau must be positive");
    }
    if !improvement_threshold.is_finite() {
        anyhow::bail!("improvement_threshold must be finite");
    }
    if !max_sensitivity_median.is_finite() || max_sensitivity_median < 0.0 {
        anyhow::bail!("max_sensitivity_median must be finite and nonnegative");
    }

    let m_grid = parse_csv_usize_list(&m_raw, "m")?;
    let tau_raw = tau.to_string();
    let keep_reports = reports_dir.is_some();
    let work_dir = match reports_dir {
        Some(dir) => {
            fs::create_dir_all(&dir)?;
            dir
        }
        None => {
            let tmp = std::env::temp_dir().join(format!(
                "threebody_context_window_{}_{}",
                std::process::id(),
                seed
            ));
            fs::create_dir_all(&tmp)?;
            tmp
        }
    };

    let mut points = Vec::with_capacity(m_grid.len());
    for m in &m_grid {
        let report_path = work_dir.join(format!("takens_m{m}_tau{tau}.json"));
        takens::run_takens(
            input.clone(),
            column.clone(),
            sensors_raw.clone(),
            report_path.clone(),
            tau_raw.clone(),
            m.to_string(),
            k_raw.clone(),
            lambda_raw.clone(),
            model_raw.clone(),
            split_mode_raw.clone(),
            seed,
            train_frac,
            val_frac,
            sensitivity_weight,
        )?;

        let raw = fs::read_to_string(&report_path)?;
        let rep: InputTakensReport = serde_json::from_str(&raw).map_err(|e| {
            anyhow::anyhow!(
                "failed to parse Takens report {} as JSON: {}",
                report_path.display(),
                e
            )
        })?;
        let mut rel = rep.best.holdout_relative_improvement;
        if rel == 0.0 && rep.best.holdout_baseline_mse > 0.0 {
            rel = (rep.best.holdout_baseline_mse - rep.best.holdout_mse)
                / rep.best.holdout_baseline_mse;
        }
        let sens = rep.best.holdout_sensitivity.median_rel_error;
        let n_sensors = rep.sensors.len().max(1);
        let context_time_window = rep.best.config.tau.saturating_mul(rep.best.config.m);
        let context_volume = context_time_window.saturating_mul(n_sensors);
        let effective =
            rel > improvement_threshold && sens.is_finite() && sens <= max_sensitivity_median;
        points.push(ContextPoint {
            m: rep.best.config.m.max(*m),
            tau: rep.best.config.tau.max(tau),
            n_sensors,
            context_time_window,
            context_volume,
            model: if rep.best.config.model.trim().is_empty() {
                "unknown".to_string()
            } else {
                rep.best.config.model
            },
            holdout_mse: rep.best.holdout_mse,
            holdout_baseline_mse: rep.best.holdout_baseline_mse,
            holdout_relative_improvement: rel,
            holdout_sensitivity_median: sens,
            effective,
            takens_report_path: report_path.display().to_string(),
        });
    }

    points.sort_by(|a, b| {
        a.context_volume
            .cmp(&b.context_volume)
            .then_with(|| a.m.cmp(&b.m))
            .then_with(|| {
                b.holdout_relative_improvement
                    .partial_cmp(&a.holdout_relative_improvement)
                    .unwrap_or(Ordering::Equal)
            })
    });
    if points.is_empty() {
        anyhow::bail!("no context points were produced");
    }

    let summary = build_summary(&points);
    let report = ContextWindowReport {
        version: PREDICTABILITY_VERSION.to_string(),
        input_csv: input.display().to_string(),
        column,
        sensors_raw,
        m_grid,
        tau,
        thresholds: ContextThresholds {
            improvement_threshold,
            max_sensitivity_median,
        },
        points,
        summary,
        notes: vec![
            "Predictability context-window hypothesis: there exists a minimum effective information window W* needed for strict holdout prediction quality."
                .to_string(),
            "Context time window is tau * m. Context volume is (tau * m) * n_sensors.".to_string(),
            "A point is effective iff holdout_relative_improvement > improvement_threshold and holdout_sensitivity_median <= max_sensitivity_median."
                .to_string(),
            "threshold_law_supported=true when at least one lower-context point fails while a higher-context point succeeds."
                .to_string(),
        ],
    };

    fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    if let Some(md_path) = markdown_out {
        fs::write(&md_path, render_markdown(&report))?;
        eprintln!(
            "wrote context-window report: {} and {}",
            out.display(),
            md_path.display()
        );
    } else {
        eprintln!("wrote context-window report: {}", out.display());
    }

    if !keep_reports {
        let _ = fs::remove_dir_all(work_dir);
    }
    Ok(())
}

fn build_summary(points: &[ContextPoint]) -> ContextSummary {
    let n_points = points.len();
    let n_effective = points.iter().filter(|p| p.effective).count();
    let effective_rate = if n_points > 0 {
        n_effective as f64 / n_points as f64
    } else {
        0.0
    };
    let min_effective = points.iter().filter(|p| p.effective).min_by(|a, b| {
        a.context_volume
            .cmp(&b.context_volume)
            .then_with(|| a.m.cmp(&b.m))
    });
    let best = points
        .iter()
        .max_by(|a, b| {
            a.holdout_relative_improvement
                .partial_cmp(&b.holdout_relative_improvement)
                .unwrap_or(Ordering::Equal)
        })
        .expect("points non-empty");
    let threshold_law_supported = min_effective.is_some_and(|minp| {
        points
            .iter()
            .any(|p| p.context_volume < minp.context_volume && !p.effective)
    });

    ContextSummary {
        n_points,
        n_effective,
        effective_rate,
        min_effective_m: min_effective.map(|p| p.m),
        min_effective_context_time_window: min_effective.map(|p| p.context_time_window),
        min_effective_context_volume: min_effective.map(|p| p.context_volume),
        best_relative_improvement: best.holdout_relative_improvement,
        best_m: best.m,
        best_context_volume: best.context_volume,
        threshold_law_supported,
    }
}

fn parse_csv_usize_list(raw: &str, field: &str) -> anyhow::Result<Vec<usize>> {
    let mut out = Vec::new();
    for tok in raw.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        let v: usize = tok
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid {field} value {tok:?}: {e}"))?;
        if v == 0 {
            anyhow::bail!("{field} values must be positive");
        }
        out.push(v);
    }
    if out.is_empty() {
        anyhow::bail!("{field} list is empty");
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

fn render_markdown(report: &ContextWindowReport) -> String {
    let mut out = String::new();
    out.push_str("# Context Window Report\n\n");
    out.push_str(&format!(
        "- column: `{}`\n- tau: `{}`\n- thresholds: improvement>{:.6e}, sensitivity<={:.6e}\n- points: {}\n- effective: {} ({:.1}%)\n",
        report.column,
        report.tau,
        report.thresholds.improvement_threshold,
        report.thresholds.max_sensitivity_median,
        report.summary.n_points,
        report.summary.n_effective,
        100.0 * report.summary.effective_rate
    ));
    if let Some(v) = report.summary.min_effective_context_time_window {
        out.push_str(&format!("- min effective context time window: `{}`\n", v));
    }
    if let Some(v) = report.summary.min_effective_context_volume {
        out.push_str(&format!("- min effective context volume: `{}`\n", v));
    }
    out.push_str(&format!(
        "- threshold law supported: `{}`\n\n",
        report.summary.threshold_law_supported
    ));
    out.push_str("| m | tau | sensors | context volume | model | rel improvement | sens median | effective |\n|---:|---:|---:|---:|---|---:|---:|---:|\n");
    for p in &report.points {
        out.push_str(&format!(
            "| {} | {} | {} | {} | `{}` | {:.6e} | {:.6e} | `{}` |\n",
            p.m,
            p.tau,
            p.n_sensors,
            p.context_volume,
            p.model,
            p.holdout_relative_improvement,
            p.holdout_sensitivity_median,
            p.effective
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{build_summary, ContextPoint};

    fn point(m: usize, v: usize, rel: f64, effective: bool) -> ContextPoint {
        ContextPoint {
            m,
            tau: 1,
            n_sensors: 2,
            context_time_window: m,
            context_volume: v,
            model: "linear".to_string(),
            holdout_mse: 1.0,
            holdout_baseline_mse: 2.0,
            holdout_relative_improvement: rel,
            holdout_sensitivity_median: 0.01,
            effective,
            takens_report_path: "x.json".to_string(),
        }
    }

    #[test]
    fn summary_finds_minimum_effective_context() {
        let pts = vec![
            point(1, 2, -0.1, false),
            point(2, 4, 0.3, true),
            point(3, 6, 0.4, true),
        ];
        let s = build_summary(&pts);
        assert_eq!(s.n_points, 3);
        assert_eq!(s.n_effective, 2);
        assert_eq!(s.min_effective_m, Some(2));
        assert_eq!(s.min_effective_context_volume, Some(4));
        assert!(s.threshold_law_supported);
    }

    #[test]
    fn summary_handles_no_effective_points() {
        let pts = vec![point(1, 2, -0.2, false), point(2, 4, -0.1, false)];
        let s = build_summary(&pts);
        assert_eq!(s.n_effective, 0);
        assert_eq!(s.min_effective_m, None);
        assert!(!s.threshold_law_supported);
    }
}
