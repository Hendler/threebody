use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::predictability::PREDICTABILITY_VERSION;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ChannelKind {
    RawState,
    DerivedDiagnostic,
    Unknown,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct InputSensitivity {
    #[serde(default)]
    median_rel_error: f64,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct InputBestConfig {
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
    column: String,
    #[serde(default)]
    best: InputBest,
}

#[derive(Debug, Clone, Serialize)]
struct EfficacyThresholds {
    improvement_threshold: f64,
    max_sensitivity_median: f64,
}

#[derive(Debug, Clone, Serialize)]
struct EfficacyChannel {
    report_path: String,
    column: String,
    channel_kind: ChannelKind,
    model: String,
    holdout_mse: f64,
    holdout_baseline_mse: f64,
    holdout_relative_improvement: f64,
    holdout_sensitivity_median: f64,
    effective: bool,
}

#[derive(Debug, Clone, Serialize)]
struct EfficacyAggregate {
    n_channels: usize,
    n_effective: usize,
    effective_rate: f64,
    median_relative_improvement_all: f64,
    median_relative_improvement_raw: Option<f64>,
    median_relative_improvement_derived: Option<f64>,
    info_value_delta_raw_minus_derived: Option<f64>,
    best_channel: Option<String>,
    worst_channel: Option<String>,
    claim_status: String,
}

#[derive(Debug, Clone, Serialize)]
struct EfficacyReport {
    version: String,
    thresholds: EfficacyThresholds,
    channels: Vec<EfficacyChannel>,
    aggregate: EfficacyAggregate,
    notes: Vec<String>,
}

pub(crate) fn run_report(
    reports: Vec<PathBuf>,
    out: PathBuf,
    markdown_out: Option<PathBuf>,
    improvement_threshold: f64,
    max_sensitivity_median: f64,
) -> anyhow::Result<()> {
    if reports.is_empty() {
        anyhow::bail!("--reports is empty");
    }
    if !improvement_threshold.is_finite() {
        anyhow::bail!("improvement_threshold must be finite");
    }
    if !max_sensitivity_median.is_finite() || max_sensitivity_median < 0.0 {
        anyhow::bail!("max_sensitivity_median must be finite and nonnegative");
    }

    let thresholds = EfficacyThresholds {
        improvement_threshold,
        max_sensitivity_median,
    };

    let mut channels = Vec::new();
    for path in reports {
        if !path.exists() {
            anyhow::bail!("report not found: {}", path.display());
        }
        let raw = fs::read_to_string(&path)?;
        let input: InputTakensReport = serde_json::from_str(&raw).map_err(|e| {
            anyhow::anyhow!(
                "failed to parse report {} as JSON: {}",
                path.display(),
                e
            )
        })?;

        let column = input.column.trim().to_string();
        if column.is_empty() {
            anyhow::bail!("report {} missing column", path.display());
        }
        let kind = classify_column(&column);
        let mut rel = input.best.holdout_relative_improvement;
        if rel == 0.0 && input.best.holdout_baseline_mse > 0.0 {
            rel = (input.best.holdout_baseline_mse - input.best.holdout_mse)
                / input.best.holdout_baseline_mse;
        }
        let sensitivity = input.best.holdout_sensitivity.median_rel_error;
        let effective = rel > thresholds.improvement_threshold
            && sensitivity.is_finite()
            && sensitivity <= thresholds.max_sensitivity_median;

        channels.push(EfficacyChannel {
            report_path: path.display().to_string(),
            column,
            channel_kind: kind,
            model: if input.best.config.model.trim().is_empty() {
                "unknown".to_string()
            } else {
                input.best.config.model
            },
            holdout_mse: input.best.holdout_mse,
            holdout_baseline_mse: input.best.holdout_baseline_mse,
            holdout_relative_improvement: rel,
            holdout_sensitivity_median: sensitivity,
            effective,
        });
    }

    channels.sort_by(|a, b| {
        b.holdout_relative_improvement
            .partial_cmp(&a.holdout_relative_improvement)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.column.cmp(&b.column))
    });

    let aggregate = build_aggregate(&channels);
    let report = EfficacyReport {
        version: PREDICTABILITY_VERSION.to_string(),
        thresholds,
        channels,
        aggregate,
        notes: vec![
            "Efficacy is reported from simulation-extracted channels under strict holdout metrics."
                .to_string(),
            "A channel is marked effective iff holdout_relative_improvement > improvement_threshold and holdout_sensitivity_median <= max_sensitivity_median."
                .to_string(),
            "info_value_delta_raw_minus_derived compares median relative improvement for raw-state channels vs derived diagnostic channels."
                .to_string(),
        ],
    };

    fs::write(&out, serde_json::to_string_pretty(&report)?)?;

    if let Some(md_path) = markdown_out {
        let md = render_markdown(&report);
        fs::write(&md_path, md)?;
        eprintln!(
            "wrote efficacy report: {} and {}",
            out.display(),
            md_path.display()
        );
    } else {
        eprintln!("wrote efficacy report: {}", out.display());
    }

    Ok(())
}

fn build_aggregate(channels: &[EfficacyChannel]) -> EfficacyAggregate {
    let n_channels = channels.len();
    let n_effective = channels.iter().filter(|c| c.effective).count();
    let effective_rate = if n_channels > 0 {
        n_effective as f64 / n_channels as f64
    } else {
        0.0
    };

    let all: Vec<f64> = channels
        .iter()
        .map(|c| c.holdout_relative_improvement)
        .collect();
    let raw: Vec<f64> = channels
        .iter()
        .filter(|c| c.channel_kind == ChannelKind::RawState)
        .map(|c| c.holdout_relative_improvement)
        .collect();
    let derived: Vec<f64> = channels
        .iter()
        .filter(|c| c.channel_kind == ChannelKind::DerivedDiagnostic)
        .map(|c| c.holdout_relative_improvement)
        .collect();

    let median_all = median_or_nan(&all);
    let median_raw = median_opt(&raw);
    let median_derived = median_opt(&derived);
    let info_delta = match (median_raw, median_derived) {
        (Some(r), Some(d)) => Some(r - d),
        _ => None,
    };

    let best_channel = channels
        .iter()
        .max_by(|a, b| {
            a.holdout_relative_improvement
                .partial_cmp(&b.holdout_relative_improvement)
                .unwrap_or(Ordering::Equal)
        })
        .map(|c| c.column.clone());
    let worst_channel = channels
        .iter()
        .min_by(|a, b| {
            a.holdout_relative_improvement
                .partial_cmp(&b.holdout_relative_improvement)
                .unwrap_or(Ordering::Equal)
        })
        .map(|c| c.column.clone());

    let claim_status = if n_effective == 0 {
        "no_information_gain".to_string()
    } else if info_delta.unwrap_or(0.0) > 0.0 {
        "information_helpful_in_some_channels".to_string()
    } else {
        "mixed_or_channel_specific".to_string()
    };

    EfficacyAggregate {
        n_channels,
        n_effective,
        effective_rate,
        median_relative_improvement_all: median_all,
        median_relative_improvement_raw: median_raw,
        median_relative_improvement_derived: median_derived,
        info_value_delta_raw_minus_derived: info_delta,
        best_channel,
        worst_channel,
        claim_status,
    }
}

fn classify_column(column: &str) -> ChannelKind {
    if is_raw_state_column(column) {
        return ChannelKind::RawState;
    }
    if matches!(
        column,
        "step"
            | "t"
            | "dt"
            | "energy_proxy"
            | "P_x"
            | "P_y"
            | "P_z"
            | "L_x"
            | "L_y"
            | "L_z"
            | "min_pair_dist"
            | "max_speed"
            | "max_accel"
            | "dt_ratio"
    ) {
        return ChannelKind::DerivedDiagnostic;
    }
    ChannelKind::Unknown
}

fn is_raw_state_column(column: &str) -> bool {
    is_component(column, 'r')
        || is_component(column, 'v')
        || is_component(column, 'a')
        || is_component(column, 'e')
        || is_component(column, 'b')
        || is_component(column, 'A')
        || is_phi_scalar(column)
}

fn is_component(column: &str, prefix: char) -> bool {
    let (left, right) = match column.split_once('_') {
        Some(v) => v,
        None => return false,
    };
    if right.len() != 1 {
        return false;
    }
    let axis = right.as_bytes()[0];
    if axis != b'x' && axis != b'y' && axis != b'z' {
        return false;
    }
    let mut chars = left.chars();
    if chars.next() != Some(prefix) {
        return false;
    }
    let digits: String = chars.collect();
    !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
}

fn is_phi_scalar(column: &str) -> bool {
    if !column.starts_with("phi") || !column.ends_with("_scalar") {
        return false;
    }
    let mid = &column[3..(column.len() - 7)];
    !mid.is_empty() && mid.chars().all(|c| c.is_ascii_digit())
}

fn median_opt(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    Some(median_or_nan(xs))
}

fn median_or_nan(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut ys = xs.to_vec();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = ys.len();
    if n % 2 == 1 {
        ys[n / 2]
    } else {
        0.5 * (ys[n / 2 - 1] + ys[n / 2])
    }
}

fn render_markdown(report: &EfficacyReport) -> String {
    let mut out = String::new();
    out.push_str("# Efficacy Report\n\n");
    out.push_str(&format!(
        "- channels: {}\n- effective channels: {} ({:.1}%)\n- claim status: `{}`\n\n",
        report.aggregate.n_channels,
        report.aggregate.n_effective,
        100.0 * report.aggregate.effective_rate,
        report.aggregate.claim_status
    ));
    out.push_str(&format!(
        "- median improvement (all): {:.6e}\n",
        report.aggregate.median_relative_improvement_all
    ));
    if let Some(v) = report.aggregate.median_relative_improvement_raw {
        out.push_str(&format!("- median improvement (raw): {:.6e}\n", v));
    }
    if let Some(v) = report.aggregate.median_relative_improvement_derived {
        out.push_str(&format!("- median improvement (derived): {:.6e}\n", v));
    }
    if let Some(v) = report.aggregate.info_value_delta_raw_minus_derived {
        out.push_str(&format!("- info delta (raw - derived): {:.6e}\n", v));
    }
    out.push_str("\n## Channels\n\n");
    out.push_str(
        "| column | kind | model | rel improvement | sens median | effective |\n|---|---|---:|---:|---:|---:|\n",
    );
    for c in &report.channels {
        out.push_str(&format!(
            "| `{}` | `{:?}` | `{}` | {:.6e} | {:.6e} | `{}` |\n",
            c.column,
            c.channel_kind,
            c.model,
            c.holdout_relative_improvement,
            c.holdout_sensitivity_median,
            c.effective
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        ChannelKind, EfficacyChannel, build_aggregate, classify_column, is_raw_state_column,
    };

    fn row(
        column: &str,
        kind: ChannelKind,
        rel: f64,
        sens: f64,
        effective: bool,
    ) -> EfficacyChannel {
        EfficacyChannel {
            report_path: "x.json".to_string(),
            column: column.to_string(),
            channel_kind: kind,
            model: "linear".to_string(),
            holdout_mse: 1.0,
            holdout_baseline_mse: 2.0,
            holdout_relative_improvement: rel,
            holdout_sensitivity_median: sens,
            effective,
        }
    }

    #[test]
    fn classify_column_separates_raw_and_derived() {
        assert!(is_raw_state_column("a1_x"));
        assert!(is_raw_state_column("r3_z"));
        assert!(is_raw_state_column("phi2_scalar"));
        assert_eq!(classify_column("a1_x"), ChannelKind::RawState);
        assert_eq!(
            classify_column("min_pair_dist"),
            ChannelKind::DerivedDiagnostic
        );
        assert_eq!(classify_column("mystery_feature"), ChannelKind::Unknown);
    }

    #[test]
    fn aggregate_reports_information_delta_and_claim_status() {
        let channels = vec![
            row("a1_x", ChannelKind::RawState, 0.90, 0.01, true),
            row("v1_x", ChannelKind::RawState, 0.80, 0.02, true),
            row("min_pair_dist", ChannelKind::DerivedDiagnostic, -0.5, 1.0, false),
            row("max_accel", ChannelKind::DerivedDiagnostic, -0.4, 1.0, false),
        ];

        let agg = build_aggregate(&channels);
        assert_eq!(agg.n_channels, 4);
        assert_eq!(agg.n_effective, 2);
        assert!(agg.effective_rate > 0.4);
        assert_eq!(agg.best_channel.as_deref(), Some("a1_x"));
        assert_eq!(agg.worst_channel.as_deref(), Some("min_pair_dist"));
        let delta = agg.info_value_delta_raw_minus_derived.unwrap();
        assert!(delta > 1.0, "expected strong raw-vs-derived gap, got {delta}");
        assert_eq!(agg.claim_status, "information_helpful_in_some_channels");
    }

    #[test]
    fn aggregate_no_effective_channels_is_no_information_gain() {
        let channels = vec![
            row("min_pair_dist", ChannelKind::DerivedDiagnostic, -0.2, 0.9, false),
            row("max_accel", ChannelKind::DerivedDiagnostic, -0.1, 0.8, false),
        ];
        let agg = build_aggregate(&channels);
        assert_eq!(agg.n_effective, 0);
        assert_eq!(agg.claim_status, "no_information_gain");
    }
}
