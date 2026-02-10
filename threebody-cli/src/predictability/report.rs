use std::cmp::Ordering;
use std::collections::BTreeMap;
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

#[derive(Debug, Clone, Deserialize, Default)]
struct InputEfficacyAggregate {
    #[serde(default)]
    n_effective: usize,
    #[serde(default)]
    effective_rate: f64,
    #[serde(default)]
    median_relative_improvement_all: f64,
    #[serde(default)]
    median_relative_improvement_raw: Option<f64>,
    #[serde(default)]
    median_relative_improvement_derived: Option<f64>,
    #[serde(default)]
    info_value_delta_raw_minus_derived: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct InputEfficacyChannel {
    #[serde(default)]
    column: String,
    #[serde(default)]
    holdout_relative_improvement: f64,
    #[serde(default)]
    effective: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct InputEfficacyReport {
    #[serde(default)]
    aggregate: InputEfficacyAggregate,
    #[serde(default)]
    channels: Vec<InputEfficacyChannel>,
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
    median_relative_improvement_all_ci_low: Option<f64>,
    median_relative_improvement_all_ci_high: Option<f64>,
    median_relative_improvement_raw: Option<f64>,
    median_relative_improvement_raw_ci_low: Option<f64>,
    median_relative_improvement_raw_ci_high: Option<f64>,
    median_relative_improvement_derived: Option<f64>,
    median_relative_improvement_derived_ci_low: Option<f64>,
    median_relative_improvement_derived_ci_high: Option<f64>,
    info_value_delta_raw_minus_derived: Option<f64>,
    best_channel: Option<String>,
    worst_channel: Option<String>,
    claim_status: String,
    evidence_tier: String,
    science_claim_allowed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct EfficacyReport {
    version: String,
    thresholds: EfficacyThresholds,
    bootstrap: BootstrapConfig,
    channels: Vec<EfficacyChannel>,
    aggregate: EfficacyAggregate,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct BootstrapConfig {
    enabled: bool,
    ci_level: f64,
    resamples: usize,
    seed: u64,
}

#[derive(Debug, Clone, Serialize)]
struct CompareAggregateDelta {
    median_relative_improvement_all_before: f64,
    median_relative_improvement_all_after: f64,
    median_relative_improvement_all_delta: f64,
    median_relative_improvement_raw_before: Option<f64>,
    median_relative_improvement_raw_after: Option<f64>,
    median_relative_improvement_raw_delta: Option<f64>,
    median_relative_improvement_derived_before: Option<f64>,
    median_relative_improvement_derived_after: Option<f64>,
    median_relative_improvement_derived_delta: Option<f64>,
    info_value_delta_before: Option<f64>,
    info_value_delta_after: Option<f64>,
    info_value_delta_change: Option<f64>,
    n_effective_before: usize,
    n_effective_after: usize,
    n_effective_delta: isize,
    effective_rate_before: f64,
    effective_rate_after: f64,
    effective_rate_delta: f64,
}

#[derive(Debug, Clone, Serialize)]
struct CompareChannelDelta {
    column: String,
    before_relative_improvement: Option<f64>,
    after_relative_improvement: Option<f64>,
    relative_improvement_delta: Option<f64>,
    before_effective: Option<bool>,
    after_effective: Option<bool>,
    channel_non_regression: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
struct CompareFlags {
    aggregate_non_regression: bool,
    effective_count_non_regression: bool,
    channel_non_regression_rate: f64,
    overall_non_regression: bool,
}

#[derive(Debug, Clone, Serialize)]
struct EfficacyCompareReport {
    version: String,
    before_report_path: String,
    after_report_path: String,
    non_regression_tol: f64,
    aggregate: CompareAggregateDelta,
    channels: Vec<CompareChannelDelta>,
    flags: CompareFlags,
    notes: Vec<String>,
}

pub(crate) fn run_report(
    reports: Vec<PathBuf>,
    out: PathBuf,
    markdown_out: Option<PathBuf>,
    improvement_threshold: f64,
    max_sensitivity_median: f64,
    bootstrap_resamples: usize,
    bootstrap_ci: f64,
    bootstrap_seed: u64,
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
    if !bootstrap_ci.is_finite() || bootstrap_ci <= 0.0 || bootstrap_ci >= 1.0 {
        anyhow::bail!("bootstrap_ci must be finite and in (0,1)");
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
            anyhow::anyhow!("failed to parse report {} as JSON: {}", path.display(), e)
        })?;
        validate_input_takens_report(&path, &input)?;

        let column = input.column.trim().to_string();
        let kind = classify_column(&column);
        let mut rel = input.best.holdout_relative_improvement;
        if (!rel.is_finite() || rel == 0.0) && input.best.holdout_baseline_mse > 0.0 {
            rel = (input.best.holdout_baseline_mse - input.best.holdout_mse)
                / input.best.holdout_baseline_mse;
        }
        let sensitivity = if input.best.holdout_sensitivity.median_rel_error.is_finite() {
            input.best.holdout_sensitivity.median_rel_error
        } else {
            f64::INFINITY
        };
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

    let aggregate = build_aggregate(&channels, bootstrap_ci, bootstrap_resamples, bootstrap_seed);
    let report = EfficacyReport {
        version: PREDICTABILITY_VERSION.to_string(),
        thresholds,
        bootstrap: BootstrapConfig {
            enabled: bootstrap_resamples > 0,
            ci_level: bootstrap_ci,
            resamples: bootstrap_resamples,
            seed: bootstrap_seed,
        },
        channels,
        aggregate,
        notes: vec![
            "Efficacy is reported from simulation-extracted channels under strict holdout metrics."
                .to_string(),
            "A channel is marked effective iff holdout_relative_improvement > improvement_threshold and holdout_sensitivity_median <= max_sensitivity_median."
                .to_string(),
            "Median bootstrap confidence intervals are deterministic (fixed seed) and summarize uncertainty from finite-channel sampling."
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

fn validate_input_takens_report(path: &std::path::Path, input: &InputTakensReport) -> anyhow::Result<()> {
    if input.column.trim().is_empty() {
        anyhow::bail!("report {} missing column", path.display());
    }
    if !input.best.holdout_mse.is_finite() || input.best.holdout_mse < 0.0 {
        anyhow::bail!(
            "report {} has invalid holdout_mse={}",
            path.display(),
            input.best.holdout_mse
        );
    }
    if !input.best.holdout_baseline_mse.is_finite() || input.best.holdout_baseline_mse < 0.0 {
        anyhow::bail!(
            "report {} has invalid holdout_baseline_mse={}",
            path.display(),
            input.best.holdout_baseline_mse
        );
    }
    if !input.best.holdout_relative_improvement.is_finite() && input.best.holdout_baseline_mse <= 0.0 {
        anyhow::bail!(
            "report {} has non-finite holdout_relative_improvement and baseline<=0",
            path.display()
        );
    }
    Ok(())
}

fn build_aggregate(
    channels: &[EfficacyChannel],
    bootstrap_ci: f64,
    bootstrap_resamples: usize,
    bootstrap_seed: u64,
) -> EfficacyAggregate {
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
    let ci_all = bootstrap_ci_for_median(&all, bootstrap_ci, bootstrap_resamples, bootstrap_seed);
    let ci_raw =
        bootstrap_ci_for_median(&raw, bootstrap_ci, bootstrap_resamples, bootstrap_seed + 1);
    let ci_derived = bootstrap_ci_for_median(
        &derived,
        bootstrap_ci,
        bootstrap_resamples,
        bootstrap_seed + 2,
    );
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
    let (evidence_tier, science_claim_allowed) =
        classify_evidence_tier(n_channels, n_effective, median_all, ci_all.map(|(lo, _)| lo));

    EfficacyAggregate {
        n_channels,
        n_effective,
        effective_rate,
        median_relative_improvement_all: median_all,
        median_relative_improvement_all_ci_low: ci_all.map(|(lo, _)| lo),
        median_relative_improvement_all_ci_high: ci_all.map(|(_, hi)| hi),
        median_relative_improvement_raw: median_raw,
        median_relative_improvement_raw_ci_low: ci_raw.map(|(lo, _)| lo),
        median_relative_improvement_raw_ci_high: ci_raw.map(|(_, hi)| hi),
        median_relative_improvement_derived: median_derived,
        median_relative_improvement_derived_ci_low: ci_derived.map(|(lo, _)| lo),
        median_relative_improvement_derived_ci_high: ci_derived.map(|(_, hi)| hi),
        info_value_delta_raw_minus_derived: info_delta,
        best_channel,
        worst_channel,
        claim_status,
        evidence_tier: evidence_tier.to_string(),
        science_claim_allowed,
    }
}

fn classify_evidence_tier(
    n_channels: usize,
    n_effective: usize,
    median_all: f64,
    ci_low: Option<f64>,
) -> (&'static str, bool) {
    if n_channels == 0 || n_effective == 0 {
        return ("engineering_only", false);
    }
    if n_effective == n_channels && median_all > 0.0 && ci_low.map_or(false, |v| v > 0.0) {
        return ("physics_candidate", true);
    }
    ("predictive_signal", false)
}

pub(crate) fn run_compare(
    before: PathBuf,
    after: PathBuf,
    out: PathBuf,
    markdown_out: Option<PathBuf>,
    non_regression_tol: f64,
) -> anyhow::Result<()> {
    if !non_regression_tol.is_finite() || non_regression_tol < 0.0 {
        anyhow::bail!("non_regression_tol must be finite and nonnegative");
    }

    let before_rep = read_efficacy_report(&before)?;
    let after_rep = read_efficacy_report(&after)?;

    let before_by_column = before_rep
        .channels
        .iter()
        .map(|c| {
            (
                c.column.clone(),
                (c.holdout_relative_improvement, c.effective),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let after_by_column = after_rep
        .channels
        .iter()
        .map(|c| {
            (
                c.column.clone(),
                (c.holdout_relative_improvement, c.effective),
            )
        })
        .collect::<BTreeMap<_, _>>();

    let mut all_columns = before_by_column
        .keys()
        .chain(after_by_column.keys())
        .cloned()
        .collect::<Vec<_>>();
    all_columns.sort();
    all_columns.dedup();

    let mut channels = Vec::with_capacity(all_columns.len());
    let mut shared_count = 0usize;
    let mut shared_non_reg_count = 0usize;
    for column in all_columns {
        let before_row = before_by_column.get(&column).copied();
        let after_row = after_by_column.get(&column).copied();
        let rel_delta = match (before_row, after_row) {
            (Some((b, _)), Some((a, _))) => Some(a - b),
            _ => None,
        };
        let channel_non_regression = match (before_row, after_row) {
            (Some((b, _)), Some((a, _))) => {
                shared_count += 1;
                let ok = a + non_regression_tol >= b;
                if ok {
                    shared_non_reg_count += 1;
                }
                Some(ok)
            }
            _ => None,
        };
        channels.push(CompareChannelDelta {
            column,
            before_relative_improvement: before_row.map(|v| v.0),
            after_relative_improvement: after_row.map(|v| v.0),
            relative_improvement_delta: rel_delta,
            before_effective: before_row.map(|v| v.1),
            after_effective: after_row.map(|v| v.1),
            channel_non_regression,
        });
    }

    channels.sort_by(|a, b| {
        b.relative_improvement_delta
            .unwrap_or(f64::NEG_INFINITY)
            .partial_cmp(&a.relative_improvement_delta.unwrap_or(f64::NEG_INFINITY))
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.column.cmp(&b.column))
    });

    let aggregate = CompareAggregateDelta {
        median_relative_improvement_all_before: before_rep
            .aggregate
            .median_relative_improvement_all,
        median_relative_improvement_all_after: after_rep.aggregate.median_relative_improvement_all,
        median_relative_improvement_all_delta: after_rep.aggregate.median_relative_improvement_all
            - before_rep.aggregate.median_relative_improvement_all,
        median_relative_improvement_raw_before: before_rep
            .aggregate
            .median_relative_improvement_raw,
        median_relative_improvement_raw_after: after_rep.aggregate.median_relative_improvement_raw,
        median_relative_improvement_raw_delta: delta_opt(
            before_rep.aggregate.median_relative_improvement_raw,
            after_rep.aggregate.median_relative_improvement_raw,
        ),
        median_relative_improvement_derived_before: before_rep
            .aggregate
            .median_relative_improvement_derived,
        median_relative_improvement_derived_after: after_rep
            .aggregate
            .median_relative_improvement_derived,
        median_relative_improvement_derived_delta: delta_opt(
            before_rep.aggregate.median_relative_improvement_derived,
            after_rep.aggregate.median_relative_improvement_derived,
        ),
        info_value_delta_before: before_rep.aggregate.info_value_delta_raw_minus_derived,
        info_value_delta_after: after_rep.aggregate.info_value_delta_raw_minus_derived,
        info_value_delta_change: delta_opt(
            before_rep.aggregate.info_value_delta_raw_minus_derived,
            after_rep.aggregate.info_value_delta_raw_minus_derived,
        ),
        n_effective_before: before_rep.aggregate.n_effective,
        n_effective_after: after_rep.aggregate.n_effective,
        n_effective_delta: after_rep.aggregate.n_effective as isize
            - before_rep.aggregate.n_effective as isize,
        effective_rate_before: before_rep.aggregate.effective_rate,
        effective_rate_after: after_rep.aggregate.effective_rate,
        effective_rate_delta: after_rep.aggregate.effective_rate
            - before_rep.aggregate.effective_rate,
    };

    let aggregate_non_regression = aggregate.median_relative_improvement_all_after
        + non_regression_tol
        >= aggregate.median_relative_improvement_all_before;
    let effective_count_non_regression =
        aggregate.n_effective_after >= aggregate.n_effective_before;
    let channel_non_regression_rate = if shared_count > 0 {
        shared_non_reg_count as f64 / shared_count as f64
    } else {
        1.0
    };
    let overall_non_regression = aggregate_non_regression
        && effective_count_non_regression
        && channel_non_regression_rate >= 1.0;

    let compare = EfficacyCompareReport {
        version: PREDICTABILITY_VERSION.to_string(),
        before_report_path: before.display().to_string(),
        after_report_path: after.display().to_string(),
        non_regression_tol,
        aggregate,
        channels,
        flags: CompareFlags {
            aggregate_non_regression,
            effective_count_non_regression,
            channel_non_regression_rate,
            overall_non_regression,
        },
        notes: vec![
            "Comparison is strictly before-vs-after on persisted efficacy reports produced from holdout metrics."
                .to_string(),
            "overall_non_regression requires: no aggregate median decline beyond tolerance, no effective-channel loss, and no per-channel decline on shared channels."
                .to_string(),
        ],
    };

    fs::write(&out, serde_json::to_string_pretty(&compare)?)?;
    if let Some(md_path) = markdown_out {
        let md = render_compare_markdown(&compare);
        fs::write(&md_path, md)?;
        eprintln!(
            "wrote efficacy comparison: {} and {}",
            out.display(),
            md_path.display()
        );
    } else {
        eprintln!("wrote efficacy comparison: {}", out.display());
    }

    Ok(())
}

fn read_efficacy_report(path: &PathBuf) -> anyhow::Result<InputEfficacyReport> {
    if !path.exists() {
        anyhow::bail!("report not found: {}", path.display());
    }
    let raw = fs::read_to_string(path)?;
    let parsed: InputEfficacyReport = serde_json::from_str(&raw).map_err(|e| {
        anyhow::anyhow!(
            "failed to parse efficacy report {} as JSON: {}",
            path.display(),
            e
        )
    })?;
    Ok(parsed)
}

fn delta_opt(before: Option<f64>, after: Option<f64>) -> Option<f64> {
    match (before, after) {
        (Some(b), Some(a)) => Some(a - b),
        _ => None,
    }
}

fn render_compare_markdown(report: &EfficacyCompareReport) -> String {
    let mut out = String::new();
    out.push_str("# Efficacy Comparison\n\n");
    out.push_str(&format!(
        "- before: `{}`\n- after: `{}`\n- tolerance: {:.3e}\n\n",
        report.before_report_path, report.after_report_path, report.non_regression_tol
    ));
    out.push_str("## Aggregate\n\n");
    out.push_str(&format!(
        "- median improvement (all): before={:.6e} after={:.6e} delta={:.6e}\n",
        report.aggregate.median_relative_improvement_all_before,
        report.aggregate.median_relative_improvement_all_after,
        report.aggregate.median_relative_improvement_all_delta
    ));
    out.push_str(&format!(
        "- effective channels: before={} after={} delta={}\n",
        report.aggregate.n_effective_before,
        report.aggregate.n_effective_after,
        report.aggregate.n_effective_delta
    ));
    out.push_str(&format!(
        "- effective rate: before={:.6e} after={:.6e} delta={:.6e}\n",
        report.aggregate.effective_rate_before,
        report.aggregate.effective_rate_after,
        report.aggregate.effective_rate_delta
    ));
    out.push_str("\n## Flags\n\n");
    out.push_str(&format!(
        "- aggregate non-regression: `{}`\n- effective-count non-regression: `{}`\n- channel non-regression rate: {:.3}\n- overall non-regression: `{}`\n",
        report.flags.aggregate_non_regression,
        report.flags.effective_count_non_regression,
        report.flags.channel_non_regression_rate,
        report.flags.overall_non_regression
    ));
    out.push_str("\n## Channels\n\n");
    out.push_str(
        "| column | before rel | after rel | delta | non-regression |\n|---|---:|---:|---:|---:|\n",
    );
    for c in &report.channels {
        out.push_str(&format!(
            "| `{}` | {} | {} | {} | `{}` |\n",
            c.column,
            fmt_opt(c.before_relative_improvement),
            fmt_opt(c.after_relative_improvement),
            fmt_opt(c.relative_improvement_delta),
            c.channel_non_regression
                .map(|v| v.to_string())
                .unwrap_or_else(|| "n/a".to_string())
        ));
    }
    out
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.6e}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn bootstrap_ci_for_median(
    xs: &[f64],
    ci_level: f64,
    resamples: usize,
    seed: u64,
) -> Option<(f64, f64)> {
    if xs.len() < 2 || resamples == 0 {
        return None;
    }
    if !ci_level.is_finite() || ci_level <= 0.0 || ci_level >= 1.0 {
        return None;
    }
    let alpha = 0.5 * (1.0 - ci_level);
    let n = xs.len();
    let mut rng = XorShift64::new(seed);
    let mut meds = Vec::with_capacity(resamples);
    let mut sample = vec![0.0; n];
    for _ in 0..resamples {
        for item in &mut sample {
            let idx = rng.next_index(n);
            *item = xs[idx];
        }
        sample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        meds.push(median_sorted(&sample));
    }
    meds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n_m = meds.len();
    let lo_idx = ((alpha * (n_m.saturating_sub(1) as f64)).floor() as usize).min(n_m - 1);
    let hi_idx = ((((1.0 - alpha) * (n_m.saturating_sub(1) as f64)).ceil()) as usize).min(n_m - 1);
    Some((meds[lo_idx], meds[hi_idx]))
}

fn median_sorted(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

#[derive(Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let init = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state: init }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_index(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
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
        "- bootstrap: enabled=`{}` ci_level={:.3} resamples={} seed={}\n",
        report.bootstrap.enabled,
        report.bootstrap.ci_level,
        report.bootstrap.resamples,
        report.bootstrap.seed
    ));
    out.push_str(&format!(
        "- channels: {}\n- effective channels: {} ({:.1}%)\n- claim status: `{}`\n- evidence tier: `{}`\n- science claim allowed: `{}`\n\n",
        report.aggregate.n_channels,
        report.aggregate.n_effective,
        100.0 * report.aggregate.effective_rate,
        report.aggregate.claim_status,
        report.aggregate.evidence_tier,
        report.aggregate.science_claim_allowed
    ));
    out.push_str(&format!(
        "- median improvement (all): {:.6e} (CI: {} to {})\n",
        report.aggregate.median_relative_improvement_all,
        fmt_opt(report.aggregate.median_relative_improvement_all_ci_low),
        fmt_opt(report.aggregate.median_relative_improvement_all_ci_high)
    ));
    if let Some(v) = report.aggregate.median_relative_improvement_raw {
        out.push_str(&format!(
            "- median improvement (raw): {:.6e} (CI: {} to {})\n",
            v,
            fmt_opt(report.aggregate.median_relative_improvement_raw_ci_low),
            fmt_opt(report.aggregate.median_relative_improvement_raw_ci_high)
        ));
    }
    if let Some(v) = report.aggregate.median_relative_improvement_derived {
        out.push_str(&format!(
            "- median improvement (derived): {:.6e} (CI: {} to {})\n",
            v,
            fmt_opt(report.aggregate.median_relative_improvement_derived_ci_low),
            fmt_opt(report.aggregate.median_relative_improvement_derived_ci_high)
        ));
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
    use std::env;
    use std::fs;

    use super::{
        build_aggregate, classify_column, is_raw_state_column, run_compare, run_report, ChannelKind,
        EfficacyChannel,
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
            row(
                "min_pair_dist",
                ChannelKind::DerivedDiagnostic,
                -0.5,
                1.0,
                false,
            ),
            row(
                "max_accel",
                ChannelKind::DerivedDiagnostic,
                -0.4,
                1.0,
                false,
            ),
        ];

        let agg = build_aggregate(&channels, 0.95, 512, 42);
        assert_eq!(agg.n_channels, 4);
        assert_eq!(agg.n_effective, 2);
        assert!(agg.effective_rate > 0.4);
        assert_eq!(agg.best_channel.as_deref(), Some("a1_x"));
        assert_eq!(agg.worst_channel.as_deref(), Some("min_pair_dist"));
        let delta = agg.info_value_delta_raw_minus_derived.unwrap();
        assert!(
            delta > 1.0,
            "expected strong raw-vs-derived gap, got {delta}"
        );
        assert_eq!(agg.claim_status, "information_helpful_in_some_channels");
        assert_eq!(agg.evidence_tier, "predictive_signal");
        assert!(!agg.science_claim_allowed);
    }

    #[test]
    fn aggregate_no_effective_channels_is_no_information_gain() {
        let channels = vec![
            row(
                "min_pair_dist",
                ChannelKind::DerivedDiagnostic,
                -0.2,
                0.9,
                false,
            ),
            row(
                "max_accel",
                ChannelKind::DerivedDiagnostic,
                -0.1,
                0.8,
                false,
            ),
        ];
        let agg = build_aggregate(&channels, 0.95, 512, 42);
        assert_eq!(agg.n_effective, 0);
        assert_eq!(agg.claim_status, "no_information_gain");
        assert_eq!(agg.evidence_tier, "engineering_only");
        assert!(!agg.science_claim_allowed);
    }

    #[test]
    fn aggregate_bootstrap_ci_brackets_median() {
        let channels = vec![
            row("a1_x", ChannelKind::RawState, 0.95, 0.01, true),
            row("v1_x", ChannelKind::RawState, 0.92, 0.02, true),
            row(
                "min_pair_dist",
                ChannelKind::DerivedDiagnostic,
                0.89,
                0.03,
                true,
            ),
            row(
                "max_accel",
                ChannelKind::DerivedDiagnostic,
                0.91,
                0.03,
                true,
            ),
        ];
        let agg = build_aggregate(&channels, 0.95, 2000, 7);
        let lo = agg
            .median_relative_improvement_all_ci_low
            .expect("all ci low");
        let hi = agg
            .median_relative_improvement_all_ci_high
            .expect("all ci high");
        assert!(lo <= agg.median_relative_improvement_all);
        assert!(agg.median_relative_improvement_all <= hi);
    }

    #[test]
    fn aggregate_marks_physics_candidate_when_all_channels_clear_positive_ci() {
        let channels = vec![
            row("a1_x", ChannelKind::RawState, 0.95, 0.01, true),
            row("v1_x", ChannelKind::RawState, 0.92, 0.02, true),
            row("r1_x", ChannelKind::RawState, 0.90, 0.02, true),
            row(
                "min_pair_dist",
                ChannelKind::DerivedDiagnostic,
                0.89,
                0.03,
                true,
            ),
        ];
        let agg = build_aggregate(&channels, 0.95, 2000, 11);
        assert_eq!(agg.n_channels, agg.n_effective);
        assert_eq!(agg.evidence_tier, "physics_candidate");
        assert!(agg.science_claim_allowed);
    }

    #[test]
    fn run_report_rejects_missing_column_schema() {
        let tmp_dir = env::temp_dir().join(format!(
            "threebody_report_schema_unit_{}",
            std::process::id()
        ));
        if tmp_dir.exists() {
            let _ = fs::remove_dir_all(&tmp_dir);
        }
        fs::create_dir_all(&tmp_dir).expect("create temp dir");
        let bad = tmp_dir.join("bad_report.json");
        let out = tmp_dir.join("out.json");
        fs::write(
            &bad,
            r#"{
  "best": {
    "config": {"model":"linear"},
    "holdout_mse": 1.0,
    "holdout_baseline_mse": 2.0,
    "holdout_relative_improvement": 0.5,
    "holdout_sensitivity": {"median_rel_error": 0.01}
  }
}"#,
        )
        .expect("write bad report");
        let err = run_report(vec![bad], out, None, 0.0, 0.1, 128, 0.95, 42).unwrap_err();
        assert!(
            err.to_string().contains("missing column"),
            "unexpected error: {err}"
        );
        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn compare_flags_detect_improvement_and_regression() {
        let tmp_dir = env::temp_dir().join(format!(
            "threebody_report_compare_unit_{}",
            std::process::id()
        ));
        if tmp_dir.exists() {
            let _ = fs::remove_dir_all(&tmp_dir);
        }
        fs::create_dir_all(&tmp_dir).expect("create temp dir");

        let before = tmp_dir.join("before.json");
        let after = tmp_dir.join("after.json");
        let compare_out = tmp_dir.join("compare.json");
        fs::write(
            &before,
            r#"{
  "aggregate": {
    "n_effective": 1,
    "effective_rate": 0.5,
    "median_relative_improvement_all": 0.2,
    "median_relative_improvement_raw": 0.9,
    "median_relative_improvement_derived": -0.5,
    "info_value_delta_raw_minus_derived": 1.4
  },
  "channels": [
    {"column":"a1_x","holdout_relative_improvement":0.9,"effective":true},
    {"column":"min_pair_dist","holdout_relative_improvement":-0.5,"effective":false}
  ]
}"#,
        )
        .expect("write before");
        fs::write(
            &after,
            r#"{
  "aggregate": {
    "n_effective": 2,
    "effective_rate": 1.0,
    "median_relative_improvement_all": 0.95,
    "median_relative_improvement_raw": 0.97,
    "median_relative_improvement_derived": 0.92,
    "info_value_delta_raw_minus_derived": 0.05
  },
  "channels": [
    {"column":"a1_x","holdout_relative_improvement":0.95,"effective":true},
    {"column":"min_pair_dist","holdout_relative_improvement":0.92,"effective":true}
  ]
}"#,
        )
        .expect("write after");

        run_compare(
            before.clone(),
            after.clone(),
            compare_out.clone(),
            None,
            0.0,
        )
        .expect("run compare");
        let compare_json = fs::read_to_string(&compare_out).expect("read compare");
        let value: serde_json::Value = serde_json::from_str(&compare_json).expect("valid compare");
        let flags = value.get("flags").expect("flags");
        assert_eq!(
            flags
                .get("overall_non_regression")
                .and_then(|v| v.as_bool()),
            Some(true)
        );
        let agg = value.get("aggregate").expect("aggregate");
        assert!(
            agg.get("median_relative_improvement_all_delta")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
                > 0.0
        );

        // Swapping before/after should trigger regression failure.
        run_compare(after, before, compare_out.clone(), None, 0.0).expect("run compare reverse");
        let reverse_json = fs::read_to_string(&compare_out).expect("read reverse compare");
        let reverse: serde_json::Value = serde_json::from_str(&reverse_json).expect("valid json");
        let reverse_flags = reverse.get("flags").expect("flags");
        assert_eq!(
            reverse_flags
                .get("overall_non_regression")
                .and_then(|v| v.as_bool()),
            Some(false)
        );
    }
}
