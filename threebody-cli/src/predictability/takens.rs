use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufRead};
use std::path::PathBuf;

use serde::Serialize;
use threebody_core::output::parse::{parse_header, require_columns};

use crate::predictability::PREDICTABILITY_VERSION;

#[derive(Debug, Clone)]
struct DelaySample {
    z: Vec<f64>,
    x_t: f64,
    y: f64,
    target_step: usize,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum SplitMode {
    Chronological,
    Shuffled,
}

impl SplitMode {
    fn parse(raw: &str) -> anyhow::Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "chronological" | "chrono" | "time" => Ok(Self::Chronological),
            "shuffled" | "shuffle" | "random" => Ok(Self::Shuffled),
            _ => anyhow::bail!("unknown split mode: {raw} (expected chronological|shuffled)"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ModelKind {
    Linear,
    Rational,
    DeltaLinear,
    DeltaRational,
    DeltaMlp,
    DeltaTcn,
}

impl ModelKind {
    fn parse_list(raw: &str) -> anyhow::Result<Vec<Self>> {
        let mut out = Vec::new();
        for tok in raw.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
            match tok.to_ascii_lowercase().as_str() {
                "linear" => push_unique_model(&mut out, Self::Linear),
                "rational" => push_unique_model(&mut out, Self::Rational),
                "delta_linear" | "residual_linear" => push_unique_model(&mut out, Self::DeltaLinear),
                "delta_rational" | "residual_rational" => {
                    push_unique_model(&mut out, Self::DeltaRational)
                }
                "delta_mlp" | "residual_mlp" | "mlp" | "nn" | "neural" => {
                    push_unique_model(&mut out, Self::DeltaMlp)
                }
                "delta_tcn" | "residual_tcn" | "tcn" => push_unique_model(&mut out, Self::DeltaTcn),
                "delta" | "residual" => {
                    push_unique_model(&mut out, Self::DeltaLinear);
                    push_unique_model(&mut out, Self::DeltaRational);
                }
                "both" | "all" => {
                    push_unique_model(&mut out, Self::Linear);
                    push_unique_model(&mut out, Self::Rational);
                    if tok.eq_ignore_ascii_case("all") {
                        push_unique_model(&mut out, Self::DeltaLinear);
                        push_unique_model(&mut out, Self::DeltaRational);
                        push_unique_model(&mut out, Self::DeltaMlp);
                        push_unique_model(&mut out, Self::DeltaTcn);
                    }
                }
                _ => anyhow::bail!(
                    "unknown model kind: {tok} (expected linear|rational|delta_linear|delta_rational|delta_mlp|delta_tcn|both|all or CSV list)"
                ),
            }
        }
        if out.is_empty() {
            anyhow::bail!("model list is empty");
        }
        Ok(out)
    }
}

fn push_unique_model(dst: &mut Vec<ModelKind>, model: ModelKind) {
    if !dst.contains(&model) {
        dst.push(model);
    }
}

const SENSOR_GATE_ACTIVE_THRESHOLD: f64 = 0.35;
const DELTA_MLP_ENERGY_PENALTY: f64 = 0.05;

fn model_kind_name(model: ModelKind) -> &'static str {
    match model {
        ModelKind::Linear => "linear",
        ModelKind::Rational => "rational",
        ModelKind::DeltaLinear => "delta_linear",
        ModelKind::DeltaRational => "delta_rational",
        ModelKind::DeltaMlp => "delta_mlp",
        ModelKind::DeltaTcn => "delta_tcn",
    }
}

#[derive(Debug, Clone, Serialize)]
struct ModelArchitecture {
    model_family: String,
    feature_map: String,
    input_dim: usize,
    feature_dim: usize,
    depth: usize,
    width: usize,
    hidden_layers: usize,
    hidden_widths: Vec<usize>,
    parameter_count: usize,
}

#[derive(Debug, Clone)]
struct SplitIndices {
    train: Vec<usize>,
    val: Vec<usize>,
    holdout: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct SplitCounts {
    train: usize,
    val: usize,
    holdout: usize,
}

#[derive(Debug, Clone, Serialize)]
struct TakensConfig {
    tau: usize,
    m: usize,
    k: usize,
    lambda: f64,
    model: ModelKind,
    split_mode: SplitMode,
}

fn infer_model_architecture(config: &TakensConfig, n_sensors: usize) -> anyhow::Result<ModelArchitecture> {
    if n_sensors == 0 {
        anyhow::bail!("n_sensors must be positive for architecture inference");
    }
    let input_dim = config
        .m
        .checked_mul(n_sensors)
        .ok_or_else(|| anyhow::anyhow!("input_dim overflow for m={} sensors={}", config.m, n_sensors))?;
    let affine_dim = input_dim
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("affine_dim overflow for input_dim={input_dim}"))?;

    let arch = match config.model {
        ModelKind::Linear => ModelArchitecture {
            model_family: "local_linear".to_string(),
            feature_map: "affine".to_string(),
            input_dim,
            feature_dim: affine_dim,
            depth: 1,
            width: affine_dim,
            hidden_layers: 0,
            hidden_widths: Vec::new(),
            parameter_count: affine_dim,
        },
        ModelKind::DeltaLinear => ModelArchitecture {
            model_family: "delta_local_linear".to_string(),
            feature_map: "affine".to_string(),
            input_dim,
            feature_dim: affine_dim,
            depth: 1,
            width: affine_dim,
            hidden_layers: 0,
            hidden_widths: Vec::new(),
            parameter_count: affine_dim,
        },
        ModelKind::Rational => ModelArchitecture {
            model_family: "local_rational".to_string(),
            feature_map: "affine_rational".to_string(),
            input_dim,
            feature_dim: affine_dim,
            depth: 1,
            width: affine_dim,
            hidden_layers: 0,
            hidden_widths: Vec::new(),
            parameter_count: affine_dim
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("parameter_count overflow for rational model"))?,
        },
        ModelKind::DeltaRational => ModelArchitecture {
            model_family: "delta_local_rational".to_string(),
            feature_map: "affine_rational".to_string(),
            input_dim,
            feature_dim: affine_dim,
            depth: 1,
            width: affine_dim,
            hidden_layers: 0,
            hidden_widths: Vec::new(),
            parameter_count: affine_dim
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("parameter_count overflow for delta_rational model"))?,
        },
        ModelKind::DeltaMlp | ModelKind::DeltaTcn => {
            let delta_dim = n_sensors
                .checked_mul(config.m.saturating_sub(1))
                .ok_or_else(|| anyhow::anyhow!("delta feature dimension overflow"))?;
            let tcn_extra = if matches!(config.model, ModelKind::DeltaTcn) {
                n_sensors
                    .checked_mul(config.m.saturating_sub(2))
                    .and_then(|v| v.checked_mul(2))
                    .ok_or_else(|| anyhow::anyhow!("tcn feature dimension overflow"))?
            } else {
                0
            };
            let feature_dim = input_dim
                .checked_add(delta_dim)
                .and_then(|v| v.checked_add(tcn_extra))
                .ok_or_else(|| anyhow::anyhow!("feature_dim overflow for neural model"))?;
            let h1 = config.k.clamp(4, 64);
            let h2 = (h1 / 2).clamp(4, 32);
            let layer1 = h1
                .checked_mul(feature_dim)
                .and_then(|v| v.checked_add(h1))
                .ok_or_else(|| anyhow::anyhow!("layer1 parameter overflow"))?;
            let layer2 = h2
                .checked_mul(h1)
                .and_then(|v| v.checked_add(h2))
                .ok_or_else(|| anyhow::anyhow!("layer2 parameter overflow"))?;
            let layer3 = h2
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("layer3 parameter overflow"))?;
            let parameter_count = layer1
                .checked_add(layer2)
                .and_then(|v| v.checked_add(layer3))
                .ok_or_else(|| anyhow::anyhow!("total neural parameter overflow"))?;
            ModelArchitecture {
                model_family: "delta_neural".to_string(),
                feature_map: if matches!(config.model, ModelKind::DeltaTcn) {
                    "temporal_conv".to_string()
                } else {
                    "delta_window".to_string()
                },
                input_dim,
                feature_dim,
                depth: 3,
                width: h1.max(h2),
                hidden_layers: 2,
                hidden_widths: vec![h1, h2],
                parameter_count,
            }
        }
    };
    Ok(arch)
}

#[derive(Debug, Clone, Serialize)]
struct SensitivitySummary {
    n_pairs: usize,
    median_rel_error: f64,
    p90_rel_error: f64,
    max_rel_error: f64,
}

#[derive(Debug, Clone, Serialize)]
struct GridEval {
    config: TakensConfig,
    architecture: ModelArchitecture,
    n_embedded: usize,
    split: SplitCounts,
    rank_score: f64,
    val_mse: f64,
    val_sensitivity_median: f64,
    holdout_mse: f64,
    holdout_baseline_mse: f64,
    holdout_sensitivity_median: f64,
}

#[derive(Debug, Clone, Serialize)]
struct BestEval {
    config: TakensConfig,
    architecture: ModelArchitecture,
    n_embedded: usize,
    split: SplitCounts,
    rank_score: f64,
    val_mse: f64,
    val_sensitivity: SensitivitySummary,
    holdout_mse: f64,
    holdout_baseline_mse: f64,
    holdout_relative_improvement: f64,
    holdout_sensitivity: SensitivitySummary,
    horizon_metrics: Vec<HorizonMetric>,
}

#[derive(Debug, Clone, Serialize)]
struct HorizonMetric {
    horizon: usize,
    n_eval: usize,
    holdout_mse: f64,
    holdout_baseline_mse: f64,
    holdout_relative_improvement: f64,
}

#[derive(Debug, Clone, Serialize)]
struct SensorGateValue {
    sensor: String,
    gate: f64,
}

#[derive(Debug, Clone, Serialize)]
struct SensorGateSummary {
    active_threshold: f64,
    active_count: usize,
    active_fraction: f64,
    min_gate: f64,
    max_gate: f64,
    mean_gate: f64,
    effective_sensor_count: f64,
}

#[derive(Debug, Clone, Serialize)]
struct DeltaMlpDiagnostics {
    config: TakensConfig,
    architecture: ModelArchitecture,
    n_embedded: usize,
    split: SplitCounts,
    val_mse: f64,
    holdout_mse: f64,
    holdout_baseline_mse: f64,
    holdout_relative_improvement: f64,
    val_sensitivity_median: f64,
    holdout_sensitivity_median: f64,
    target_is_energy: bool,
    energy_penalty: f64,
    gate_summary: SensorGateSummary,
    sensor_gates_ranked: Vec<SensorGateValue>,
}

#[derive(Debug, Clone, Serialize)]
struct TakensReport {
    version: String,
    input_csv: String,
    column: String,
    sensors: Vec<String>,
    n_raw: usize,
    grid_evals: usize,
    model_eval_counts: BTreeMap<String, usize>,
    sensitivity_weight: f64,
    best: BestEval,
    delta_mlp_diagnostics: Option<DeltaMlpDiagnostics>,
    notes: Vec<String>,
}

pub(crate) fn run_takens(
    input: PathBuf,
    column: String,
    sensors_raw: String,
    out: PathBuf,
    tau_raw: String,
    m_raw: String,
    k_raw: String,
    lambda_raw: String,
    model_raw: String,
    split_mode_raw: String,
    seed: u64,
    train_frac: f64,
    val_frac: f64,
    sensitivity_weight: f64,
) -> anyhow::Result<()> {
    if !input.exists() {
        anyhow::bail!("input CSV not found: {}", input.display());
    }
    if !sensitivity_weight.is_finite() || sensitivity_weight < 0.0 {
        anyhow::bail!(
            "sensitivity_weight must be finite and nonnegative, got {}",
            sensitivity_weight
        );
    }

    let split_mode = SplitMode::parse(&split_mode_raw)?;
    let taus = parse_csv_usize_list(&tau_raw, "tau")?;
    let ms = parse_csv_usize_list(&m_raw, "m")?;
    let ks = parse_csv_usize_list(&k_raw, "k")?;
    let lambdas = parse_csv_f64_list(&lambda_raw, "lambda")?;
    let models = ModelKind::parse_list(&model_raw)?;

    let mut sensors = parse_csv_string_list(&sensors_raw)?;
    if sensors.is_empty() {
        sensors.push(column.clone());
    } else if !sensors.iter().any(|s| s == &column) {
        sensors.insert(0, column.clone());
    }
    let n_sensors = sensors.len().max(1);
    let target_is_energy = column == "energy_proxy";

    let aligned = read_aligned_target_and_sensors(&input, &column, &sensors)?;
    if aligned.target.len() < 16 {
        anyhow::bail!(
            "series too short in column {column}: {} (need at least 16 samples)",
            aligned.target.len()
        );
    }

    let mut all_rows = Vec::new();
    let mut best_rank_score = f64::INFINITY;
    let mut best_val_sensitivity = None;
    let mut best_holdout_sensitivity = None;

    for &tau in &taus {
        for &m in &ms {
            let samples = build_delay_embedding_multi(&aligned.target, &aligned.sensors, tau, m)?;
            if samples.len() < 8 {
                continue;
            }
            let split = split_indices(samples.len(), split_mode, seed, train_frac, val_frac)?;
            if matches!(split_mode, SplitMode::Chronological) {
                assert_no_chronological_leakage(&samples, &split)?;
            }
            for &k in &ks {
                for &lambda in &lambdas {
                    for &model in &models {
                        let val_mse = evaluate_model_mse(
                            &samples,
                            &split.train,
                            &split.val,
                            k,
                            lambda,
                            model,
                            n_sensors,
                            target_is_energy,
                        )?;
                        let holdout_mse = evaluate_model_mse(
                            &samples,
                            &split.train,
                            &split.holdout,
                            k,
                            lambda,
                            model,
                            n_sensors,
                            target_is_energy,
                        )?;
                        let holdout_baseline_mse =
                            evaluate_persistence_mse(&samples, &split.holdout)?;
                        let val_sensitivity = evaluate_sensitivity_summary(
                            &samples,
                            &split.train,
                            &split.val,
                            k,
                            lambda,
                            model,
                            n_sensors,
                            target_is_energy,
                        )?;
                        let holdout_sensitivity = evaluate_sensitivity_summary(
                            &samples,
                            &split.train,
                            &split.holdout,
                            k,
                            lambda,
                            model,
                            n_sensors,
                            target_is_energy,
                        )?;
                        let rank_score = val_mse
                            * (1.0
                                + sensitivity_weight * val_sensitivity.median_rel_error.max(0.0));

                        let config = TakensConfig {
                            tau,
                            m,
                            k,
                            lambda,
                            model,
                            split_mode,
                        };
                        let architecture = infer_model_architecture(&config, n_sensors)?;
                        let row = GridEval {
                            config,
                            architecture,
                            n_embedded: samples.len(),
                            split: SplitCounts {
                                train: split.train.len(),
                                val: split.val.len(),
                                holdout: split.holdout.len(),
                            },
                            rank_score,
                            val_mse,
                            val_sensitivity_median: val_sensitivity.median_rel_error,
                            holdout_mse,
                            holdout_baseline_mse,
                            holdout_sensitivity_median: holdout_sensitivity.median_rel_error,
                        };

                        if rank_score < best_rank_score {
                            best_rank_score = rank_score;
                            best_val_sensitivity = Some(val_sensitivity);
                            best_holdout_sensitivity = Some(holdout_sensitivity);
                        }
                        all_rows.push(row);
                    }
                }
            }
        }
    }

    if all_rows.is_empty() {
        anyhow::bail!("no valid Takens grid rows were produced");
    }
    all_rows.sort_by(|a, b| {
        a.rank_score
            .partial_cmp(&b.rank_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.val_mse.partial_cmp(&b.val_mse).unwrap_or(Ordering::Equal))
    });

    let best_row = &all_rows[0];
    let best_val_sensitivity = best_val_sensitivity
        .ok_or_else(|| anyhow::anyhow!("missing validation sensitivity for best model"))?;
    let best_holdout_sensitivity = best_holdout_sensitivity
        .ok_or_else(|| anyhow::anyhow!("missing holdout sensitivity for best model"))?;

    let rel = if best_row.holdout_baseline_mse > 0.0 {
        (best_row.holdout_baseline_mse - best_row.holdout_mse) / best_row.holdout_baseline_mse
    } else {
        0.0
    };
    let mut horizon_metrics = Vec::new();
    let best_samples = build_delay_embedding_multi(
        &aligned.target,
        &aligned.sensors,
        best_row.config.tau,
        best_row.config.m,
    )?;
    let best_split = split_indices(
        best_samples.len(),
        best_row.config.split_mode,
        seed,
        train_frac,
        val_frac,
    )?;
    if matches!(best_row.config.split_mode, SplitMode::Chronological) {
        assert_no_chronological_leakage(&best_samples, &best_split)?;
    }
    for horizon in [1usize, 2, 4, 8] {
        if let Some(metric) = evaluate_recursive_horizon_metric(
            &best_samples,
            &aligned.target,
            &best_split.train,
            &best_split.holdout,
            &best_row.config,
            n_sensors,
            target_is_energy,
            horizon,
        )? {
            horizon_metrics.push(metric);
        }
    }

    let best = BestEval {
        config: best_row.config.clone(),
        architecture: best_row.architecture.clone(),
        n_embedded: best_row.n_embedded,
        split: best_row.split.clone(),
        rank_score: best_row.rank_score,
        val_mse: best_row.val_mse,
        val_sensitivity: best_val_sensitivity,
        holdout_mse: best_row.holdout_mse,
        holdout_baseline_mse: best_row.holdout_baseline_mse,
        holdout_relative_improvement: rel,
        holdout_sensitivity: best_holdout_sensitivity,
        horizon_metrics,
    };

    let mut model_eval_counts = BTreeMap::<String, usize>::new();
    for row in &all_rows {
        *model_eval_counts
            .entry(model_kind_name(row.config.model).to_string())
            .or_insert(0) += 1;
    }

    let mut notes = vec![
        "Takens embedding uses one-step target x_{t+1} and delay coordinates from selected sensor columns."
            .to_string(),
        "Model families include local linear, local rational, residual local variants (delta_*), and temporal neural baselines (delta_mlp, delta_tcn)."
            .to_string(),
        "Each row now includes architecture metadata (model_family, depth/width, feature_dim, parameter_count)."
            .to_string(),
        "Primary score is strict holdout MSE; baseline is persistence y_hat=x_t.".to_string(),
        "Selection rank_score = val_mse * (1 + sensitivity_weight * val_sensitivity_median)."
            .to_string(),
        "Chronological split enforces no-future-leakage across train/val/holdout.".to_string(),
    ];
    let delta_mlp_row = all_rows
        .iter()
        .find(|row| matches!(row.config.model, ModelKind::DeltaMlp))
        .cloned();
    let delta_mlp_diagnostics = if let Some(delta_row) = delta_mlp_row {
        let samples =
            build_delay_embedding_multi(&aligned.target, &aligned.sensors, delta_row.config.tau, delta_row.config.m)?;
        let split = split_indices(
            samples.len(),
            delta_row.config.split_mode,
            seed,
            train_frac,
            val_frac,
        )?;
        if matches!(delta_row.config.split_mode, SplitMode::Chronological) {
            assert_no_chronological_leakage(&samples, &split)?;
        }
        match build_delta_mlp_diagnostics(
            &samples,
            &split,
            &delta_row,
            &sensors,
            target_is_energy,
        ) {
            Ok(diag) => Some(diag),
            Err(err) => {
                notes.push(format!("delta_mlp diagnostics unavailable: {}", err));
                None
            }
        }
    } else {
        None
    };
    if best.horizon_metrics.is_empty() {
        notes.push(
            "multi-horizon metrics unavailable for this best config (requires single-sensor embedding with tau=1 and sufficient holdout continuity).".to_string(),
        );
    }

    let report = TakensReport {
        version: PREDICTABILITY_VERSION.to_string(),
        input_csv: input.display().to_string(),
        column: column.clone(),
        sensors,
        n_raw: aligned.target.len(),
        grid_evals: all_rows.len(),
        model_eval_counts,
        sensitivity_weight,
        best,
        delta_mlp_diagnostics,
        notes,
    };

    fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    eprintln!(
        "wrote Takens report: {} | sensors={} model={:?} holdout_mse={:.6e} baseline={:.6e} rel_impr={:.6} sens_med={:.6}",
        out.display(),
        report.sensors.len(),
        report.best.config.model,
        report.best.holdout_mse,
        report.best.holdout_baseline_mse,
        report.best.holdout_relative_improvement,
        report.best.holdout_sensitivity.median_rel_error,
    );
    Ok(())
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

fn parse_csv_f64_list(raw: &str, field: &str) -> anyhow::Result<Vec<f64>> {
    let mut out = Vec::new();
    for tok in raw.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        let v: f64 = tok
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid {field} value {tok:?}: {e}"))?;
        if !v.is_finite() || v < 0.0 {
            anyhow::bail!("{field} values must be finite and nonnegative");
        }
        out.push(v);
    }
    if out.is_empty() {
        anyhow::bail!("{field} list is empty");
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    out.dedup_by(|a, b| (*a - *b).abs() <= 1e-15);
    Ok(out)
}

fn parse_csv_string_list(raw: &str) -> anyhow::Result<Vec<String>> {
    let mut out = Vec::new();
    for tok in raw.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        if tok.contains(',') {
            anyhow::bail!("invalid sensor token with comma: {tok:?}");
        }
        let candidate = tok.to_string();
        if !out.iter().any(|s| s == &candidate) {
            out.push(candidate);
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
struct AlignedSeries {
    target: Vec<f64>,
    sensors: Vec<Vec<f64>>,
}

fn read_aligned_target_and_sensors(
    input: &PathBuf,
    target_column: &str,
    sensors: &[String],
) -> anyhow::Result<AlignedSeries> {
    if sensors.is_empty() {
        anyhow::bail!("sensor list is empty");
    }

    let mut requested = Vec::<String>::new();
    requested.push(target_column.to_string());
    for s in sensors {
        if !requested.iter().any(|x| x == s) {
            requested.push(s.clone());
        }
    }

    let file = fs::File::open(input)?;
    let mut reader = io::BufReader::new(file);
    let mut header_line = String::new();
    if reader.read_line(&mut header_line)? == 0 {
        anyhow::bail!("empty CSV: {}", input.display());
    }
    let header = parse_header(&header_line);
    let requested_ref: Vec<&str> = requested.iter().map(|s| s.as_str()).collect();
    let map = require_columns(&header, &requested_ref).map_err(anyhow::Error::msg)?;

    let mut cols_out = vec![Vec::<f64>::new(); requested.len()];
    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row_cols: Vec<&str> = line.split(',').collect();
        let mut row_values = Vec::with_capacity(requested.len());
        let mut finite = true;
        for name in &requested {
            let idx = *map
                .get(name.as_str())
                .ok_or_else(|| anyhow::anyhow!("missing required column mapping for {}", name))?;
            let raw = row_cols.get(idx).copied().ok_or_else(|| {
                anyhow::anyhow!(
                    "row {} missing column {} in {}",
                    line_no + 2,
                    name,
                    input.display()
                )
            })?;
            let v: f64 = raw.parse().map_err(|e| {
                anyhow::anyhow!(
                    "row {} invalid {}={:?} in {}: {}",
                    line_no + 2,
                    name,
                    raw,
                    input.display(),
                    e
                )
            })?;
            if !v.is_finite() {
                finite = false;
                break;
            }
            row_values.push(v);
        }
        if finite && row_values.len() == requested.len() {
            for (dst, v) in cols_out.iter_mut().zip(row_values.into_iter()) {
                dst.push(v);
            }
        }
    }

    let target = cols_out
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("target column data not found"))?;
    if target.is_empty() {
        anyhow::bail!(
            "no finite aligned rows for target={} sensors={} in {}",
            target_column,
            sensors.join(","),
            input.display()
        );
    }

    let mut sensor_out = Vec::with_capacity(sensors.len());
    for sensor in sensors {
        let idx = requested
            .iter()
            .position(|name| name == sensor)
            .ok_or_else(|| anyhow::anyhow!("sensor {} not found in requested columns", sensor))?;
        let series = cols_out
            .get(idx)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("sensor index out of bounds for {}", sensor))?;
        sensor_out.push(series);
    }

    Ok(AlignedSeries {
        target,
        sensors: sensor_out,
    })
}

#[cfg(test)]
fn build_delay_embedding(series: &[f64], tau: usize, m: usize) -> anyhow::Result<Vec<DelaySample>> {
    let sensors = vec![series.to_vec()];
    build_delay_embedding_multi(series, &sensors, tau, m)
}

fn build_delay_embedding_multi(
    target: &[f64],
    sensors: &[Vec<f64>],
    tau: usize,
    m: usize,
) -> anyhow::Result<Vec<DelaySample>> {
    if tau == 0 || m == 0 {
        anyhow::bail!("tau and m must be positive");
    }
    if sensors.is_empty() {
        anyhow::bail!("need at least one sensor column");
    }
    for (idx, s) in sensors.iter().enumerate() {
        if s.len() != target.len() {
            anyhow::bail!(
                "sensor length mismatch at index {}: sensor_len={} target_len={}",
                idx,
                s.len(),
                target.len()
            );
        }
    }
    let lag = (m - 1) * tau;
    if target.len() <= lag + 1 {
        anyhow::bail!(
            "series too short for tau={}, m={} (need > {})",
            tau,
            m,
            lag + 1
        );
    }
    let mut out = Vec::new();
    for t in lag..(target.len() - 1) {
        let mut z = Vec::with_capacity(m * sensors.len());
        for j in 0..m {
            let src = t - j * tau;
            for s in sensors {
                z.push(s[src]);
            }
        }
        let x_t = target[t];
        let y = target[t + 1];
        if z.iter().all(|v| v.is_finite()) && y.is_finite() && x_t.is_finite() {
            out.push(DelaySample {
                z,
                x_t,
                y,
                target_step: t + 1,
            });
        }
    }
    Ok(out)
}

fn split_indices(
    n: usize,
    mode: SplitMode,
    seed: u64,
    train_frac: f64,
    val_frac: f64,
) -> anyhow::Result<SplitIndices> {
    if n < 6 {
        anyhow::bail!("need at least 6 embedded samples, got {n}");
    }
    if !(0.0..1.0).contains(&train_frac) {
        anyhow::bail!("train_frac must be in (0,1), got {train_frac}");
    }
    if !(0.0..1.0).contains(&val_frac) {
        anyhow::bail!("val_frac must be in (0,1), got {val_frac}");
    }
    if train_frac + val_frac >= 1.0 {
        anyhow::bail!("train_frac + val_frac must be < 1");
    }

    let mut idx: Vec<usize> = (0..n).collect();
    if matches!(mode, SplitMode::Shuffled) {
        let mut rng = threebody_discover::ga::Lcg::new(seed);
        for i in (1..n).rev() {
            let j = rng.gen_range_usize(0, i);
            idx.swap(i, j);
        }
    }

    let mut n_train = (train_frac * n as f64).floor() as usize;
    let mut n_val = (val_frac * n as f64).floor() as usize;
    n_train = n_train.clamp(2, n.saturating_sub(3));
    n_val = n_val.clamp(2, n.saturating_sub(n_train + 1));
    let n_holdout = n.saturating_sub(n_train + n_val);
    if n_holdout < 2 {
        n_val = n_val.saturating_sub(1);
    }

    let train = idx[..n_train].to_vec();
    let val = idx[n_train..(n_train + n_val)].to_vec();
    let holdout = idx[(n_train + n_val)..].to_vec();
    if train.is_empty() || val.is_empty() || holdout.is_empty() {
        anyhow::bail!(
            "invalid split sizes: train={}, val={}, holdout={}",
            train.len(),
            val.len(),
            holdout.len()
        );
    }

    Ok(SplitIndices {
        train,
        val,
        holdout,
    })
}

fn assert_no_chronological_leakage(
    samples: &[DelaySample],
    split: &SplitIndices,
) -> anyhow::Result<()> {
    let max_train = split
        .train
        .iter()
        .filter_map(|&i| samples.get(i).map(|s| s.target_step))
        .max()
        .ok_or_else(|| anyhow::anyhow!("empty train split"))?;
    let min_val = split
        .val
        .iter()
        .filter_map(|&i| samples.get(i).map(|s| s.target_step))
        .min()
        .ok_or_else(|| anyhow::anyhow!("empty val split"))?;
    let min_holdout = split
        .holdout
        .iter()
        .filter_map(|&i| samples.get(i).map(|s| s.target_step))
        .min()
        .ok_or_else(|| anyhow::anyhow!("empty holdout split"))?;
    if max_train >= min_val || max_train >= min_holdout {
        anyhow::bail!(
            "chronological leakage detected (max_train_target_step={} min_val={} min_holdout={})",
            max_train,
            min_val,
            min_holdout
        );
    }
    Ok(())
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn feature_vector(z: &[f64]) -> Vec<f64> {
    let mut phi = Vec::with_capacity(z.len() + 1);
    phi.push(1.0);
    phi.extend_from_slice(z);
    phi
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 || a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = a[col][col].abs();
        for (r, row) in a.iter().enumerate().take(n).skip(col + 1) {
            let v = row[col].abs();
            if v > pivot_abs {
                pivot_abs = v;
                pivot = r;
            }
        }
        if pivot_abs <= 1e-15 || !pivot_abs.is_finite() {
            return None;
        }
        if pivot != col {
            a.swap(col, pivot);
            b.swap(col, pivot);
        }

        let diag = a[col][col];
        for j in col..n {
            a[col][j] /= diag;
        }
        b[col] /= diag;

        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r][col];
            if factor == 0.0 {
                continue;
            }
            for j in col..n {
                a[r][j] -= factor * a[col][j];
            }
            b[r] -= factor * b[col];
        }
    }
    Some(b)
}

fn nearest_neighbors(
    samples: &[DelaySample],
    train_indices: &[usize],
    query: &[f64],
    k: usize,
) -> Vec<usize> {
    let mut ranked: Vec<(f64, usize)> = train_indices
        .iter()
        .filter_map(|&i| samples.get(i).map(|s| (sq_dist(&s.z, query), i)))
        .collect();
    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    ranked.into_iter().take(k.max(1)).map(|(_, i)| i).collect()
}

#[derive(Debug, Clone)]
enum LocalModel {
    Linear {
        beta: Vec<f64>,
        add_x_t: bool,
    },
    Rational {
        beta: Vec<f64>,
        delta: Vec<f64>,
        add_x_t: bool,
    },
}

impl LocalModel {
    fn predict(&self, query_z: &[f64]) -> Option<f64> {
        let phi = feature_vector(query_z);
        match self {
            Self::Linear { beta, add_x_t } => {
                let mut pred = dot(beta, &phi);
                if *add_x_t {
                    pred += *query_z.first().unwrap_or(&0.0);
                }
                if pred.is_finite() {
                    Some(pred)
                } else {
                    None
                }
            }
            Self::Rational {
                beta,
                delta,
                add_x_t,
            } => {
                let num = dot(beta, &phi);
                let raw_den = 1.0 + dot(delta, &phi);
                let den = if raw_den.abs() < 1e-9 {
                    if raw_den.is_sign_negative() {
                        -1e-9
                    } else {
                        1e-9
                    }
                } else {
                    raw_den
                };
                let mut pred = num / den;
                if *add_x_t {
                    pred += *query_z.first().unwrap_or(&0.0);
                }
                if pred.is_finite() {
                    Some(pred)
                } else {
                    None
                }
            }
        }
    }

    fn gradient(&self, query_z: &[f64]) -> Option<Vec<f64>> {
        let d = query_z.len();
        match self {
            Self::Linear { beta, add_x_t } => {
                if beta.len() != d + 1 {
                    return None;
                }
                let mut out = beta[1..].to_vec();
                if *add_x_t && !out.is_empty() {
                    out[0] += 1.0;
                }
                Some(out)
            }
            Self::Rational {
                beta,
                delta,
                add_x_t,
            } => {
                if beta.len() != d + 1 || delta.len() != d + 1 {
                    return None;
                }
                let phi = feature_vector(query_z);
                let num = dot(beta, &phi);
                let den = 1.0 + dot(delta, &phi);
                if den.abs() < 1e-12 {
                    return None;
                }
                let mut grad = vec![0.0; d];
                for j in 0..d {
                    let num_j = beta[j + 1] * den - num * delta[j + 1];
                    let g = num_j / (den * den);
                    if !g.is_finite() {
                        return None;
                    }
                    grad[j] = g;
                }
                if *add_x_t && !grad.is_empty() {
                    grad[0] += 1.0;
                }
                Some(grad)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum NeuralFeatureMap {
    DeltaWindow,
    TemporalConv,
}

#[derive(Debug, Clone)]
struct DeltaNeuralModel {
    n_sensors: usize,
    sensor_gates: Vec<f64>,
    feature_map: NeuralFeatureMap,
    input_mean: Vec<f64>,
    input_scale: Vec<f64>,
    target_mean: f64,
    target_scale: f64,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    w3: Vec<f64>,
    b3: f64,
}

impl DeltaNeuralModel {
    fn predict(&self, query_z: &[f64]) -> Option<f64> {
        if query_z.len() != self.input_mean.len() || query_z.is_empty() || self.n_sensors == 0 {
            return None;
        }
        let x_norm = normalize_input(query_z, &self.input_mean, &self.input_scale)?;
        let x_feat = build_neural_features(
            &x_norm,
            self.n_sensors,
            &self.sensor_gates,
            self.feature_map,
        )?;
        let (_, _, _, h2) = self.forward_hidden(&x_feat)?;
        let y_norm = dot(&self.w3, &h2) + self.b3;
        if !y_norm.is_finite() {
            return None;
        }
        let delta = y_norm * self.target_scale + self.target_mean;
        let y = query_z[0] + delta;
        if y.is_finite() { Some(y) } else { None }
    }

    fn gradient(&self, query_z: &[f64]) -> Option<Vec<f64>> {
        if query_z.len() != self.input_mean.len() || query_z.is_empty() || self.n_sensors == 0 {
            return None;
        }
        let x_norm = normalize_input(query_z, &self.input_mean, &self.input_scale)?;
        let x_feat = build_neural_features(
            &x_norm,
            self.n_sensors,
            &self.sensor_gates,
            self.feature_map,
        )?;
        let (_, h1, _, h2) = self.forward_hidden(&x_feat)?;
        let h1_n = h1.len();
        let h2_n = h2.len();
        let feat_d = x_feat.len();
        let raw_d = query_z.len();
        if self.w2.len() != h2_n || self.w3.len() != h2_n || self.w1.len() != h1_n {
            return None;
        }

        let mut g_u2 = vec![0.0; h2_n];
        for j in 0..h2_n {
            let one_minus_sq = 1.0 - h2[j] * h2[j];
            g_u2[j] = self.w3[j] * one_minus_sq;
            if !g_u2[j].is_finite() {
                return None;
            }
        }

        let mut g_u1 = vec![0.0; h1_n];
        for k in 0..h1_n {
            let mut acc = 0.0;
            for (j, row) in self.w2.iter().enumerate().take(h2_n) {
                let w = *row.get(k)?;
                acc += g_u2[j] * w;
            }
            let one_minus_sq = 1.0 - h1[k] * h1[k];
            g_u1[k] = acc * one_minus_sq;
            if !g_u1[k].is_finite() {
                return None;
            }
        }

        let mut g_feat = vec![0.0; feat_d];
        for (m, slot) in g_feat.iter_mut().enumerate().take(feat_d) {
            let mut acc = 0.0;
            for (k, row) in self.w1.iter().enumerate().take(h1_n) {
                let w = *row.get(m)?;
                acc += g_u1[k] * w;
            }
            if !acc.is_finite() {
                return None;
            }
            *slot = acc;
        }

        let g_xn = backprop_neural_features(
            &g_feat,
            raw_d,
            self.n_sensors,
            &self.sensor_gates,
            self.feature_map,
        )?;
        let mut grad = vec![0.0; raw_d];
        for j in 0..raw_d {
            let scale = self.input_scale[j].abs().max(1e-12);
            grad[j] = self.target_scale * g_xn[j] / scale;
            if j == 0 {
                grad[j] += 1.0;
            }
            if !grad[j].is_finite() {
                return None;
            }
        }
        Some(grad)
    }

    fn forward_hidden(&self, x_norm: &[f64]) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        let h1_n = self.w1.len();
        let h2_n = self.w2.len();
        if h1_n == 0 || h2_n == 0 || x_norm.is_empty() {
            return None;
        }

        let mut u1 = vec![0.0; h1_n];
        let mut h1 = vec![0.0; h1_n];
        for (i, (u, h)) in u1.iter_mut().zip(h1.iter_mut()).enumerate().take(h1_n) {
            let mut acc = *self.b1.get(i)?;
            let row = self.w1.get(i)?;
            if row.len() != x_norm.len() {
                return None;
            }
            for (w, x) in row.iter().zip(x_norm.iter()) {
                acc += w * x;
            }
            if !acc.is_finite() {
                return None;
            }
            *u = acc;
            *h = acc.tanh();
        }

        let mut u2 = vec![0.0; h2_n];
        let mut h2 = vec![0.0; h2_n];
        for (i, (u, h)) in u2.iter_mut().zip(h2.iter_mut()).enumerate().take(h2_n) {
            let mut acc = *self.b2.get(i)?;
            let row = self.w2.get(i)?;
            if row.len() != h1_n {
                return None;
            }
            for (w, x) in row.iter().zip(h1.iter()) {
                acc += w * x;
            }
            if !acc.is_finite() {
                return None;
            }
            *u = acc;
            *h = acc.tanh();
        }
        Some((u1, h1, u2, h2))
    }
}

#[derive(Debug, Clone)]
struct DeltaMlpTrainConfig {
    lambda: f64,
    energy_penalty: f64,
    epochs: usize,
    learning_rate: f64,
}

fn infer_lag_count(input_dim: usize, n_sensors: usize) -> Option<usize> {
    if n_sensors == 0 || input_dim == 0 || input_dim % n_sensors != 0 {
        return None;
    }
    Some(input_dim / n_sensors)
}

fn build_temporal_window_features(
    x_norm: &[f64],
    n_sensors: usize,
    sensor_gates: &[f64],
) -> Option<Vec<f64>> {
    let n_lags = infer_lag_count(x_norm.len(), n_sensors)?;
    if sensor_gates.len() != n_sensors {
        return None;
    }
    let mut out = Vec::with_capacity(x_norm.len() + n_sensors * n_lags.saturating_sub(1));
    for lag in 0..n_lags {
        let base = lag * n_sensors;
        for (s, gate) in sensor_gates.iter().enumerate().take(n_sensors) {
            out.push(*gate * x_norm[base + s]);
        }
    }
    for lag in 0..n_lags.saturating_sub(1) {
        let a = lag * n_sensors;
        let b = (lag + 1) * n_sensors;
        for (s, gate) in sensor_gates.iter().enumerate().take(n_sensors) {
            out.push(*gate * (x_norm[a + s] - x_norm[b + s]));
        }
    }
    Some(out)
}

fn backprop_temporal_window_features(
    grad_feat: &[f64],
    raw_input_dim: usize,
    n_sensors: usize,
    sensor_gates: &[f64],
) -> Option<Vec<f64>> {
    let n_lags = infer_lag_count(raw_input_dim, n_sensors)?;
    if sensor_gates.len() != n_sensors {
        return None;
    }
    let raw_len = raw_input_dim;
    let diff_len = n_sensors * n_lags.saturating_sub(1);
    if grad_feat.len() != raw_len + diff_len {
        return None;
    }
    let mut grad_x = vec![0.0; raw_input_dim];
    for lag in 0..n_lags {
        let base = lag * n_sensors;
        for (s, gate) in sensor_gates.iter().enumerate().take(n_sensors) {
            grad_x[base + s] += *gate * grad_feat[base + s];
        }
    }
    let diff_base = raw_len;
    for lag in 0..n_lags.saturating_sub(1) {
        let a = lag * n_sensors;
        let b = (lag + 1) * n_sensors;
        for (s, gate) in sensor_gates.iter().enumerate().take(n_sensors) {
            let gd = grad_feat[diff_base + lag * n_sensors + s];
            grad_x[a + s] += *gate * gd;
            grad_x[b + s] -= *gate * gd;
        }
    }
    Some(grad_x)
}

fn build_tcn_features(
    x_norm: &[f64],
    n_sensors: usize,
    sensor_gates: &[f64],
) -> Option<Vec<f64>> {
    let n_lags = infer_lag_count(x_norm.len(), n_sensors)?;
    if sensor_gates.len() != n_sensors {
        return None;
    }
    let mut out = Vec::with_capacity(
        x_norm.len()
            + n_sensors * n_lags.saturating_sub(1)
            + n_sensors * n_lags.saturating_sub(2) * 2,
    );
    for lag in 0..n_lags {
        let base = lag * n_sensors;
        for (s, gate) in sensor_gates.iter().enumerate().take(n_sensors) {
            out.push(*gate * x_norm[base + s]);
        }
    }
    for lag in 0..n_lags.saturating_sub(1) {
        let a = lag * n_sensors;
        let b = (lag + 1) * n_sensors;
        for (s, gate) in sensor_gates.iter().enumerate().take(n_sensors) {
            out.push(*gate * (x_norm[a + s] - x_norm[b + s]));
        }
    }
    for lag in 0..n_lags.saturating_sub(2) {
        let a = lag * n_sensors;
        let b = (lag + 1) * n_sensors;
        let c = (lag + 2) * n_sensors;
        for (s, gate) in sensor_gates.iter().enumerate().take(n_sensors) {
            out.push(*gate * (x_norm[a + s] - x_norm[c + s]));
            out.push(*gate * (x_norm[a + s] - 2.0 * x_norm[b + s] + x_norm[c + s]));
        }
    }
    Some(out)
}

fn backprop_tcn_features(
    grad_feat: &[f64],
    raw_dim: usize,
    n_sensors: usize,
    sensor_gates: &[f64],
) -> Option<Vec<f64>> {
    let n_lags = infer_lag_count(raw_dim, n_sensors)?;
    if sensor_gates.len() != n_sensors {
        return None;
    }
    let mut grad_x = vec![0.0; raw_dim];
    let mut pos = 0usize;
    for lag in 0..n_lags {
        let base = lag * n_sensors;
        for (s, gate) in sensor_gates.iter().enumerate().take(n_sensors) {
            let g = *grad_feat.get(pos)?;
            grad_x[base + s] += *gate * g;
            pos += 1;
        }
    }
    for lag in 0..n_lags.saturating_sub(1) {
        let a = lag * n_sensors;
        let b = (lag + 1) * n_sensors;
        for (s, gate) in sensor_gates.iter().enumerate().take(n_sensors) {
            let g = *grad_feat.get(pos)?;
            grad_x[a + s] += *gate * g;
            grad_x[b + s] -= *gate * g;
            pos += 1;
        }
    }
    for lag in 0..n_lags.saturating_sub(2) {
        let a = lag * n_sensors;
        let b = (lag + 1) * n_sensors;
        let c = (lag + 2) * n_sensors;
        for (s, gate) in sensor_gates.iter().enumerate().take(n_sensors) {
            let g_dilated = *grad_feat.get(pos)?;
            pos += 1;
            grad_x[a + s] += *gate * g_dilated;
            grad_x[c + s] -= *gate * g_dilated;

            let g_curvature = *grad_feat.get(pos)?;
            pos += 1;
            grad_x[a + s] += *gate * g_curvature;
            grad_x[b + s] -= 2.0 * *gate * g_curvature;
            grad_x[c + s] += *gate * g_curvature;
        }
    }
    if pos != grad_feat.len() {
        return None;
    }
    Some(grad_x)
}

fn build_neural_features(
    x_norm: &[f64],
    n_sensors: usize,
    sensor_gates: &[f64],
    feature_map: NeuralFeatureMap,
) -> Option<Vec<f64>> {
    match feature_map {
        NeuralFeatureMap::DeltaWindow => {
            build_temporal_window_features(x_norm, n_sensors, sensor_gates)
        }
        NeuralFeatureMap::TemporalConv => build_tcn_features(x_norm, n_sensors, sensor_gates),
    }
}

fn backprop_neural_features(
    grad_feat: &[f64],
    raw_dim: usize,
    n_sensors: usize,
    sensor_gates: &[f64],
    feature_map: NeuralFeatureMap,
) -> Option<Vec<f64>> {
    match feature_map {
        NeuralFeatureMap::DeltaWindow => {
            backprop_temporal_window_features(grad_feat, raw_dim, n_sensors, sensor_gates)
        }
        NeuralFeatureMap::TemporalConv => {
            backprop_tcn_features(grad_feat, raw_dim, n_sensors, sensor_gates)
        }
    }
}

fn learn_sparse_sensor_gates(x_norm: &[Vec<f64>], y_norm: &[f64], n_sensors: usize) -> Vec<f64> {
    if x_norm.is_empty() || y_norm.is_empty() || n_sensors == 0 {
        return vec![1.0; n_sensors.max(1)];
    }
    let d = x_norm[0].len();
    let n_lags = infer_lag_count(d, n_sensors).unwrap_or(1);
    let n = x_norm.len().min(y_norm.len());
    if n == 0 {
        return vec![1.0; n_sensors];
    }

    let y_mean = y_norm.iter().take(n).sum::<f64>() / n as f64;
    let y_var = y_norm
        .iter()
        .take(n)
        .map(|v| {
            let dv = *v - y_mean;
            dv * dv
        })
        .sum::<f64>()
        / n as f64;
    let y_std = y_var.sqrt().max(1e-9);

    let mut scores = vec![0.0; n_sensors];
    for s in 0..n_sensors {
        let mut x_vals = Vec::with_capacity(n);
        for row in x_norm.iter().take(n) {
            let mut acc = 0.0;
            for lag in 0..n_lags {
                acc += row[lag * n_sensors + s];
            }
            x_vals.push(acc / n_lags as f64);
        }
        let x_mean = x_vals.iter().sum::<f64>() / n as f64;
        let mut cov = 0.0;
        let mut xv = 0.0;
        for i in 0..n {
            let dx = x_vals[i] - x_mean;
            let dy = y_norm[i] - y_mean;
            cov += dx * dy;
            xv += dx * dx;
        }
        cov /= n as f64;
        xv /= n as f64;
        let x_std = xv.sqrt().max(1e-9);
        scores[s] = (cov / (x_std * y_std)).abs();
    }

    let max_score = scores
        .iter()
        .copied()
        .fold(0.0f64, |a, b| if b > a { b } else { a })
        .max(1e-9);
    let sparse_threshold = 0.08;
    let min_gate = 0.20;
    let prior_gate = 0.45;
    let prior_mix = (2.0 / n_sensors as f64).clamp(0.08, 0.22);
    let mut gates = vec![0.0; n_sensors];
    for s in 0..n_sensors {
        let norm = (scores[s] / max_score).clamp(0.0, 1.0);
        let sparse = ((norm - sparse_threshold) / (1.0 - sparse_threshold)).clamp(0.0, 1.0);
        let sparse_gate = (min_gate + (1.0 - min_gate) * sparse).clamp(min_gate, 1.0);
        gates[s] = ((1.0 - prior_mix) * sparse_gate + prior_mix * prior_gate).clamp(min_gate, 1.0);
    }

    // Keep a minimum active sensor budget so the temporal model does not collapse to one channel.
    let active_threshold = SENSOR_GATE_ACTIVE_THRESHOLD;
    let max_active = n_sensors.min(8).max(1);
    let min_active = max_active.min(2);
    let target_active = if n_sensors <= 2 {
        1
    } else {
        ((n_sensors as f64) * 0.30).ceil() as usize
    }
    .clamp(min_active, max_active);
    let active_now = gates.iter().filter(|g| **g >= active_threshold).count();
    if active_now < target_active {
        let mut ranked = (0..n_sensors).collect::<Vec<_>>();
        ranked.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &idx in ranked.iter().take(target_active) {
            gates[idx] = gates[idx].max(active_threshold);
        }
    }

    if gates.iter().all(|g| *g <= min_gate + 1e-9) {
        let mut best = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (i, s) in scores.iter().enumerate() {
            if *s > best_score {
                best_score = *s;
                best = i;
            }
        }
        gates[best] = 1.0;
    }
    gates
}

fn summarize_sensor_gates(
    sensors: &[String],
    sensor_gates: &[f64],
) -> anyhow::Result<(SensorGateSummary, Vec<SensorGateValue>)> {
    if sensors.is_empty() || sensor_gates.is_empty() || sensors.len() != sensor_gates.len() {
        anyhow::bail!(
            "sensor gate shape mismatch: sensors={} gates={}",
            sensors.len(),
            sensor_gates.len()
        );
    }
    if sensor_gates.iter().any(|g| !g.is_finite()) {
        anyhow::bail!("non-finite sensor gate encountered");
    }

    let mut ranked = sensors
        .iter()
        .zip(sensor_gates.iter())
        .map(|(s, g)| SensorGateValue {
            sensor: s.clone(),
            gate: *g,
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.gate.partial_cmp(&a.gate).unwrap_or(Ordering::Equal));

    let n = sensor_gates.len();
    let sum = sensor_gates.iter().sum::<f64>();
    let sum_sq = sensor_gates.iter().map(|g| g * g).sum::<f64>();
    let active_count = sensor_gates
        .iter()
        .filter(|g| **g >= SENSOR_GATE_ACTIVE_THRESHOLD)
        .count();
    let min_gate = sensor_gates
        .iter()
        .copied()
        .fold(f64::INFINITY, |a, b| a.min(b));
    let max_gate = sensor_gates
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let summary = SensorGateSummary {
        active_threshold: SENSOR_GATE_ACTIVE_THRESHOLD,
        active_count,
        active_fraction: active_count as f64 / n as f64,
        min_gate,
        max_gate,
        mean_gate: sum / n as f64,
        effective_sensor_count: if sum_sq > 0.0 { (sum * sum) / sum_sq } else { 0.0 },
    };
    Ok((summary, ranked))
}

fn normalize_input(x: &[f64], mean: &[f64], scale: &[f64]) -> Option<Vec<f64>> {
    if x.len() != mean.len() || x.len() != scale.len() {
        return None;
    }
    let mut out = Vec::with_capacity(x.len());
    for j in 0..x.len() {
        let s = scale[j].abs().max(1e-12);
        let v = (x[j] - mean[j]) / s;
        if !v.is_finite() {
            return None;
        }
        out.push(v);
    }
    Some(out)
}

fn build_delta_mlp_diagnostics(
    samples: &[DelaySample],
    split: &SplitIndices,
    row: &GridEval,
    sensors: &[String],
    target_is_energy: bool,
) -> anyhow::Result<DeltaMlpDiagnostics> {
    let n_sensors = sensors.len().max(1);
    let mlp = fit_delta_mlp_model(
        samples,
        &split.train,
        n_sensors,
        row.config.k,
        row.config.lambda,
        target_is_energy,
    )
    .ok_or_else(|| anyhow::anyhow!("delta_mlp fit failed for diagnostics"))?;
    let (gate_summary, sensor_gates_ranked) = summarize_sensor_gates(sensors, &mlp.sensor_gates)?;

    let rel = if row.holdout_baseline_mse > 0.0 {
        (row.holdout_baseline_mse - row.holdout_mse) / row.holdout_baseline_mse
    } else {
        0.0
    };
    Ok(DeltaMlpDiagnostics {
        config: row.config.clone(),
        architecture: row.architecture.clone(),
        n_embedded: row.n_embedded,
        split: row.split.clone(),
        val_mse: row.val_mse,
        holdout_mse: row.holdout_mse,
        holdout_baseline_mse: row.holdout_baseline_mse,
        holdout_relative_improvement: rel,
        val_sensitivity_median: row.val_sensitivity_median,
        holdout_sensitivity_median: row.holdout_sensitivity_median,
        target_is_energy,
        energy_penalty: if target_is_energy {
            DELTA_MLP_ENERGY_PENALTY
        } else {
            0.0
        },
        gate_summary,
        sensor_gates_ranked,
    })
}

fn fit_delta_mlp_model(
    samples: &[DelaySample],
    train_indices: &[usize],
    n_sensors: usize,
    hidden_hint: usize,
    lambda: f64,
    target_is_energy: bool,
) -> Option<DeltaNeuralModel> {
    fit_delta_neural_model(
        samples,
        train_indices,
        n_sensors,
        hidden_hint,
        lambda,
        target_is_energy,
        NeuralFeatureMap::DeltaWindow,
    )
}

fn fit_delta_tcn_model(
    samples: &[DelaySample],
    train_indices: &[usize],
    n_sensors: usize,
    hidden_hint: usize,
    lambda: f64,
    target_is_energy: bool,
) -> Option<DeltaNeuralModel> {
    fit_delta_neural_model(
        samples,
        train_indices,
        n_sensors,
        hidden_hint,
        lambda,
        target_is_energy,
        NeuralFeatureMap::TemporalConv,
    )
}

fn fit_delta_neural_model(
    samples: &[DelaySample],
    train_indices: &[usize],
    n_sensors: usize,
    hidden_hint: usize,
    lambda: f64,
    target_is_energy: bool,
    feature_map: NeuralFeatureMap,
) -> Option<DeltaNeuralModel> {
    let first = samples.get(*train_indices.first()?)?;
    let d = first.z.len();
    if d == 0 || n_sensors == 0 {
        return None;
    }
    let _n_lags = infer_lag_count(d, n_sensors)?;

    let train_n = train_indices.len();
    if train_n < 8 {
        return None;
    }

    let mut input_mean = vec![0.0; d];
    let mut target_mean = 0.0;
    let mut x_train = Vec::with_capacity(train_n);
    let mut y_train = Vec::with_capacity(train_n);
    for &idx in train_indices {
        let s = samples.get(idx)?;
        if s.z.len() != d {
            return None;
        }
        for (j, mu) in input_mean.iter_mut().enumerate().take(d) {
            *mu += s.z[j];
        }
        let residual = s.y - s.x_t;
        if !residual.is_finite() {
            return None;
        }
        target_mean += residual;
        x_train.push(s.z.clone());
        y_train.push(residual);
    }
    for mu in input_mean.iter_mut().take(d) {
        *mu /= train_n as f64;
    }
    target_mean /= train_n as f64;

    let mut input_var = vec![0.0; d];
    let mut target_var = 0.0;
    for (x, y) in x_train.iter().zip(y_train.iter()) {
        for j in 0..d {
            let dv = x[j] - input_mean[j];
            input_var[j] += dv * dv;
        }
        let dy = *y - target_mean;
        target_var += dy * dy;
    }
    let mut input_scale = vec![0.0; d];
    for j in 0..d {
        input_scale[j] = (input_var[j] / train_n as f64).sqrt().max(1e-6);
    }
    let target_scale = (target_var / train_n as f64).sqrt().max(1e-6);

    let mut x_norm = Vec::with_capacity(train_n);
    let mut y_norm = Vec::with_capacity(train_n);
    for (x, y) in x_train.iter().zip(y_train.iter()) {
        x_norm.push(normalize_input(x, &input_mean, &input_scale)?);
        let yn = (*y - target_mean) / target_scale;
        if !yn.is_finite() {
            return None;
        }
        y_norm.push(yn);
    }

    let sensor_gates = learn_sparse_sensor_gates(&x_norm, &y_norm, n_sensors);
    let mut x_feat = Vec::with_capacity(train_n);
    for x in x_norm.iter().take(train_n) {
        x_feat.push(build_neural_features(x, n_sensors, &sensor_gates, feature_map)?);
    }
    let p = x_feat.first()?.len();
    if p == 0 {
        return None;
    }

    let h1 = hidden_hint.clamp(4, 64);
    let h2 = (h1 / 2).clamp(4, 32);
    let cfg = DeltaMlpTrainConfig {
        lambda: lambda.max(0.0),
        energy_penalty: if target_is_energy {
            DELTA_MLP_ENERGY_PENALTY
        } else {
            0.0
        },
        epochs: 260,
        learning_rate: 1e-2,
    };

    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    seed ^= (d as u64).wrapping_mul(0xA24B_AED4_963E_E407);
    seed ^= (train_n as u64).wrapping_mul(0x9FB2_1C65_1E98_DF25);
    seed ^= (h1 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    seed ^= cfg.lambda.to_bits().wrapping_mul(0x1656_67B1_9E37_79F9);
    let mut rng = threebody_discover::ga::Lcg::new(seed);

    let s1 = (1.0 / p as f64).sqrt();
    let s2 = (1.0 / h1 as f64).sqrt();
    let s3 = (1.0 / h2 as f64).sqrt();
    let mut w1 = vec![vec![0.0; p]; h1];
    let mut b1 = vec![0.0; h1];
    let mut w2 = vec![vec![0.0; h1]; h2];
    let mut b2 = vec![0.0; h2];
    let mut w3 = vec![0.0; h2];
    let mut b3 = 0.0f64;
    for row in w1.iter_mut().take(h1) {
        for w in row.iter_mut().take(p) {
            *w = rng.gen_range_f64(-s1, s1);
        }
    }
    for row in w2.iter_mut().take(h2) {
        for w in row.iter_mut().take(h1) {
            *w = rng.gen_range_f64(-s2, s2);
        }
    }
    for w in w3.iter_mut().take(h2) {
        *w = rng.gen_range_f64(-s3, s3);
    }

    // Adam state
    let (beta1, beta2, eps) = (0.9, 0.999, 1e-8);
    let mut mw1 = vec![vec![0.0; p]; h1];
    let mut vw1 = vec![vec![0.0; p]; h1];
    let mut mb1 = vec![0.0; h1];
    let mut vb1 = vec![0.0; h1];
    let mut mw2 = vec![vec![0.0; h1]; h2];
    let mut vw2 = vec![vec![0.0; h1]; h2];
    let mut mb2 = vec![0.0; h2];
    let mut vb2 = vec![0.0; h2];
    let mut mw3 = vec![0.0; h2];
    let mut vw3 = vec![0.0; h2];
    let mut mb3 = 0.0f64;
    let mut vb3 = 0.0f64;

    let mut gw1 = vec![vec![0.0; p]; h1];
    let mut gb1 = vec![0.0; h1];
    let mut gw2 = vec![vec![0.0; h1]; h2];
    let mut gb2 = vec![0.0; h2];
    let mut gw3 = vec![0.0; h2];
    let mut gb3: f64;

    for epoch in 1..=cfg.epochs {
        for row in gw1.iter_mut().take(h1) {
            row.fill(0.0);
        }
        gb1.fill(0.0);
        for row in gw2.iter_mut().take(h2) {
            row.fill(0.0);
        }
        gb2.fill(0.0);
        gw3.fill(0.0);
        gb3 = 0.0;

        let inv_n = 1.0 / train_n as f64;
        for (x, y) in x_feat.iter().zip(y_norm.iter()) {
            let mut u1 = vec![0.0; h1];
            let mut h1v = vec![0.0; h1];
            for i in 0..h1 {
                let mut acc = b1[i];
                for (w, xv) in w1[i].iter().zip(x.iter()) {
                    acc += w * xv;
                }
                u1[i] = acc;
                h1v[i] = acc.tanh();
            }
            let mut u2 = vec![0.0; h2];
            let mut h2v = vec![0.0; h2];
            for i in 0..h2 {
                let mut acc = b2[i];
                for (w, h) in w2[i].iter().zip(h1v.iter()) {
                    acc += w * h;
                }
                u2[i] = acc;
                h2v[i] = acc.tanh();
            }
            let y_hat = dot(&w3, &h2v) + b3;
            let mut dy = (y_hat - *y) * (2.0 * inv_n);
            if cfg.energy_penalty > 0.0 {
                let delta_raw = y_hat * target_scale + target_mean;
                dy += 2.0 * cfg.energy_penalty * delta_raw * target_scale * inv_n;
            }
            if !dy.is_finite() {
                return None;
            }

            for j in 0..h2 {
                gw3[j] += dy * h2v[j];
            }
            gb3 += dy;

            let mut du2 = vec![0.0; h2];
            for j in 0..h2 {
                du2[j] = dy * w3[j] * (1.0 - h2v[j] * h2v[j]);
            }
            let mut dh1 = vec![0.0; h1];
            for j in 0..h2 {
                gb2[j] += du2[j];
                for k in 0..h1 {
                    gw2[j][k] += du2[j] * h1v[k];
                    dh1[k] += du2[j] * w2[j][k];
                }
            }
            for k in 0..h1 {
                let du1 = dh1[k] * (1.0 - h1v[k] * h1v[k]);
                gb1[k] += du1;
                for (m, xv) in x.iter().enumerate().take(p) {
                    gw1[k][m] += du1 * xv;
                }
            }
        }

        if cfg.lambda > 0.0 {
            let wd = 2.0 * cfg.lambda;
            for i in 0..h1 {
                for j in 0..p {
                    gw1[i][j] += wd * w1[i][j];
                }
            }
            for i in 0..h2 {
                for j in 0..h1 {
                    gw2[i][j] += wd * w2[i][j];
                }
            }
            for j in 0..h2 {
                gw3[j] += wd * w3[j];
            }
        }

        let lr = cfg.learning_rate / (1.0 + 0.01 * epoch as f64);
        let t = epoch as i32;
        for i in 0..h1 {
            for j in 0..p {
                let g = gw1[i][j].clamp(-5.0, 5.0);
                mw1[i][j] = beta1 * mw1[i][j] + (1.0 - beta1) * g;
                vw1[i][j] = beta2 * vw1[i][j] + (1.0 - beta2) * g * g;
                let mhat = mw1[i][j] / (1.0 - beta1.powi(t));
                let vhat = vw1[i][j] / (1.0 - beta2.powi(t));
                w1[i][j] -= lr * mhat / (vhat.sqrt() + eps);
            }
            let g = gb1[i].clamp(-5.0, 5.0);
            mb1[i] = beta1 * mb1[i] + (1.0 - beta1) * g;
            vb1[i] = beta2 * vb1[i] + (1.0 - beta2) * g * g;
            let mhat = mb1[i] / (1.0 - beta1.powi(t));
            let vhat = vb1[i] / (1.0 - beta2.powi(t));
            b1[i] -= lr * mhat / (vhat.sqrt() + eps);
        }
        for i in 0..h2 {
            for j in 0..h1 {
                let g = gw2[i][j].clamp(-5.0, 5.0);
                mw2[i][j] = beta1 * mw2[i][j] + (1.0 - beta1) * g;
                vw2[i][j] = beta2 * vw2[i][j] + (1.0 - beta2) * g * g;
                let mhat = mw2[i][j] / (1.0 - beta1.powi(t));
                let vhat = vw2[i][j] / (1.0 - beta2.powi(t));
                w2[i][j] -= lr * mhat / (vhat.sqrt() + eps);
            }
            let g = gb2[i].clamp(-5.0, 5.0);
            mb2[i] = beta1 * mb2[i] + (1.0 - beta1) * g;
            vb2[i] = beta2 * vb2[i] + (1.0 - beta2) * g * g;
            let mhat = mb2[i] / (1.0 - beta1.powi(t));
            let vhat = vb2[i] / (1.0 - beta2.powi(t));
            b2[i] -= lr * mhat / (vhat.sqrt() + eps);
        }
        for j in 0..h2 {
            let g = gw3[j].clamp(-5.0, 5.0);
            mw3[j] = beta1 * mw3[j] + (1.0 - beta1) * g;
            vw3[j] = beta2 * vw3[j] + (1.0 - beta2) * g * g;
            let mhat = mw3[j] / (1.0 - beta1.powi(t));
            let vhat = vw3[j] / (1.0 - beta2.powi(t));
            w3[j] -= lr * mhat / (vhat.sqrt() + eps);
        }
        let g = gb3.clamp(-5.0, 5.0);
        mb3 = beta1 * mb3 + (1.0 - beta1) * g;
        vb3 = beta2 * vb3 + (1.0 - beta2) * g * g;
        let mhat = mb3 / (1.0 - beta1.powi(t));
        let vhat = vb3 / (1.0 - beta2.powi(t));
        b3 -= lr * mhat / (vhat.sqrt() + eps);
    }

    // Basic finite guard.
    let mut all_finite = b3.is_finite();
    all_finite &= b1.iter().all(|v| v.is_finite());
    all_finite &= b2.iter().all(|v| v.is_finite());
    all_finite &= w1.iter().flatten().all(|v| v.is_finite());
    all_finite &= w2.iter().flatten().all(|v| v.is_finite());
    all_finite &= w3.iter().all(|v| v.is_finite());
    if !all_finite {
        return None;
    }

    Some(DeltaNeuralModel {
        n_sensors,
        sensor_gates,
        feature_map,
        input_mean,
        input_scale,
        target_mean,
        target_scale,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
    })
}

fn fit_linear_model(
    samples: &[DelaySample],
    neighbors: &[usize],
    lambda: f64,
    fit_residual: bool,
) -> Option<LocalModel> {
    let first = samples.get(*neighbors.first()?)?;
    let p = first.z.len() + 1;
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![0.0; p];

    for &idx in neighbors {
        let s = samples.get(idx)?;
        let phi = feature_vector(&s.z);
        let target = if fit_residual { s.y - s.x_t } else { s.y };
        for i in 0..p {
            xty[i] += phi[i] * target;
            for j in 0..p {
                xtx[i][j] += phi[i] * phi[j];
            }
        }
    }

    let lambda = lambda.max(0.0);
    for (i, row) in xtx.iter_mut().enumerate().take(p).skip(1) {
        row[i] += lambda;
    }

    let beta = solve_linear_system(xtx, xty)?;
    Some(LocalModel::Linear {
        beta,
        add_x_t: fit_residual,
    })
}

fn fit_rational_model(
    samples: &[DelaySample],
    neighbors: &[usize],
    lambda: f64,
    fit_residual: bool,
) -> Option<LocalModel> {
    let first = samples.get(*neighbors.first()?)?;
    let p = first.z.len() + 1;
    let n_param = 2 * p;
    let mut ata = vec![vec![0.0; n_param]; n_param];
    let mut atb = vec![0.0; n_param];

    for &idx in neighbors {
        let s = samples.get(idx)?;
        let phi = feature_vector(&s.z);
        let y = if fit_residual { s.y - s.x_t } else { s.y };
        let mut row = vec![0.0; n_param];
        for j in 0..p {
            row[j] = phi[j];
            row[p + j] = -y * phi[j];
        }

        for i in 0..n_param {
            atb[i] += row[i] * y;
            for j in 0..n_param {
                ata[i][j] += row[i] * row[j];
            }
        }
    }

    let lambda = lambda.max(0.0);
    for (i, row) in ata.iter_mut().enumerate().take(n_param).skip(1) {
        row[i] += lambda;
    }

    let theta = solve_linear_system(ata, atb)?;
    let beta = theta[..p].to_vec();
    let delta = theta[p..].to_vec();
    Some(LocalModel::Rational {
        beta,
        delta,
        add_x_t: fit_residual,
    })
}

fn fit_local_model(
    samples: &[DelaySample],
    train_indices: &[usize],
    query: &[f64],
    k: usize,
    lambda: f64,
    model: ModelKind,
) -> Option<(LocalModel, f64)> {
    if train_indices.is_empty() || query.is_empty() {
        return None;
    }

    let neighbors = nearest_neighbors(samples, train_indices, query, k);
    let fallback = samples.get(*neighbors.first()?)?.y;
    let local = match model {
        ModelKind::Linear => fit_linear_model(samples, &neighbors, lambda, false)?,
        ModelKind::Rational => fit_rational_model(samples, &neighbors, lambda, false)?,
        ModelKind::DeltaLinear => fit_linear_model(samples, &neighbors, lambda, true)?,
        ModelKind::DeltaRational => fit_rational_model(samples, &neighbors, lambda, true)?,
        ModelKind::DeltaMlp | ModelKind::DeltaTcn => return None,
    };
    Some((local, fallback))
}

fn fit_delta_neural_for_model(
    samples: &[DelaySample],
    train_indices: &[usize],
    k: usize,
    lambda: f64,
    model: ModelKind,
    n_sensors: usize,
    target_is_energy: bool,
) -> anyhow::Result<DeltaNeuralModel> {
    match model {
        ModelKind::DeltaMlp => fit_delta_mlp_model(
            samples,
            train_indices,
            n_sensors,
            k,
            lambda,
            target_is_energy,
        )
        .ok_or_else(|| anyhow::anyhow!("delta_mlp fit failed")),
        ModelKind::DeltaTcn => fit_delta_tcn_model(
            samples,
            train_indices,
            n_sensors,
            k,
            lambda,
            target_is_energy,
        )
        .ok_or_else(|| anyhow::anyhow!("delta_tcn fit failed")),
        _ => anyhow::bail!("model is not neural: {}", model_kind_name(model)),
    }
}

fn evaluate_model_mse(
    samples: &[DelaySample],
    train_indices: &[usize],
    eval_indices: &[usize],
    k: usize,
    lambda: f64,
    model: ModelKind,
    n_sensors: usize,
    target_is_energy: bool,
) -> anyhow::Result<f64> {
    if eval_indices.is_empty() {
        anyhow::bail!("empty evaluation split");
    }

    if matches!(model, ModelKind::DeltaMlp | ModelKind::DeltaTcn) {
        let neural = fit_delta_neural_for_model(
            samples,
            train_indices,
            k,
            lambda,
            model,
            n_sensors,
            target_is_energy,
        )?;
        let mut se = 0.0;
        let mut n = 0usize;
        for &i in eval_indices {
            let s = samples
                .get(i)
                .ok_or_else(|| anyhow::anyhow!("evaluation index out of bounds: {i}"))?;
            let pred = neural
                .predict(&s.z)
                .ok_or_else(|| anyhow::anyhow!("neural prediction failed for index {i}"))?;
            let err = pred - s.y;
            se += err * err;
            n += 1;
        }
        return Ok(se / n as f64);
    }

    let mut se = 0.0;
    let mut n = 0usize;
    for &i in eval_indices {
        let s = samples
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("evaluation index out of bounds: {i}"))?;

        let pred = fit_local_model(samples, train_indices, &s.z, k, lambda, model)
            .and_then(|(local, fallback)| local.predict(&s.z).or(Some(fallback)))
            .ok_or_else(|| anyhow::anyhow!("prediction failed for index {i}"))?;

        let err = pred - s.y;
        se += err * err;
        n += 1;
    }
    Ok(se / n as f64)
}

fn evaluate_sensitivity_summary(
    samples: &[DelaySample],
    train_indices: &[usize],
    eval_indices: &[usize],
    k: usize,
    lambda: f64,
    model: ModelKind,
    n_sensors: usize,
    target_is_energy: bool,
) -> anyhow::Result<SensitivitySummary> {
    if eval_indices.len() < 2 {
        anyhow::bail!("sensitivity evaluation requires at least 2 points");
    }

    let fitted_neural = if matches!(model, ModelKind::DeltaMlp | ModelKind::DeltaTcn) {
        Some(
            fit_delta_neural_for_model(
                samples,
                train_indices,
                k,
                lambda,
                model,
                n_sensors,
                target_is_energy,
            )?,
        )
    } else {
        None
    };

    let mut rel_errors = Vec::new();
    for &i in eval_indices {
        let s_i = samples
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("evaluation index out of bounds: {i}"))?;

        let (_, j) = eval_indices
            .iter()
            .filter(|&&j| j != i)
            .filter_map(|&j| {
                let s_j = samples.get(j)?;
                Some((sq_dist(&s_i.z, &s_j.z), j))
            })
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
            .ok_or_else(|| anyhow::anyhow!("failed to find sensitivity peer for index {i}"))?;

        let s_j = samples
            .get(j)
            .ok_or_else(|| anyhow::anyhow!("sensitivity peer index out of bounds: {j}"))?;

        let grad = if let Some(neural) = fitted_neural.as_ref() {
            neural
                .gradient(&s_i.z)
                .ok_or_else(|| anyhow::anyhow!("neural gradient failed for index {i}"))?
        } else {
            let (local, _) = fit_local_model(samples, train_indices, &s_i.z, k, lambda, model)
                .ok_or_else(|| anyhow::anyhow!("local model fit failed for index {i}"))?;
            local
                .gradient(&s_i.z)
                .ok_or_else(|| anyhow::anyhow!("gradient failed for index {i}"))?
        };

        let mut dz = vec![0.0; s_i.z.len()];
        for (slot, (a, b)) in dz.iter_mut().zip(s_j.z.iter().zip(s_i.z.iter())) {
            *slot = a - b;
        }
        let dy_obs = s_j.y - s_i.y;
        let dy_pred = dot(&grad, &dz);

        let denom = dy_obs.abs().max(1e-12);
        let rel = (dy_pred - dy_obs).abs() / denom;
        if rel.is_finite() {
            rel_errors.push(rel);
        }
    }

    if rel_errors.is_empty() {
        anyhow::bail!("no finite sensitivity errors computed");
    }

    rel_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median = quantile_sorted(&rel_errors, 0.5);
    let p90 = quantile_sorted(&rel_errors, 0.9);
    let max_rel = *rel_errors.last().unwrap_or(&median);

    Ok(SensitivitySummary {
        n_pairs: rel_errors.len(),
        median_rel_error: median,
        p90_rel_error: p90,
        max_rel_error: max_rel,
    })
}

fn quantile_sorted(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    if xs.len() == 1 {
        return xs[0];
    }

    let q = q.clamp(0.0, 1.0);
    let pos = q * (xs.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        xs[lo]
    } else {
        let w = pos - lo as f64;
        xs[lo] * (1.0 - w) + xs[hi] * w
    }
}

fn evaluate_persistence_mse(
    samples: &[DelaySample],
    eval_indices: &[usize],
) -> anyhow::Result<f64> {
    if eval_indices.is_empty() {
        anyhow::bail!("empty evaluation split");
    }
    let mut se = 0.0;
    let mut n = 0usize;
    for &i in eval_indices {
        let s = samples
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("evaluation index out of bounds: {i}"))?;
        let pred = s.x_t;
        let err = pred - s.y;
        se += err * err;
        n += 1;
    }
    Ok(se / n as f64)
}

fn build_recursive_query_single_sensor(
    target: &[f64],
    start_t: usize,
    current_t: usize,
    tau: usize,
    m: usize,
    predicted: &BTreeMap<usize, f64>,
) -> anyhow::Result<Vec<f64>> {
    let mut z = Vec::with_capacity(m);
    for j in 0..m {
        let lag_t = current_t
            .checked_sub(j * tau)
            .ok_or_else(|| anyhow::anyhow!("lag underflow while building recursive query"))?;
        let v = if lag_t > start_t {
            predicted.get(&lag_t).copied().ok_or_else(|| {
                anyhow::anyhow!("missing predicted value for recursive lag time {lag_t}")
            })?
        } else {
            *target
                .get(lag_t)
                .ok_or_else(|| anyhow::anyhow!("target index out of bounds for lag time {lag_t}"))?
        };
        if !v.is_finite() {
            anyhow::bail!("non-finite value in recursive query at lag time {}", lag_t);
        }
        z.push(v);
    }
    Ok(z)
}

fn evaluate_recursive_horizon_metric(
    samples: &[DelaySample],
    target: &[f64],
    train_indices: &[usize],
    eval_indices: &[usize],
    config: &TakensConfig,
    n_sensors: usize,
    target_is_energy: bool,
    horizon: usize,
) -> anyhow::Result<Option<HorizonMetric>> {
    if horizon == 0 || eval_indices.is_empty() || n_sensors != 1 {
        return Ok(None);
    }
    let fitted_neural = if matches!(config.model, ModelKind::DeltaMlp | ModelKind::DeltaTcn) {
        Some(fit_delta_neural_for_model(
            samples,
            train_indices,
            config.k,
            config.lambda,
            config.model,
            n_sensors,
            target_is_energy,
        )?)
    } else {
        None
    };

    let mut se_model = 0.0;
    let mut se_baseline = 0.0;
    let mut n_eval = 0usize;
    for &idx in eval_indices {
        let s0 = samples
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("evaluation index out of bounds: {idx}"))?;
        let t0 = s0.target_step.saturating_sub(1);
        let t_h = t0 + horizon;
        let Some(&truth_h) = target.get(t_h) else {
            continue;
        };

        let mut predicted = BTreeMap::<usize, f64>::new();
        for step in 1..=horizon {
            let current_t = t0 + step - 1;
            let query = build_recursive_query_single_sensor(
                target,
                t0,
                current_t,
                config.tau,
                config.m,
                &predicted,
            )?;
            let pred = if let Some(neural) = fitted_neural.as_ref() {
                neural
                    .predict(&query)
                    .ok_or_else(|| anyhow::anyhow!("neural recursive prediction failed"))?
            } else {
                fit_local_model(
                    samples,
                    train_indices,
                    &query,
                    config.k,
                    config.lambda,
                    config.model,
                )
                .and_then(|(local, fallback)| local.predict(&query).or(Some(fallback)))
                .ok_or_else(|| anyhow::anyhow!("local recursive prediction failed"))?
            };
            predicted.insert(current_t + 1, pred);
        }

        let pred_h = predicted.get(&t_h).copied().ok_or_else(|| {
            anyhow::anyhow!("missing recursive prediction at horizon time {t_h}")
        })?;
        let err = pred_h - truth_h;
        se_model += err * err;

        let base_err = s0.x_t - truth_h;
        se_baseline += base_err * base_err;
        n_eval += 1;
    }

    if n_eval == 0 {
        return Ok(None);
    }
    let holdout_mse = se_model / n_eval as f64;
    let holdout_baseline_mse = se_baseline / n_eval as f64;
    let holdout_relative_improvement = if holdout_baseline_mse > 0.0 {
        (holdout_baseline_mse - holdout_mse) / holdout_baseline_mse
    } else {
        0.0
    };
    Ok(Some(HorizonMetric {
        horizon,
        n_eval,
        holdout_mse,
        holdout_baseline_mse,
        holdout_relative_improvement,
    }))
}

#[cfg(test)]
mod tests {
    use super::{
        assert_no_chronological_leakage, build_delay_embedding, build_delay_embedding_multi,
        evaluate_model_mse, evaluate_persistence_mse, evaluate_recursive_horizon_metric,
        evaluate_sensitivity_summary, infer_model_architecture, split_indices, ModelKind,
        SplitMode, TakensConfig,
    };

    fn logistic_series(n: usize, r: f64, x0: f64) -> Vec<f64> {
        let mut out = Vec::with_capacity(n);
        let mut x = x0;
        for _ in 0..n {
            out.push(x);
            x = r * x * (1.0 - x);
        }
        out
    }

    fn rational_series(n: usize, a0: f64, a1: f64, b1: f64, x0: f64) -> Vec<f64> {
        let mut out = Vec::with_capacity(n);
        let mut x = x0;
        for _ in 0..n {
            out.push(x);
            let den = (1.0 + b1 * x).abs().max(1e-9);
            x = (a0 + a1 * x) / den;
        }
        out
    }

    fn ar2_series(n: usize, a1: f64, a2: f64, b: f64, x0: f64, x1: f64) -> Vec<f64> {
        let mut out = Vec::with_capacity(n);
        let mut prev2 = x0;
        let mut prev1 = x1;
        out.push(prev2);
        out.push(prev1);
        while out.len() < n {
            let x = a1 * prev1 + a2 * prev2 + b;
            out.push(x);
            prev2 = prev1;
            prev1 = x;
        }
        out
    }

    fn coupled_series(n: usize, y0: f64) -> (Vec<f64>, Vec<f64>) {
        let mut y = Vec::with_capacity(n);
        let mut u = Vec::with_capacity(n);
        let mut prev_y = y0;
        for t in 0..n {
            let tt = t as f64;
            let sensor = (0.07 * tt).sin() + 0.3 * (0.11 * tt).cos();
            y.push(prev_y);
            u.push(sensor);
            prev_y = 0.7 * prev_y + 0.2 * sensor;
        }
        (y, u)
    }

    fn persistence_dominated_series(n: usize, x0: f64) -> (Vec<f64>, Vec<f64>) {
        let mut x = Vec::with_capacity(n);
        let mut u = Vec::with_capacity(n);
        let mut prev_x = x0;
        for t in 0..n {
            let tt = t as f64;
            let sensor = (0.05 * tt).sin() + 0.25 * (0.09 * tt).cos();
            x.push(prev_x);
            u.push(sensor);
            prev_x = prev_x + 0.03 * sensor - 0.005 * sensor * sensor;
        }
        (x, u)
    }

    fn nonlinear_delta_series(n: usize, x0: f64) -> (Vec<f64>, Vec<f64>) {
        let mut x = Vec::with_capacity(n);
        let mut u = Vec::with_capacity(n);
        let mut prev_x = x0;
        for t in 0..n {
            let tt = t as f64;
            let sensor = (0.03 * tt).sin() + 0.35 * (0.07 * tt).cos();
            x.push(prev_x);
            u.push(sensor);
            let delta = 0.18 * (1.4 * prev_x).sin() + 0.07 * sensor * sensor + 0.03 * sensor;
            prev_x += delta;
        }
        (x, u)
    }

    #[test]
    fn delay_embedding_indexing_is_correct() {
        let series: Vec<f64> = (0..10).map(|v| v as f64).collect();
        let emb = build_delay_embedding(&series, 2, 3).unwrap();
        assert_eq!(emb.len(), 5);
        assert_eq!(emb[0].z, vec![4.0, 2.0, 0.0]);
        assert_eq!(emb[0].x_t, 4.0);
        assert_eq!(emb[0].y, 5.0);
        assert_eq!(emb[0].target_step, 5);
        let last = emb.last().unwrap();
        assert_eq!(last.z, vec![8.0, 6.0, 4.0]);
        assert_eq!(last.x_t, 8.0);
        assert_eq!(last.y, 9.0);
        assert_eq!(last.target_step, 9);
    }

    #[test]
    fn multi_sensor_embedding_dimension_matches_m_times_sensors() {
        let target: Vec<f64> = (0..12).map(|v| v as f64).collect();
        let sensor_b: Vec<f64> = (0..12).map(|v| (v as f64) * 10.0).collect();
        let emb = build_delay_embedding_multi(&target, &[target.clone(), sensor_b], 1, 3).unwrap();
        assert!(!emb.is_empty());
        assert_eq!(emb[0].z.len(), 6);
        assert_eq!(emb[0].z, vec![2.0, 20.0, 1.0, 10.0, 0.0, 0.0]);
    }

    #[test]
    fn chronological_split_is_disjoint_and_ordered() {
        let split = split_indices(30, SplitMode::Chronological, 42, 0.7, 0.15).unwrap();
        assert!(!split.train.is_empty());
        assert!(!split.val.is_empty());
        assert!(!split.holdout.is_empty());
        assert!(
            split.train.iter().all(|&i| i < split.val[0]),
            "train indices should come before val"
        );
        assert!(
            split
                .val
                .iter()
                .all(|&i| i > *split.train.last().unwrap() && i < split.holdout[0]),
            "val indices should sit between train and holdout"
        );
    }

    #[test]
    fn shuffled_split_is_reproducible_with_seed() {
        let a = split_indices(40, SplitMode::Shuffled, 123, 0.7, 0.15).unwrap();
        let b = split_indices(40, SplitMode::Shuffled, 123, 0.7, 0.15).unwrap();
        assert_eq!(a.train, b.train);
        assert_eq!(a.val, b.val);
        assert_eq!(a.holdout, b.holdout);
    }

    #[test]
    fn chronological_split_has_no_future_leakage() {
        let series = logistic_series(120, 3.9, 0.231);
        let emb = build_delay_embedding(&series, 1, 4).unwrap();
        let split = split_indices(emb.len(), SplitMode::Chronological, 42, 0.7, 0.15).unwrap();
        assert_no_chronological_leakage(&emb, &split).unwrap();
    }

    #[test]
    fn local_linear_map_beats_persistence_on_logistic_fixture() {
        let series = logistic_series(500, 3.9, 0.231);
        let emb = build_delay_embedding(&series, 1, 4).unwrap();
        let split = split_indices(emb.len(), SplitMode::Chronological, 42, 0.7, 0.15).unwrap();
        assert_no_chronological_leakage(&emb, &split).unwrap();

        let holdout_mse = evaluate_model_mse(
            &emb,
            &split.train,
            &split.holdout,
            12,
            1e-8,
            ModelKind::Linear,
            1,
            false,
        )
        .unwrap();
        let baseline_mse = evaluate_persistence_mse(&emb, &split.holdout).unwrap();
        assert!(
            holdout_mse < baseline_mse * 0.9,
            "expected local map MSE < persistence MSE (got local={} baseline={})",
            holdout_mse,
            baseline_mse
        );
    }

    #[test]
    fn local_rational_map_beats_local_linear_on_rational_fixture() {
        let series = rational_series(500, 0.05, 1.4, 2.0, 0.2);
        let emb = build_delay_embedding(&series, 1, 1).unwrap();
        let split = split_indices(emb.len(), SplitMode::Chronological, 7, 0.7, 0.15).unwrap();
        assert_no_chronological_leakage(&emb, &split).unwrap();

        let k_global = split.train.len();
        let linear_mse = evaluate_model_mse(
            &emb,
            &split.train,
            &split.holdout,
            k_global,
            1e-8,
            ModelKind::Linear,
            1,
            false,
        )
        .unwrap();
        let rational_mse = evaluate_model_mse(
            &emb,
            &split.train,
            &split.holdout,
            k_global,
            1e-8,
            ModelKind::Rational,
            1,
            false,
        )
        .unwrap();

        assert!(
            rational_mse < linear_mse * 0.5,
            "expected rational local map to improve over linear (linear={} rational={})",
            linear_mse,
            rational_mse
        );
    }

    #[test]
    fn sensitivity_summary_is_low_for_linear_ar2_system() {
        let series = ar2_series(600, 0.6, -0.2, 0.1, 0.11, 0.27);
        let emb = build_delay_embedding(&series, 1, 2).unwrap();
        let split = split_indices(emb.len(), SplitMode::Chronological, 11, 0.7, 0.15).unwrap();
        assert_no_chronological_leakage(&emb, &split).unwrap();

        let sens = evaluate_sensitivity_summary(
            &emb,
            &split.train,
            &split.holdout,
            18,
            1e-8,
            ModelKind::Linear,
            1,
            false,
        )
        .unwrap();

        assert!(
            sens.median_rel_error < 0.15,
            "expected low linear sensitivity error, got {:?}",
            sens
        );
    }

    #[test]
    fn richer_sensors_improve_prediction_for_coupled_target() {
        let (target, sensor_u) = coupled_series(800, 0.13);
        let emb_target_only =
            build_delay_embedding_multi(&target, &[target.clone()], 1, 1).unwrap();
        let emb_with_sensor =
            build_delay_embedding_multi(&target, &[target.clone(), sensor_u], 1, 1).unwrap();

        let split = split_indices(
            emb_target_only.len(),
            SplitMode::Chronological,
            42,
            0.7,
            0.15,
        )
        .unwrap();
        let k_global = split.train.len();
        let mse_target_only = evaluate_model_mse(
            &emb_target_only,
            &split.train,
            &split.holdout,
            k_global,
            1e-8,
            ModelKind::Linear,
            1,
            false,
        )
        .unwrap();
        let mse_with_sensor = evaluate_model_mse(
            &emb_with_sensor,
            &split.train,
            &split.holdout,
            k_global,
            1e-8,
            ModelKind::Linear,
            2,
            false,
        )
        .unwrap();

        assert!(
            mse_with_sensor < mse_target_only * 0.1,
            "expected richer sensors to improve prediction (target_only={} with_sensor={})",
            mse_target_only,
            mse_with_sensor
        );
    }

    #[test]
    fn delta_linear_beats_linear_when_persistence_dominates() {
        let (target, sensor_u) = persistence_dominated_series(900, 0.2);
        let emb = build_delay_embedding_multi(&target, &[target.clone(), sensor_u], 1, 1).unwrap();
        let split = split_indices(emb.len(), SplitMode::Chronological, 99, 0.7, 0.15).unwrap();
        let k_global = split.train.len();

        let linear_mse = evaluate_model_mse(
            &emb,
            &split.train,
            &split.holdout,
            k_global,
            1.0,
            ModelKind::Linear,
            2,
            false,
        )
        .unwrap();
        let delta_mse = evaluate_model_mse(
            &emb,
            &split.train,
            &split.holdout,
            k_global,
            1.0,
            ModelKind::DeltaLinear,
            2,
            false,
        )
        .unwrap();
        assert!(
            delta_mse < linear_mse * 0.5,
            "expected delta linear to outperform linear on persistence-dominated target (linear={} delta={})",
            linear_mse,
            delta_mse
        );
    }

    #[test]
    fn delta_mlp_beats_persistence_on_nonlinear_residual_fixture() {
        let (target, sensor_u) = nonlinear_delta_series(1200, 0.1);
        let emb = build_delay_embedding_multi(&target, &[target.clone(), sensor_u], 1, 1).unwrap();
        let split = split_indices(emb.len(), SplitMode::Chronological, 21, 0.7, 0.15).unwrap();

        let delta_mlp_mse = evaluate_model_mse(
            &emb,
            &split.train,
            &split.holdout,
            12,
            1e-6,
            ModelKind::DeltaMlp,
            2,
            false,
        )
        .unwrap();
        let baseline_mse = evaluate_persistence_mse(&emb, &split.holdout).unwrap();

        assert!(
            delta_mlp_mse < baseline_mse * 0.5,
            "expected delta_mlp to beat persistence on nonlinear residual fixture (baseline={} delta_mlp={})",
            baseline_mse,
            delta_mlp_mse
        );
    }

    #[test]
    fn delta_tcn_beats_persistence_on_nonlinear_residual_fixture() {
        let (target, sensor_u) = nonlinear_delta_series(1200, 0.1);
        let emb = build_delay_embedding_multi(&target, &[target.clone(), sensor_u], 1, 1).unwrap();
        let split = split_indices(emb.len(), SplitMode::Chronological, 21, 0.7, 0.15).unwrap();

        let delta_tcn_mse = evaluate_model_mse(
            &emb,
            &split.train,
            &split.holdout,
            12,
            1e-6,
            ModelKind::DeltaTcn,
            2,
            false,
        )
        .unwrap();
        let baseline_mse = evaluate_persistence_mse(&emb, &split.holdout).unwrap();

        assert!(
            delta_tcn_mse < baseline_mse * 0.75,
            "expected delta_tcn to beat persistence on nonlinear residual fixture (baseline={} delta_tcn={})",
            baseline_mse,
            delta_tcn_mse
        );
    }

    #[test]
    fn recursive_horizon_metric_is_available_for_single_sensor() {
        let series = logistic_series(800, 3.75, 0.23);
        let emb = build_delay_embedding(&series, 1, 3).unwrap();
        let split = split_indices(emb.len(), SplitMode::Chronological, 7, 0.7, 0.15).unwrap();
        let cfg = TakensConfig {
            tau: 1,
            m: 3,
            k: 12,
            lambda: 1e-6,
            model: ModelKind::DeltaLinear,
            split_mode: SplitMode::Chronological,
        };
        let metric = evaluate_recursive_horizon_metric(
            &emb,
            &series,
            &split.train,
            &split.holdout,
            &cfg,
            1,
            false,
            4,
        )
        .unwrap()
        .expect("horizon metric");
        assert_eq!(metric.horizon, 4);
        assert!(metric.n_eval > 0);
        assert!(metric.holdout_mse.is_finite());
        assert!(metric.holdout_baseline_mse.is_finite());
    }

    #[test]
    fn infer_model_architecture_reports_expected_counts_for_local_models() {
        let linear_cfg = TakensConfig {
            tau: 1,
            m: 5,
            k: 10,
            lambda: 1e-6,
            model: ModelKind::Linear,
            split_mode: SplitMode::Chronological,
        };
        let linear = infer_model_architecture(&linear_cfg, 2).expect("linear arch");
        assert_eq!(linear.model_family, "local_linear");
        assert_eq!(linear.feature_map, "affine");
        assert_eq!(linear.input_dim, 10);
        assert_eq!(linear.feature_dim, 11);
        assert_eq!(linear.depth, 1);
        assert_eq!(linear.width, 11);
        assert_eq!(linear.hidden_layers, 0);
        assert!(linear.hidden_widths.is_empty());
        assert_eq!(linear.parameter_count, 11);

        let rational_cfg = TakensConfig {
            model: ModelKind::DeltaRational,
            ..linear_cfg
        };
        let rational = infer_model_architecture(&rational_cfg, 2).expect("delta rational arch");
        assert_eq!(rational.model_family, "delta_local_rational");
        assert_eq!(rational.feature_dim, 11);
        assert_eq!(rational.parameter_count, 22);
    }

    #[test]
    fn infer_model_architecture_reports_expected_counts_for_neural_models() {
        let mlp_cfg = TakensConfig {
            tau: 1,
            m: 4,
            k: 12,
            lambda: 1e-6,
            model: ModelKind::DeltaMlp,
            split_mode: SplitMode::Chronological,
        };
        let mlp = infer_model_architecture(&mlp_cfg, 3).expect("delta mlp arch");
        assert_eq!(mlp.model_family, "delta_neural");
        assert_eq!(mlp.feature_map, "delta_window");
        assert_eq!(mlp.input_dim, 12);
        assert_eq!(mlp.feature_dim, 21);
        assert_eq!(mlp.depth, 3);
        assert_eq!(mlp.width, 12);
        assert_eq!(mlp.hidden_layers, 2);
        assert_eq!(mlp.hidden_widths, vec![12, 6]);
        assert_eq!(mlp.parameter_count, 349);

        let tcn_cfg = TakensConfig {
            model: ModelKind::DeltaTcn,
            ..mlp_cfg
        };
        let tcn = infer_model_architecture(&tcn_cfg, 3).expect("delta tcn arch");
        assert_eq!(tcn.model_family, "delta_neural");
        assert_eq!(tcn.feature_map, "temporal_conv");
        assert_eq!(tcn.feature_dim, 33);
        assert_eq!(tcn.hidden_widths, vec![12, 6]);
        assert_eq!(tcn.parameter_count, 493);
    }

    #[test]
    fn sparse_sensor_gates_keep_multiple_channels_active() {
        let n = 240usize;
        let n_sensors = 10usize;
        let n_lags = 2usize;
        let mut x_norm = Vec::with_capacity(n);
        let mut y_norm = Vec::with_capacity(n);
        for t in 0..n {
            let tt = t as f64;
            let mut row = Vec::with_capacity(n_sensors * n_lags);
            for lag in 0..n_lags {
                for s in 0..n_sensors {
                    let sf = s as f64 + 1.0;
                    let lf = lag as f64;
                    let val =
                        (0.017 * tt + 0.11 * sf + 0.07 * lf).sin() + 0.35 * (0.013 * tt * sf).cos();
                    row.push(val);
                }
            }
            let y = 0.30 * row[0]
                + 0.22 * row[1]
                + 0.18 * row[2]
                + 0.12 * row[3]
                + 0.06 * row[4]
                + 0.02 * row[8];
            x_norm.push(row);
            y_norm.push(y);
        }

        let gates = super::learn_sparse_sensor_gates(&x_norm, &y_norm, n_sensors);
        assert_eq!(gates.len(), n_sensors);
        assert!(gates.iter().all(|g| g.is_finite() && *g >= 0.0 && *g <= 1.0));
        let active = gates.iter().filter(|g| **g >= 0.35).count();
        assert!(
            active >= 3,
            "expected at least 3 active sensors, got {} gates={:?}",
            active,
            gates
        );
    }

    #[test]
    fn sparse_sensor_gates_handles_single_sensor_without_panic() {
        let n = 128usize;
        let mut x_norm = Vec::with_capacity(n);
        let mut y_norm = Vec::with_capacity(n);
        for t in 0..n {
            let tt = t as f64;
            let x = (0.05 * tt).sin() + 0.3 * (0.011 * tt).cos();
            x_norm.push(vec![x, x * 0.9]);
            y_norm.push(0.7 * x + 0.1 * x.sin());
        }
        let gates = super::learn_sparse_sensor_gates(&x_norm, &y_norm, 1);
        assert_eq!(gates.len(), 1);
        assert!(gates[0].is_finite());
        assert!(gates[0] >= 0.0 && gates[0] <= 1.0);
    }

    #[test]
    fn summarize_sensor_gates_reports_ranked_and_active_counts() {
        let sensors = vec![
            "s0".to_string(),
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
            "s4".to_string(),
        ];
        let gates = vec![0.15, 0.72, 0.38, 0.21, 0.64];
        let (summary, ranked) = super::summarize_sensor_gates(&sensors, &gates).unwrap();
        assert_eq!(ranked.len(), sensors.len());
        assert_eq!(ranked[0].sensor, "s1");
        assert_eq!(ranked[1].sensor, "s4");
        assert_eq!(summary.active_threshold, super::SENSOR_GATE_ACTIVE_THRESHOLD);
        assert_eq!(summary.active_count, 3);
        assert!((summary.active_fraction - 0.6).abs() < 1e-12);
        assert!(summary.effective_sensor_count >= 1.0);
        assert!(summary.effective_sensor_count <= sensors.len() as f64);
    }
}
