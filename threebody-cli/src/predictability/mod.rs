use std::path::PathBuf;

use clap::Subcommand;

mod detect;
mod ensemble;
mod extract;
mod features;
mod report;
mod takens;
mod train_map;

pub const PREDICTABILITY_VERSION: &str = "v1";

#[derive(Subcommand, Debug)]
pub enum PredictabilityCommand {
    /// Extract encounter events from a single trajectory (event-driven view of chaos).
    Extract {
        /// Input CSV path.
        #[arg(long, default_value = "traj.csv")]
        input: PathBuf,
        /// Sidecar JSON path (defaults to <input>.json when omitted).
        #[arg(long)]
        sidecar: Option<PathBuf>,
        /// Output JSONL path.
        #[arg(long, default_value = "encounters.jsonl")]
        out: PathBuf,
        /// Output summary JSON path.
        #[arg(long, default_value = "encounters_summary.json")]
        summary_out: PathBuf,
        /// Minimum separation (local minima) threshold; 0 disables filtering.
        #[arg(long, default_value_t = 0.0)]
        min_event_dist_max: f64,
        /// Pre-window duration (seconds) used to compute features.
        #[arg(long, default_value_t = 0.05)]
        pre_window: f64,
        /// Post-window duration (seconds) used to compute labels.
        #[arg(long, default_value_t = 0.05)]
        post_window: f64,
    },
    /// Run a small ensemble of nearby initial conditions (conditioning probe).
    Ensemble {
        /// Config JSON path (optional; defaults to Config::default()).
        #[arg(long)]
        config: Option<PathBuf>,
        /// Base initial condition JSON path.
        #[arg(long)]
        ic: PathBuf,
        /// Output directory.
        #[arg(long, default_value = "ensemble_out")]
        out_dir: PathBuf,
        /// Number of ensemble members.
        #[arg(long, default_value_t = 16)]
        n: usize,
        /// Position perturbation σ (applied per component).
        #[arg(long, default_value_t = 1e-3)]
        sigma_pos: f64,
        /// Velocity perturbation σ (applied per component).
        #[arg(long, default_value_t = 1e-3)]
        sigma_vel: f64,
        /// RNG seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Steps to simulate.
        #[arg(long, default_value_t = 300)]
        steps: usize,
        /// Base dt (RK45 may adapt in truth mode).
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
        /// Mode: standard or truth.
        #[arg(long, default_value = "truth")]
        mode: String,
        /// Enable EM (overrides config default).
        #[arg(long)]
        em: bool,
        /// Disable EM.
        #[arg(long)]
        no_em: bool,
        /// Disable gravity.
        #[arg(long)]
        no_gravity: bool,
    },
    /// Detect when predictions become well-conditioned ("lock points") from an ensemble directory.
    Detect {
        /// Ensemble directory created by `predictability ensemble`.
        #[arg(long, default_value = "ensemble_out")]
        ensemble_dir: PathBuf,
        /// Output JSONL path for per-time lock points.
        #[arg(long, default_value = "lock_points.jsonl")]
        out: PathBuf,
        /// Output summary JSON path.
        #[arg(long, default_value = "predictability.json")]
        summary_out: PathBuf,
        /// Minimum mode fraction to accept categorical outcome lock.
        #[arg(long, default_value_t = 0.9)]
        min_mode_frac: f64,
        /// Window length in steps for sustained lock.
        #[arg(long, default_value_t = 20)]
        window: usize,
    },
    /// Train a lightweight encounter map model from extracted events.
    TrainMap {
        /// Input encounters JSONL path.
        #[arg(long, default_value = "encounters.jsonl")]
        encounters: PathBuf,
        /// Output model JSON path.
        #[arg(long, default_value = "encounter_map.json")]
        out: PathBuf,
        /// RNG seed for split.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Fit a Takens delay-embedding local map with strict holdout evaluation.
    Takens {
        /// Input CSV path.
        #[arg(long, default_value = "traj.csv")]
        input: PathBuf,
        /// Numeric column name to model.
        #[arg(long, default_value = "min_pair_dist")]
        column: String,
        /// Optional sensor columns (comma-separated). When omitted, uses only --column.
        #[arg(long, default_value = "")]
        sensors: String,
        /// Output report JSON path.
        #[arg(long, default_value = "takens_report.json")]
        out: PathBuf,
        /// Delay values (comma-separated positive integers).
        #[arg(long, default_value = "1,2,3")]
        tau: String,
        /// Embedding dimensions (comma-separated positive integers).
        #[arg(long, default_value = "3,4,5")]
        m: String,
        /// kNN neighborhood sizes (comma-separated positive integers).
        #[arg(long, default_value = "8,12,16")]
        k: String,
        /// Ridge lambdas (comma-separated nonnegative floats).
        #[arg(long, default_value = "1e-8,1e-6,1e-4,1e-2")]
        lambda: String,
        /// Model family: linear | rational | delta_linear | delta_rational | both | all.
        #[arg(long, default_value = "all")]
        model: String,
        /// Split mode: chronological | shuffled.
        #[arg(long, default_value = "chronological")]
        split_mode: String,
        /// RNG seed used when split mode is shuffled.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Train fraction in (0,1).
        #[arg(long, default_value_t = 0.7)]
        train_frac: f64,
        /// Validation fraction in [0,1); holdout is the remainder.
        #[arg(long, default_value_t = 0.15)]
        val_frac: f64,
        /// Weight for sensitivity consistency in selection score.
        #[arg(long, default_value_t = 0.1)]
        sensitivity_weight: f64,
    },
    /// Summarize efficacy from one or more predictability report JSON files.
    Report {
        /// Input report JSON files (comma-separated or repeated).
        #[arg(long, value_delimiter = ',', num_args = 1.., required = true)]
        reports: Vec<PathBuf>,
        /// Output summary JSON path.
        #[arg(long, default_value = "efficacy_report.json")]
        out: PathBuf,
        /// Optional Markdown summary output path.
        #[arg(long)]
        markdown_out: Option<PathBuf>,
        /// Minimum relative-improvement threshold for marking a channel effective.
        #[arg(long, default_value_t = 0.0)]
        improvement_threshold: f64,
        /// Maximum allowed sensitivity-median threshold for marking a channel effective.
        #[arg(long, default_value_t = 0.1)]
        max_sensitivity_median: f64,
        /// Number of bootstrap resamples for median CI estimates (0 disables CIs).
        #[arg(long, default_value_t = 2000)]
        bootstrap_resamples: usize,
        /// Confidence level for bootstrap CIs in (0,1), typically 0.95.
        #[arg(long, default_value_t = 0.95)]
        bootstrap_ci: f64,
        /// RNG seed for deterministic bootstrap resampling.
        #[arg(long, default_value_t = 42)]
        bootstrap_seed: u64,
    },
    /// Compare two efficacy reports (before vs after) with effect-size and non-regression flags.
    Compare {
        /// Baseline efficacy report JSON path.
        #[arg(long)]
        before: PathBuf,
        /// Candidate efficacy report JSON path.
        #[arg(long)]
        after: PathBuf,
        /// Output comparison JSON path.
        #[arg(long, default_value = "efficacy_compare.json")]
        out: PathBuf,
        /// Optional Markdown summary output path.
        #[arg(long)]
        markdown_out: Option<PathBuf>,
        /// Allowed negative delta tolerance for non-regression checks.
        #[arg(long, default_value_t = 0.0)]
        non_regression_tol: f64,
    },
    /// Forecast with lock detection and optional encounter map (reserved for v1).
    Forecast {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        ic: PathBuf,
        #[arg(long, default_value_t = 300)]
        steps: usize,
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
        #[arg(long, default_value = "truth")]
        mode: String,
    },
}

pub fn run(cmd: PredictabilityCommand) -> anyhow::Result<()> {
    match cmd {
        PredictabilityCommand::Extract {
            input,
            sidecar,
            out,
            summary_out,
            min_event_dist_max,
            pre_window,
            post_window,
        } => extract::run_extract(
            input,
            sidecar,
            out,
            summary_out,
            min_event_dist_max,
            pre_window,
            post_window,
        ),
        PredictabilityCommand::Ensemble {
            config,
            ic,
            out_dir,
            n,
            sigma_pos,
            sigma_vel,
            seed,
            steps,
            dt,
            mode,
            em,
            no_em,
            no_gravity,
        } => ensemble::run_ensemble(
            config, ic, out_dir, n, sigma_pos, sigma_vel, seed, steps, dt, mode, em, no_em,
            no_gravity,
        ),
        PredictabilityCommand::Detect {
            ensemble_dir,
            out,
            summary_out,
            min_mode_frac,
            window,
        } => detect::run_detect(ensemble_dir, out, summary_out, min_mode_frac, window),
        PredictabilityCommand::TrainMap {
            encounters,
            out,
            seed,
        } => train_map::run_train_map(encounters, out, seed),
        PredictabilityCommand::Takens {
            input,
            column,
            sensors,
            out,
            tau,
            m,
            k,
            lambda,
            model,
            split_mode,
            seed,
            train_frac,
            val_frac,
            sensitivity_weight,
        } => takens::run_takens(
            input,
            column,
            sensors,
            out,
            tau,
            m,
            k,
            lambda,
            model,
            split_mode,
            seed,
            train_frac,
            val_frac,
            sensitivity_weight,
        ),
        PredictabilityCommand::Report {
            reports,
            out,
            markdown_out,
            improvement_threshold,
            max_sensitivity_median,
            bootstrap_resamples,
            bootstrap_ci,
            bootstrap_seed,
        } => report::run_report(
            reports,
            out,
            markdown_out,
            improvement_threshold,
            max_sensitivity_median,
            bootstrap_resamples,
            bootstrap_ci,
            bootstrap_seed,
        ),
        PredictabilityCommand::Compare {
            before,
            after,
            out,
            markdown_out,
            non_regression_tol,
        } => report::run_compare(before, after, out, markdown_out, non_regression_tol),
        PredictabilityCommand::Forecast { .. } => {
            anyhow::bail!("predictability forecast not implemented yet")
        }
    }
}
