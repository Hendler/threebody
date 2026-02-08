use std::path::PathBuf;

use clap::Subcommand;

mod detect;
mod ensemble;
mod extract;
mod features;
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
        PredictabilityCommand::Forecast { .. } => {
            anyhow::bail!("predictability forecast not implemented yet")
        }
    }
}
