use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand};
use threebody_core::config::{Config, IntegratorKind};
use threebody_core::frames::to_barycentric;
use threebody_core::forces::{compute_accel, ForceConfig};
use threebody_core::integrators::{boris::Boris, implicit_midpoint::ImplicitMidpoint, leapfrog::Leapfrog, rk45::Rk45, Integrator};
use threebody_core::math::vec3::Vec3;
use threebody_core::output::csv::write_csv;
use threebody_core::output::sidecar::{build_sidecar, write_sidecar};
use threebody_core::sim::{simulate, SimOptions};
use threebody_core::state::{Body, State, System};
use threebody_discover::ga::DiscoveryConfig;
use threebody_discover::library::FeatureLibrary;
use threebody_discover::judge::{
    CandidateMetrics, CandidateSummary, DatasetSummary, FeatureDescription, IcBounds, IcRequest, InitialConditionSpec,
    JudgeInput, JudgeResponse, Rubric, SimulationSummary,
};
use threebody_discover::llm::{LlmClient, MockLlm, OpenAIClient};
use threebody_discover::{grid_search, run_search, Dataset};

#[derive(Parser)]
#[command(name = "threebody-cli")]
#[command(about = "Three-body simulator CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Emit an example config JSON.
    ExampleConfig {
        /// Output path (stdout if omitted).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Run a simulation.
    Simulate {
        /// Config JSON path.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Output CSV path.
        #[arg(long, default_value = "traj.csv")]
        output: PathBuf,
        /// Steps to simulate.
        #[arg(long, default_value_t = 100)]
        steps: usize,
        /// Timestep.
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
        /// Mode: standard or truth (adaptive RK45 with strict tolerances).
        #[arg(long, default_value = "standard")]
        mode: String,
        /// Preset initial conditions.
        #[arg(long, default_value = "two-body")]
        preset: String,
        /// Override integrator.
        #[arg(long)]
        integrator: Option<String>,
        /// Disable EM.
        #[arg(long)]
        no_em: bool,
        /// Disable gravity.
        #[arg(long)]
        no_gravity: bool,
        /// Output format (csv only for now).
        #[arg(long, default_value = "csv")]
        format: String,
        /// Dry run (no files written).
        #[arg(long)]
        dry_run: bool,
        /// Print summary diagnostics.
        #[arg(long)]
        summary: bool,
    },
    /// Alias for simulate.
    Run {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, default_value = "traj.csv")]
        output: PathBuf,
        #[arg(long, default_value_t = 100)]
        steps: usize,
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
        #[arg(long, default_value = "standard")]
        mode: String,
        #[arg(long, default_value = "two-body")]
        preset: String,
        #[arg(long)]
        integrator: Option<String>,
        #[arg(long)]
        no_em: bool,
        #[arg(long)]
        no_gravity: bool,
        #[arg(long, default_value = "csv")]
        format: String,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        summary: bool,
    },
    /// Run the equation discovery loop.
    Discover {
        /// Number of GA runs.
        #[arg(long, default_value_t = 50)]
        runs: usize,
        /// Population size.
        #[arg(long, default_value_t = 20)]
        population: usize,
        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Output JSON path.
        #[arg(long, default_value = "top_equations.json")]
        out: PathBuf,
        /// Enable LLM ranking/interpretation.
        #[arg(long)]
        llm: bool,
        /// LLM model name.
        #[arg(long, default_value = "gpt-5")]
        model: String,
    },
    /// Run the LLM-assisted factory loop (ICs -> sim -> discovery -> judge).
    Factory {
        /// Output directory for all artifacts.
        #[arg(long, default_value = "factory_out")]
        out_dir: PathBuf,
        /// Max iterations to run.
        #[arg(long, default_value_t = 1)]
        max_iters: usize,
        /// Run without prompts.
        #[arg(long)]
        auto: bool,
        /// Config JSON path.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Steps to simulate.
        #[arg(long, default_value_t = 200)]
        steps: usize,
        /// Timestep.
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
        /// Mode: standard or truth (adaptive RK45).
        #[arg(long, default_value = "standard")]
        mode: String,
        /// Preset used when LLM is off or fails.
        #[arg(long, default_value = "two-body")]
        preset: String,
        /// Disable EM.
        #[arg(long)]
        no_em: bool,
        /// Disable gravity.
        #[arg(long)]
        no_gravity: bool,
        /// Number of GA runs.
        #[arg(long, default_value_t = 50)]
        runs: usize,
        /// Population size.
        #[arg(long, default_value_t = 20)]
        population: usize,
        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// LLM mode: off, mock, openai.
        #[arg(long, default_value = "off")]
        llm_mode: String,
        /// LLM model name (openai mode).
        #[arg(long, default_value = "gpt-5")]
        model: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::ExampleConfig { out } => {
            let cfg = Config::default();
            let json = serde_json::to_string_pretty(&cfg)?;
            if let Some(path) = out {
                fs::write(path, json)?;
            } else {
                println!("{json}");
            }
        }
        Commands::Simulate {
            config,
            output,
            steps,
            dt,
            mode,
            preset,
            integrator,
            no_em,
            no_gravity,
            format,
            dry_run,
            summary,
        } => {
            run_simulation(
                config,
                output,
                steps,
                dt,
                mode,
                preset,
                integrator,
                no_em,
                no_gravity,
                format,
                dry_run,
                summary,
            )?;
        }
        Commands::Run {
            config,
            output,
            steps,
            dt,
            mode,
            preset,
            integrator,
            no_em,
            no_gravity,
            format,
            dry_run,
            summary,
        } => {
            run_simulation(
                config,
                output,
                steps,
                dt,
                mode,
                preset,
                integrator,
                no_em,
                no_gravity,
                format,
                dry_run,
                summary,
            )?;
        }
        Commands::Discover {
            runs,
            population,
            seed,
            out,
            llm,
            model,
        } => {
            run_discovery(runs, population, seed, out, llm, model)?;
        }
        Commands::Factory {
            out_dir,
            max_iters,
            auto,
            config,
            steps,
            dt,
            mode,
            preset,
            no_em,
            no_gravity,
            runs,
            population,
            seed,
            llm_mode,
            model,
        } => {
            run_factory(
                out_dir,
                max_iters,
                auto,
                config,
                steps,
                dt,
                mode,
                preset,
                no_em,
                no_gravity,
                runs,
                population,
                seed,
                llm_mode,
                model,
            )?;
        }
    }
    Ok(())
}

fn run_simulation(
    config_path: Option<PathBuf>,
    output: PathBuf,
    steps: usize,
    dt: f64,
    mode: String,
    preset: String,
    integrator_override: Option<String>,
    no_em: bool,
    no_gravity: bool,
    format: String,
    dry_run: bool,
    summary: bool,
) -> anyhow::Result<()> {
    if format != "csv" {
        anyhow::bail!("unsupported format: {format}");
    }
    let mut cfg: Config = if let Some(path) = config_path.clone() {
        if path.exists() {
            let cfg_json = fs::read_to_string(&path)?;
            serde_json::from_str(&cfg_json)?
        } else {
            eprintln!("config not found, using defaults: {}", path.display());
            Config::default()
        }
    } else {
        Config::default()
    };
    if no_em {
        cfg.enable_em = false;
    }
    if no_gravity {
        cfg.enable_gravity = false;
    }
    if let Some(name) = integrator_override {
        cfg.integrator.kind = match name.as_str() {
            "leapfrog" => IntegratorKind::Leapfrog,
            "rk45" => IntegratorKind::Rk45,
            "boris" => IntegratorKind::Boris,
            "implicit_midpoint" => IntegratorKind::ImplicitMidpoint,
            _ => anyhow::bail!("unknown integrator: {name}"),
        };
    }

    if mode == "truth" {
        cfg.integrator.kind = IntegratorKind::Rk45;
        cfg.integrator.adaptive = true;
        cfg.integrator.rtol = 1e-12;
        cfg.integrator.atol = 1e-14;
        cfg.integrator.dt_min = cfg.integrator.dt_min.min(1e-6);
        cfg.integrator.dt_max = cfg.integrator.dt_max.max(0.05);
    } else if mode != "standard" {
        anyhow::bail!("unknown mode: {mode}");
    }
    cfg.validate().map_err(anyhow::Error::msg)?;

    let system = preset_system(&preset)?;
    let options = SimOptions { steps, dt };
    let (integrator, encounter_integrator): (Box<dyn Integrator>, Option<Box<dyn Integrator>>) =
        match cfg.integrator.kind {
            IntegratorKind::Leapfrog => (Box::new(Leapfrog), None),
            IntegratorKind::Rk45 => (Box::new(Rk45), None),
            IntegratorKind::Boris => (Box::new(Boris), None),
            IntegratorKind::ImplicitMidpoint => (Box::new(ImplicitMidpoint), None),
        };

    let result = simulate(system, &cfg, integrator.as_ref(), encounter_integrator.as_deref(), options);

    if summary {
        if let Some(step) = result.steps.last() {
            println!("energy_proxy={}", step.diagnostics.energy_proxy);
            println!("min_pair_dist={}", step.regime.min_pair_dist);
        }
    }

    if dry_run {
        return Ok(());
    }

    let mut csv_file = fs::File::create(&output)?;
    write_csv(&mut csv_file, &result.steps, &cfg)?;

    let header = threebody_core::output::csv::csv_header(&cfg);
    let sidecar = build_sidecar(&cfg, &header, &result);
    let sidecar_path = output.with_extension("json");
    let mut sidecar_file = fs::File::create(sidecar_path)?;
    write_sidecar(&mut sidecar_file, &sidecar)?;

    Ok(())
}

fn preset_system(name: &str) -> anyhow::Result<System> {
    match name {
        "two-body" => {
            let bodies = [Body::new(1.0, 0.0), Body::new(1.0, 0.0), Body::new(0.0, 0.0)];
            let pos = [Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0), Vec3::zero()];
            let v = (0.5_f64).sqrt();
            let vel = [Vec3::new(0.0, v, 0.0), Vec3::new(0.0, -v, 0.0), Vec3::zero()];
            Ok(System::new(bodies, State::new(pos, vel)))
        }
        "static" => {
            let bodies = [Body::new(1.0, 0.0), Body::new(1.0, 0.0), Body::new(1.0, 0.0)];
            let pos = [Vec3::zero(); 3];
            let vel = [Vec3::zero(); 3];
            Ok(System::new(bodies, State::new(pos, vel)))
        }
        _ => anyhow::bail!("unknown preset: {name}"),
    }
}

fn run_discovery(
    runs: usize,
    population: usize,
    seed: u64,
    out: PathBuf,
    llm: bool,
    model: String,
) -> anyhow::Result<()> {
    let dataset = discovery_dataset();
    let library = FeatureLibrary::default_physics();
    let cfg = DiscoveryConfig {
        runs,
        population,
        seed,
        ..DiscoveryConfig::default()
    };
    let llm_client = if llm { Some(OpenAIClient::from_env(&model)?) } else { None };

    let topk = run_search(&dataset, &library, &cfg);
    let candidates: Vec<_> = topk.entries.iter().map(|e| e.equation.clone()).collect();
    let grid_topk = grid_search(&candidates, &dataset);

    let judge = if let Some(client) = llm_client.as_ref() {
        let judge_input = build_judge_input(&dataset, &library, &topk.entries, None, "toy_dataset");
        client.judge_candidates(&judge_input).ok().map(|r| r.value)
    } else {
        None
    };

    #[derive(serde::Serialize)]
    struct Output {
        top3: Vec<threebody_discover::EquationScore>,
        grid_top3: Vec<threebody_discover::EquationScore>,
        judge: Option<JudgeResponse>,
    }

    let output = Output {
        top3: topk.entries.clone(),
        grid_top3: grid_topk.entries.clone(),
        judge,
    };
    let json = serde_json::to_string_pretty(&output)?;
    fs::write(out, json)?;
    Ok(())
}

fn discovery_dataset() -> Dataset {
    let feature_names = vec!["x".to_string(), "x2".to_string()];
    let mut samples = Vec::new();
    let mut targets = Vec::new();
    for i in 1..=5 {
        let x = i as f64;
        samples.push(vec![x, x * x]);
        targets.push(x);
    }
    Dataset::new(feature_names, samples, targets)
}
