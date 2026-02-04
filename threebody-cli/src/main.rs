use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use threebody_core::config::{Config, IntegratorKind};
use threebody_core::diagnostics::compute_diagnostics;
use threebody_core::frames::to_barycentric;
use threebody_core::forces::{compute_accel, ForceConfig};
use threebody_core::integrators::{boris::Boris, implicit_midpoint::ImplicitMidpoint, leapfrog::Leapfrog, rk45::Rk45, Integrator};
use threebody_core::math::vec3::Vec3;
use threebody_core::output::csv::write_csv;
use threebody_core::output::parse::{parse_header, require_columns};
use threebody_core::output::sidecar::{build_sidecar, write_sidecar};
use threebody_core::regime::compute_regime;
use threebody_core::sim::{simulate, SimOptions};
use threebody_core::state::{Body, State, System};
use threebody_discover::ga::DiscoveryConfig;
use threebody_discover::library::FeatureLibrary;
use threebody_discover::judge::{
    CandidateMetrics, CandidateSummary, DatasetSummary, FeatureDescription, IcBounds, IcRequest, InitialConditionSpec,
    JudgeInput, JudgeResponse, Rubric, SimulationSummary,
};
use threebody_discover::llm::{LlmClient, MockLlm, OpenAIClient};
use threebody_discover::{
    grid_search, lasso_path_search, run_search, stls_path_search, Dataset, FitnessHeuristic, LassoConfig, StlsConfig,
};

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
    /// Emit an example initial-conditions JSON.
    ExampleIc {
        /// Preset initial conditions to use.
        #[arg(long, default_value = "three-body")]
        preset: String,
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
        /// Initial conditions JSON path (overrides --preset).
        #[arg(long)]
        ic: Option<PathBuf>,
        /// Override integrator.
        #[arg(long)]
        integrator: Option<String>,
        /// Enable EM (overrides config default).
        #[arg(long)]
        em: bool,
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
        /// Initial conditions JSON path (overrides --preset).
        #[arg(long)]
        ic: Option<PathBuf>,
        #[arg(long)]
        integrator: Option<String>,
        /// Enable EM (overrides config default).
        #[arg(long)]
        em: bool,
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
        /// GA runs (only used with --solver ga).
        #[arg(long, default_value_t = 50)]
        runs: usize,
        /// GA population size (only used with --solver ga).
        #[arg(long, default_value_t = 20)]
        population: usize,
        /// GA random seed (only used with --solver ga).
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Output JSON path.
        #[arg(long, default_value = "top_equations.json")]
        out: PathBuf,
        /// Input CSV path (defaults to ./traj.csv).
        #[arg(long, default_value = "traj.csv")]
        input: PathBuf,
        /// Sidecar JSON path (defaults to <input>.json).
        #[arg(long)]
        sidecar: Option<PathBuf>,
        /// Discover from a single body only (0, 1, or 2). If omitted, uses all bodies.
        #[arg(long)]
        body: Option<usize>,
        /// Enable LLM ranking/interpretation.
        #[arg(long)]
        llm: bool,
        /// LLM model name.
        #[arg(long, default_value = "gpt-5")]
        model: String,
        /// OpenAI API key file (overrides OPENAI_API_KEY).
        #[arg(long)]
        openai_key_file: Option<PathBuf>,
        /// Fitness heuristic: mse | mse_parsimony.
        #[arg(long, default_value = "mse")]
        fitness: String,
        /// Rollout integrator for evaluation: euler | leapfrog.
        #[arg(long, default_value = "euler")]
        rollout_integrator: String,
        /// Discovery solver: stls | lasso | ga.
        #[arg(long, default_value = "stls")]
        solver: String,
        /// Disable column normalization for STLS/LASSO.
        #[arg(long)]
        no_normalize: bool,
        /// Ridge penalty λ for STLS.
        #[arg(long, default_value_t = 1e-8)]
        ridge_lambda: f64,
        /// STLS active-set refit iterations.
        #[arg(long, default_value_t = 25)]
        stls_max_iter: usize,
        /// Comma-separated STLS thresholds (overrides auto grid).
        #[arg(long)]
        stls_thresholds: Option<String>,
        /// LASSO coordinate-descent iterations.
        #[arg(long, default_value_t = 2000)]
        lasso_max_iter: usize,
        /// LASSO coordinate-descent tolerance.
        #[arg(long, default_value_t = 1e-6)]
        lasso_tol: f64,
        /// Comma-separated LASSO alphas (overrides auto grid).
        #[arg(long)]
        lasso_alphas: Option<String>,
    },
    /// Run the LLM-assisted factory loop (ICs -> sim -> discovery -> judge).
    #[command(alias = "experiment")]
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
        /// Enable EM (overrides config default).
        #[arg(long)]
        em: bool,
        /// Disable EM.
        #[arg(long)]
        no_em: bool,
        /// Disable gravity.
        #[arg(long)]
        no_gravity: bool,
        /// GA runs (only used with --solver ga).
        #[arg(long, default_value_t = 50)]
        runs: usize,
        /// GA population size (only used with --solver ga).
        #[arg(long, default_value_t = 20)]
        population: usize,
        /// Random seed (used for IC proposals; also used with --solver ga).
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Fitness heuristic: mse | mse_parsimony.
        #[arg(long, default_value = "mse")]
        fitness: String,
        /// Rollout integrator for evaluation: euler | leapfrog.
        #[arg(long, default_value = "euler")]
        rollout_integrator: String,
        /// Discovery solver: stls | lasso | ga.
        #[arg(long, default_value = "stls")]
        solver: String,
        /// Disable column normalization for STLS/LASSO.
        #[arg(long)]
        no_normalize: bool,
        /// Ridge penalty λ for STLS.
        #[arg(long, default_value_t = 1e-8)]
        ridge_lambda: f64,
        /// STLS active-set refit iterations.
        #[arg(long, default_value_t = 25)]
        stls_max_iter: usize,
        /// Comma-separated STLS thresholds (overrides auto grid).
        #[arg(long)]
        stls_thresholds: Option<String>,
        /// LASSO coordinate-descent iterations.
        #[arg(long, default_value_t = 2000)]
        lasso_max_iter: usize,
        /// LASSO coordinate-descent tolerance.
        #[arg(long, default_value_t = 1e-6)]
        lasso_tol: f64,
        /// Comma-separated LASSO alphas (overrides auto grid).
        #[arg(long)]
        lasso_alphas: Option<String>,
        /// LLM mode: off, mock, openai.
        #[arg(long, default_value = "mock")]
        llm_mode: String,
        /// LLM model name (openai mode).
        #[arg(long, default_value = "gpt-5")]
        model: String,
        /// OpenAI API key file (overrides OPENAI_API_KEY).
        #[arg(long)]
        openai_key_file: Option<PathBuf>,
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
        Commands::ExampleIc { preset, out } => {
            let ic = initial_conditions_from_preset(&preset)?;
            let json = serde_json::to_string_pretty(&ic)?;
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
            ic,
            integrator,
            em,
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
                ic,
                integrator,
                em,
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
            ic,
            integrator,
            em,
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
                ic,
                integrator,
                em,
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
            input,
            sidecar,
            body,
            llm,
            model,
            openai_key_file,
            fitness,
            rollout_integrator,
            solver,
            no_normalize,
            ridge_lambda,
            stls_max_iter,
            stls_thresholds,
            lasso_max_iter,
            lasso_tol,
            lasso_alphas,
        } => {
            let solver_settings = DiscoverySolverSettings {
                solver: parse_discovery_solver(&solver)?,
                normalize: !no_normalize,
                stls_thresholds: stls_thresholds
                    .as_deref()
                    .map(parse_csv_f64_list)
                    .transpose()?
                    .unwrap_or_default(),
                stls_ridge_lambda: ridge_lambda,
                stls_max_iter,
                lasso_alphas: lasso_alphas
                    .as_deref()
                    .map(parse_csv_f64_list)
                    .transpose()?
                    .unwrap_or_default(),
                lasso_max_iter,
                lasso_tol,
            };
            run_discovery(
                runs,
                population,
                seed,
                out,
                input,
                sidecar,
                body,
                llm,
                model,
                openai_key_file,
                fitness,
                rollout_integrator,
                solver_settings,
            )?;
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
            em,
            no_em,
            no_gravity,
            runs,
            population,
            seed,
            fitness,
            rollout_integrator,
            solver,
            no_normalize,
            ridge_lambda,
            stls_max_iter,
            stls_thresholds,
            lasso_max_iter,
            lasso_tol,
            lasso_alphas,
            llm_mode,
            model,
            openai_key_file,
        } => {
            let solver_settings = DiscoverySolverSettings {
                solver: parse_discovery_solver(&solver)?,
                normalize: !no_normalize,
                stls_thresholds: stls_thresholds
                    .as_deref()
                    .map(parse_csv_f64_list)
                    .transpose()?
                    .unwrap_or_default(),
                stls_ridge_lambda: ridge_lambda,
                stls_max_iter,
                lasso_alphas: lasso_alphas
                    .as_deref()
                    .map(parse_csv_f64_list)
                    .transpose()?
                    .unwrap_or_default(),
                lasso_max_iter,
                lasso_tol,
            };
            run_factory(
                out_dir,
                max_iters,
                auto,
                config,
                steps,
                dt,
                mode,
                preset,
                em,
                no_em,
                no_gravity,
                runs,
                population,
                seed,
                fitness,
                rollout_integrator,
                solver_settings,
                llm_mode,
                model,
                openai_key_file,
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
    ic_path: Option<PathBuf>,
    integrator_override: Option<String>,
    em: bool,
    no_em: bool,
    no_gravity: bool,
    format: String,
    dry_run: bool,
    summary: bool,
) -> anyhow::Result<()> {
    if format != "csv" {
        anyhow::bail!("unsupported format: {format}");
    }
    let cfg = build_config(config_path, &mode, integrator_override, em, no_em, no_gravity)?;
    let system = if let Some(path) = ic_path {
        let json = fs::read_to_string(&path)?;
        let spec: InitialConditionSpec = serde_json::from_str(&json)?;
        system_from_ic(&spec, &default_ic_bounds())?
    } else {
        preset_system(&preset)?
    };
    let options = SimOptions { steps, dt };
    let result = simulate_with_cfg(system, &cfg, options);

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

fn build_config(
    config_path: Option<PathBuf>,
    mode: &str,
    integrator_override: Option<String>,
    em: bool,
    no_em: bool,
    no_gravity: bool,
) -> anyhow::Result<Config> {
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
    if em && no_em {
        anyhow::bail!("conflicting flags: --em and --no-em");
    }
    if em {
        cfg.enable_em = true;
    }
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
    Ok(cfg)
}

fn simulate_with_cfg(system: System, cfg: &Config, options: SimOptions) -> threebody_core::sim::SimResult {
    let (integrator, encounter_integrator): (Box<dyn Integrator>, Option<Box<dyn Integrator>>) =
        match cfg.integrator.kind {
            IntegratorKind::Leapfrog => (Box::new(Leapfrog), None),
            IntegratorKind::Rk45 => (Box::new(Rk45), None),
            IntegratorKind::Boris => (Box::new(Boris), None),
            IntegratorKind::ImplicitMidpoint => (Box::new(ImplicitMidpoint), None),
        };
    simulate(system, cfg, integrator.as_ref(), encounter_integrator.as_deref(), options)
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
        "three-body" => {
            // Lagrange equilateral solution (normalized): three equal masses at an equilateral triangle,
            // rotating about the center of mass. Assumes G=1, m=1, side length L=1 -> speed v = sqrt(G*m/L) = 1.
            let bodies = [Body::new(1.0, 0.0), Body::new(1.0, 0.0), Body::new(1.0, 0.0)];
            let r = 1.0 / 3.0_f64.sqrt(); // circumradius for side length 1
            let pos = [
                Vec3::new(r, 0.0, 0.0),
                Vec3::new(-0.5 * r, 0.5, 0.0),
                Vec3::new(-0.5 * r, -0.5, 0.0),
            ];
            let vel = [
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(-0.866_025_403_784_438_6, -0.5, 0.0),
                Vec3::new(0.866_025_403_784_438_6, -0.5, 0.0),
            ];
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
    input: PathBuf,
    sidecar_path: Option<PathBuf>,
    body: Option<usize>,
    llm: bool,
    model: String,
    openai_key_file: Option<PathBuf>,
    fitness: String,
    rollout_integrator: String,
    solver_settings: DiscoverySolverSettings,
) -> anyhow::Result<()> {
    let library = FeatureLibrary::default_physics();
    let fitness = parse_fitness_heuristic(&fitness)?;
    let rollout_integrator = parse_rollout_integrator(&rollout_integrator)?;
    let cfg = DiscoveryConfig {
        runs,
        population,
        seed,
        fitness,
        ..DiscoveryConfig::default()
    };
    let llm_client = if llm {
        Some(OpenAIClient::from_env_or_file(&model, openai_key_file.as_deref())?)
    } else {
        None
    };

    if let Some(b) = body {
        if b > 2 {
            anyhow::bail!("--body must be 0, 1, or 2");
        }
    }
    if !input.exists() {
        anyhow::bail!("input CSV not found: {}", input.display());
    }

    let sidecar_path = sidecar_path.unwrap_or_else(|| input.with_extension("json"));
    if !sidecar_path.exists() {
        anyhow::bail!(
            "sidecar JSON not found: {} (expected alongside CSV)",
            sidecar_path.display()
        );
    }
    let sidecar_json = fs::read_to_string(&sidecar_path)?;
    let sidecar: threebody_core::output::sidecar::Sidecar = serde_json::from_str(&sidecar_json)?;
    let sim_cfg = sidecar.config;
    sim_cfg.validate().map_err(anyhow::Error::msg)?;

    let init = sidecar
        .initial_state
        .ok_or_else(|| anyhow::anyhow!("sidecar missing initial_state; rerun simulate to regenerate"))?;
    let mut bodies = [Body::new(0.0, 0.0); 3];
    for i in 0..3 {
        bodies[i] = Body::new(init.mass[i], init.charge[i]);
    }
    let steps = load_steps_from_csv(&input, bodies, &sim_cfg)?;
    let result = threebody_core::sim::SimResult {
        steps,
        encounter: None,
        encounter_action: None,
        warnings: sidecar.warnings.clone(),
        terminated_early: false,
        termination_reason: None,
        stats: sidecar.sim_stats,
    };

    let regime = if sim_cfg.enable_em { "em_quasistatic" } else { "gravity_only" };

    let vector_data = build_vector_dataset(&result, &sim_cfg, &library.features, body);
    let [dataset_x, dataset_y, dataset_z] = component_datasets(&vector_data);

    let stls_cfg = StlsConfig {
        thresholds: solver_settings.stls_thresholds.clone(),
        ridge_lambda: solver_settings.stls_ridge_lambda,
        max_iter: solver_settings.stls_max_iter,
        normalize: solver_settings.normalize,
    };
    let lasso_cfg = LassoConfig {
        alphas: solver_settings.lasso_alphas.clone(),
        max_iter: solver_settings.lasso_max_iter,
        tol: solver_settings.lasso_tol,
        normalize: solver_settings.normalize,
    };
    let (topk_x, topk_y, topk_z) = match solver_settings.solver {
        DiscoverySolver::Ga => (
            run_search(&dataset_x, &library, &cfg),
            run_search(&dataset_y, &library, &cfg),
            run_search(&dataset_z, &library, &cfg),
        ),
        DiscoverySolver::Stls => (
            stls_path_search(&dataset_x, &stls_cfg, fitness),
            stls_path_search(&dataset_y, &stls_cfg, fitness),
            stls_path_search(&dataset_z, &stls_cfg, fitness),
        ),
        DiscoverySolver::Lasso => (
            lasso_path_search(&dataset_x, &lasso_cfg, fitness),
            lasso_path_search(&dataset_y, &lasso_cfg, fitness),
            lasso_path_search(&dataset_z, &lasso_cfg, fitness),
        ),
    };

    let vector_candidates = build_vector_candidates(
        &topk_x.entries,
        &topk_y.entries,
        &topk_z.entries,
        &vector_data.feature_names,
        &result,
        &sim_cfg,
        regime,
        rollout_integrator,
    );

    let grid_x = grid_search(
        &topk_x.entries.iter().map(|e| e.equation.clone()).collect::<Vec<_>>(),
        &dataset_x,
    );
    let grid_y = grid_search(
        &topk_y.entries.iter().map(|e| e.equation.clone()).collect::<Vec<_>>(),
        &dataset_y,
    );
    let grid_z = grid_search(
        &topk_z.entries.iter().map(|e| e.equation.clone()).collect::<Vec<_>>(),
        &dataset_z,
    );

    let simulation = build_sim_summary(&result, rollout_integrator);
    let dataset_summary = build_dataset_summary(
        &vector_data.feature_names,
        &library,
        vector_data.samples.len(),
        "accel_components_body_all",
    );
    let mut judge_input = build_judge_input_from_candidates(
        dataset_summary.clone(),
        vector_candidates.clone(),
        Some(simulation.clone()),
        regime,
    );
    judge_input
        .notes
        .push(format!("fitness_heuristic={}", fitness.as_str()));
    judge_input
        .notes
        .push(format!("discovery_solver={}", discovery_solver_label(solver_settings.solver)));
    judge_input.notes.push(format!("normalize={}", solver_settings.normalize));
    match solver_settings.solver {
        DiscoverySolver::Ga => {
            judge_input.notes.push(format!("ga_runs={runs}"));
            judge_input.notes.push(format!("ga_population={population}"));
        }
        DiscoverySolver::Stls => {
            judge_input
                .notes
                .push(format!("stls_ridge_lambda={}", solver_settings.stls_ridge_lambda));
            if solver_settings.stls_thresholds.is_empty() {
                judge_input.notes.push("stls_thresholds=auto".to_string());
            } else {
                judge_input
                    .notes
                    .push(format!("stls_thresholds={:?}", solver_settings.stls_thresholds));
            }
        }
        DiscoverySolver::Lasso => {
            judge_input
                .notes
                .push(format!("lasso_tol={}", solver_settings.lasso_tol));
            if solver_settings.lasso_alphas.is_empty() {
                judge_input.notes.push("lasso_alphas=auto".to_string());
            } else {
                judge_input
                    .notes
                    .push(format!("lasso_alphas={:?}", solver_settings.lasso_alphas));
            }
        }
    }

    let mut judge_prompt = None;
    let mut judge_response = None;
    let judge = if let Some(client) = llm_client.as_ref() {
        match client.judge_candidates(&judge_input) {
            Ok(result) => {
                judge_prompt = Some(result.prompt);
                judge_response = Some(result.response);
                let resp = result.value;
                if resp.validate(&judge_input).is_ok() {
                    Some(resp)
                } else {
                    eprintln!("LLM judge response failed validation");
                    None
                }
            }
            Err(err) => {
                eprintln!("LLM judge failed: {}", err);
                None
            }
        }
    } else {
        None
    };

    if let Some(prompt) = judge_prompt.as_ref() {
        fs::write(out.with_extension("judge_prompt.txt"), prompt)?;
    }
    if let Some(resp) = judge_response.as_ref() {
        fs::write(out.with_extension("judge_response.txt"), resp)?;
    }
    let solver_meta = build_solver_meta(&solver_settings, fitness, Some((runs, population, seed)));

    #[derive(serde::Serialize)]
    struct Output {
        input_csv: String,
        sidecar_json: String,
        regime: String,
        config: Config,
        dataset: DatasetSummary,
        simulation: SimulationSummary,
        solver: SolverMeta,
        top3: Vec<CandidateSummary>,
        component_top3: ComponentTop3,
        grid_top3: ComponentTop3,
        judge: Option<JudgeResponse>,
    }

    #[derive(serde::Serialize)]
    struct ComponentTop3 {
        x: Vec<threebody_discover::EquationScore>,
        y: Vec<threebody_discover::EquationScore>,
        z: Vec<threebody_discover::EquationScore>,
    }

    let output = Output {
        input_csv: input.display().to_string(),
        sidecar_json: sidecar_path.display().to_string(),
        regime: regime.to_string(),
        config: sim_cfg,
        dataset: dataset_summary,
        simulation,
        solver: solver_meta,
        top3: vector_candidates,
        component_top3: ComponentTop3 {
            x: topk_x.entries.clone(),
            y: topk_y.entries.clone(),
            z: topk_z.entries.clone(),
        },
        grid_top3: ComponentTop3 {
            x: grid_x.entries.clone(),
            y: grid_y.entries.clone(),
            z: grid_z.entries.clone(),
        },
        judge,
    };

    let json = serde_json::to_string_pretty(&output)?;
    fs::write(out, json)?;
    Ok(())
}

fn load_steps_from_csv(input: &PathBuf, bodies: [Body; 3], cfg: &Config) -> anyhow::Result<Vec<threebody_core::sim::SimStep>> {
    const REQUIRED: [&str; 20] = [
        "t",
        "dt",
        "r1_x",
        "r1_y",
        "r1_z",
        "r2_x",
        "r2_y",
        "r2_z",
        "r3_x",
        "r3_y",
        "r3_z",
        "v1_x",
        "v1_y",
        "v1_z",
        "v2_x",
        "v2_y",
        "v2_z",
        "v3_x",
        "v3_y",
        "v3_z",
    ];

    let file = fs::File::open(input)?;
    let mut reader = io::BufReader::new(file);
    let mut header_line = String::new();
    let n = reader.read_line(&mut header_line)?;
    if n == 0 {
        anyhow::bail!("empty CSV: {}", input.display());
    }
    let header = parse_header(&header_line);
    let map = require_columns(&header, &REQUIRED).map_err(anyhow::Error::msg)?;

    let force_cfg = ForceConfig {
        g: cfg.constants.g,
        k_e: cfg.constants.k_e,
        mu_0: cfg.constants.mu_0,
        epsilon: cfg.softening,
        enable_gravity: cfg.enable_gravity,
        enable_em: cfg.enable_em,
    };

    let mut steps = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        let get = |name: &str| -> anyhow::Result<&str> {
            let idx = *map
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("missing column in header: {name}"))?;
            cols.get(idx)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("row {} missing column: {name}", line_no + 2))
        };
        let parse_f64 = |name: &str| -> anyhow::Result<f64> {
            let raw = get(name)?;
            raw.parse::<f64>()
                .map_err(|e| anyhow::anyhow!("row {} invalid {name}={raw:?}: {e}", line_no + 2))
        };

        let t = parse_f64("t")?;
        let dt = parse_f64("dt")?;

        let pos = [
            Vec3::new(parse_f64("r1_x")?, parse_f64("r1_y")?, parse_f64("r1_z")?),
            Vec3::new(parse_f64("r2_x")?, parse_f64("r2_y")?, parse_f64("r2_z")?),
            Vec3::new(parse_f64("r3_x")?, parse_f64("r3_y")?, parse_f64("r3_z")?),
        ];
        let vel = [
            Vec3::new(parse_f64("v1_x")?, parse_f64("v1_y")?, parse_f64("v1_z")?),
            Vec3::new(parse_f64("v2_x")?, parse_f64("v2_y")?, parse_f64("v2_z")?),
            Vec3::new(parse_f64("v3_x")?, parse_f64("v3_y")?, parse_f64("v3_z")?),
        ];

        let system = System::new(bodies, State::new(pos, vel));
        let acc = compute_accel(&system, &force_cfg);
        let diagnostics = compute_diagnostics(&system, cfg);
        let regime = compute_regime(&system, &acc, dt);
        steps.push(threebody_core::sim::SimStep {
            system,
            diagnostics,
            regime,
            t,
            dt,
        });
    }

    if steps.is_empty() {
        anyhow::bail!("no data rows in CSV: {}", input.display());
    }
    Ok(steps)
}

fn build_judge_input_from_candidates(
    dataset_summary: DatasetSummary,
    candidates: Vec<CandidateSummary>,
    simulation: Option<SimulationSummary>,
    regime: &str,
) -> JudgeInput {
    JudgeInput {
        rubric: Rubric::default_rubric(),
        regime: regime.to_string(),
        dataset: dataset_summary,
        simulation,
        candidates,
        ic_bounds: default_ic_bounds(),
        notes: vec!["LLM is supplemental; do not override numeric ranking without evidence.".to_string()],
    }
}

fn build_dataset_summary(
    feature_names: &[String],
    library: &FeatureLibrary,
    n_samples: usize,
    target_description: &str,
) -> DatasetSummary {
    let mut desc_map = std::collections::HashMap::new();
    for fd in library.feature_descriptions() {
        desc_map.insert(fd.name.clone(), fd);
    }
    let feature_descriptions: Vec<FeatureDescription> = feature_names
        .iter()
        .map(|name| {
            desc_map.get(name).cloned().unwrap_or(FeatureDescription {
                name: name.clone(),
                description: "feature".to_string(),
                tags: vec![],
            })
        })
        .collect();
    DatasetSummary {
        n_samples,
        target_description: target_description.to_string(),
        feature_names: feature_names.to_vec(),
        feature_descriptions,
    }
}

fn stability_flags_for(eq: &threebody_discover::Equation, regime: &str) -> Vec<String> {
    let uses_velocity = eq.terms.iter().any(|t| t.feature.contains('v'));
    let mut flags = Vec::new();
    if regime == "gravity_only" && uses_velocity {
        flags.push("velocity_terms_in_gravity_only".to_string());
    }
    flags
}

fn default_ic_bounds() -> IcBounds {
    IcBounds {
        mass_min: 0.1,
        mass_max: 5.0,
        charge_min: -1.0,
        charge_max: 1.0,
        pos_min: -1.5,
        pos_max: 1.5,
        vel_min: -1.5,
        vel_max: 1.5,
        min_pair_dist: 0.2,
        recommend_barycentric: true,
    }
}

fn system_from_ic(spec: &InitialConditionSpec, bounds: &IcBounds) -> anyhow::Result<System> {
    if spec.bodies.len() != 3 {
        anyhow::bail!("initial conditions must include exactly 3 bodies");
    }
    let clamp = |v: f64, min: f64, max: f64| {
        if !v.is_finite() {
            min
        } else {
            v.min(max).max(min)
        }
    };
    let mut bodies = [Body::new(1.0, 0.0); 3];
    let mut pos = [Vec3::zero(); 3];
    let mut vel = [Vec3::zero(); 3];
    for (i, b) in spec.bodies.iter().enumerate() {
        let mass = clamp(b.mass, bounds.mass_min, bounds.mass_max);
        let charge = clamp(b.charge, bounds.charge_min, bounds.charge_max);
        bodies[i] = Body::new(mass, charge);
        pos[i] = Vec3::new(
            clamp(b.pos[0], bounds.pos_min, bounds.pos_max),
            clamp(b.pos[1], bounds.pos_min, bounds.pos_max),
            clamp(b.pos[2], bounds.pos_min, bounds.pos_max),
        );
        vel[i] = Vec3::new(
            clamp(b.vel[0], bounds.vel_min, bounds.vel_max),
            clamp(b.vel[1], bounds.vel_min, bounds.vel_max),
            clamp(b.vel[2], bounds.vel_min, bounds.vel_max),
        );
    }
    let mut system = System::new(bodies, State::new(pos, vel));
    if spec.barycentric {
        system = to_barycentric(system);
    }
    let min_dist = min_pair_distance(&system.state.pos);
    if min_dist < bounds.min_pair_dist {
        anyhow::bail!("initial conditions too close: min_pair_dist={}", min_dist);
    }
    Ok(system)
}

fn min_pair_distance(pos: &[Vec3; 3]) -> f64 {
    let mut min = f64::INFINITY;
    for i in 0..3 {
        for j in (i + 1)..3 {
            let d = (pos[j] - pos[i]).norm();
            if d < min {
                min = d;
            }
        }
    }
    min
}

struct VectorDataset {
    feature_names: Vec<String>,
    samples: Vec<Vec<f64>>,
    targets: Vec<[f64; 3]>,
}

fn build_vector_dataset(
    result: &threebody_core::sim::SimResult,
    cfg: &Config,
    feature_names: &[String],
    body_filter: Option<usize>,
) -> VectorDataset {
    let mut samples = Vec::new();
    let mut targets = Vec::new();
    let force_cfg = ForceConfig {
        g: cfg.constants.g,
        k_e: cfg.constants.k_e,
        mu_0: cfg.constants.mu_0,
        epsilon: cfg.softening,
        enable_gravity: cfg.enable_gravity,
        enable_em: cfg.enable_em,
    };
    for step in &result.steps {
        let system = &step.system;
        let acc = compute_accel(system, &force_cfg);
        for body in 0..3 {
            if let Some(only) = body_filter {
                if body != only {
                    continue;
                }
            }
            let features = compute_feature_vector(system, body, cfg, feature_names);
            samples.push(features);
            targets.push([acc[body].x, acc[body].y, acc[body].z]);
        }
    }
    VectorDataset {
        feature_names: feature_names.to_vec(),
        samples,
        targets,
    }
}

fn component_datasets(vec_data: &VectorDataset) -> [Dataset; 3] {
    let mut targets_x = Vec::new();
    let mut targets_y = Vec::new();
    let mut targets_z = Vec::new();
    for t in &vec_data.targets {
        targets_x.push(t[0]);
        targets_y.push(t[1]);
        targets_z.push(t[2]);
    }
    [
        Dataset::new(vec_data.feature_names.clone(), vec_data.samples.clone(), targets_x),
        Dataset::new(vec_data.feature_names.clone(), vec_data.samples.clone(), targets_y),
        Dataset::new(vec_data.feature_names.clone(), vec_data.samples.clone(), targets_z),
    ]
}

fn compute_feature_vector(
    system: &System,
    body: usize,
    cfg: &Config,
    feature_names: &[String],
) -> Vec<f64> {
    fn softened_inv_r3(r2: f64, epsilon: f64) -> f64 {
        if r2 == 0.0 {
            return 0.0;
        }
        let soft2 = if epsilon == 0.0 { r2 } else { r2 + epsilon * epsilon };
        let r = soft2.sqrt();
        1.0 / (r * r * r)
    }

    let epsilon = cfg.softening;
    let mut grav = Vec3::zero(); // Σ m_j (r_j - r_i) / |r|^3  (no G)
    let mut elec = Vec3::zero(); // (q_i/m_i) Σ q_j (r_i - r_j) / |r|^3 (no k_e)
    let mut mag = Vec3::zero(); // (q_i/m_i) (v_i × B_basis) where B_basis=(1/4π)Σ q_j(v_j×(r_i-r_j))/|r|^3 (no μ0)

    if cfg.enable_gravity {
        for j in 0..3 {
            if j == body {
                continue;
            }
            let r_ji = system.state.pos[j] - system.state.pos[body];
            let inv_r3 = softened_inv_r3(r_ji.norm_sq(), epsilon);
            grav = grav + r_ji * (system.bodies[j].mass * inv_r3);
        }
    }

    if cfg.enable_em {
        let qi = system.bodies[body].charge;
        let mi = system.bodies[body].mass;
        if qi != 0.0 && mi != 0.0 {
            let q_over_m = qi / mi;
            let mut e_basis = Vec3::zero();
            let mut b_basis = Vec3::zero();
            let inv_4pi = 1.0 / (4.0 * std::f64::consts::PI);
            for j in 0..3 {
                if j == body {
                    continue;
                }
                // Use (r_i - r_j) to match the electrostatic field direction.
                let r = system.state.pos[body] - system.state.pos[j];
                let inv_r3 = softened_inv_r3(r.norm_sq(), epsilon);
                let qj = system.bodies[j].charge;
                e_basis = e_basis + r * (qj * inv_r3);

                // Magnetic basis uses v_j × (r_i - r_j).
                let vj_cross_r = system.state.vel[j].cross(r);
                b_basis = b_basis + vj_cross_r * (qj * inv_r3 * inv_4pi);
            }
            elec = e_basis * q_over_m;
            mag = system.state.vel[body].cross(b_basis) * q_over_m;
        }
    }
    feature_names
        .iter()
        .map(|name| match name.as_str() {
            "grav_x" => grav.x,
            "grav_y" => grav.y,
            "grav_z" => grav.z,
            "elec_x" => elec.x,
            "elec_y" => elec.y,
            "elec_z" => elec.z,
            "mag_x" => mag.x,
            "mag_y" => mag.y,
            "mag_z" => mag.z,
            _ => 0.0,
        })
        .collect()
}

#[derive(Clone)]
struct VectorModel {
    eq_x: threebody_discover::Equation,
    eq_y: threebody_discover::Equation,
    eq_z: threebody_discover::Equation,
}

fn format_vector_model(model: &VectorModel) -> String {
    format!(
        "ax={} ; ay={} ; az={}",
        model.eq_x.format(),
        model.eq_y.format(),
        model.eq_z.format()
    )
}

fn rollout_metrics(
    model: &VectorModel,
    feature_names: &[String],
    result: &threebody_core::sim::SimResult,
    cfg: &Config,
    rollout_integrator: RolloutIntegrator,
) -> (f64, Option<f64>) {
    let mut system = result.steps.first().map(|s| s.system).unwrap_or_else(|| {
        let bodies = [Body::new(1.0, 0.0), Body::new(1.0, 0.0), Body::new(1.0, 0.0)];
        let pos = [Vec3::zero(); 3];
        let vel = [Vec3::zero(); 3];
        System::new(bodies, State::new(pos, vel))
    });
    let feature_dataset = Dataset::new(feature_names.to_vec(), vec![], vec![]);
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    let mut t = 0.0;
    let threshold = (0.5 * min_pair_distance(&result.steps[0].system.state.pos)).max(0.1);
    let mut divergence_time = None;
    for i in 0..(result.steps.len().saturating_sub(1)) {
        let dt = result.steps[i].dt;
        system = rollout_step(&system, &feature_dataset, model, cfg, dt, rollout_integrator);
        t += dt;
        let err = rms_pos_error(&system, &result.steps[i + 1].system);
        sum_sq += err * err;
        count += 1;
        if divergence_time.is_none() && err > threshold {
            divergence_time = Some(t);
        }
    }
    let rmse = if count > 0 { (sum_sq / count as f64).sqrt() } else { 0.0 };
    (rmse, divergence_time)
}

fn rollout_trace(
    model: &VectorModel,
    feature_names: &[String],
    result: &threebody_core::sim::SimResult,
    cfg: &Config,
    rollout_integrator: RolloutIntegrator,
) -> Vec<RolloutTraceStep> {
    let mut system = result.steps.first().map(|s| s.system).unwrap_or_else(|| {
        let bodies = [Body::new(1.0, 0.0), Body::new(1.0, 0.0), Body::new(1.0, 0.0)];
        let pos = [Vec3::zero(); 3];
        let vel = [Vec3::zero(); 3];
        System::new(bodies, State::new(pos, vel))
    });
    let feature_dataset = Dataset::new(feature_names.to_vec(), vec![], vec![]);
    let mut trace = Vec::new();
    let mut t = 0.0;
    for i in 0..(result.steps.len().saturating_sub(1)) {
        let dt = result.steps[i].dt;
        system = rollout_step(&system, &feature_dataset, model, cfg, dt, rollout_integrator);
        t += dt;
        let err = rms_pos_error(&system, &result.steps[i + 1].system);
        trace.push(RolloutTraceStep {
            t,
            pos: system
                .state
                .pos
                .iter()
                .map(|p| [p.x, p.y, p.z])
                .collect(),
            rmse_pos: err,
        });
    }
    trace
}

#[derive(serde::Serialize)]
struct RolloutTraceStep {
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
        let features = compute_feature_vector(system, body, cfg, &feature_dataset.feature_names);
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
            let mut new_pos = system.state.pos;
            let mut new_vel = system.state.vel;
            for b in 0..3 {
                new_vel[b] = new_vel[b] + acc[b] * dt;
                new_pos[b] = new_pos[b] + new_vel[b] * dt;
            }
            System::new(system.bodies, State::new(new_pos, new_vel))
        }
        RolloutIntegrator::Leapfrog => {
            let mut v_half = system.state.vel;
            for b in 0..3 {
                v_half[b] = v_half[b] + acc[b] * (0.5 * dt);
            }
            let mut new_pos = system.state.pos;
            for b in 0..3 {
                new_pos[b] = new_pos[b] + v_half[b] * dt;
            }
            let interim = System::new(system.bodies, State::new(new_pos, v_half));
            let acc_new = predict_accel(&interim, feature_dataset, model, cfg);
            let mut new_vel = v_half;
            for b in 0..3 {
                new_vel[b] = new_vel[b] + acc_new[b] * (0.5 * dt);
            }
            System::new(system.bodies, State::new(interim.state.pos, new_vel))
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

fn build_vector_candidates(
    topk_x: &[threebody_discover::EquationScore],
    topk_y: &[threebody_discover::EquationScore],
    topk_z: &[threebody_discover::EquationScore],
    feature_names: &[String],
    result: &threebody_core::sim::SimResult,
    cfg: &Config,
    regime: &str,
    rollout_integrator: RolloutIntegrator,
) -> Vec<CandidateSummary> {
    let n = topk_x.len().min(topk_y.len()).min(topk_z.len()).min(3);
    let mut candidates = Vec::new();
    for i in 0..n {
        let model = VectorModel {
            eq_x: topk_x[i].equation.clone(),
            eq_y: topk_y[i].equation.clone(),
            eq_z: topk_z[i].equation.clone(),
        };
        let mse = (topk_x[i].score + topk_y[i].score + topk_z[i].score) / 3.0;
        let complexity = model.eq_x.complexity() + model.eq_y.complexity() + model.eq_z.complexity();
        let (rmse, divergence_time) = rollout_metrics(&model, feature_names, result, cfg, rollout_integrator);
        let mut flags = Vec::new();
        flags.extend(stability_flags_for(&model.eq_x, regime));
        flags.extend(stability_flags_for(&model.eq_y, regime));
        flags.extend(stability_flags_for(&model.eq_z, regime));
        flags.sort();
        flags.dedup();
        candidates.push(CandidateSummary {
            id: i,
            equation: model.eq_x.clone(),
            equation_text: format_vector_model(&model),
            metrics: CandidateMetrics {
                mse,
                complexity,
                rollout_rmse: Some(rmse),
                divergence_time,
                stability_flags: flags,
            },
            notes: vec![],
        });
    }
    candidates
}

fn build_sim_summary(result: &threebody_core::sim::SimResult, rollout_integrator: RolloutIntegrator) -> SimulationSummary {
    let energy_start = result.steps.first().map(|s| s.diagnostics.energy_proxy);
    let energy_end = result.steps.last().map(|s| s.diagnostics.energy_proxy);
    let energy_drift = match (energy_start, energy_end) {
        (Some(a), Some(b)) => Some(b - a),
        _ => None,
    };
    let mut min_pair: Option<f64> = None;
    let mut max_speed: Option<f64> = None;
    let mut max_accel: Option<f64> = None;
    for step in &result.steps {
        min_pair = Some(match min_pair {
            Some(v) => v.min(step.regime.min_pair_dist),
            None => step.regime.min_pair_dist,
        });
        max_speed = Some(match max_speed {
            Some(v) => v.max(step.regime.max_speed),
            None => step.regime.max_speed,
        });
        max_accel = Some(match max_accel {
            Some(v) => v.max(step.regime.max_accel),
            None => step.regime.max_accel,
        });
    }
    SimulationSummary {
        steps: result.steps.len(),
        energy_start,
        energy_end,
        energy_drift,
        min_pair_dist: min_pair,
        max_speed,
        max_accel,
        dt_min: result.stats.dt_min,
        dt_max: result.stats.dt_max,
        dt_avg: result.stats.dt_avg,
        warnings: result.warnings.clone(),
        rollout_integrator: rollout_integrator_label(rollout_integrator).to_string(),
    }
}

#[derive(Clone, Copy, Debug)]
enum LlmMode {
    Off,
    Mock,
    OpenAI,
}

#[derive(Clone, Copy, Debug)]
enum RolloutIntegrator {
    Euler,
    Leapfrog,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DiscoverySolver {
    Ga,
    Stls,
    Lasso,
}

#[derive(Clone, Debug)]
struct DiscoverySolverSettings {
    solver: DiscoverySolver,
    normalize: bool,
    stls_thresholds: Vec<f64>, // empty => auto grid
    stls_ridge_lambda: f64,
    stls_max_iter: usize,
    lasso_alphas: Vec<f64>, // empty => auto grid
    lasso_max_iter: usize,
    lasso_tol: f64,
}

#[derive(serde::Serialize)]
struct SolverMeta {
    name: String,
    normalize: bool,
    fitness: String,
    stls: Option<StlsMeta>,
    lasso: Option<LassoMeta>,
    ga: Option<GaMeta>,
}

#[derive(serde::Serialize)]
struct StlsMeta {
    auto_thresholds: bool,
    thresholds: Vec<f64>,
    ridge_lambda: f64,
    max_iter: usize,
}

#[derive(serde::Serialize)]
struct LassoMeta {
    auto_alphas: bool,
    alphas: Vec<f64>,
    max_iter: usize,
    tol: f64,
}

#[derive(serde::Serialize)]
struct GaMeta {
    runs: usize,
    population: usize,
    seed: u64,
}

fn build_solver_meta(
    solver_settings: &DiscoverySolverSettings,
    fitness: FitnessHeuristic,
    ga_cfg: Option<(usize, usize, u64)>,
) -> SolverMeta {
    SolverMeta {
        name: discovery_solver_label(solver_settings.solver).to_string(),
        normalize: solver_settings.normalize,
        fitness: fitness.as_str().to_string(),
        stls: matches!(solver_settings.solver, DiscoverySolver::Stls).then(|| StlsMeta {
            auto_thresholds: solver_settings.stls_thresholds.is_empty(),
            thresholds: solver_settings.stls_thresholds.clone(),
            ridge_lambda: solver_settings.stls_ridge_lambda,
            max_iter: solver_settings.stls_max_iter,
        }),
        lasso: matches!(solver_settings.solver, DiscoverySolver::Lasso).then(|| LassoMeta {
            auto_alphas: solver_settings.lasso_alphas.is_empty(),
            alphas: solver_settings.lasso_alphas.clone(),
            max_iter: solver_settings.lasso_max_iter,
            tol: solver_settings.lasso_tol,
        }),
        ga: matches!(solver_settings.solver, DiscoverySolver::Ga).then(|| {
            let (runs, population, seed) = ga_cfg.unwrap_or((0, 0, 0));
            GaMeta {
                runs,
                population,
                seed,
            }
        }),
    }
}

fn parse_discovery_solver(name: &str) -> anyhow::Result<DiscoverySolver> {
    match name {
        "ga" => Ok(DiscoverySolver::Ga),
        "stls" => Ok(DiscoverySolver::Stls),
        "lasso" => Ok(DiscoverySolver::Lasso),
        _ => anyhow::bail!("unknown discovery solver: {name} (expected stls|lasso|ga)"),
    }
}

fn discovery_solver_label(solver: DiscoverySolver) -> &'static str {
    match solver {
        DiscoverySolver::Ga => "ga",
        DiscoverySolver::Stls => "stls",
        DiscoverySolver::Lasso => "lasso",
    }
}

fn apply_discovery_recommendations(
    settings: &mut DiscoverySolverSettings,
    rec: &threebody_discover::JudgeRecommendations,
) {
    fn centered_triple(v: f64) -> Vec<f64> {
        let mut out = vec![v * 0.5, v, v * 2.0];
        out.retain(|x| x.is_finite() && *x >= 0.0);
        out.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out.dedup();
        out
    }

    if let Some(name) = rec.next_discovery_solver.as_deref() {
        if let Ok(parsed) = parse_discovery_solver(name) {
            settings.solver = parsed;
        }
    }
    if let Some(norm) = rec.next_normalize {
        settings.normalize = norm;
    }
    if let Some(lambda) = rec.next_ridge_lambda {
        if lambda.is_finite() && lambda >= 0.0 {
            settings.stls_ridge_lambda = lambda;
        }
    }
    if let Some(tau) = rec.next_stls_threshold {
        if tau.is_finite() && tau > 0.0 {
            settings.stls_thresholds = centered_triple(tau);
        }
    }
    if let Some(alpha) = rec.next_lasso_alpha {
        if alpha.is_finite() && alpha > 0.0 {
            settings.lasso_alphas = centered_triple(alpha);
        }
    }
}

fn parse_llm_mode(mode: &str) -> anyhow::Result<LlmMode> {
    match mode {
        "off" => Ok(LlmMode::Off),
        "mock" => Ok(LlmMode::Mock),
        "openai" => Ok(LlmMode::OpenAI),
        _ => anyhow::bail!("unknown llm mode: {mode}"),
    }
}

fn parse_rollout_integrator(name: &str) -> anyhow::Result<RolloutIntegrator> {
    match name {
        "euler" => Ok(RolloutIntegrator::Euler),
        "leapfrog" => Ok(RolloutIntegrator::Leapfrog),
        _ => anyhow::bail!("unknown rollout integrator: {name}"),
    }
}

fn rollout_integrator_label(intg: RolloutIntegrator) -> &'static str {
    match intg {
        RolloutIntegrator::Euler => "euler",
        RolloutIntegrator::Leapfrog => "leapfrog",
    }
}

fn parse_fitness_heuristic(name: &str) -> anyhow::Result<FitnessHeuristic> {
    match name {
        "mse" => Ok(FitnessHeuristic::Mse),
        "mse_parsimony" => Ok(FitnessHeuristic::MseParsimony),
        _ => anyhow::bail!("unknown fitness heuristic: {name}"),
    }
}

fn llm_mode_label(mode: LlmMode) -> &'static str {
    match mode {
        LlmMode::Off => "off",
        LlmMode::Mock => "mock",
        LlmMode::OpenAI => "openai",
    }
}

fn select_llm_client(
    mode: LlmMode,
    model: &str,
    openai_key_file: Option<&std::path::Path>,
) -> anyhow::Result<Option<Box<dyn LlmClient>>> {
    match mode {
        LlmMode::Off => Ok(None),
        LlmMode::Mock => Ok(Some(Box::new(MockLlm))),
        LlmMode::OpenAI => Ok(Some(Box::new(OpenAIClient::from_env_or_file(
            model,
            openai_key_file,
        )?))),
    }
}

fn run_factory(
    out_dir: PathBuf,
    max_iters: usize,
    auto: bool,
    config: Option<PathBuf>,
    steps: usize,
    dt: f64,
    mode: String,
    preset: String,
    em: bool,
    no_em: bool,
    no_gravity: bool,
    runs: usize,
    population: usize,
    seed: u64,
    fitness: String,
    rollout_integrator: String,
    solver_settings: DiscoverySolverSettings,
    llm_mode: String,
    model: String,
    openai_key_file: Option<PathBuf>,
) -> anyhow::Result<()> {
    fs::create_dir_all(&out_dir)?;
    let mut next_ic: Option<InitialConditionSpec> = None;
    let llm_mode = parse_llm_mode(&llm_mode)?;
    let llm_client = select_llm_client(llm_mode, &model, openai_key_file.as_deref())?;
    let mut current_fitness = parse_fitness_heuristic(&fitness)?;
    let mut current_rollout = parse_rollout_integrator(&rollout_integrator)?;
    let mut current_solver = solver_settings;

    for iter in 0..max_iters {
        let run_id = format!("run_{:03}", iter + 1);
        let run_dir = out_dir.join(&run_id);
        fs::create_dir_all(&run_dir)?;

        let cfg = build_config(config.clone(), &mode, None, em, no_em, no_gravity)?;
        let regime = if cfg.enable_em { "em_quasistatic" } else { "gravity_only" };
        let ic_bounds = default_ic_bounds();
        let ic_request = IcRequest {
            bounds: ic_bounds.clone(),
            regime: regime.to_string(),
            notes: vec!["avoid close encounters".to_string(), "prefer bounded motion".to_string()],
            seed: Some(seed + iter as u64),
        };

        let mut ic_prompt = None;
        let mut ic_response = None;
        let ic_spec = if let Some(spec) = next_ic.take() {
            spec
        } else if let Some(client) = llm_client.as_ref() {
            match client.propose_initial_conditions(&ic_request) {
                Ok(result) => {
                    ic_prompt = Some(result.prompt);
                    ic_response = Some(result.response);
                    result.value
                }
                Err(err) => {
                    eprintln!("LLM IC proposal failed: {}", err);
                    initial_conditions_from_preset(&preset)?
                }
            }
        } else {
            initial_conditions_from_preset(&preset)?
        };

        let ic_request_path = run_dir.join("ic_request.json");
        fs::write(&ic_request_path, serde_json::to_string_pretty(&ic_request)?)?;
        let ic_spec_path = run_dir.join("initial_conditions.json");
        fs::write(&ic_spec_path, serde_json::to_string_pretty(&ic_spec)?)?;
        let cfg_path = run_dir.join("config.json");
        fs::write(&cfg_path, serde_json::to_string_pretty(&cfg)?)?;

        let system = match system_from_ic(&ic_spec, &ic_bounds) {
            Ok(sys) => sys,
            Err(err) => {
                eprintln!("IC validation failed ({err}); falling back to preset.");
                preset_system(&preset)?
            }
        };

        let options = SimOptions { steps, dt };
        let result = simulate_with_cfg(system, &cfg, options);

        let traj_path = run_dir.join("traj.csv");
        let mut csv_file = fs::File::create(&traj_path)?;
        write_csv(&mut csv_file, &result.steps, &cfg)?;
        let header = threebody_core::output::csv::csv_header(&cfg);
        let sidecar = build_sidecar(&cfg, &header, &result);
        let sidecar_path = run_dir.join("traj.json");
        let mut sidecar_file = fs::File::create(&sidecar_path)?;
        write_sidecar(&mut sidecar_file, &sidecar)?;

        let library = FeatureLibrary::default_physics();
        let vector_data = build_vector_dataset(&result, &cfg, &library.features, None);
        let [dataset_x, dataset_y, dataset_z] = component_datasets(&vector_data);
        let disc_cfg = DiscoveryConfig {
            runs,
            population,
            seed: seed + iter as u64,
            fitness: current_fitness,
            ..DiscoveryConfig::default()
        };
        let stls_cfg = StlsConfig {
            thresholds: current_solver.stls_thresholds.clone(),
            ridge_lambda: current_solver.stls_ridge_lambda,
            max_iter: current_solver.stls_max_iter,
            normalize: current_solver.normalize,
        };
        let lasso_cfg = LassoConfig {
            alphas: current_solver.lasso_alphas.clone(),
            max_iter: current_solver.lasso_max_iter,
            tol: current_solver.lasso_tol,
            normalize: current_solver.normalize,
        };
        let (topk_x, topk_y, topk_z) = match current_solver.solver {
            DiscoverySolver::Ga => (
                run_search(&dataset_x, &library, &disc_cfg),
                run_search(&dataset_y, &library, &disc_cfg),
                run_search(&dataset_z, &library, &disc_cfg),
            ),
            DiscoverySolver::Stls => (
                stls_path_search(&dataset_x, &stls_cfg, current_fitness),
                stls_path_search(&dataset_y, &stls_cfg, current_fitness),
                stls_path_search(&dataset_z, &stls_cfg, current_fitness),
            ),
            DiscoverySolver::Lasso => (
                lasso_path_search(&dataset_x, &lasso_cfg, current_fitness),
                lasso_path_search(&dataset_y, &lasso_cfg, current_fitness),
                lasso_path_search(&dataset_z, &lasso_cfg, current_fitness),
            ),
        };
        let vector_candidates = build_vector_candidates(
            &topk_x.entries,
            &topk_y.entries,
            &topk_z.entries,
            &vector_data.feature_names,
            &result,
            &cfg,
            regime,
            current_rollout,
        );
        let grid_topk = grid_search(
            &vector_candidates
                .iter()
                .map(|c| c.equation.clone())
                .collect::<Vec<_>>(),
            &dataset_x,
        );

        let mut trace_written = false;
        if !vector_candidates.is_empty() {
            let mut best_idx = 0usize;
            let mut best_mse = vector_candidates[0].metrics.mse;
            for (i, cand) in vector_candidates.iter().enumerate().skip(1) {
                if cand.metrics.mse < best_mse {
                    best_mse = cand.metrics.mse;
                    best_idx = i;
                }
            }
            if best_idx < topk_x.entries.len()
                && best_idx < topk_y.entries.len()
                && best_idx < topk_z.entries.len()
            {
                let best_model = VectorModel {
                    eq_x: topk_x.entries[best_idx].equation.clone(),
                    eq_y: topk_y.entries[best_idx].equation.clone(),
                    eq_z: topk_z.entries[best_idx].equation.clone(),
                };
                let trace = rollout_trace(
                    &best_model,
                    &vector_data.feature_names,
                    &result,
                    &cfg,
                    current_rollout,
                );
                let trace_path = run_dir.join("rollout_trace.json");
                fs::write(&trace_path, serde_json::to_string_pretty(&trace)?)?;
                trace_written = true;
            }
        }

        let sim_summary = build_sim_summary(&result, current_rollout);
        let dataset_summary = build_dataset_summary(
            &vector_data.feature_names,
            &library,
            vector_data.samples.len(),
            "accel_components_body_all",
        );
        let mut judge_input = build_judge_input_from_candidates(
            dataset_summary,
            vector_candidates.clone(),
            Some(sim_summary.clone()),
            regime,
        );
        judge_input
            .notes
            .push(format!("fitness_heuristic={}", current_fitness.as_str()));
        judge_input
            .notes
            .push(format!("discovery_solver={}", discovery_solver_label(current_solver.solver)));
        judge_input
            .notes
            .push(format!("normalize={}", current_solver.normalize));
        match current_solver.solver {
            DiscoverySolver::Ga => {
                judge_input.notes.push(format!("ga_runs={runs}"));
                judge_input.notes.push(format!("ga_population={population}"));
            }
            DiscoverySolver::Stls => {
                judge_input.notes.push(format!("stls_ridge_lambda={}", current_solver.stls_ridge_lambda));
                if current_solver.stls_thresholds.is_empty() {
                    judge_input.notes.push("stls_thresholds=auto".to_string());
                } else {
                    judge_input.notes.push(format!("stls_thresholds={:?}", current_solver.stls_thresholds));
                }
            }
            DiscoverySolver::Lasso => {
                judge_input.notes.push(format!("lasso_tol={}", current_solver.lasso_tol));
                if current_solver.lasso_alphas.is_empty() {
                    judge_input.notes.push("lasso_alphas=auto".to_string());
                } else {
                    judge_input.notes.push(format!("lasso_alphas={:?}", current_solver.lasso_alphas));
                }
            }
        }
        judge_input
            .notes
            .push(format!("rollout_integrator={}", rollout_integrator_label(current_rollout)));
        let judge_input_path = run_dir.join("judge_input.json");
        fs::write(&judge_input_path, serde_json::to_string_pretty(&judge_input)?)?;
        let mut judge_prompt = None;
        let mut judge_response = None;
        let judge = if let Some(client) = llm_client.as_ref() {
            match client.judge_candidates(&judge_input) {
                Ok(result) => {
                    judge_prompt = Some(result.prompt);
                    judge_response = Some(result.response);
                    let resp = result.value;
                    if resp.validate(&judge_input).is_ok() {
                        Some(resp)
                    } else {
                        eprintln!("LLM judge response failed validation");
                        None
                    }
                }
                Err(err) => {
                    eprintln!("LLM judge failed: {}", err);
                    None
                }
            }
        } else {
            None
        };

        if let Some(prompt) = ic_prompt.as_ref() {
            fs::write(run_dir.join("ic_prompt.txt"), prompt)?;
        }
        if let Some(resp) = ic_response.as_ref() {
            fs::write(run_dir.join("ic_response.txt"), resp)?;
        }
        if let Some(prompt) = judge_prompt.as_ref() {
            fs::write(run_dir.join("judge_prompt.txt"), prompt)?;
        }
        if let Some(resp) = judge_response.as_ref() {
            fs::write(run_dir.join("judge_response.txt"), resp)?;
        }

        let discovery_out = run_dir.join("discovery.json");
        let ga_seed = seed + iter as u64;
        let solver_meta = build_solver_meta(
            &current_solver,
            current_fitness,
            matches!(current_solver.solver, DiscoverySolver::Ga).then_some((runs, population, ga_seed)),
        );
        let discovery_json = serde_json::to_string_pretty(&serde_json::json!({
            "solver": &solver_meta,
            "top3_x": topk_x.entries,
            "top3_y": topk_y.entries,
            "top3_z": topk_z.entries,
            "vector_candidates": vector_candidates,
            "grid_top3": grid_topk.entries,
        }))?;
        fs::write(&discovery_out, discovery_json)?;

        #[derive(serde::Serialize)]
        struct FactoryReport {
            iteration: usize,
            run_id: String,
            regime: String,
            config: Config,
            initial_conditions: InitialConditionSpec,
            simulation: SimulationSummary,
            solver: SolverMeta,
            discovery_top3: Vec<threebody_discover::EquationScore>,
            vector_candidates: Vec<CandidateSummary>,
            grid_top3: Vec<threebody_discover::EquationScore>,
            judge: Option<JudgeResponse>,
            llm_mode: String,
            llm_model: Option<String>,
            fitness_heuristic: String,
            rollout_integrator: String,
            rollout_trace: Option<String>,
        }

        let report = FactoryReport {
            iteration: iter + 1,
            run_id: run_id.clone(),
            regime: regime.to_string(),
            config: cfg,
            initial_conditions: ic_spec.clone(),
            simulation: sim_summary.clone(),
            solver: solver_meta,
            discovery_top3: topk_x.entries.clone(),
            vector_candidates: vector_candidates.clone(),
            grid_top3: grid_topk.entries.clone(),
            judge: judge.clone(),
            llm_mode: llm_mode_label(llm_mode).to_string(),
            llm_model: if matches!(llm_mode, LlmMode::OpenAI) { Some(model.clone()) } else { None },
            fitness_heuristic: current_fitness.as_str().to_string(),
            rollout_integrator: rollout_integrator_label(current_rollout).to_string(),
            rollout_trace: if trace_written {
                Some("rollout_trace.json".to_string())
            } else {
                None
            },
        };
        let report_json = serde_json::to_string_pretty(&report)?;
        fs::write(run_dir.join("report.json"), report_json)?;

        let mut md = String::new();
        md.push_str(&format!("# Factory Report {}\n\n", run_id));
        md.push_str(&format!("- Regime: {}\n", regime));
        md.push_str(&format!("- Steps: {}\n", sim_summary.steps));
        md.push_str(&format!("- Energy drift: {:?}\n", sim_summary.energy_drift));
        md.push_str(&format!("- Min pair dist: {:?}\n", sim_summary.min_pair_dist));
        if let Some(best_vec) = vector_candidates.first() {
            md.push_str(&format!(
                "- Best vector model: mse={:.6}, rollout_rmse={:?}\n",
                best_vec.metrics.mse,
                best_vec.metrics.rollout_rmse
            ));
            md.push_str(&format!("  eq: {}\n", best_vec.equation_text));
        }
        if let Some(j) = judge.as_ref() {
            md.push_str("\n## LLM Judge Summary\n");
            md.push_str(&format!("{}\n", j.summary));
            md.push_str(&format!("Ranking: {:?}\n", j.ranking));
        }
        fs::write(run_dir.join("report.md"), md)?;

        println!("Factory iteration {} complete -> {}", iter + 1, run_dir.display());
        if let Some(j) = judge.as_ref() {
            println!("LLM summary: {}", j.summary);
        }

        let mut continue_loop = auto;
        if !auto && iter + 1 < max_iters {
            continue_loop = prompt_continue()?;
        }
        if !continue_loop {
            break;
        }
        if let Some(j) = judge {
            apply_discovery_recommendations(&mut current_solver, &j.recommendations);
            if let Some(next) = j.recommendations.next_initial_conditions {
                next_ic = Some(next);
            }
            if let Some(next) = j.recommendations.next_ga_heuristic.as_ref() {
                if let Ok(parsed) = parse_fitness_heuristic(next) {
                    current_fitness = parsed;
                }
            }
            if let Some(next) = j.recommendations.next_rollout_integrator.as_ref() {
                if let Ok(parsed) = parse_rollout_integrator(next) {
                    current_rollout = parsed;
                }
            }
        }
    }
    Ok(())
}

fn parse_csv_f64_list(raw: &str) -> anyhow::Result<Vec<f64>> {
    let mut values = Vec::new();
    for part in raw.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        let v: f64 = t.parse().map_err(|e| anyhow::anyhow!("invalid f64 '{t}': {e}"))?;
        values.push(v);
    }
    Ok(values)
}

fn prompt_continue() -> anyhow::Result<bool> {
    print!("Continue to next iteration? [y/N]: ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let answer = input.trim().to_ascii_lowercase();
    Ok(answer == "y" || answer == "yes")
}

fn initial_conditions_from_preset(preset: &str) -> anyhow::Result<InitialConditionSpec> {
    let system = preset_system(preset)?;
    Ok(InitialConditionSpec {
        bodies: system
            .bodies
            .iter()
            .zip(system.state.pos.iter().zip(system.state.vel.iter()))
            .map(|(b, (p, v))| threebody_discover::BodyInit {
                mass: b.mass,
                charge: b.charge,
                pos: [p.x, p.y, p.z],
                vel: [v.x, v.y, v.z],
            })
            .collect(),
        barycentric: true,
        notes: format!("preset:{preset}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use threebody_discover::Equation;

    fn unique_temp_path(name: &str, ext: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("threebody_cli_test_{name}_{}_{}.{}", std::process::id(), n, ext))
    }

    #[test]
    fn vector_dataset_includes_all_bodies() {
        let cfg = Config::default();
        let system = preset_system("two-body").unwrap();
        let result = simulate_with_cfg(system, &cfg, SimOptions { steps: 2, dt: 0.01 });
        let library = FeatureLibrary::default_physics();
        let vec_data = build_vector_dataset(&result, &cfg, &library.features, None);
        let expected = result.steps.len() * 3;
        assert_eq!(vec_data.samples.len(), expected);
        assert_eq!(vec_data.targets.len(), expected);
    }

    #[test]
    fn vector_dataset_body_filter_reduces_samples() {
        let cfg = Config::default();
        let system = preset_system("two-body").unwrap();
        let result = simulate_with_cfg(system, &cfg, SimOptions { steps: 3, dt: 0.01 });
        let library = FeatureLibrary::default_physics();
        let vec_data = build_vector_dataset(&result, &cfg, &library.features, Some(1));
        let expected = result.steps.len();
        assert_eq!(vec_data.samples.len(), expected);
        assert_eq!(vec_data.targets.len(), expected);
    }

    #[test]
    fn rollout_metrics_are_finite() {
        let cfg = Config::default();
        let system = preset_system("two-body").unwrap();
        let result = simulate_with_cfg(system, &cfg, SimOptions { steps: 3, dt: 0.01 });
        let library = FeatureLibrary::default_physics();
        let vec_data = build_vector_dataset(&result, &cfg, &library.features, None);
        let model = VectorModel {
            eq_x: Equation { terms: vec![] },
            eq_y: Equation { terms: vec![] },
            eq_z: Equation { terms: vec![] },
        };
        let (rmse_e, _div_e) =
            rollout_metrics(&model, &vec_data.feature_names, &result, &cfg, RolloutIntegrator::Euler);
        let (rmse_l, _div_l) =
            rollout_metrics(&model, &vec_data.feature_names, &result, &cfg, RolloutIntegrator::Leapfrog);
        assert!(rmse_e.is_finite());
        assert!(rmse_l.is_finite());
    }

    #[test]
    fn gravity_features_match_simple_analytic_case() {
        let mut cfg = Config::default();
        cfg.enable_gravity = true;
        cfg.enable_em = false;
        cfg.softening = 0.0;

        let bodies = [Body::new(1.0, 0.0), Body::new(2.0, 0.0), Body::new(3.0, 0.0)];
        let pos = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let vel = [Vec3::zero(); 3];
        let system = System::new(bodies, State::new(pos, vel));

        let feature_names = FeatureLibrary::default_physics().features;
        let f0 = compute_feature_vector(&system, 0, &cfg, &feature_names);
        let idx = |name: &str| feature_names.iter().position(|n| n == name).unwrap();

        // grav = Σ m_j (r_j - r_i) / |r|^3 (no G)
        // body0: from body1: 2*(1,0,0)/1^3 -> (2,0,0)
        //        from body2: 3*(0,2,0)/2^3 -> 3*(0,2,0)/8 -> (0,0.75,0)
        assert!((f0[idx("grav_x")] - 2.0).abs() < 1e-12);
        assert!((f0[idx("grav_y")] - 0.75).abs() < 1e-12);
        assert!((f0[idx("grav_z")] - 0.0).abs() < 1e-12);

        // EM disabled => elec/mag features are zero.
        for name in ["elec_x", "elec_y", "elec_z", "mag_x", "mag_y", "mag_z"] {
            assert!((f0[idx(name)] - 0.0).abs() < 1e-12);
        }
    }

    #[test]
    fn electric_features_match_simple_analytic_case() {
        let mut cfg = Config::default();
        cfg.enable_gravity = false;
        cfg.enable_em = true;
        cfg.softening = 0.0;

        let bodies = [Body::new(1.0, 1.0), Body::new(1.0, 2.0), Body::new(1.0, 0.0)];
        let pos = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let vel = [Vec3::zero(); 3];
        let system = System::new(bodies, State::new(pos, vel));

        let feature_names = FeatureLibrary::default_physics().features;
        let idx = |name: &str| feature_names.iter().position(|n| n == name).unwrap();

        // elec = (q_i/m_i) Σ q_j (r_i - r_j) / |r|^3 (no k_e)
        let f0 = compute_feature_vector(&system, 0, &cfg, &feature_names);
        // body0: q0/m0=1.  only body1 contributes: q1*(r0-r1) = 2*(-1,0,0) -> (-2,0,0)
        assert!((f0[idx("elec_x")] - (-2.0)).abs() < 1e-12);
        assert!((f0[idx("elec_y")] - 0.0).abs() < 1e-12);

        let f1 = compute_feature_vector(&system, 1, &cfg, &feature_names);
        // body1: q1/m1=2. body0 contributes: q0*(r1-r0) = 1*(1,0,0) -> (1,0,0) then *2 -> (2,0,0)
        assert!((f1[idx("elec_x")] - 2.0).abs() < 1e-12);

        // With zero velocities, magnetic contribution must be zero even when EM is enabled.
        for name in ["mag_x", "mag_y", "mag_z"] {
            assert!((f0[idx(name)] - 0.0).abs() < 1e-12);
            assert!((f1[idx(name)] - 0.0).abs() < 1e-12);
        }
    }

    #[test]
    fn magnetic_features_match_simple_analytic_case() {
        let mut cfg = Config::default();
        cfg.enable_gravity = false;
        cfg.enable_em = true;
        cfg.softening = 0.0;

        // body0 at origin, body1 at x=1. Choose velocities so that v1 × (r0-r1) points +z.
        // r = r0 - r1 = (-1,0,0), v1=(0,1,0) => v1×r=(0,0,1).
        // B_basis_z = (1/4π) * q1 * 1/|r|^3 = q1/(4π).
        // mag = (q0/m0) * v0 × B_basis, with v0=(1,0,0) and B_basis=(0,0,q1/(4π))
        // => mag_y = -(q0/m0) * q1/(4π).
        let bodies = [Body::new(1.0, 1.0), Body::new(1.0, 2.0), Body::new(1.0, 0.0)];
        let pos = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
        ];
        let vel = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::zero(),
        ];
        let system = System::new(bodies, State::new(pos, vel));

        let feature_names = FeatureLibrary::default_physics().features;
        let idx = |name: &str| feature_names.iter().position(|n| n == name).unwrap();

        let f0 = compute_feature_vector(&system, 0, &cfg, &feature_names);
        let expected_mag_y = -(1.0_f64) * (2.0 / (4.0 * std::f64::consts::PI));
        assert!((f0[idx("mag_x")] - 0.0).abs() < 1e-12);
        assert!((f0[idx("mag_y")] - expected_mag_y).abs() < 1e-12);
        assert!((f0[idx("mag_z")] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn load_steps_from_csv_parses_minimal_file() {
        let mut cfg = Config::default();
        cfg.enable_gravity = true;
        cfg.enable_em = false;
        cfg.softening = 0.0;

        let bodies = [Body::new(1.0, 0.0), Body::new(2.0, 0.0), Body::new(3.0, 0.0)];
        let csv_path = unique_temp_path("load_steps", "csv");

        let header = [
            "t",
            "dt",
            "r1_x",
            "r1_y",
            "r1_z",
            "r2_x",
            "r2_y",
            "r2_z",
            "r3_x",
            "r3_y",
            "r3_z",
            "v1_x",
            "v1_y",
            "v1_z",
            "v2_x",
            "v2_y",
            "v2_z",
            "v3_x",
            "v3_y",
            "v3_z",
        ]
        .join(",");
        let row = [
            "0.0",
            "0.1",
            "0.0",
            "0.0",
            "0.0",
            "1.0",
            "0.0",
            "0.0",
            "0.0",
            "2.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
        ]
        .join(",");
        fs::write(&csv_path, format!("{header}\n{row}\n")).unwrap();

        let steps = load_steps_from_csv(&csv_path, bodies, &cfg).unwrap();
        assert_eq!(steps.len(), 1);
        assert!((steps[0].t - 0.0).abs() < 1e-12);
        assert!((steps[0].dt - 0.1).abs() < 1e-12);
        assert!(steps[0].system.state.pos[1].approx_eq(Vec3::new(1.0, 0.0, 0.0), 1e-12, 1e-12));
        assert!(steps[0].diagnostics.energy_proxy.is_finite());
    }

    #[test]
    fn apply_discovery_recommendations_updates_solver_and_params() {
        let mut settings = DiscoverySolverSettings {
            solver: DiscoverySolver::Ga,
            normalize: true,
            stls_thresholds: Vec::new(),
            stls_ridge_lambda: 1e-8,
            stls_max_iter: 25,
            lasso_alphas: Vec::new(),
            lasso_max_iter: 2000,
            lasso_tol: 1e-6,
        };
        let rec = threebody_discover::JudgeRecommendations {
            next_initial_conditions: None,
            next_rollout_integrator: None,
            next_ga_heuristic: None,
            next_discovery_solver: Some("stls".to_string()),
            next_normalize: Some(false),
            next_stls_threshold: Some(0.2),
            next_ridge_lambda: Some(1e-6),
            next_lasso_alpha: None,
            next_search_directions: vec![],
            notes: String::new(),
        };
        apply_discovery_recommendations(&mut settings, &rec);
        assert_eq!(settings.solver, DiscoverySolver::Stls);
        assert!(!settings.normalize);
        assert!((settings.stls_ridge_lambda - 1e-6).abs() < 1e-18);
        assert_eq!(settings.stls_thresholds, vec![0.1, 0.2, 0.4]);
    }
}
