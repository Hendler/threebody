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
    JudgeInput, JudgeRecommendationsLite, JudgeResponse, Rubric, SimulationSummary,
};
use threebody_discover::llm::{AutoLlmClient, LlmClient, MockLlm, OpenAIClient};
use threebody_discover::{
    grid_search, lasso_path_search, run_search, stls_path_search, Dataset, FactoryEvaluationCandidate,
    FactoryEvaluationInput, FactoryEvaluationIteration, FactoryEvaluationIterationJudge, FitnessHeuristic,
    GaSolverSummary, LassoConfig, LassoSolverSummary, StlsConfig, StlsSolverSummary, DiscoverySolverSummary,
    FACTORY_EVALUATION_VERSION,
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
        #[arg(long, default_value = "gpt-5.2")]
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
    /// Run an end-to-end workflow and write artifacts under a single directory.
    Quickstart {
        /// Output directory root (defaults to `results/quickstart_<timestamp>/`).
        #[arg(long)]
        out_dir: Option<PathBuf>,
        /// Steps to simulate per run.
        #[arg(long, default_value_t = 200)]
        steps: usize,
        /// Max factory iterations (default: 10).
        #[arg(long, default_value_t = 10, hide = true)]
        max_iters: usize,
        /// Require a reachable OpenAI-compatible LLM (fail instead of falling back to mock).
        #[arg(long, default_value_t = false, hide = true)]
        require_llm: bool,
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
        /// LLM mode: off, mock, auto, openai.
        #[arg(long, default_value = "mock")]
        llm_mode: String,
        /// LLM model name (openai mode).
        #[arg(long, default_value = "gpt-5.2")]
        model: String,
        /// OpenAI API key file (overrides OPENAI_API_KEY).
        #[arg(long)]
        openai_key_file: Option<PathBuf>,
        /// Require a reachable OpenAI-compatible LLM (fail instead of falling back to mock).
        #[arg(long, default_value_t = false)]
        require_llm: bool,
    },
    /// Generate a LaTeX (and optionally PDF) findings report from `results/best_results.json`.
    Findings {
        /// Results directory (contains `best_results.json`).
        #[arg(long, default_value = "results")]
        results_dir: PathBuf,
        /// Output TeX path.
        #[arg(long, default_value = "results/findings.tex")]
        out_tex: PathBuf,
        /// Do not attempt to build a PDF via pdflatex.
        #[arg(long)]
        no_pdf: bool,
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
        Commands::Quickstart {
            out_dir,
            steps,
            max_iters,
            require_llm,
        } => {
            run_quickstart(out_dir, steps, max_iters, require_llm)?;
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
            require_llm,
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
                require_llm,
            )?;
        }
        Commands::Findings {
            results_dir,
            out_tex,
            no_pdf,
        } => {
            run_findings(results_dir, out_tex, !no_pdf)?;
        }
    }
    Ok(())
}

fn run_quickstart(out_dir: Option<PathBuf>, steps: usize, max_iters: usize, require_llm: bool) -> anyhow::Result<()> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let out_dir = out_dir.unwrap_or_else(|| {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        std::path::Path::new("results").join(format!("quickstart_{ts}"))
    });
    fs::create_dir_all(&out_dir)?;

    let config_path = out_dir.join("config.json");

    // Emit a starter config (so users can tweak constants/integrators if desired).
    fs::write(&config_path, serde_json::to_string_pretty(&Config::default())?)?;

    let solver_settings = DiscoverySolverSettings {
        solver: DiscoverySolver::Stls,
        normalize: true,
        stls_thresholds: Vec::new(),
        stls_ridge_lambda: 1e-8,
        stls_max_iter: 25,
        lasso_alphas: Vec::new(),
        lasso_max_iter: 2000,
        lasso_tol: 1e-6,
    };

    let factory_dir = out_dir.join("factory");
    run_factory(
        factory_dir.clone(),
        max_iters,
        true,
        Some(config_path.clone()),
        steps,
        0.01,
        "standard".to_string(),
        "three-body".to_string(),
        false,
        false,
        false,
        50,
        20,
        42,
        "mse".to_string(),
        "euler".to_string(),
        solver_settings,
        "auto".to_string(),
        "gpt-5.2".to_string(),
        None,
        require_llm,
    )?;

    // Copy a single novice-friendly artifact to the quickstart root.
    let eval_md = factory_dir.join("evaluation.md");
    if eval_md.exists() {
        let _ = fs::copy(&eval_md, out_dir.join("RESULTS.md"));
    }
    let eval_pdf = factory_dir.join("evaluation.pdf");
    if eval_pdf.exists() {
        let _ = fs::copy(&eval_pdf, out_dir.join("RESULTS.pdf"));
    }

    update_best_results_index_best_effort(std::path::Path::new("results"));

    println!("Quickstart complete: {}", out_dir.display());
    println!("Key outputs:");
    println!("- {}/RESULTS.md", out_dir.display());
    println!("- {}/RESULTS.pdf (if built)", out_dir.display());
    println!("- {}/factory/ (evidence + per-iteration artifacts)", out_dir.display());
    Ok(())
}

fn run_findings(results_dir: PathBuf, out_tex: PathBuf, build_pdf: bool) -> anyhow::Result<()> {
    update_best_results_index_best_effort(&results_dir);
    let index = load_best_results_index(&results_dir)
        .or_else(|| scan_best_results(&results_dir).ok())
        .ok_or_else(|| anyhow::anyhow!("no best_results found; run quickstart/factory first"))?;

    if let Some(parent) = out_tex.parent() {
        fs::create_dir_all(parent)?;
    }
    let progress = compute_bucket_progress(&results_dir, &index).unwrap_or_default();
    let tex = render_findings_tex(&index, &progress);
    fs::write(&out_tex, tex)?;
    println!("Findings TeX written -> {}", out_tex.display());

    if !build_pdf {
        return Ok(());
    }
    let Some(pdflatex) = find_pdflatex() else {
        eprintln!("pdflatex not found; skipped findings PDF build.");
        return Ok(());
    };

    let tex_dir = out_tex.parent().unwrap_or_else(|| std::path::Path::new("."));
    let tex_name = out_tex
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid out_tex filename"))?;
    let output = std::process::Command::new(&pdflatex)
        .current_dir(tex_dir)
        .args(["-interaction=nonstopmode", tex_name])
        .output();
    let pdf_path = out_tex.with_extension("pdf");
    match output {
        Ok(out) => {
            if !out.status.success() && !pdf_path.exists() {
                let stem = out_tex
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("findings");
                let err_path = tex_dir.join(format!("{stem}_pdf_error.txt"));
                let mut msg = String::new();
                msg.push_str("pdflatex failed\n");
                msg.push_str("stdout:\n");
                msg.push_str(&String::from_utf8_lossy(&out.stdout));
                msg.push_str("\nstderr:\n");
                msg.push_str(&String::from_utf8_lossy(&out.stderr));
                let _ = fs::write(err_path, msg);
            }
        }
        Err(err) => {
            if !pdf_path.exists() {
                let stem = out_tex
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("findings");
                let err_path = tex_dir.join(format!("{stem}_pdf_error.txt"));
                let _ = fs::write(err_path, format!("failed to run pdflatex: {err}"));
            }
        }
    }
    if pdf_path.exists() {
        println!("Findings PDF written -> {}", pdf_path.display());
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let stem = out_tex
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("findings");
        let stamped = tex_dir.join(format!("{stem}_{secs}.pdf"));
        let _ = fs::copy(&pdf_path, &stamped);
        println!("Findings PDF (timestamped) -> {}", stamped.display());
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
    let mut flags = Vec::new();
    if regime == "gravity_only" {
        let uses_em = eq
            .terms
            .iter()
            .any(|t| t.feature.starts_with("elec_") || t.feature.starts_with("mag_"));
        if uses_em {
            flags.push("em_terms_in_gravity_only".to_string());
        }
        let uses_velocity = eq.terms.iter().any(|t| {
            t.feature.starts_with("mag_")
                || t.feature.starts_with("vel")
                || t.feature.starts_with("v_")
                || t.feature.contains("_vel")
                || t.feature.contains("vel_")
                || t.feature.contains("vrel")
                || t.feature.contains("dv")
        });
        if uses_velocity {
            flags.push("velocity_terms_in_gravity_only".to_string());
        }
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
    Auto,
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
        "auto" => Ok(LlmMode::Auto),
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
        LlmMode::Auto => "auto",
        LlmMode::OpenAI => "openai",
    }
}

fn select_llm_client(
    mode: LlmMode,
    model: &str,
    openai_key_file: Option<&std::path::Path>,
    require_llm: bool,
) -> anyhow::Result<Option<Box<dyn LlmClient>>> {
    match mode {
        LlmMode::Off => {
            if require_llm {
                anyhow::bail!("LLM required but --llm-mode off was selected");
            }
            Ok(None)
        }
        LlmMode::Mock => {
            if require_llm {
                anyhow::bail!("LLM required but --llm-mode mock was selected");
            }
            Ok(Some(Box::new(MockLlm)))
        }
        LlmMode::Auto => {
            if require_llm {
                Ok(Some(Box::new(OpenAIClient::from_env_or_file(
                    model,
                    openai_key_file,
                )?)))
            } else {
                Ok(Some(Box::new(AutoLlmClient::from_env_or_file(
                    model,
                    openai_key_file,
                ))))
            }
        }
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
    require_llm: bool,
) -> anyhow::Result<()> {
    fs::create_dir_all(&out_dir)?;
    let mut next_ic: Option<InitialConditionSpec> = None;
    let llm_mode = parse_llm_mode(&llm_mode)?;
    let llm_client = select_llm_client(llm_mode, &model, openai_key_file.as_deref(), require_llm)?;
    let mut current_fitness = parse_fitness_heuristic(&fitness)?;
    let mut current_rollout = parse_rollout_integrator(&rollout_integrator)?;
    let mut current_solver = solver_settings;
    let mut evaluation_iterations: Vec<FactoryEvaluationIteration> = Vec::new();
    let incumbent_bucket = BucketKey { steps, dt };
    let incumbent = load_incumbent_for_bucket(std::path::Path::new("results"), &incumbent_bucket);

    if require_llm {
        let Some(client) = llm_client.as_ref() else {
            anyhow::bail!("LLM required but no client is configured");
        };
        let preflight = IcRequest {
            bounds: default_ic_bounds(),
            regime: "gravity_only".to_string(),
            notes: vec!["llm_preflight=true".to_string()],
            seed: Some(seed),
        };
        client
            .propose_initial_conditions(&preflight)
            .map_err(|e| anyhow::anyhow!("LLM preflight failed: {e}"))?;
    }

    for iter in 0..max_iters {
        let run_id = format!("run_{:03}", iter + 1);
        let run_dir = out_dir.join(&run_id);
        fs::create_dir_all(&run_dir)?;

        let cfg = build_config(config.clone(), &mode, None, em, no_em, no_gravity)?;
        let regime = if cfg.enable_em { "em_quasistatic" } else { "gravity_only" };
        let ic_bounds = default_ic_bounds();
        let mut ic_notes = vec!["avoid close encounters".to_string(), "prefer bounded motion".to_string()];
        if let Some(rec) = incumbent.as_ref() {
            ic_notes.extend(incumbent_prompt_notes(rec));
        }

        let mut ic_prompt = None;
        let mut ic_response = None;
        let max_ic_attempts = if require_llm { 3 } else { 1 };
        let mut ic_request: IcRequest;
        let mut ic_spec: InitialConditionSpec;
        let system: System;
        let mut last_err: Option<String> = None;
        let mut ic_attempt: usize = 0;

        'ic_loop: loop {
            ic_attempt += 1;
            let mut attempt_notes = ic_notes.clone();
            if let Some(err) = last_err.as_ref() {
                attempt_notes.push(format!("previous_ic_validation_error={}", single_line(err)));
            }
            ic_request = IcRequest {
                bounds: ic_bounds.clone(),
                regime: regime.to_string(),
                notes: attempt_notes,
                seed: Some(seed + iter as u64 + (ic_attempt as u64).saturating_sub(1)),
            };

            let candidate = if let Some(spec) = next_ic.take() {
                spec
            } else if let Some(client) = llm_client.as_ref() {
                match client.propose_initial_conditions(&ic_request) {
                    Ok(result) => {
                        ic_prompt = Some(result.prompt);
                        ic_response = Some(result.response);
                        result.value
                    }
                    Err(err) => {
                        if require_llm {
                            return Err(anyhow::anyhow!("LLM IC proposal failed: {err}"));
                        }
                        eprintln!("LLM IC proposal failed: {}", err);
                        initial_conditions_from_preset(&preset)?
                    }
                }
            } else if require_llm {
                anyhow::bail!("LLM required but no client is configured");
            } else {
                initial_conditions_from_preset(&preset)?
            };
            ic_spec = candidate;

            match system_from_ic(&ic_spec, &ic_bounds) {
                Ok(sys) => {
                    system = sys;
                    break 'ic_loop;
                }
                Err(err) => {
                    if !require_llm {
                        eprintln!("IC validation failed ({err}); falling back to preset.");
                        ic_spec = initial_conditions_from_preset(&preset)?;
                        system = preset_system(&preset)?;
                        break 'ic_loop;
                    }
                    last_err = Some(err.to_string());
                    if ic_attempt >= max_ic_attempts {
                        anyhow::bail!("IC validation failed after {max_ic_attempts} attempts: {err}");
                    }
                    eprintln!("IC validation failed ({err}); retrying with LLM.");
                    continue 'ic_loop;
                }
            }
        }

        let ic_request_path = run_dir.join("ic_request.json");
        fs::write(&ic_request_path, serde_json::to_string_pretty(&ic_request)?)?;
        let ic_spec_path = run_dir.join("initial_conditions.json");
        fs::write(&ic_spec_path, serde_json::to_string_pretty(&ic_spec)?)?;
        let cfg_path = run_dir.join("config.json");
        fs::write(&cfg_path, serde_json::to_string_pretty(&cfg)?)?;

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
        if let Some(rec) = incumbent.as_ref() {
            judge_input.notes.extend(incumbent_prompt_notes(rec));
        }
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
                        if require_llm {
                            anyhow::bail!("LLM judge response failed validation");
                        }
                        eprintln!("LLM judge response failed validation");
                        None
                    }
                }
                Err(err) => {
                    if require_llm {
                        return Err(anyhow::anyhow!("LLM judge failed: {err}"));
                    }
                    eprintln!("LLM judge failed: {}", err);
                    None
                }
            }
        } else if require_llm {
            anyhow::bail!("LLM required but no client is configured");
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

        let solver_summary = DiscoverySolverSummary {
            name: discovery_solver_label(current_solver.solver).to_string(),
            normalize: current_solver.normalize,
            fitness_heuristic: current_fitness.as_str().to_string(),
            stls: matches!(current_solver.solver, DiscoverySolver::Stls).then(|| StlsSolverSummary {
                auto_thresholds: current_solver.stls_thresholds.is_empty(),
                thresholds: current_solver.stls_thresholds.clone(),
                ridge_lambda: current_solver.stls_ridge_lambda,
                max_iter: current_solver.stls_max_iter,
            }),
            lasso: matches!(current_solver.solver, DiscoverySolver::Lasso).then(|| LassoSolverSummary {
                auto_alphas: current_solver.lasso_alphas.is_empty(),
                alphas: current_solver.lasso_alphas.clone(),
                max_iter: current_solver.lasso_max_iter,
                tol: current_solver.lasso_tol,
            }),
            ga: matches!(current_solver.solver, DiscoverySolver::Ga)
                .then(|| GaSolverSummary {
                    runs,
                    population,
                    seed: ga_seed,
                }),
        };
        let mut candidates_sorted = vector_candidates.clone();
        candidates_sorted.sort_by(|a, b| {
            a.metrics
                .mse
                .partial_cmp(&b.metrics.mse)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_candidates: Vec<FactoryEvaluationCandidate> = candidates_sorted
            .into_iter()
            .take(3)
            .map(|c| FactoryEvaluationCandidate {
                id: c.id,
                equation_text: c.equation_text,
                metrics: c.metrics,
            })
            .collect();
        let judge_summary = judge.as_ref().map(|j| FactoryEvaluationIterationJudge {
            summary: j.summary.clone(),
            ranking: j.ranking.clone(),
            recommendations: JudgeRecommendationsLite::from(&j.recommendations),
        });
        evaluation_iterations.push(FactoryEvaluationIteration {
            iteration: iter + 1,
            run_id: run_id.clone(),
            regime: regime.to_string(),
            solver: solver_summary,
            simulation: Some(sim_summary.clone()),
            top_candidates,
            judge: judge_summary,
        });

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

    let mut eval_notes = Vec::new();
    eval_notes.push(format!("max_iters={}", max_iters));
    eval_notes.push(format!("steps={}", steps));
    eval_notes.push(format!("dt={}", dt));
    eval_notes.push(format!("mode={}", mode));
    eval_notes.push(format!("preset={}", preset));
    eval_notes.push(format!("seed={}", seed));
    eval_notes.push(format!("auto={}", auto));
    eval_notes.push(format!("llm_mode={}", llm_mode_label(llm_mode)));
    if matches!(llm_mode, LlmMode::OpenAI | LlmMode::Auto) {
        eval_notes.push(format!("llm_model={}", model));
    }
    let eval_input = FactoryEvaluationInput {
        version: FACTORY_EVALUATION_VERSION.to_string(),
        notes: eval_notes,
        iterations: evaluation_iterations,
    };
    let eval_input_path = out_dir.join("evaluation_input.json");
    fs::write(&eval_input_path, serde_json::to_string_pretty(&eval_input)?)?;

    let current_best = best_candidate_from_eval_input(&eval_input);
    let prior_filter = PriorAttemptFilter::from_notes(&eval_input.notes);
    let prior_best = find_best_prior_attempt(std::path::Path::new("results"), &eval_input_path, &prior_filter)?;
    let best_ic: Option<InitialConditionSpec> = current_best
        .as_ref()
        .and_then(|b| fs::read_to_string(out_dir.join(&b.run_id).join("initial_conditions.json")).ok())
        .and_then(|raw| serde_json::from_str(&raw).ok());
    let incumbent_on_best_run = (|| -> anyhow::Result<Option<IncumbentEvalOnBestRun>> {
        let (best, prior) = match (current_best.as_ref(), prior_best.as_ref()) {
            (Some(b), Some(p)) => (b, p),
            _ => return Ok(None),
        };
        let best_run_dir = out_dir.join(&best.run_id);
        let oracle = match load_oracle_run_from_dir(&best_run_dir) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };

        let rollout_integrator = eval_input
            .iterations
            .iter()
            .find(|it| it.run_id == best.run_id)
            .and_then(|it| it.simulation.as_ref())
            .and_then(|sim| parse_rollout_integrator(&sim.rollout_integrator).ok())
            .unwrap_or(RolloutIntegrator::Euler);

        let incumbent_model = match vector_model_from_equation_text(&prior.best.candidate.equation_text) {
            Some(v) => v,
            None => return Ok(None),
        };
        let complexity = incumbent_model.eq_x.complexity() + incumbent_model.eq_y.complexity() + incumbent_model.eq_z.complexity();
        let feature_names = FeatureLibrary::default_physics().features;
        let (rmse, div) = rollout_metrics(
            &incumbent_model,
            &feature_names,
            &oracle.result,
            &oracle.cfg,
            rollout_integrator,
        );
        Ok(Some(IncumbentEvalOnBestRun {
            equation_text: prior.best.candidate.equation_text.clone(),
            rollout_rmse: rmse,
            divergence_time: div,
            complexity,
        }))
    })()?;

    #[derive(serde::Serialize)]
    struct HistoryAttempt {
        factory_dir: String,
        eval_input: String,
        run_id: String,
        regime: String,
        equation_text: String,
        metrics: CandidateMetrics,
    }

    #[derive(serde::Serialize)]
    struct HistoryDelta {
        rollout_rmse: Option<f64>,
        mse: Option<f64>,
        complexity: Option<i64>,
    }

    #[derive(serde::Serialize)]
    struct EvaluationHistory {
        version: String,
        comparator: String,
        current: Option<HistoryAttempt>,
        prior_best: Option<HistoryAttempt>,
        delta: Option<HistoryDelta>,
    }

    let current_attempt = current_best.as_ref().map(|b| HistoryAttempt {
        factory_dir: out_dir.display().to_string(),
        eval_input: eval_input_path.display().to_string(),
        run_id: b.run_id.clone(),
        regime: b.regime.clone(),
        equation_text: b.candidate.equation_text.clone(),
        metrics: b.candidate.metrics.clone(),
    });
    let prior_attempt = prior_best.as_ref().map(|p| HistoryAttempt {
        factory_dir: p.factory_dir.display().to_string(),
        eval_input: p.eval_input_path.display().to_string(),
        run_id: p.best.run_id.clone(),
        regime: p.best.regime.clone(),
        equation_text: p.best.candidate.equation_text.clone(),
        metrics: p.best.candidate.metrics.clone(),
    });
    let delta = match (current_best.as_ref(), prior_best.as_ref()) {
        (Some(c), Some(p)) => {
            let curr_roll = c.candidate.metrics.rollout_rmse.unwrap_or(f64::INFINITY);
            let prior_roll = p.best.candidate.metrics.rollout_rmse.unwrap_or(f64::INFINITY);
            let rollout_rmse = (curr_roll.is_finite() && prior_roll.is_finite()).then_some(curr_roll - prior_roll);
            let mse = (c.candidate.metrics.mse.is_finite() && p.best.candidate.metrics.mse.is_finite())
                .then_some(c.candidate.metrics.mse - p.best.candidate.metrics.mse);
            let complexity = Some(c.candidate.metrics.complexity as i64 - p.best.candidate.metrics.complexity as i64);
            Some(HistoryDelta {
                rollout_rmse,
                mse,
                complexity,
            })
        }
        _ => None,
    };
    let history = EvaluationHistory {
        version: "v1".to_string(),
        comparator: "rollout_rmse -> mse -> complexity (prior filtered by steps/dt when present)".to_string(),
        current: current_attempt,
        prior_best: prior_attempt,
        delta,
    };
    fs::write(
        out_dir.join("evaluation_history.json"),
        serde_json::to_string_pretty(&history)?,
    )?;

    let eval_tex_path = out_dir.join("evaluation.tex");
    let eval_tex = render_evaluation_tex(
        &eval_input,
        current_best.as_ref(),
        prior_best.as_ref(),
        best_ic.as_ref(),
    );
    fs::write(&eval_tex_path, eval_tex)?;

    // Best-effort PDF build when pdflatex is available.
    if let Some(pdflatex) = find_pdflatex() {
        let output = std::process::Command::new(&pdflatex)
            .current_dir(&out_dir)
            .args(["-interaction=nonstopmode", "evaluation.tex"])
            .output();
        match output {
            Ok(out) => {
                if !out.status.success() && !out_dir.join("evaluation.pdf").exists() {
                    let mut msg = String::new();
                    msg.push_str("pdflatex failed\n");
                    msg.push_str("stdout:\n");
                    msg.push_str(&String::from_utf8_lossy(&out.stdout));
                    msg.push_str("\nstderr:\n");
                    msg.push_str(&String::from_utf8_lossy(&out.stderr));
                    fs::write(out_dir.join("evaluation_pdf_error.txt"), msg)?;
                }
            }
            Err(err) => {
                if !out_dir.join("evaluation.pdf").exists() {
                    fs::write(
                        out_dir.join("evaluation_pdf_error.txt"),
                        format!("failed to run pdflatex: {err}"),
                    )?;
                }
            }
        }
    }

    let eval_prompt_path = out_dir.join("evaluation_prompt.txt");
    let eval_md_path = out_dir.join("evaluation.md");
    let eval_llm_md_path = out_dir.join("evaluation_llm.md");
    let exec_md = render_executive_summary_md(
        &eval_input,
        current_best.as_ref(),
        prior_best.as_ref(),
        best_ic.as_ref(),
        incumbent_on_best_run.as_ref(),
    );

    let mut raw_md = String::new();
    let prompt_text: String;
    if let Some(client) = llm_client.as_ref() {
        match client.explain_factory_evaluation(&eval_input) {
            Ok(res) => {
                prompt_text = res.prompt;
                raw_md = res.value;
            }
            Err(err) => {
                if require_llm {
                    return Err(anyhow::anyhow!("LLM evaluation failed: {err}"));
                }
                eprintln!("LLM evaluation failed: {}", err);
                fs::write(out_dir.join("evaluation_error.txt"), err.to_string())?;
                let fallback = MockLlm.explain_factory_evaluation(&eval_input)?;
                prompt_text = fallback.prompt;
                raw_md.push_str("<!-- WARNING: LLM evaluation failed; using local fallback. -->\n\n");
                raw_md.push_str(&fallback.value);
            }
        }
    } else if require_llm {
        anyhow::bail!("LLM required but `--llm-mode off` was selected");
    } else {
        prompt_text = "LLM evaluation disabled (`--llm-mode off`).".to_string();
        raw_md.push_str("# Factory Evaluation\n\n");
        raw_md.push_str("LLM evaluation was disabled (`--llm-mode off`).\n\n");
        raw_md.push_str("Open per-run reports in `run_###/report.md` and compare:\n");
        raw_md.push_str("- `mse` (lower is better)\n");
        raw_md.push_str("- `rollout_rmse` (lower is better)\n");
        raw_md.push_str("- `divergence_time` (higher is better)\n");
        raw_md.push_str("- `complexity` (lower is simpler)\n");
    }

    fs::write(&eval_prompt_path, prompt_text)?;
    fs::write(&eval_llm_md_path, &raw_md)?;
    fs::write(&eval_md_path, exec_md)?;

    update_best_results_index_best_effort(std::path::Path::new("results"));

    println!("Factory evaluation written -> {}", eval_md_path.display());
    Ok(())
}

#[derive(Clone, Debug)]
struct BestFactoryCandidate {
    run_id: String,
    regime: String,
    candidate: FactoryEvaluationCandidate,
}

fn metrics_sort_key(metrics: &CandidateMetrics) -> (f64, f64, usize) {
    let rollout_rmse = metrics.rollout_rmse.unwrap_or(f64::INFINITY);
    let rollout_rmse = if rollout_rmse.is_finite() { rollout_rmse } else { f64::INFINITY };
    let mse = if metrics.mse.is_finite() { metrics.mse } else { f64::INFINITY };
    (rollout_rmse, mse, metrics.complexity)
}

fn best_candidate_from_eval_input(input: &FactoryEvaluationInput) -> Option<BestFactoryCandidate> {
    let mut best: Option<BestFactoryCandidate> = None;
    let mut best_key: Option<(f64, f64, usize)> = None;
    for iter in &input.iterations {
        for cand in &iter.top_candidates {
            let key = metrics_sort_key(&cand.metrics);
            if best_key.map_or(true, |bk| key < bk) {
                best_key = Some(key);
                best = Some(BestFactoryCandidate {
                    run_id: iter.run_id.clone(),
                    regime: iter.regime.clone(),
                    candidate: cand.clone(),
                });
            }
        }
    }
    best
}

#[derive(Clone, Debug)]
struct PriorAttemptBest {
    factory_dir: PathBuf,
    eval_input_path: PathBuf,
    best: BestFactoryCandidate,
}

#[derive(Clone, Debug, Default)]
struct PriorAttemptFilter {
    steps: Option<usize>,
    dt: Option<f64>,
}

impl PriorAttemptFilter {
    fn from_notes(notes: &[String]) -> Self {
        fn find_value<'a>(notes: &'a [String], key: &str) -> Option<&'a str> {
            let prefix = format!("{key}=");
            for note in notes {
                if let Some(rest) = note.strip_prefix(&prefix) {
                    return Some(rest);
                }
            }
            None
        }

        let steps = find_value(notes, "steps").and_then(|v| v.parse::<usize>().ok());
        let dt = find_value(notes, "dt").and_then(|v| v.parse::<f64>().ok());
        Self { steps, dt }
    }

    fn matches(&self, input: &FactoryEvaluationInput) -> bool {
        let other = PriorAttemptFilter::from_notes(&input.notes);
        if let Some(steps) = self.steps {
            if other.steps != Some(steps) {
                return false;
            }
        }
        if let Some(dt) = self.dt {
            match other.dt {
                Some(v) if (v - dt).abs() <= 1e-12 => {}
                _ => return false,
            }
        }
        true
    }
}

fn collect_eval_input_paths(root: &std::path::Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    if !root.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_eval_input_paths(&path, out)?;
            continue;
        }
        if path.file_name().and_then(|s| s.to_str()) == Some("evaluation_input.json") {
            out.push(path);
        }
    }
    Ok(())
}

fn find_best_prior_attempt(
    results_root: &std::path::Path,
    exclude_eval_input: &std::path::Path,
    filter: &PriorAttemptFilter,
) -> anyhow::Result<Option<PriorAttemptBest>> {
    let mut paths = Vec::new();
    collect_eval_input_paths(results_root, &mut paths)?;
    let mut best_prior: Option<PriorAttemptBest> = None;
    let mut best_key: Option<(f64, f64, usize)> = None;

    for path in paths {
        if path == exclude_eval_input {
            continue;
        }
        let raw = match fs::read_to_string(&path) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let input: FactoryEvaluationInput = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if !filter.matches(&input) {
            continue;
        }
        let best = match best_candidate_from_eval_input(&input) {
            Some(v) => v,
            None => continue,
        };
        let key = metrics_sort_key(&best.candidate.metrics);
        if best_key.map_or(true, |bk| key < bk) {
            best_key = Some(key);
            let factory_dir = path.parent().unwrap_or_else(|| std::path::Path::new(".")).to_path_buf();
            best_prior = Some(PriorAttemptBest {
                factory_dir,
                eval_input_path: path,
                best,
            });
        }
    }

    Ok(best_prior)
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
struct BucketKey {
    steps: usize,
    dt: f64,
}

impl BucketKey {
    fn matches(&self, other: &BucketKey) -> bool {
        self.steps == other.steps && (self.dt - other.dt).abs() <= 1e-12
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
struct BestRecord {
    bucket: BucketKey,
    run_dir: String,
    factory_dir: String,
    eval_input_path: String,
    run_id: String,
    regime: String,
    equation_text: String,
    metrics: CandidateMetrics,
    ic: Option<InitialConditionSpec>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
struct BestResultsIndexV1 {
    version: String,
    updated_at_utc: String,
    buckets: Vec<BestRecord>,
    notes: Vec<String>,
}

fn updated_at_unix_utc() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix_seconds={secs}")
}

fn bucket_from_notes(notes: &[String]) -> Option<BucketKey> {
    fn find_value<'a>(notes: &'a [String], key: &str) -> Option<&'a str> {
        let prefix = format!("{key}=");
        for note in notes {
            if let Some(rest) = note.strip_prefix(&prefix) {
                return Some(rest);
            }
        }
        None
    }
    let steps = find_value(notes, "steps").and_then(|v| v.parse::<usize>().ok())?;
    let dt = find_value(notes, "dt").and_then(|v| v.parse::<f64>().ok())?;
    Some(BucketKey { steps, dt })
}

fn collect_factory_eval_input_paths(root: &std::path::Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    if !root.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_factory_eval_input_paths(&path, out)?;
            continue;
        }
        if path.file_name().and_then(|s| s.to_str()) != Some("evaluation_input.json") {
            continue;
        }
        let parent_is_factory = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            == Some("factory");
        if parent_is_factory {
            out.push(path);
        }
    }
    Ok(())
}

fn best_record_from_eval_input_path(path: &std::path::Path) -> Option<BestRecord> {
    let raw = fs::read_to_string(path).ok()?;
    let input: FactoryEvaluationInput = serde_json::from_str(&raw).ok()?;
    let bucket = bucket_from_notes(&input.notes)?;
    let best = best_candidate_from_eval_input(&input)?;

    let factory_dir = path.parent()?.to_path_buf();
    let run_dir = factory_dir
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .to_path_buf();

    let ic: Option<InitialConditionSpec> = fs::read_to_string(factory_dir.join(&best.run_id).join("initial_conditions.json"))
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok());

    Some(BestRecord {
        bucket,
        run_dir: run_dir.to_string_lossy().to_string(),
        factory_dir: factory_dir.to_string_lossy().to_string(),
        eval_input_path: path.to_string_lossy().to_string(),
        run_id: best.run_id,
        regime: best.regime,
        equation_text: best.candidate.equation_text,
        metrics: best.candidate.metrics,
        ic,
    })
}

fn scan_best_results(results_root: &std::path::Path) -> anyhow::Result<BestResultsIndexV1> {
    let mut paths = Vec::new();
    collect_factory_eval_input_paths(results_root, &mut paths)?;
    let mut buckets: Vec<BestRecord> = Vec::new();

    for path in paths {
        let record = match best_record_from_eval_input_path(&path) {
            Some(v) => v,
            None => continue,
        };
        let key = metrics_sort_key(&record.metrics);
        let existing_idx = buckets
            .iter()
            .position(|e| e.bucket.matches(&record.bucket));
        if let Some(idx) = existing_idx {
            let existing_key = metrics_sort_key(&buckets[idx].metrics);
            if key < existing_key || (key == existing_key && record.eval_input_path < buckets[idx].eval_input_path) {
                buckets[idx] = record;
            }
        } else {
            buckets.push(record);
        }
    }

    buckets.sort_by(|a, b| {
        a.bucket
            .steps
            .cmp(&b.bucket.steps)
            .then_with(|| a.bucket.dt.partial_cmp(&b.bucket.dt).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.eval_input_path.cmp(&b.eval_input_path))
    });

    Ok(BestResultsIndexV1 {
        version: "v1".to_string(),
        updated_at_utc: updated_at_unix_utc(),
        buckets,
        notes: vec![
            "Selection rule: minimize (rollout_rmse, mse, complexity).".to_string(),
            "Buckets are grouped by (steps, dt).".to_string(),
            "Only */factory/evaluation_input.json files are considered.".to_string(),
        ],
    })
}

fn format_metric_opt(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.6}"),
        Some(_) => "nan".to_string(),
        None => "n/a".to_string(),
    }
}

fn render_best_results_md(index: &BestResultsIndexV1) -> String {
    let mut md = String::new();
    md.push_str("# Best Results (auto-updated)\n\n");
    md.push_str(&format!("Last updated: `{}`\n\n", index.updated_at_utc));

    md.push_str("## Executive Summary\n\n");
    md.push_str("- This file is the single top-level place to see the best discovered equations so far.\n");
    md.push_str("- Comparator: minimize `rollout_rmse`, then `mse`, then `complexity`.\n");
    md.push_str("- Buckets: grouped by `(steps, dt)` to avoid apples-to-oranges comparisons.\n");
    if index.buckets.is_empty() {
        md.push_str("\nNo runs found yet under `results/**/factory/evaluation_input.json`.\n");
        return md;
    }
    md.push_str("\nTracked buckets:\n");
    for rec in &index.buckets {
        md.push_str(&format!("- steps={}, dt={}\n", rec.bucket.steps, rec.bucket.dt));
    }

    md.push_str("\n## Current Incumbents\n\n");
    for rec in &index.buckets {
        md.push_str(&format!("### steps={}, dt={}\n\n", rec.bucket.steps, rec.bucket.dt));
        md.push_str(&format!(
            "- Best metrics: rollout_rmse={}, mse={:.6e}, complexity={}, divergence_time={}\n",
            format_metric_opt(rec.metrics.rollout_rmse),
            rec.metrics.mse,
            rec.metrics.complexity,
            format_metric_opt(rec.metrics.divergence_time),
        ));
        md.push_str("- Best equation (exact):\n");
        md.push_str("```text\n");
        md.push_str(&rec.equation_text);
        md.push_str("\n```\n\n");
        md.push_str(&format!("- Evidence: `{}/RESULTS.md`\n", rec.run_dir));
        md.push_str(&format!("- Raw: `{}`\n\n", rec.eval_input_path));
    }

    md.push_str("## Next Experiment (for humans and LLMs)\n\n");
    md.push_str("- Goal: reduce `rollout_rmse` without increasing `complexity`.\n");
    md.push_str("- If `mse` is tiny but `rollout_rmse` is large, the model fits accelerations but rolls out poorly (instability).\n");
    md.push_str("- Report improvements by attaching:\n");
    md.push_str("  - `results/best_results.md`\n");
    md.push_str("  - the run’s `RESULTS.md`\n");
    md.push_str("  - the run’s `factory/evaluation_input.json`\n");

    md
}

#[derive(Clone, Debug)]
struct BucketProgress {
    bucket: BucketKey,
    attempts: usize,
    first: BestRecord,
    best: BestRecord,
}

fn compute_bucket_progress(
    results_root: &std::path::Path,
    index: &BestResultsIndexV1,
) -> anyhow::Result<Vec<BucketProgress>> {
    let mut paths = Vec::new();
    collect_factory_eval_input_paths(results_root, &mut paths)?;
    let mut records: Vec<BestRecord> = Vec::new();
    for path in paths {
        if let Some(rec) = best_record_from_eval_input_path(&path) {
            records.push(rec);
        }
    }

    let mut out: Vec<BucketProgress> = Vec::new();
    for best in &index.buckets {
        let mut bucket_records: Vec<BestRecord> = records
            .iter()
            .filter(|r| r.bucket.matches(&best.bucket))
            .cloned()
            .collect();
        if bucket_records.is_empty() {
            continue;
        }
        bucket_records.sort_by(|a, b| a.eval_input_path.cmp(&b.eval_input_path));
        let first = bucket_records[0].clone();
        out.push(BucketProgress {
            bucket: best.bucket.clone(),
            attempts: bucket_records.len(),
            first,
            best: best.clone(),
        });
    }
    out.sort_by(|a, b| {
        a.bucket
            .steps
            .cmp(&b.bucket.steps)
            .then_with(|| a.bucket.dt.partial_cmp(&b.bucket.dt).unwrap_or(std::cmp::Ordering::Equal))
    });
    Ok(out)
}

fn render_findings_tex(index: &BestResultsIndexV1, progress: &[BucketProgress]) -> String {
    let mut tex = String::new();
    tex.push_str("\\documentclass[11pt]{article}\n");
    tex.push_str("\\usepackage[margin=1in]{geometry}\n");
    tex.push_str("\\usepackage{amsmath}\n");
    tex.push_str("\\usepackage{booktabs}\n");
    tex.push_str("\\usepackage{longtable}\n");
    tex.push_str("\\usepackage{hyperref}\n");
    tex.push_str("\\title{Threebody: Best Discovered Equations (Auto-Generated)}\n");
    tex.push_str(&format!("\\date{{{}}}\n", escape_latex(&index.updated_at_utc)));
    tex.push_str("\\begin{document}\n");
    tex.push_str("\\maketitle\n\n");

    tex.push_str("\\section*{Executive Summary}\n");
    tex.push_str("This report summarizes the best equation(s) discovered by local runs of the threebody system.\\\\\n");
    tex.push_str("Comparator: minimize rollout\\_rmse, then mse, then complexity (lower is better).\\\\\n");
    tex.push_str("Buckets are grouped by (steps, dt) to avoid apples-to-oranges comparisons.\\\\\n\n");

    tex.push_str("\\section*{Plain-language context: state of the art}\n");
    tex.push_str("The modern (``state of the art'') way to discover equations from data usually looks like this:\\\\\n");
    tex.push_str("\\begin{itemize}\n");
    tex.push_str("  \\item Choose a menu (a \\emph{library}) of candidate building blocks (e.g., gravity-like terms).\n");
    tex.push_str("  \\item Fit a sparse combination using sparse regression (STLS/LASSO), so most coefficients become exactly zero.\n");
    tex.push_str("  \\item Validate the equation by simulating it forward (a \\emph{rollout}) and comparing trajectories, not just instantaneous fit.\n");
    tex.push_str("\\end{itemize}\n");
    tex.push_str("In plain terms: a good equation is one that stays accurate \\emph{over time}, not just one that matches one step.\\\\\n\n");

    tex.push_str("\\section*{Did we improve (over local history)?}\n");
    tex.push_str("For each bucket (steps, dt), we compare the first recorded attempt to the best found so far.\\\\\n");
    tex.push_str("\\begin{longtable}{lrrrr}\n");
    tex.push_str("\\toprule\n");
    tex.push_str("Bucket & Runs & First rollout\\_rmse & Best rollout\\_rmse & Improvement\\\\\n");
    tex.push_str("\\midrule\n");
    for p in progress {
        let first = p.first.metrics.rollout_rmse.unwrap_or(f64::INFINITY);
        let best = p.best.metrics.rollout_rmse.unwrap_or(f64::INFINITY);
        let improvement = if first.is_finite() && best.is_finite() && first > 0.0 {
            let delta = first - best;
            if delta.abs() <= 1e-12 {
                "no change".to_string()
            } else {
                format!("{:+.6} ({:+.1}\\%)", delta, (delta / first) * 100.0)
            }
        } else {
            "n/a".to_string()
        };
        tex.push_str(&format!(
            "steps={} dt={} & {} & {} & {} & {}\\\\\n",
            p.bucket.steps,
            p.bucket.dt,
            p.attempts,
            format_opt_f64(p.first.metrics.rollout_rmse),
            format_opt_f64(p.best.metrics.rollout_rmse),
            improvement
        ));
    }
    tex.push_str("\\bottomrule\n");
    tex.push_str("\\end{longtable}\n\n");
    tex.push_str("If the improvement says ``no change'', it typically means we are deterministically rediscovering the same law with the same settings, or that the discovery problem is already saturated under the current feature library.\\\\\n\n");

    if index.buckets.is_empty() {
        tex.push_str("No runs found yet under \\texttt{results/**/factory/evaluation\\_input.json}.\\\\\n\n");
        tex.push_str("\\end{document}\n");
        return tex;
    }

    let highlight = index
        .buckets
        .iter()
        .max_by(|a, b| a.bucket.steps.cmp(&b.bucket.steps).then_with(|| a.bucket.dt.partial_cmp(&b.bucket.dt).unwrap_or(std::cmp::Ordering::Equal)));
    if let Some(best) = highlight {
        tex.push_str("\\subsection*{Most stringent bucket (largest steps)}\n");
        tex.push_str(&format!(
            "steps={}, dt={}. Best metrics: rollout\\_rmse={}, mse={:.6e}, complexity={}.\\\\\n",
            best.bucket.steps,
            best.bucket.dt,
            format_opt_f64(best.metrics.rollout_rmse),
            best.metrics.mse,
            best.metrics.complexity
        ));
        tex.push_str("Best equation (exact):\n");
        tex.push_str("\\begin{verbatim}\n");
        tex.push_str(&best.equation_text);
        tex.push_str("\n\\end{verbatim}\n\n");
    }

    tex.push_str("\\section*{Best Results by Bucket}\n\n");
    for rec in &index.buckets {
        tex.push_str(&format!(
            "\\subsection*{{steps={}, dt={}}}\n",
            rec.bucket.steps, rec.bucket.dt
        ));
        tex.push_str(&format!(
            "Metrics: rollout\\_rmse={}, mse={:.6e}, complexity={}, divergence\\_time={}.\\\\\n\n",
            format_opt_f64(rec.metrics.rollout_rmse),
            rec.metrics.mse,
            rec.metrics.complexity,
            format_opt_f64(rec.metrics.divergence_time)
        ));

        tex.push_str("\\paragraph{Best equation (exact)}\n");
        tex.push_str("\\begin{verbatim}\n");
        tex.push_str(&rec.equation_text);
        tex.push_str("\n\\end{verbatim}\n\n");

        tex.push_str("\\paragraph{Math formula (expanded)}\n");
        tex.push_str(&render_expanded_math_tex(&rec.equation_text));

        tex.push_str("\\paragraph{Evidence}\\\\\n");
        tex.push_str(&format!(
            "Run directory: \\texttt{{{}}}.\\\\\n",
            escape_latex(&rec.run_dir)
        ));
        tex.push_str(&format!(
            "Eval input: \\texttt{{{}}}.\\\\\n\n",
            escape_latex(&rec.eval_input_path)
        ));

        tex.push_str("\\paragraph{Initial conditions (best run)}\n");
        if let Some(ic) = rec.ic.as_ref() {
            tex.push_str(&format!(
                "Barycentric: \\texttt{{{}}}.\\\\\n",
                if ic.barycentric { "true" } else { "false" }
            ));
            tex.push_str(&format!("Notes: \\texttt{{{}}}.\\\\\n\n", escape_latex(&ic.notes)));
            tex.push_str("\\begin{longtable}{lrrrrrrrr}\n");
            tex.push_str("\\toprule\n");
            tex.push_str("Body & m & q & x & y & z & v_x & v_y & v_z\\\\\n");
            tex.push_str("\\midrule\n");
            for (i, b) in ic.bodies.iter().enumerate() {
                tex.push_str(&format!(
                    "{} & {:.6} & {:.6} & {:.6} & {:.6} & {:.6} & {:.6} & {:.6} & {:.6}\\\\\n",
                    i,
                    b.mass,
                    b.charge,
                    b.pos[0],
                    b.pos[1],
                    b.pos[2],
                    b.vel[0],
                    b.vel[1],
                    b.vel[2],
                ));
            }
            tex.push_str("\\bottomrule\n");
            tex.push_str("\\end{longtable}\n\n");
        } else {
            tex.push_str("N/A.\\\\\n\n");
        }
    }

    tex.push_str("\\section*{How to Reproduce}\n");
    tex.push_str("The simplest workflow runs 10 quickstarts and then regenerates this report:\\\\\n");
    tex.push_str("\\begin{verbatim}\n");
    tex.push_str("just quickstart10 1000\n");
    tex.push_str("cargo run -p threebody-cli -- findings\n");
    tex.push_str("\\end{verbatim}\n\n");

    tex.push_str("\\section*{How to Report Improvements (Novice-Friendly)}\n");
    tex.push_str("An improvement is typically a lower rollout\\_rmse at the same (or lower) complexity in the same bucket (steps, dt).\\\\\n");
    tex.push_str("Attach these files when reporting:\\\\\n");
    tex.push_str("\\begin{itemize}\n");
    tex.push_str("  \\item \\texttt{results/best\\_results.md}\n");
    tex.push_str("  \\item \\texttt{results/<run>/RESULTS.md}\n");
    tex.push_str("  \\item \\texttt{results/<run>/factory/evaluation\\_input.json}\n");
    tex.push_str("\\end{itemize}\n\n");

    tex.push_str("\\end{document}\n");
    tex
}

fn write_best_results(results_root: &std::path::Path, index: &BestResultsIndexV1) -> anyhow::Result<()> {
    fs::create_dir_all(results_root)?;
    let json_path = results_root.join("best_results.json");
    let md_path = results_root.join("best_results.md");
    fs::write(&json_path, serde_json::to_string_pretty(index)?)?;
    fs::write(&md_path, render_best_results_md(index))?;
    Ok(())
}

fn update_best_results_index_best_effort(results_root: &std::path::Path) {
    match scan_best_results(results_root).and_then(|idx| write_best_results(results_root, &idx)) {
        Ok(_) => {}
        Err(err) => eprintln!("WARN: failed to update best_results index: {err}"),
    }
}

fn truncate_to_chars(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    let mut out: String = input.chars().take(max_chars).collect();
    out.push_str("…");
    out
}

fn single_line(input: &str) -> String {
    input
        .replace(['\r', '\n', '\t'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn find_pdflatex() -> Option<PathBuf> {
    for candidate in ["pdflatex", "/Library/TeX/texbin/pdflatex"] {
        let ok = std::process::Command::new(candidate)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return Some(PathBuf::from(candidate));
        }
    }
    None
}

fn load_best_results_index(results_root: &std::path::Path) -> Option<BestResultsIndexV1> {
    let path = results_root.join("best_results.json");
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

fn load_incumbent_for_bucket(results_root: &std::path::Path, bucket: &BucketKey) -> Option<BestRecord> {
    if let Some(index) = load_best_results_index(results_root) {
        if let Some(hit) = index.buckets.into_iter().find(|r| r.bucket.matches(bucket)) {
            return Some(hit);
        }
    }
    // Fallback: scan the tree (best-effort).
    if let Ok(index) = scan_best_results(results_root) {
        if let Some(hit) = index.buckets.into_iter().find(|r| r.bucket.matches(bucket)) {
            return Some(hit);
        }
    }
    None
}

fn incumbent_prompt_notes(record: &BestRecord) -> Vec<String> {
    let eq = truncate_to_chars(&single_line(&record.equation_text), 500);
    vec![
        format!("INCUMBENT_BEST_EQUATION: {eq}"),
        format!(
            "INCUMBENT_BEST_METRICS: rollout_rmse={}, mse={:.6e}, complexity={}",
            format_metric_opt(record.metrics.rollout_rmse),
            record.metrics.mse,
            record.metrics.complexity
        ),
        format!("INCUMBENT_SOURCE_RUN_DIR: {}", record.run_dir),
        "GOAL: reduce rollout_rmse without increasing complexity.".to_string(),
    ]
}

fn escape_latex(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\textbackslash{}"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            '$' => out.push_str("\\$"),
            '&' => out.push_str("\\&"),
            '#' => out.push_str("\\#"),
            '_' => out.push_str("\\_"),
            '%' => out.push_str("\\%"),
            '^' => out.push_str("\\^{}"),
            '~' => out.push_str("\\~{}"),
            _ => out.push(ch),
        }
    }
    out
}

fn format_opt_f64(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.6}"),
        Some(_) => "nan".to_string(),
        None => "n/a".to_string(),
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct BasisUsage {
    grav: bool,
    elec: bool,
    mag: bool,
}

fn basis_usage_from_equation_text(text: &str) -> BasisUsage {
    BasisUsage {
        grav: text.contains("grav_"),
        elec: text.contains("elec_"),
        mag: text.contains("mag_"),
    }
}

fn parse_vector_equation_terms(equation_text: &str) -> [Vec<(f64, String)>; 3] {
    let mut x_terms: Vec<(f64, String)> = Vec::new();
    let mut y_terms: Vec<(f64, String)> = Vec::new();
    let mut z_terms: Vec<(f64, String)> = Vec::new();

    for part in equation_text.split(';') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (lhs, rhs) = match part.split_once('=') {
            Some(v) => v,
            None => continue,
        };
        let axis = lhs.trim().chars().last().unwrap_or(' ');
        let rhs = rhs.trim();
        if rhs.is_empty() || rhs == "0" {
            continue;
        }
        let target = match axis {
            'x' => &mut x_terms,
            'y' => &mut y_terms,
            'z' => &mut z_terms,
            _ => continue,
        };
        for token in rhs.split_whitespace() {
            let (coeff_raw, feature) = match token.split_once('*') {
                Some(v) => v,
                None => continue,
            };
            let coeff: f64 = match coeff_raw.parse::<f64>() {
                Ok(v) if v.is_finite() => v,
                _ => continue,
            };
            target.push((coeff, feature.to_string()));
        }
    }

    [x_terms, y_terms, z_terms]
}

fn feature_to_math_symbol(feature: &str, axis: char) -> String {
    let basis = if feature.starts_with("grav_") {
        "g"
    } else if feature.starts_with("elec_") {
        "e"
    } else if feature.starts_with("mag_") {
        "b"
    } else {
        return feature.to_string();
    };
    format!("{basis}_{{i,{axis}}}^*")
}

fn feature_to_tex_symbol(feature: &str, axis: char) -> String {
    if feature.starts_with("grav_") {
        return format!("g_{{i,{axis}}}^*");
    }
    if feature.starts_with("elec_") {
        return format!("e_{{i,{axis}}}^*");
    }
    if feature.starts_with("mag_") {
        return format!("b_{{i,{axis}}}^*");
    }
    format!("\\text{{\\texttt{{{}}}}}", escape_latex(feature))
}

fn render_expanded_math_md(equation_text: &str) -> String {
    let usage = basis_usage_from_equation_text(equation_text);
    let [tx, ty, tz] = parse_vector_equation_terms(equation_text);

    fn render_component(axis: char, terms: &[(f64, String)]) -> String {
        if terms.is_empty() {
            return "0".to_string();
        }
        let mut out = String::new();
        for (i, (coeff, feature)) in terms.iter().enumerate() {
            let sign = if *coeff >= 0.0 { "+" } else { "-" };
            let abs = coeff.abs();
            if i == 0 {
                if sign == "-" {
                    out.push_str("-");
                }
            } else {
                out.push(' ');
                out.push_str(sign);
                out.push(' ');
            }
            out.push_str(&format!("{abs:.6} * {}", feature_to_math_symbol(feature, axis)));
        }
        out
    }

    let mut md = String::new();
    md.push_str("\n## Math Formula (Expanded)\n");
    md.push_str("Same best model, written in standard physics-ish notation (per body `i`):\n\n");
    md.push_str("```text\n");
    md.push_str(&format!("a_{{i,x}} ≈ {}\n", render_component('x', &tx)));
    md.push_str(&format!("a_{{i,y}} ≈ {}\n", render_component('y', &ty)));
    md.push_str(&format!("a_{{i,z}} ≈ {}\n", render_component('z', &tz)));
    md.push_str("```\n\n");

    md.push_str("Basis definitions used by the feature library:\n\n");
    md.push_str("```text\n");
    if usage.grav {
        md.push_str("g_i^* = Σ_{j≠i} m_j * (r_j - r_i) / ||r_j - r_i||^3    (no G)\n");
    }
    if usage.elec {
        md.push_str("e_i^* = (q_i/m_i) * Σ_{j≠i} q_j * (r_i - r_j) / ||r_i - r_j||^3    (no k_e)\n");
    }
    if usage.mag {
        md.push_str("B_i^* = (1/4π) * Σ_{j≠i} q_j * (v_j × (r_i - r_j)) / ||r_i - r_j||^3    (no μ0)\n");
        md.push_str("b_i^* = (q_i/m_i) * (v_i × B_i^*)    (no μ0)\n");
    }
    md.push_str("```\n\n");

    md.push_str("Note: the basis terms omit physical constants, so learned coefficients roughly match `config.constants.g`, `k_e`, and `mu_0`.\n");
    md
}

fn best_candidate_in_iteration(iter: &FactoryEvaluationIteration) -> Option<&FactoryEvaluationCandidate> {
    let mut best: Option<&FactoryEvaluationCandidate> = None;
    let mut best_key: Option<(f64, f64, usize)> = None;
    for cand in &iter.top_candidates {
        let key = metrics_sort_key(&cand.metrics);
        if best_key.map_or(true, |bk| key < bk) {
            best_key = Some(key);
            best = Some(cand);
        }
    }
    best
}

#[derive(Clone, Debug)]
struct OracleRun {
    cfg: Config,
    result: threebody_core::sim::SimResult,
}

fn load_oracle_run_from_dir(run_dir: &std::path::Path) -> anyhow::Result<OracleRun> {
    let cfg_json = fs::read_to_string(run_dir.join("config.json"))?;
    let cfg: Config = serde_json::from_str(&cfg_json)?;
    cfg.validate().map_err(anyhow::Error::msg)?;

    let ic_json = fs::read_to_string(run_dir.join("initial_conditions.json"))?;
    let ic: InitialConditionSpec = serde_json::from_str(&ic_json)?;
    if ic.bodies.len() != 3 {
        anyhow::bail!(
            "expected 3 bodies in initial_conditions.json (found {})",
            ic.bodies.len()
        );
    }
    let mut bodies = [Body::new(0.0, 0.0); 3];
    for i in 0..3 {
        bodies[i] = Body::new(ic.bodies[i].mass, ic.bodies[i].charge);
    }

    let steps = load_steps_from_csv(&run_dir.join("traj.csv").to_path_buf(), bodies, &cfg)?;
    let sidecar_json = fs::read_to_string(run_dir.join("traj.json"))?;
    let sidecar: threebody_core::output::sidecar::Sidecar = serde_json::from_str(&sidecar_json)?;

    Ok(OracleRun {
        cfg,
        result: threebody_core::sim::SimResult {
            steps,
            encounter: None,
            encounter_action: None,
            warnings: sidecar.warnings,
            terminated_early: false,
            termination_reason: None,
            stats: sidecar.sim_stats,
        },
    })
}

fn vector_model_from_equation_text(equation_text: &str) -> Option<VectorModel> {
    let [tx, ty, tz] = parse_vector_equation_terms(equation_text);
    let to_eq = |terms: Vec<(f64, String)>| threebody_discover::Equation {
        terms: terms
            .into_iter()
            .map(|(coeff, feature)| threebody_discover::equation::Term { feature, coeff })
            .collect(),
    };
    Some(VectorModel {
        eq_x: to_eq(tx),
        eq_y: to_eq(ty),
        eq_z: to_eq(tz),
    })
}

#[derive(Clone, Debug)]
struct IncumbentEvalOnBestRun {
    equation_text: String,
    rollout_rmse: f64,
    divergence_time: Option<f64>,
    complexity: usize,
}

fn render_expanded_math_tex(equation_text: &str) -> String {
    let usage = basis_usage_from_equation_text(equation_text);
    let [tx, ty, tz] = parse_vector_equation_terms(equation_text);

    fn render_component(axis: char, terms: &[(f64, String)]) -> String {
        if terms.is_empty() {
            return "0".to_string();
        }
        let mut out = String::new();
        for (i, (coeff, feature)) in terms.iter().enumerate() {
            let sign = if *coeff >= 0.0 { "+" } else { "-" };
            let abs = coeff.abs();
            if i == 0 {
                if sign == "-" {
                    out.push_str("-");
                }
            } else {
                out.push_str(" ");
                out.push_str(sign);
                out.push_str(" ");
            }
            let symbol = feature_to_tex_symbol(feature, axis);
            out.push_str(&format!("{abs:.6}\\,{symbol}"));
        }
        out
    }

    let mut tex = String::new();
    tex.push_str("\\begin{align*}\n");
    tex.push_str(&format!(
        "a_{{i,x}} &\\approx {}\\\\\n",
        render_component('x', &tx)
    ));
    tex.push_str(&format!(
        "a_{{i,y}} &\\approx {}\\\\\n",
        render_component('y', &ty)
    ));
    tex.push_str(&format!(
        "a_{{i,z}} &\\approx {}\n",
        render_component('z', &tz)
    ));
    tex.push_str("\\end{align*}\n\n");

    tex.push_str("\\noindent Basis definitions used by the feature library:\n\n");
    tex.push_str("\\begin{align*}\n");
    if usage.grav {
        tex.push_str(
            "g_i^* &= \\sum_{j\\ne i} m_j\\, \\frac{r_j - r_i}{\\lVert r_j - r_i \\rVert^3} \\quad (\\text{no }G)\\\\\n",
        );
    }
    if usage.elec {
        tex.push_str("e_i^* &= (q_i/m_i)\\, \\sum_{j\\ne i} q_j\\, \\frac{r_i - r_j}{\\lVert r_i - r_j \\rVert^3} \\quad (\\text{no }k_e)\\\\\n");
    }
    if usage.mag {
        tex.push_str("B_i^* &= (1/4\\pi)\\, \\sum_{j\\ne i} q_j\\, \\frac{v_j \\times (r_i - r_j)}{\\lVert r_i - r_j \\rVert^3} \\quad (\\text{no }\\mu_0)\\\\\n");
        tex.push_str("b_i^* &= (q_i/m_i)\\, (v_i \\times B_i^*) \\quad (\\text{no }\\mu_0)\\\\\n");
    }
    tex.push_str("\\end{align*}\n\n");
    tex.push_str(
        "\\noindent\\textbf{Note:} basis terms omit physical constants, so learned coefficients roughly match $G$, $k_e$, and $\\mu_0$ from the config.\n\n",
    );
    tex
}

fn render_evaluation_tex(
    eval_input: &FactoryEvaluationInput,
    current_best: Option<&BestFactoryCandidate>,
    prior_best: Option<&PriorAttemptBest>,
    best_ic: Option<&InitialConditionSpec>,
) -> String {
    let mut tex = String::new();
    tex.push_str("\\documentclass[11pt]{article}\n");
    tex.push_str("\\usepackage[margin=1in]{geometry}\n");
    tex.push_str("\\usepackage{amsmath}\n");
    tex.push_str("\\usepackage{booktabs}\n");
    tex.push_str("\\usepackage{longtable}\n");
    tex.push_str("\\usepackage{hyperref}\n");
    tex.push_str("\\begin{document}\n\n");

    tex.push_str("\\section*{Executive Summary}\n");
    match current_best {
        Some(best) => {
            let m = &best.candidate.metrics;
            tex.push_str(&format!(
                "Best run: \\texttt{{{}}} (regime: \\texttt{{{}}}).\\\\\n",
                escape_latex(&best.run_id),
                escape_latex(&best.regime)
            ));
            tex.push_str(&format!(
                "Metrics: rollout\\_rmse={}, mse={:.6e}, complexity={}.\\\\\n",
                format_opt_f64(m.rollout_rmse),
                m.mse,
                m.complexity
            ));
            if let Some(prior) = prior_best {
                let pm = &prior.best.candidate.metrics;
                let curr_roll = m.rollout_rmse.unwrap_or(f64::INFINITY);
                let prior_roll = pm.rollout_rmse.unwrap_or(f64::INFINITY);
                if curr_roll.is_finite() && prior_roll.is_finite() {
                    let delta = curr_roll - prior_roll;
                    tex.push_str(&format!(
                        "Compared to previous best ({}): rollout\\_rmse change = {:+.6}.\\\\\n",
                        escape_latex(&prior.eval_input_path.display().to_string()),
                        delta
                    ));
                }
            } else {
                tex.push_str("No prior attempts found under \\texttt{results/}.\\\\\n");
            }
        }
        None => {
            tex.push_str("No candidates were recorded.\\\\\n");
        }
    }
    tex.push_str("\n");

    tex.push_str("\\section*{Run Notes}\n");
    if eval_input.notes.is_empty() {
        tex.push_str("No notes.\\\\\n\n");
    } else {
        tex.push_str("\\begin{itemize}\n");
        for note in &eval_input.notes {
            tex.push_str(&format!("  \\item \\texttt{{{}}}\n", escape_latex(note)));
        }
        tex.push_str("\\end{itemize}\n\n");
    }

    tex.push_str("\\section*{Best Equation (Exact)}\n");
    if let Some(best) = current_best {
        tex.push_str("\\begin{verbatim}\n");
        tex.push_str(&best.candidate.equation_text);
        tex.push_str("\n\\end{verbatim}\n\n");
    } else {
        tex.push_str("N/A.\n\n");
    }

    tex.push_str("\\section*{Math Formula (Expanded)}\n");
    if let Some(best) = current_best {
        tex.push_str(&render_expanded_math_tex(&best.candidate.equation_text));
    } else {
        tex.push_str("N/A.\n\n");
    }

    tex.push_str("\\section*{Initial Conditions (Best Run)}\n");
    if let Some(ic) = best_ic {
        tex.push_str(&format!(
            "Barycentric: \\texttt{{{}}}.\\\\\n",
            if ic.barycentric { "true" } else { "false" }
        ));
        tex.push_str(&format!("Notes: \\texttt{{{}}}.\\\\\n\n", escape_latex(&ic.notes)));
        tex.push_str("\\begin{longtable}{lrrrrrrrr}\n");
        tex.push_str("\\toprule\n");
        tex.push_str("Body & m & q & x & y & z & v_x & v_y & v_z\\\\\n");
        tex.push_str("\\midrule\n");
        for (i, b) in ic.bodies.iter().enumerate() {
            tex.push_str(&format!(
                "{} & {:.6} & {:.6} & {:.6} & {:.6} & {:.6} & {:.6} & {:.6} & {:.6}\\\\\n",
                i,
                b.mass,
                b.charge,
                b.pos[0],
                b.pos[1],
                b.pos[2],
                b.vel[0],
                b.vel[1],
                b.vel[2],
            ));
        }
        tex.push_str("\\bottomrule\n");
        tex.push_str("\\end{longtable}\n\n");
    } else {
        tex.push_str("N/A.\n\n");
    }

    tex.push_str("\\section*{Comparison to Prior Attempts}\n");
    if let (Some(best), Some(prior)) = (current_best, prior_best) {
        let cm = &best.candidate.metrics;
        let pm = &prior.best.candidate.metrics;
        tex.push_str("\\begin{tabular}{lrr}\n");
        tex.push_str("\\toprule\n");
        tex.push_str("Metric & Prior best & Current\\\\\n");
        tex.push_str("\\midrule\n");
        tex.push_str(&format!(
            "rollout\\_rmse & {} & {}\\\\\n",
            format_opt_f64(pm.rollout_rmse),
            format_opt_f64(cm.rollout_rmse)
        ));
        tex.push_str(&format!("mse & {:.6e} & {:.6e}\\\\\n", pm.mse, cm.mse));
        tex.push_str(&format!("complexity & {} & {}\\\\\n", pm.complexity, cm.complexity));
        tex.push_str("\\bottomrule\n");
        tex.push_str("\\end{tabular}\n\n");
        tex.push_str(&format!(
            "Prior artifact: \\texttt{{{}}}.\\\\\n",
            escape_latex(&prior.eval_input_path.display().to_string())
        ));
    } else {
        tex.push_str("No prior attempts found (or no candidates in this run).\n\n");
    }

    tex.push_str("\\end{document}\n");
    tex
}

fn render_executive_summary_md(
    eval_input: &FactoryEvaluationInput,
    current_best: Option<&BestFactoryCandidate>,
    prior_best: Option<&PriorAttemptBest>,
    best_ic: Option<&InitialConditionSpec>,
    incumbent_on_best_run: Option<&IncumbentEvalOnBestRun>,
) -> String {
    let mut md = String::new();
    md.push_str("# Executive Summary\n\n");
    md.push_str(&format!("- Iterations: {}\n", eval_input.iterations.len()));

    match current_best {
        Some(best) => {
            let m = &best.candidate.metrics;
            md.push_str(&format!("- Best run: `{}` (regime: `{}`)\n", best.run_id, best.regime));
            md.push_str(&format!(
                "- Best metrics: rollout_rmse={}, mse={:.6e}, complexity={}, divergence_time={}\n",
                format_opt_f64(m.rollout_rmse),
                m.mse,
                m.complexity,
                format_opt_f64(m.divergence_time)
            ));
            md.push_str("- Best equation (exact):\n");
            md.push_str("```text\n");
            md.push_str(&best.candidate.equation_text);
            md.push_str("\n```\n");
            md.push_str(&render_expanded_math_md(&best.candidate.equation_text));

            if let Some(iter) = eval_input.iterations.iter().find(|it| it.run_id == best.run_id) {
                if let Some(sim) = iter.simulation.as_ref() {
                    md.push_str("\n## Best-Run Physics Diagnostics\n");
                    md.push_str(&format!("- steps={}\n", sim.steps));
                    md.push_str(&format!("- min_pair_dist={}\n", format_opt_f64(sim.min_pair_dist)));
                    md.push_str(&format!("- energy_drift={}\n", format_opt_f64(sim.energy_drift)));
                    md.push_str(&format!("- rollout_integrator={}\n", sim.rollout_integrator));
                    if !sim.warnings.is_empty() {
                        md.push_str(&format!("- warnings: {}\n", sim.warnings.join("; ")));
                    }
                }
            }

            if let Some(ic) = best_ic {
                md.push_str("\n## Initial Conditions (Best Run)\n");
                md.push_str(&format!("- barycentric={}\n", ic.barycentric));
                if !ic.notes.trim().is_empty() {
                    md.push_str(&format!("- notes: {}\n", ic.notes.trim()));
                }
                md.push_str("- bodies:\n");
                for (i, b) in ic.bodies.iter().enumerate() {
                    md.push_str(&format!(
                        "  - body {i}: m={:.6}, q={:.6}, r=({:.6},{:.6},{:.6}), v=({:.6},{:.6},{:.6})\n",
                        b.mass, b.charge, b.pos[0], b.pos[1], b.pos[2], b.vel[0], b.vel[1], b.vel[2]
                    ));
                }
            }
        }
        None => {
            md.push_str("- No candidate equations were recorded.\n");
        }
    }

    md.push_str("\n## What the Metrics Mean (No Math Required)\n");
    md.push_str("- `rollout_rmse` (lower is better): how far the learned model’s trajectory drifts from the oracle.\n");
    md.push_str("- `divergence_time` (higher is better): how long the rollout stays “close enough” before blowing up.\n");
    md.push_str("- `mse` (lower is better): pointwise acceleration-fit error (useful, but not sufficient).\n");
    md.push_str("- `complexity` (lower is simpler): number of non-zero terms across x/y/z.\n");

    md.push_str("\n## Gains vs Previous Attempts\n");
    if let (Some(best), Some(prior)) = (current_best, prior_best) {
        let cm = &best.candidate.metrics;
        let pm = &prior.best.candidate.metrics;
        let curr_roll = cm.rollout_rmse.unwrap_or(f64::INFINITY);
        let prior_roll = pm.rollout_rmse.unwrap_or(f64::INFINITY);
        if curr_roll.is_finite() && prior_roll.is_finite() {
            let delta = curr_roll - prior_roll;
            let eps = 1e-9;
            let verdict = if delta < -eps {
                "improved"
            } else if delta > eps {
                "worse"
            } else {
                "no change"
            };
            let delta_disp = if delta.abs() <= eps { 0.0 } else { delta };
            md.push_str(&format!(
                "- Compared to prior best: {verdict} (Δ rollout_rmse = {delta_disp:+.6}).\n"
            ));
            md.push_str(&format!(
                "- Prior best artifact: `{}`\n",
                prior.eval_input_path.display()
            ));
            md.push_str("- Note: this comparison is approximate because each attempt may use different initial conditions.\n");
        } else {
            md.push_str("- Prior/current rollout_rmse missing; cannot compute gains.\n");
        }
    } else {
        md.push_str("- No prior attempts found under `results/` (or no candidates in this run).\n");
    }

    if let (Some(best), Some(inc)) = (current_best, incumbent_on_best_run) {
        let curr_roll = best.candidate.metrics.rollout_rmse.unwrap_or(f64::INFINITY);
        if curr_roll.is_finite() && inc.rollout_rmse.is_finite() {
            let delta = curr_roll - inc.rollout_rmse;
            let eps = 1e-9;
            let verdict = if delta < -eps {
                "better"
            } else if delta > eps {
                "worse"
            } else {
                "equal"
            };
            let delta_disp = if delta.abs() <= eps { 0.0 } else { delta };
            md.push_str("\n## Evidence: Prior Best Re-Scored on This Run\n");
            md.push_str("- This is an apples-to-apples check on the *same* oracle trajectory.\n");
            md.push_str(&format!("- Prior best complexity: {}.\n", inc.complexity));
            md.push_str("- Prior best equation (exact):\n");
            md.push_str("```text\n");
            md.push_str(&inc.equation_text);
            md.push_str("\n```\n");
            md.push_str(&format!(
                "- Prior best equation rollout_rmse on best-run oracle: {:.6} (divergence_time={}).\n",
                inc.rollout_rmse,
                format_opt_f64(inc.divergence_time)
            ));
            md.push_str(&format!(
                "- Current best rollout_rmse on best-run oracle: {}.\n",
                format_opt_f64(best.candidate.metrics.rollout_rmse)
            ));
            md.push_str(&format!(
                "- Verdict on this run: {verdict} (Δ rollout_rmse = {delta_disp:+.6}).\n"
            ));
        }
    }

    md.push_str("\n## Evidence: Per-Iteration Best Candidates\n");
    md.push_str("Selection rule: pick the lowest `rollout_rmse`, break ties by lower `mse`, then lower `complexity`.\n\n");
    md.push_str("| run | rollout_integrator | rollout_rmse | mse | complexity | equation |\n");
    md.push_str("|---|---|---:|---:|---:|---|\n");
    for iter in &eval_input.iterations {
        let best_iter = match best_candidate_in_iteration(iter) {
            Some(v) => v,
            None => continue,
        };
        let sim_int = iter
            .simulation
            .as_ref()
            .map(|s| s.rollout_integrator.as_str())
            .unwrap_or("n/a");
        md.push_str(&format!(
            "| `{}` | `{}` | {} | {:.6e} | {} | `{}` |\n",
            iter.run_id,
            sim_int,
            format_opt_f64(best_iter.metrics.rollout_rmse),
            best_iter.metrics.mse,
            best_iter.metrics.complexity,
            best_iter.equation_text.replace('`', "'")
        ));
    }

    md.push_str("\n## Where to Look Next\n");
    md.push_str("- `evaluation.tex` (and `evaluation.pdf` if built)\n");
    md.push_str("- `run_###/report.md` (per-iteration summary)\n");
    md.push_str("- `run_###/discovery.json` (solver metadata + candidates)\n");
    md.push_str("- `evaluation_llm.md` (optional narrative; not used for numeric selection)\n");
    md.push_str("\n");
    md
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
    fn stability_flags_do_not_false_positive_on_gravity_terms() {
        let eq = Equation {
            terms: vec![threebody_discover::equation::Term {
                feature: "grav_x".to_string(),
                coeff: 1.0,
            }],
        };
        let flags = stability_flags_for(&eq, "gravity_only");
        assert!(
            flags.is_empty(),
            "expected no flags for grav_x in gravity_only, got: {flags:?}"
        );
    }

    #[test]
    fn stability_flags_flag_mag_terms_in_gravity_only() {
        let eq = Equation {
            terms: vec![threebody_discover::equation::Term {
                feature: "mag_x".to_string(),
                coeff: 1.0,
            }],
        };
        let flags = stability_flags_for(&eq, "gravity_only");
        assert!(flags.contains(&"em_terms_in_gravity_only".to_string()));
        assert!(flags.contains(&"velocity_terms_in_gravity_only".to_string()));
    }

    #[test]
    fn best_candidate_prefers_lower_rollout_rmse() {
        let input = FactoryEvaluationInput {
            version: FACTORY_EVALUATION_VERSION.to_string(),
            notes: vec![],
            iterations: vec![FactoryEvaluationIteration {
                iteration: 1,
                run_id: "run_001".to_string(),
                regime: "gravity_only".to_string(),
                solver: DiscoverySolverSummary {
                    name: "stls".to_string(),
                    normalize: true,
                    fitness_heuristic: "mse".to_string(),
                    stls: None,
                    lasso: None,
                    ga: None,
                },
                simulation: None,
                top_candidates: vec![
                    FactoryEvaluationCandidate {
                        id: 0,
                        equation_text: "a = bad".to_string(),
                        metrics: CandidateMetrics {
                            mse: 0.0,
                            complexity: 1,
                            rollout_rmse: Some(1.0),
                            divergence_time: None,
                            stability_flags: vec![],
                        },
                    },
                    FactoryEvaluationCandidate {
                        id: 1,
                        equation_text: "a = good".to_string(),
                        metrics: CandidateMetrics {
                            mse: 100.0,
                            complexity: 10,
                            rollout_rmse: Some(0.1),
                            divergence_time: None,
                            stability_flags: vec![],
                        },
                    },
                ],
                judge: None,
            }],
        };

        let best = best_candidate_from_eval_input(&input).unwrap();
        assert_eq!(best.run_id, "run_001");
        assert_eq!(best.candidate.equation_text, "a = good");
    }

    #[test]
    fn find_best_prior_attempt_scans_results_tree() {
        let root = unique_temp_path("results_root", "dir");
        if root.exists() {
            let _ = fs::remove_dir_all(&root);
        }
        fs::create_dir_all(&root).unwrap();

        let run1_dir = root.join("quickstart_a").join("factory");
        let run2_dir = root.join("quickstart_b").join("factory");
        fs::create_dir_all(&run1_dir).unwrap();
        fs::create_dir_all(&run2_dir).unwrap();

        let mk_input = |rollout_rmse: f64| FactoryEvaluationInput {
            version: FACTORY_EVALUATION_VERSION.to_string(),
            notes: vec![],
            iterations: vec![FactoryEvaluationIteration {
                iteration: 1,
                run_id: "run_001".to_string(),
                regime: "gravity_only".to_string(),
                solver: DiscoverySolverSummary {
                    name: "stls".to_string(),
                    normalize: true,
                    fitness_heuristic: "mse".to_string(),
                    stls: None,
                    lasso: None,
                    ga: None,
                },
                simulation: None,
                top_candidates: vec![FactoryEvaluationCandidate {
                    id: 0,
                    equation_text: "a = 0".to_string(),
                    metrics: CandidateMetrics {
                        mse: 1.0,
                        complexity: 1,
                        rollout_rmse: Some(rollout_rmse),
                        divergence_time: None,
                        stability_flags: vec![],
                    },
                }],
                judge: None,
            }],
        };

        fs::write(
            run1_dir.join("evaluation_input.json"),
            serde_json::to_string_pretty(&mk_input(0.9)).unwrap(),
        )
        .unwrap();
        fs::write(
            run2_dir.join("evaluation_input.json"),
            serde_json::to_string_pretty(&mk_input(0.1)).unwrap(),
        )
        .unwrap();

        let exclude = root.join("current").join("factory").join("evaluation_input.json");
        let best = find_best_prior_attempt(&root, &exclude, &PriorAttemptFilter::default())
            .unwrap()
            .unwrap();
        assert!(best.eval_input_path.starts_with(&run2_dir));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn latex_escape_escapes_special_chars() {
        let s = r"results/run_001/a_b%";
        let escaped = escape_latex(s);
        assert!(escaped.contains(r"\_"));
        assert!(escaped.contains(r"\%"));
    }

    #[test]
    fn render_evaluation_tex_includes_required_sections() {
        let eval_input = FactoryEvaluationInput {
            version: FACTORY_EVALUATION_VERSION.to_string(),
            notes: vec!["steps=5".to_string()],
            iterations: vec![FactoryEvaluationIteration {
                iteration: 1,
                run_id: "run_001".to_string(),
                regime: "gravity_only".to_string(),
                solver: DiscoverySolverSummary {
                    name: "stls".to_string(),
                    normalize: true,
                    fitness_heuristic: "mse".to_string(),
                    stls: None,
                    lasso: None,
                    ga: None,
                },
                simulation: None,
                top_candidates: vec![FactoryEvaluationCandidate {
                    id: 0,
                    equation_text: "ax=+1*grav_x".to_string(),
                    metrics: CandidateMetrics {
                        mse: 1.0,
                        complexity: 1,
                        rollout_rmse: Some(0.1),
                        divergence_time: None,
                        stability_flags: vec![],
                    },
                }],
                judge: None,
            }],
        };
        let best = best_candidate_from_eval_input(&eval_input).unwrap();
        let ic = InitialConditionSpec {
            bodies: vec![
                threebody_discover::BodyInit {
                    mass: 1.0,
                    charge: 0.0,
                    pos: [0.0, 0.0, 0.0],
                    vel: [0.0, 0.0, 0.0],
                },
                threebody_discover::BodyInit {
                    mass: 1.0,
                    charge: 0.0,
                    pos: [1.0, 0.0, 0.0],
                    vel: [0.0, 0.0, 0.0],
                },
                threebody_discover::BodyInit {
                    mass: 1.0,
                    charge: 0.0,
                    pos: [0.0, 1.0, 0.0],
                    vel: [0.0, 0.0, 0.0],
                },
            ],
            barycentric: true,
            notes: "test".to_string(),
        };
        let tex = render_evaluation_tex(&eval_input, Some(&best), None, Some(&ic));
        for needle in [
            "\\section*{Executive Summary}",
            "\\section*{Best Equation (Exact)}",
            "\\section*{Initial Conditions (Best Run)}",
            "\\section*{Comparison to Prior Attempts}",
        ] {
            assert!(tex.contains(needle), "missing {needle}");
        }
        assert!(tex.contains("ax=+1*grav_x"));
    }

    fn mk_eval_input_with_notes(
        steps: usize,
        dt: f64,
        rollout_rmse: f64,
        mse: f64,
        complexity: usize,
        equation_text: &str,
    ) -> FactoryEvaluationInput {
        FactoryEvaluationInput {
            version: FACTORY_EVALUATION_VERSION.to_string(),
            notes: vec![format!("steps={steps}"), format!("dt={dt}")],
            iterations: vec![FactoryEvaluationIteration {
                iteration: 1,
                run_id: "run_001".to_string(),
                regime: "gravity_only".to_string(),
                solver: DiscoverySolverSummary {
                    name: "stls".to_string(),
                    normalize: true,
                    fitness_heuristic: "mse".to_string(),
                    stls: None,
                    lasso: None,
                    ga: None,
                },
                simulation: None,
                top_candidates: vec![FactoryEvaluationCandidate {
                    id: 0,
                    equation_text: equation_text.to_string(),
                    metrics: CandidateMetrics {
                        mse,
                        complexity,
                        rollout_rmse: Some(rollout_rmse),
                        divergence_time: None,
                        stability_flags: vec![],
                    },
                }],
                judge: None,
            }],
        }
    }

    #[test]
    fn scan_best_results_ignores_non_factory_eval_inputs() {
        let root = unique_temp_path("best_results_root", "dir");
        if root.exists() {
            let _ = fs::remove_dir_all(&root);
        }
        fs::create_dir_all(&root).unwrap();

        let non_factory = root.join("quickstart_nonfactory");
        fs::create_dir_all(&non_factory).unwrap();
        fs::write(
            non_factory.join("evaluation_input.json"),
            serde_json::to_string_pretty(&mk_eval_input_with_notes(200, 0.01, 0.9, 1.0, 1, "a=bad")).unwrap(),
        )
        .unwrap();

        let factory_dir = root.join("quickstart_factory").join("factory");
        fs::create_dir_all(&factory_dir).unwrap();
        fs::write(
            factory_dir.join("evaluation_input.json"),
            serde_json::to_string_pretty(&mk_eval_input_with_notes(200, 0.01, 0.1, 1.0, 1, "a=good")).unwrap(),
        )
        .unwrap();

        let index = scan_best_results(&root).unwrap();
        assert_eq!(index.buckets.len(), 1);
        assert!(index.buckets[0].eval_input_path.contains("quickstart_factory"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn scan_best_results_selects_best_per_bucket() {
        let root = unique_temp_path("best_results_choose", "dir");
        if root.exists() {
            let _ = fs::remove_dir_all(&root);
        }
        fs::create_dir_all(&root).unwrap();

        let run1_factory = root.join("run1").join("factory");
        let run2_factory = root.join("run2").join("factory");
        fs::create_dir_all(&run1_factory).unwrap();
        fs::create_dir_all(&run2_factory).unwrap();

        fs::write(
            run1_factory.join("evaluation_input.json"),
            serde_json::to_string_pretty(&mk_eval_input_with_notes(100, 0.01, 0.9, 1.0, 1, "a=run1")).unwrap(),
        )
        .unwrap();
        fs::write(
            run2_factory.join("evaluation_input.json"),
            serde_json::to_string_pretty(&mk_eval_input_with_notes(100, 0.01, 0.1, 2.0, 2, "a=run2")).unwrap(),
        )
        .unwrap();

        let index = scan_best_results(&root).unwrap();
        assert_eq!(index.buckets.len(), 1);
        assert!(index.buckets[0].eval_input_path.contains("run2"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn render_best_results_md_includes_equation_and_metrics() {
        let index = BestResultsIndexV1 {
            version: "v1".to_string(),
            updated_at_utc: "unix_seconds=0".to_string(),
            buckets: vec![BestRecord {
                bucket: BucketKey { steps: 200, dt: 0.01 },
                run_dir: "results/quickstart_x".to_string(),
                factory_dir: "results/quickstart_x/factory".to_string(),
                eval_input_path: "results/quickstart_x/factory/evaluation_input.json".to_string(),
                run_id: "run_001".to_string(),
                regime: "gravity_only".to_string(),
                equation_text: "ax=+1*grav_x".to_string(),
                metrics: CandidateMetrics {
                    mse: 1.0,
                    complexity: 1,
                    rollout_rmse: Some(0.1),
                    divergence_time: None,
                    stability_flags: vec![],
                },
                ic: None,
            }],
            notes: vec![],
        };
        let md = render_best_results_md(&index);
        assert!(md.contains("Best Results"));
        assert!(md.contains("ax=+1*grav_x"));
        assert!(md.contains("rollout_rmse"));
    }

    #[test]
    fn render_findings_tex_includes_required_sections_and_equation() {
        let index = BestResultsIndexV1 {
            version: "v1".to_string(),
            updated_at_utc: "unix_seconds=0".to_string(),
            buckets: vec![BestRecord {
                bucket: BucketKey { steps: 200, dt: 0.01 },
                run_dir: "results/quickstart_x".to_string(),
                factory_dir: "results/quickstart_x/factory".to_string(),
                eval_input_path: "results/quickstart_x/factory/evaluation_input.json".to_string(),
                run_id: "run_001".to_string(),
                regime: "gravity_only".to_string(),
                equation_text: "ax=+1.000000*grav_x ; ay=0 ; az=0".to_string(),
                metrics: CandidateMetrics {
                    mse: 1.0,
                    complexity: 1,
                    rollout_rmse: Some(0.1),
                    divergence_time: None,
                    stability_flags: vec![],
                },
                ic: None,
            }],
            notes: vec![],
        };
        let tex = render_findings_tex(&index, &[]);
        assert!(tex.contains("\\section*{Executive Summary}"));
        assert!(tex.contains("\\section*{Best Results by Bucket}"));
        assert!(tex.contains("ax=+1.000000*grav_x"));
    }

    #[test]
    fn write_best_results_writes_md_and_json() {
        let root = unique_temp_path("best_results_write", "dir");
        if root.exists() {
            let _ = fs::remove_dir_all(&root);
        }
        fs::create_dir_all(&root).unwrap();

        let index = BestResultsIndexV1 {
            version: "v1".to_string(),
            updated_at_utc: "unix_seconds=0".to_string(),
            buckets: vec![],
            notes: vec![],
        };
        write_best_results(&root, &index).unwrap();
        assert!(root.join("best_results.md").exists());
        assert!(root.join("best_results.json").exists());

        let _ = fs::remove_dir_all(&root);
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
