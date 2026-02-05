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

mod eval;
mod predictability;

use eval::{format_vector_model, rollout_metrics, rollout_trace, VectorModel};

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
        /// Feature library: default | extended | em_fields.
        #[arg(long, default_value = "default")]
        feature_library: String,
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
    /// Run a deterministic EM identifiability benchmark suite (sweep + multiple IC templates).
    BenchEm {
        /// Results directory root (stores global best index and is scanned for history).
        #[arg(long, default_value = "results")]
        results_dir: PathBuf,
        /// Output directory root (defaults to `results/bench_em_<timestamp>/`).
        #[arg(long)]
        out_dir: Option<PathBuf>,
        /// Steps to simulate per case.
        #[arg(long, default_value_t = 200)]
        steps: usize,
        /// Timestep (initial dt; RK45 may adapt in truth mode).
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
        /// Max cases to run (truncates the sweep grid; increase for more coverage).
        #[arg(long, default_value_t = 24)]
        max_cases: usize,
        /// Random seed (controls deterministic IC perturbations).
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Held-out IC rollouts for the selected best EM-heavy equation.
        #[arg(long, default_value_t = 2)]
        heldout: usize,
        /// Do not attempt to build PDFs via pdflatex.
        #[arg(long)]
        no_pdf: bool,
    },
    /// Predictability-aware tools (ensemble runs, encounter extraction, and lock detection).
    Predictability {
        #[command(subcommand)]
        command: predictability::PredictabilityCommand,
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
        /// Feature library: auto | default | extended | em_fields.
        #[arg(long, default_value = "auto")]
        feature_library: String,
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
    /// Verify OpenAI-compatible LLM connectivity and JSON validity.
    LlmCheck {
        /// LLM model name.
        #[arg(long, default_value = "gpt-5.2")]
        model: String,
        /// OpenAI API key file (overrides OPENAI_API_KEY).
        #[arg(long)]
        openai_key_file: Option<PathBuf>,
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
            feature_library,
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
                feature_library,
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
        Commands::BenchEm {
            results_dir,
            out_dir,
            steps,
            dt,
            max_cases,
            seed,
            heldout,
            no_pdf,
        } => {
            run_bench_em(
                results_dir,
                out_dir,
                steps,
                dt,
                max_cases,
                seed,
                heldout,
                !no_pdf,
            )?;
        }
        Commands::Predictability { command } => {
            predictability::run(command)?;
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
            feature_library,
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
                feature_library,
                llm_mode,
                model,
                openai_key_file,
                require_llm,
            )?;
        }
        Commands::LlmCheck { model, openai_key_file } => {
            run_llm_check(model, openai_key_file)?;
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

fn run_llm_check(model: String, openai_key_file: Option<PathBuf>) -> anyhow::Result<()> {
    use std::path::Path;

    let client = OpenAIClient::from_env_or_file(&model, openai_key_file.as_deref())
        .map_err(|e| anyhow::anyhow!("failed to initialize OpenAI client: {e}"))?;
    let api_style = std::env::var("THREEBODY_OPENAI_API_STYLE")
        .or_else(|_| std::env::var("OPENAI_API_STYLE"))
        .unwrap_or_else(|_| "auto".to_string());

    let api_key_source = if let Some(p) = openai_key_file.as_deref() {
        format!("file:{}", p.display())
    } else if Path::new(".openai_key").exists() {
        "file:.openai_key".to_string()
    } else if std::env::var("OPENAI_API_KEY").is_ok() {
        "env:OPENAI_API_KEY".to_string()
    } else {
        "missing".to_string()
    };
    let base_url_env = std::env::var("OPENAI_BASE_URL")
        .or_else(|_| std::env::var("THREEBODY_OPENAI_BASE_URL"))
        .unwrap_or_else(|_| "(default)".to_string());

    println!("LLM check:");
    println!("- base_url={}", client.base_url);
    println!("- model={}", client.model);
    println!("- api_style={}", api_style);
    println!("- api_key_source={}", api_key_source);
    println!("- base_url_env={}", base_url_env);

    let bounds = default_ic_bounds();
    let preflight = IcRequest {
        bounds: bounds.clone(),
        regime: "gravity_only".to_string(),
        notes: vec!["llm_check=true".to_string()],
        seed: Some(1),
    };
    let ic = client
        .propose_initial_conditions(&preflight)
        .map_err(|e| anyhow::anyhow!("propose_initial_conditions failed: {e}"))?;
    system_from_ic(&ic.value, &bounds).map_err(|e| anyhow::anyhow!("IC output invalid: {e}"))?;
    println!("- propose_initial_conditions: ok");

    let judge_input = JudgeInput {
        rubric: Rubric::default_rubric(),
        regime: "gravity_only".to_string(),
        dataset: DatasetSummary {
            n_samples: 1,
            target_description: "a_x".to_string(),
            feature_names: vec!["grav_x".to_string()],
            feature_descriptions: vec![FeatureDescription {
                name: "grav_x".to_string(),
                description: "gravity basis x-component".to_string(),
                tags: vec!["gravity".to_string()],
            }],
        },
        simulation: None,
        candidates: vec![CandidateSummary {
            id: 0,
            equation: threebody_discover::Equation { terms: vec![] },
            equation_text: "0".to_string(),
            metrics: CandidateMetrics {
                mse: 1.0,
                complexity: 0,
                rollout_rmse: None,
                divergence_time: None,
                stability_flags: vec![],
            },
            notes: vec![],
        }],
        ic_bounds: bounds.clone(),
        notes: vec!["llm_check=true".to_string()],
    };
    let judge = client
        .judge_candidates(&judge_input)
        .map_err(|e| anyhow::anyhow!("judge_candidates failed: {e}"))?;
    let mut judge_resp = judge.value;
    sanitize_judge_recommendations(&mut judge_resp.recommendations, &bounds);
    judge_resp
        .validate(&judge_input)
        .map_err(|e| anyhow::anyhow!("judge response invalid: {e}"))?;
    println!("- judge_candidates: ok");

    let eval_input = FactoryEvaluationInput {
        version: FACTORY_EVALUATION_VERSION.to_string(),
        notes: vec!["llm_check=true".to_string()],
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
                equation_text: "ax=+1*grav_x ; ay=+1*grav_y ; az=0".to_string(),
                metrics: CandidateMetrics {
                    mse: 1.0,
                    complexity: 2,
                    rollout_rmse: Some(0.1),
                    divergence_time: Some(1.0),
                    stability_flags: vec![],
                },
            }],
            judge: None,
        }],
    };
    let explainer = client
        .explain_factory_evaluation(&eval_input)
        .map_err(|e| anyhow::anyhow!("explain_factory_evaluation failed: {e}"))?;
    if explainer.value.trim().is_empty() {
        anyhow::bail!("explain_factory_evaluation returned empty response");
    }
    println!("- explain_factory_evaluation: ok");

    println!("LLM check OK.");
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

    let llm_model = std::env::var("THREEBODY_LLM_MODEL")
        .or_else(|_| std::env::var("OPENAI_MODEL"))
        .unwrap_or_else(|_| "gpt-5.2".to_string());

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
        "auto".to_string(),
        llm_model,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchEmTemplate {
    HeadOn,
    HighAngMomentum,
    NearCircular,
    NearCollisionSafe,
}

impl BenchEmTemplate {
    fn as_str(&self) -> &'static str {
        match self {
            BenchEmTemplate::HeadOn => "head_on",
            BenchEmTemplate::HighAngMomentum => "high_ang_momentum",
            BenchEmTemplate::NearCircular => "near_circular",
            BenchEmTemplate::NearCollisionSafe => "near_collision_safe",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchChargePattern {
    Like,
    AlternatingSigns,
}

impl BenchChargePattern {
    fn as_str(&self) -> &'static str {
        match self {
            BenchChargePattern::Like => "like",
            BenchChargePattern::AlternatingSigns => "alternating_signs",
        }
    }
}

#[derive(Clone, Debug)]
struct BenchEmCase {
    template: BenchEmTemplate,
    charge_pattern: BenchChargePattern,
    charge_scale: f64,
    velocity_scale: f64,
}

fn run_bench_em(
    results_dir: PathBuf,
    out_dir: Option<PathBuf>,
    steps: usize,
    dt: f64,
    max_cases: usize,
    seed: u64,
    heldout: usize,
    build_pdf: bool,
) -> anyhow::Result<()> {
    use std::time::{SystemTime, UNIX_EPOCH};

    fs::create_dir_all(&results_dir)?;
    let out_dir = out_dir.unwrap_or_else(|| {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        results_dir.join(format!("bench_em_{ts}"))
    });
    fs::create_dir_all(&out_dir)?;

    let factory_dir = out_dir.join("factory");
    fs::create_dir_all(&factory_dir)?;

    let charge_scales = [0.0, 1.0, 10.0, 100.0];
    let velocity_scales = [0.5, 1.0, 2.0, 4.0];
    let templates = [
        BenchEmTemplate::HeadOn,
        BenchEmTemplate::HighAngMomentum,
        BenchEmTemplate::NearCircular,
        BenchEmTemplate::NearCollisionSafe,
    ];
    let patterns = [BenchChargePattern::Like, BenchChargePattern::AlternatingSigns];

    let mut cases = Vec::new();
    for template in templates {
        for pattern in patterns {
            for &cs in &charge_scales {
                for &vs in &velocity_scales {
                    cases.push(BenchEmCase {
                        template,
                        charge_pattern: pattern,
                        charge_scale: cs,
                        velocity_scale: vs,
                    });
                }
            }
        }
    }

    // Deterministically shuffle so `--max-cases` samples across the grid.
    {
        let mut rng = threebody_discover::ga::Lcg::new(seed);
        for i in (1..cases.len()).rev() {
            let j = rng.gen_range_usize(0, i);
            cases.swap(i, j);
        }
    }
    if max_cases > 0 && cases.len() > max_cases {
        cases.truncate(max_cases);
    }

    let mut oracle_cfg = Config::default();
    oracle_cfg.enable_gravity = true;
    oracle_cfg.enable_em = true;
    oracle_cfg.integrator.kind = IntegratorKind::Rk45;
    oracle_cfg.integrator.adaptive = true;
    oracle_cfg.integrator.rtol = 1e-12;
    oracle_cfg.integrator.atol = 1e-14;
    oracle_cfg.integrator.dt_min = oracle_cfg.integrator.dt_min.min(1e-6);
    oracle_cfg.integrator.dt_max = oracle_cfg.integrator.dt_max.max(0.05);
    oracle_cfg.validate().map_err(anyhow::Error::msg)?;

    let library = FeatureLibrary::extended_physics();
    let rollout_integrator = RolloutIntegrator::Euler;

    let stls_settings = DiscoverySolverSettings {
        solver: DiscoverySolver::Stls,
        normalize: true,
        stls_thresholds: Vec::new(),
        stls_ridge_lambda: 1e-8,
        stls_max_iter: 25,
        lasso_alphas: Vec::new(),
        lasso_max_iter: 2000,
        lasso_tol: 1e-6,
    };
    let lasso_settings = DiscoverySolverSettings {
        solver: DiscoverySolver::Lasso,
        normalize: true,
        stls_thresholds: Vec::new(),
        stls_ridge_lambda: 1e-8,
        stls_max_iter: 25,
        lasso_alphas: Vec::new(),
        lasso_max_iter: 2000,
        lasso_tol: 1e-6,
    };

    #[derive(Clone, Debug, serde::Serialize)]
    struct CaseResult {
        run_id: String,
        template: String,
        charge_pattern: String,
        charge_scale: f64,
        velocity_scale: f64,
        sim: SimulationSummary,
        best_equation_text: String,
        best_metrics: CandidateMetrics,
        uses_em_terms: bool,
        selected_solver: String,
    }

    let mut suite_cases: Vec<CaseResult> = Vec::new();
    let mut eval_iterations: Vec<FactoryEvaluationIteration> = Vec::new();

    fn pick_best<'a>(cands: &'a [CandidateSummary]) -> Option<&'a CandidateSummary> {
        cands.iter().min_by(|a, b| {
            metrics_sort_key(&a.metrics)
                .partial_cmp(&metrics_sort_key(&b.metrics))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    for (idx, case) in cases.iter().enumerate() {
        let run_id = format!("run_{:03}", idx + 1);
        let run_dir = factory_dir.join(&run_id);
        fs::create_dir_all(&run_dir)?;

        let (system, ic_spec) = bench_em_initial_conditions(case, seed + idx as u64)?;
        let result = simulate_with_cfg(system, &oracle_cfg, SimOptions { steps, dt });

        fs::write(run_dir.join("config.json"), serde_json::to_string_pretty(&oracle_cfg)?)?;
        fs::write(
            run_dir.join("initial_conditions.json"),
            serde_json::to_string_pretty(&ic_spec)?,
        )?;

        let traj_path = run_dir.join("traj.csv");
        let mut csv_file = fs::File::create(&traj_path)?;
        write_csv(&mut csv_file, &result.steps, &oracle_cfg)?;
        let header = threebody_core::output::csv::csv_header(&oracle_cfg);
        let sidecar = build_sidecar(&oracle_cfg, &header, &result, Some(steps), Some(dt));
        let mut sidecar_file = fs::File::create(run_dir.join("traj.json"))?;
        write_sidecar(&mut sidecar_file, &sidecar)?;

        let vector_data = build_vector_dataset(&result, &oracle_cfg, &library.features, None);
        let [dataset_x, dataset_y, dataset_z] = component_datasets(&vector_data);

        let stls_cfg = StlsConfig {
            thresholds: stls_settings.stls_thresholds.clone(),
            ridge_lambda: stls_settings.stls_ridge_lambda,
            max_iter: stls_settings.stls_max_iter,
            normalize: stls_settings.normalize,
        };
        let lasso_cfg = LassoConfig {
            alphas: lasso_settings.lasso_alphas.clone(),
            max_iter: lasso_settings.lasso_max_iter,
            tol: lasso_settings.lasso_tol,
            normalize: lasso_settings.normalize,
        };

        let (stls_topk_x, stls_topk_y, stls_topk_z) = (
            stls_path_search(&dataset_x, &stls_cfg, FitnessHeuristic::Mse),
            stls_path_search(&dataset_y, &stls_cfg, FitnessHeuristic::Mse),
            stls_path_search(&dataset_z, &stls_cfg, FitnessHeuristic::Mse),
        );
        let (lasso_topk_x, lasso_topk_y, lasso_topk_z) = (
            lasso_path_search(&dataset_x, &lasso_cfg, FitnessHeuristic::Mse),
            lasso_path_search(&dataset_y, &lasso_cfg, FitnessHeuristic::Mse),
            lasso_path_search(&dataset_z, &lasso_cfg, FitnessHeuristic::Mse),
        );

        let stls_candidates = build_vector_candidates(
            &stls_topk_x.entries,
            &stls_topk_y.entries,
            &stls_topk_z.entries,
            &vector_data.feature_names,
            &result,
            &oracle_cfg,
            "em_quasistatic",
            rollout_integrator,
        );
        let lasso_candidates = build_vector_candidates(
            &lasso_topk_x.entries,
            &lasso_topk_y.entries,
            &lasso_topk_z.entries,
            &vector_data.feature_names,
            &result,
            &oracle_cfg,
            "em_quasistatic",
            rollout_integrator,
        );

        let stls_best = pick_best(&stls_candidates);
        let lasso_best = pick_best(&lasso_candidates);
        let selected_solver = match (stls_best, lasso_best) {
            (Some(a), Some(b)) => {
                if metrics_sort_key(&a.metrics) <= metrics_sort_key(&b.metrics) {
                    "stls"
                } else {
                    "lasso"
                }
            }
            (Some(_), None) => "stls",
            (None, Some(_)) => "lasso",
            (None, None) => "stls",
        };
        let selected_candidates = if selected_solver == "stls" {
            stls_candidates.clone()
        } else {
            lasso_candidates.clone()
        };

        let sim_summary = build_sim_summary(
            &result,
            &oracle_cfg,
            rollout_integrator,
            Some(steps),
            Some(dt),
        );

        let mut best_equation_text = String::new();
        let mut best_metrics = CandidateMetrics {
            mse: f64::INFINITY,
            complexity: usize::MAX,
            rollout_rmse: None,
            divergence_time: None,
            stability_flags: vec![],
        };
        if let Some(best) = pick_best(&selected_candidates) {
            best_equation_text = best.equation_text.clone();
            best_metrics = best.metrics.clone();
        }

        let usage = basis_usage_from_equation_text(&best_equation_text);
        let uses_em_terms = usage.elec || usage.mag;

        #[derive(serde::Serialize)]
        struct BenchReport<'a> {
            template: &'a str,
            charge_pattern: &'a str,
            charge_scale: f64,
            velocity_scale: f64,
            selected_solver: &'a str,
            simulation: &'a SimulationSummary,
            best_equation: &'a str,
            best_metrics: &'a CandidateMetrics,
            uses_em_terms: bool,
        }

        let report_json = serde_json::to_string_pretty(&BenchReport {
            template: case.template.as_str(),
            charge_pattern: case.charge_pattern.as_str(),
            charge_scale: case.charge_scale,
            velocity_scale: case.velocity_scale,
            selected_solver,
            simulation: &sim_summary,
            best_equation: &best_equation_text,
            best_metrics: &best_metrics,
            uses_em_terms,
        })?;
        fs::write(run_dir.join("report.json"), report_json)?;

        let mut md = String::new();
        md.push_str(&format!("# Bench-EM Case {}\n\n", run_id));
        md.push_str(&format!("- template: `{}`\n", case.template.as_str()));
        md.push_str(&format!("- charge_pattern: `{}`\n", case.charge_pattern.as_str()));
        md.push_str(&format!("- charge_scale: {}\n", case.charge_scale));
        md.push_str(&format!("- velocity_scale: {}\n", case.velocity_scale));
        md.push_str(&format!("- selected_solver: `{}`\n", selected_solver));
        md.push_str(&format!(
            "- mean(|a_em|)/mean(|a_grav|): {}\n",
            format_opt_f64(sim_summary.mean_abs_accel_ratio_em_over_grav)
        ));
        md.push_str("\n## Best equation\n");
        md.push_str("```text\n");
        md.push_str(&best_equation_text);
        md.push_str("\n```\n");
        md.push_str(&format!(
            "- metrics: rollout_rmse={}, mse={:.6e}, complexity={}\n",
            format_opt_f64(best_metrics.rollout_rmse),
            best_metrics.mse,
            best_metrics.complexity
        ));
        md.push_str(&format!("- uses_em_terms: {}\n", uses_em_terms));
        fs::write(run_dir.join("report.md"), md)?;

        let discovery_json = serde_json::to_string_pretty(&serde_json::json!({
            "selected_solver": selected_solver,
            "stls": {
                "solver": build_solver_meta(&stls_settings, FitnessHeuristic::Mse, None),
                "top3_x": stls_topk_x.entries,
                "top3_y": stls_topk_y.entries,
                "top3_z": stls_topk_z.entries,
                "vector_candidates": stls_candidates,
            },
            "lasso": {
                "solver": build_solver_meta(&lasso_settings, FitnessHeuristic::Mse, None),
                "top3_x": lasso_topk_x.entries,
                "top3_y": lasso_topk_y.entries,
                "top3_z": lasso_topk_z.entries,
                "vector_candidates": lasso_candidates,
            },
        }))?;
        fs::write(run_dir.join("discovery.json"), discovery_json)?;

        let solver_summary = DiscoverySolverSummary {
            name: selected_solver.to_string(),
            normalize: true,
            fitness_heuristic: "mse".to_string(),
            stls: (selected_solver == "stls").then(|| StlsSolverSummary {
                auto_thresholds: true,
                thresholds: Vec::new(),
                ridge_lambda: 1e-8,
                max_iter: 25,
            }),
            lasso: (selected_solver == "lasso").then(|| LassoSolverSummary {
                auto_alphas: true,
                alphas: Vec::new(),
                max_iter: 2000,
                tol: 1e-6,
            }),
            ga: None,
        };

        let mut top_candidates = selected_candidates.clone();
        top_candidates.sort_by(|a, b| {
            a.metrics
                .mse
                .partial_cmp(&b.metrics.mse)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_candidates = top_candidates
            .into_iter()
            .take(3)
            .map(|c| FactoryEvaluationCandidate {
                id: c.id,
                equation_text: c.equation_text,
                metrics: c.metrics,
            })
            .collect::<Vec<_>>();

        eval_iterations.push(FactoryEvaluationIteration {
            iteration: idx + 1,
            run_id: run_id.clone(),
            regime: "em_quasistatic".to_string(),
            solver: solver_summary,
            simulation: Some(sim_summary.clone()),
            top_candidates,
            judge: None,
        });

        suite_cases.push(CaseResult {
            run_id,
            template: case.template.as_str().to_string(),
            charge_pattern: case.charge_pattern.as_str().to_string(),
            charge_scale: case.charge_scale,
            velocity_scale: case.velocity_scale,
            sim: sim_summary,
            best_equation_text,
            best_metrics,
            uses_em_terms,
            selected_solver: selected_solver.to_string(),
        });
    }

    let mut notes = Vec::new();
    notes.push("bench_em=true".to_string());
    notes.push(format!("steps={steps}"));
    notes.push(format!("dt={dt}"));
    notes.push(format!("seed={seed}"));
    notes.push(format!("max_cases={}", suite_cases.len()));
    notes.push("mode=truth".to_string());
    notes.push("regime=em_quasistatic".to_string());

    let eval_input = FactoryEvaluationInput {
        version: FACTORY_EVALUATION_VERSION.to_string(),
        notes,
        iterations: eval_iterations,
    };
    let eval_input_path = factory_dir.join("evaluation_input.json");
    fs::write(&eval_input_path, serde_json::to_string_pretty(&eval_input)?)?;

    // Render a deterministic novice summary for this benchmark run.
    let current_best = best_candidate_from_eval_input(&eval_input);
    let prior_filter = PriorAttemptFilter::from_notes(&eval_input.notes);
    let prior_best = find_best_prior_attempt(&results_dir, &eval_input_path, &prior_filter)?;
    let best_ic: Option<InitialConditionSpec> = current_best
        .as_ref()
        .and_then(|b| fs::read_to_string(factory_dir.join(&b.run_id).join("initial_conditions.json")).ok())
        .and_then(|raw| serde_json::from_str(&raw).ok());
    let incumbent_on_best_run = (|| -> anyhow::Result<Option<IncumbentEvalOnBestRun>> {
        let (best, prior) = match (current_best.as_ref(), prior_best.as_ref()) {
            (Some(b), Some(p)) => (b, p),
            _ => return Ok(None),
        };
        let best_run_dir = factory_dir.join(&best.run_id);
        let oracle = match load_oracle_run_from_dir(&best_run_dir) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        let incumbent_model = match vector_model_from_equation_text(&prior.best.candidate.equation_text) {
            Some(v) => v,
            None => return Ok(None),
        };
        let complexity = incumbent_model.eq_x.complexity()
            + incumbent_model.eq_y.complexity()
            + incumbent_model.eq_z.complexity();
        let feature_names = feature_names_for_equation_text(&prior.best.candidate.equation_text);
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

    let exec_md = render_executive_summary_md(
        &eval_input,
        current_best.as_ref(),
        prior_best.as_ref(),
        best_ic.as_ref(),
        incumbent_on_best_run.as_ref(),
        None,
    );
    fs::write(factory_dir.join("evaluation.md"), &exec_md)?;
    fs::write(out_dir.join("RESULTS.md"), &exec_md)?;

    // LaTeX evaluation (best effort; mirrors factory).
    let eval_tex = render_evaluation_tex(
        &eval_input,
        current_best.as_ref(),
        prior_best.as_ref(),
        best_ic.as_ref(),
        EvaluationTexOptions {
            factory_dir: Some(factory_dir.as_path()),
            benchmark: None,
            incumbent_on_best_run: incumbent_on_best_run.as_ref(),
        },
    );
    fs::write(factory_dir.join("evaluation.tex"), eval_tex)?;
    if build_pdf {
        if let Some(pdflatex) = find_pdflatex() {
            let _ = std::process::Command::new(&pdflatex)
                .current_dir(&factory_dir)
                .args(["-interaction=nonstopmode", "evaluation.tex"])
                .output();
            let pdf = factory_dir.join("evaluation.pdf");
            if pdf.exists() {
                let _ = fs::copy(&pdf, out_dir.join("RESULTS.pdf"));
            }
        }
    }

    #[derive(serde::Serialize)]
    struct Suite<'a> {
        version: &'a str,
        generated_at_utc: String,
        steps: usize,
        dt: f64,
        seed: u64,
        heldout: usize,
        cases: &'a [CaseResult],
    }
    let suite = Suite {
        version: "v1",
        generated_at_utc: updated_at_unix_utc(),
        steps,
        dt,
        seed,
        heldout,
        cases: &suite_cases,
    };
    fs::write(out_dir.join("suite.json"), serde_json::to_string_pretty(&suite)?)?;

    // Update global best index for the results directory.
    update_best_results_index_best_effort(&results_dir);

    println!("Bench-EM complete: {}", out_dir.display());
    println!("Key outputs:");
    println!("- {}/RESULTS.md", out_dir.display());
    println!("- {}/suite.json", out_dir.display());
    println!("- {}/factory/ (evidence per case)", out_dir.display());
    Ok(())
}

fn bench_em_initial_conditions(case: &BenchEmCase, seed: u64) -> anyhow::Result<(System, InitialConditionSpec)> {
    let base_charge = 0.01;
    let q = base_charge * case.charge_scale;
    let charges = match case.charge_pattern {
        BenchChargePattern::Like => [q, q, q],
        BenchChargePattern::AlternatingSigns => [q, -q, q],
    };

    let (pos, vel) = match case.template {
        BenchEmTemplate::HeadOn => {
            let pos = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.8, 0.0)];
            let v = 0.3 * case.velocity_scale;
            let vel = [Vec3::new(v, 0.0, 0.0), Vec3::new(-v, 0.0, 0.0), Vec3::zero()];
            (pos, vel)
        }
        BenchEmTemplate::HighAngMomentum => {
            let pos = [Vec3::new(-0.9, 0.0, 0.0), Vec3::new(0.9, 0.0, 0.0), Vec3::new(0.0, 1.6, 0.0)];
            let v = 0.3 * case.velocity_scale;
            let vel = [Vec3::new(0.0, v, 0.0), Vec3::new(0.0, -v, 0.0), Vec3::zero()];
            (pos, vel)
        }
        BenchEmTemplate::NearCircular => {
            let r = 1.0 / 3.0_f64.sqrt();
            let pos = [
                Vec3::new(r, 0.0, 0.0),
                Vec3::new(-0.5 * r, 0.5, 0.0),
                Vec3::new(-0.5 * r, -0.5, 0.0),
            ];
            let v = 0.3 * case.velocity_scale;
            let vel = [
                Vec3::new(0.0, v, 0.0),
                Vec3::new(-0.866_025_403_784_438_6 * v, -0.5 * v, 0.0),
                Vec3::new(0.866_025_403_784_438_6 * v, -0.5 * v, 0.0),
            ];
            (pos, vel)
        }
        BenchEmTemplate::NearCollisionSafe => {
            let pos = [Vec3::new(-0.35, 0.0, 0.0), Vec3::new(0.35, 0.0, 0.0), Vec3::new(0.0, 1.2, 0.0)];
            let v = 0.3 * case.velocity_scale;
            let vel = [Vec3::new(v, 0.0, 0.0), Vec3::new(-v, 0.0, 0.0), Vec3::zero()];
            (pos, vel)
        }
    };

    let mut bodies = [Body::new(1.0, 0.0); 3];
    for i in 0..3 {
        bodies[i] = Body::new(1.0, charges[i]);
    }
    let mut system = System::new(bodies, State::new(pos, vel));

    // Small deterministic perturbation to avoid perfectly symmetric degeneracy in some grids.
    let mut rng = threebody_discover::ga::Lcg::new(seed);
    let jitter = 0.02;
    for i in 0..3 {
        system.state.pos[i].x += rng.gen_range_f64(-jitter, jitter);
        system.state.pos[i].y += rng.gen_range_f64(-jitter, jitter);
        system.state.vel[i].x += rng.gen_range_f64(-jitter, jitter) * 0.1;
        system.state.vel[i].y += rng.gen_range_f64(-jitter, jitter) * 0.1;
    }
    system = to_barycentric(system);

    let bounds = default_ic_bounds();
    let min_dist = min_pair_distance(&system.state.pos);
    if min_dist < bounds.min_pair_dist {
        anyhow::bail!(
            "generated IC too close: min_pair_dist={} (min allowed {})",
            min_dist,
            bounds.min_pair_dist
        );
    }

    let ic_spec = InitialConditionSpec {
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
        notes: format!(
            "bench_em template={} pattern={} charge_scale={} velocity_scale={}",
            case.template.as_str(),
            case.charge_pattern.as_str(),
            case.charge_scale,
            case.velocity_scale
        ),
    };
    Ok((system, ic_spec))
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
    let sidecar = build_sidecar(&cfg, &header, &result, Some(steps), Some(dt));
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
    feature_library: String,
) -> anyhow::Result<()> {
    let library_kind = parse_feature_library_kind(&feature_library).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown --feature-library={feature_library} (expected default|extended|em_fields)"
        )
    })?;
    let library = match library_kind {
        FeatureLibraryKind::DefaultPhysics => FeatureLibrary::default_physics(),
        FeatureLibraryKind::ExtendedPhysics => FeatureLibrary::extended_physics(),
        FeatureLibraryKind::EmFieldsLorentz => FeatureLibrary::em_fields_lorentz(),
    };
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
        encounter: sidecar.encounter.map(|e| threebody_core::sim::EncounterEvent {
            step: e.step,
            min_pair_dist: e.min_pair_dist,
            epsilon_before: e.epsilon_before,
            epsilon_after: e.epsilon_after,
            substeps_used: e.substeps_used,
        }),
        encounter_action: sidecar.encounter_action,
        warnings: sidecar.warnings.clone(),
        terminated_early: sidecar.terminated_early,
        termination_reason: sidecar.termination_reason.clone(),
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

    let simulation = build_sim_summary(
        &result,
        &sim_cfg,
        rollout_integrator,
        sidecar.requested_steps,
        sidecar.requested_dt,
    );
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
                let mut resp = result.value;
                sanitize_judge_recommendations(&mut resp.recommendations, &judge_input.ic_bounds);
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

fn sanitize_judge_recommendations(rec: &mut threebody_discover::JudgeRecommendations, bounds: &IcBounds) {
    // 1) Drop invalid ICs (keeps `--require-llm` runs robust; the preset fallback still works).
    if let Some(spec) = rec.next_initial_conditions.as_ref() {
        if let Err(err) = system_from_ic(spec, bounds) {
            eprintln!("Ignoring invalid LLM next_initial_conditions recommendation ({err}).");
            rec.next_initial_conditions = None;
        }
    }

    // 2) Normalize "enum-like" strings: if it's not one of our accepted values, ignore it.
    fn normalize_choice(opt: &mut Option<String>, allowed: &[&str], field: &str) {
        let Some(v) = opt.as_deref() else {
            return;
        };
        let v_trim = v.trim();
        if v_trim.is_empty() {
            *opt = None;
            return;
        }
        if !allowed.iter().any(|a| *a == v_trim) {
            eprintln!("Ignoring invalid LLM {field} recommendation ({v_trim}).");
            *opt = None;
        } else if v_trim != v {
            *opt = Some(v_trim.to_string());
        }
    }

    normalize_choice(
        &mut rec.next_rollout_integrator,
        &["euler", "leapfrog"],
        "next_rollout_integrator",
    );
    normalize_choice(
        &mut rec.next_ga_heuristic,
        &["mse", "mse_parsimony"],
        "next_ga_heuristic",
    );
    normalize_choice(
        &mut rec.next_discovery_solver,
        &["stls", "lasso", "ga"],
        "next_discovery_solver",
    );
    normalize_choice(
        &mut rec.next_feature_library,
        &["default", "extended"],
        "next_feature_library",
    );

    // 3) Numeric hyperparams: ignore invalid / negative values.
    for (name, v) in [
        ("next_stls_threshold", &mut rec.next_stls_threshold),
        ("next_lasso_alpha", &mut rec.next_lasso_alpha),
    ] {
        if let Some(x) = v.as_ref() {
            if !x.is_finite() || *x <= 0.0 {
                eprintln!("Ignoring invalid LLM {name} recommendation ({x}).");
                *v = None;
            }
        }
    }
    for (name, v) in [("next_ridge_lambda", &mut rec.next_ridge_lambda)] {
        if let Some(x) = v.as_ref() {
            if !x.is_finite() || *x < 0.0 {
                eprintln!("Ignoring invalid LLM {name} recommendation ({x}).");
                *v = None;
            }
        }
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

fn ic_is_nearly_planar(spec: &InitialConditionSpec) -> bool {
    const EPS: f64 = 1e-9;
    spec.bodies
        .iter()
        .all(|b| b.pos[2].abs() <= EPS && b.vel[2].abs() <= EPS)
}

fn force_nonplanar_ic(mut spec: InitialConditionSpec, bounds: &IcBounds, seed: u64) -> InitialConditionSpec {
    if spec.bodies.len() != 3 {
        return spec;
    }
    if !ic_is_nearly_planar(&spec) {
        return spec;
    }

    let clamp = |v: f64, min: f64, max: f64| {
        if !v.is_finite() {
            min
        } else {
            v.min(max).max(min)
        }
    };

    // Deterministic, small z-perturbation to ensure 3D dynamics so `az` is identifiable.
    // Barycentric conversion later removes COM position/velocity, so we only need a nonzero z component.
    let sign = if seed % 2 == 0 { 1.0 } else { -1.0 };
    let dz = 0.25 * sign;
    let dvz = 0.10 * if seed % 3 == 0 { 1.0 } else { -1.0 };

    spec.bodies[0].pos[2] = clamp(spec.bodies[0].pos[2] + dz, bounds.pos_min, bounds.pos_max);
    spec.bodies[1].pos[2] = clamp(spec.bodies[1].pos[2] - 0.5 * dz, bounds.pos_min, bounds.pos_max);
    spec.bodies[2].pos[2] = clamp(spec.bodies[2].pos[2] - 0.5 * dz, bounds.pos_min, bounds.pos_max);

    spec.bodies[0].vel[2] = clamp(spec.bodies[0].vel[2] + dvz, bounds.vel_min, bounds.vel_max);
    spec.bodies[1].vel[2] = clamp(spec.bodies[1].vel[2] - 0.5 * dvz, bounds.vel_min, bounds.vel_max);
    spec.bodies[2].vel[2] = clamp(spec.bodies[2].vel[2] - 0.5 * dvz, bounds.vel_min, bounds.vel_max);

    if spec.notes.trim().is_empty() {
        spec.notes = "forced_nonplanar_z=true".to_string();
    } else {
        spec.notes = format!("{} | forced_nonplanar_z=true", spec.notes.trim());
    }
    spec.barycentric = true;
    spec
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
    fn softened_inv_r3_r4(r2: f64, epsilon: f64) -> (f64, f64) {
        if r2 == 0.0 {
            return (0.0, 0.0);
        }
        let soft2 = if epsilon == 0.0 { r2 } else { r2 + epsilon * epsilon };
        let r = soft2.sqrt();
        if r == 0.0 || !r.is_finite() {
            return (0.0, 0.0);
        }
        let inv_r = 1.0 / r;
        let inv_r3 = inv_r * inv_r * inv_r;
        let inv_r4 = inv_r3 * inv_r;
        (inv_r3, inv_r4)
    }

    let epsilon = cfg.softening;
    let mut grav = Vec3::zero(); // Σ m_j (r_j - r_i) / |r|^3  (no G)
    let mut grav_r4 = Vec3::zero(); // Σ m_j (r_j - r_i) / |r|^4 (no G)
    let mut elec = Vec3::zero(); // (q_i/m_i) Σ q_j (r_i - r_j) / |r|^3 (no k_e)
    let mut elec_r4 = Vec3::zero(); // (q_i/m_i) Σ q_j (r_i - r_j) / |r|^4 (no k_e)
    let mut mag = Vec3::zero(); // (q_i/m_i) (v_i × B_basis) where B_basis=(1/4π)Σ q_j(v_j×(r_i-r_j))/|r|^3 (no μ0)
    let mut mag_r4 = Vec3::zero(); // same, with |r|^4 scaling

    if cfg.enable_gravity {
        for j in 0..3 {
            if j == body {
                continue;
            }
            let r_ji = system.state.pos[j] - system.state.pos[body];
            let (inv_r3, inv_r4) = softened_inv_r3_r4(r_ji.norm_sq(), epsilon);
            grav = grav + r_ji * (system.bodies[j].mass * inv_r3);
            grav_r4 = grav_r4 + r_ji * (system.bodies[j].mass * inv_r4);
        }
    }

    if cfg.enable_em {
        let qi = system.bodies[body].charge;
        let mi = system.bodies[body].mass;
        if qi != 0.0 && mi != 0.0 {
            let q_over_m = qi / mi;
            let mut e_basis = Vec3::zero();
            let mut e_basis_r4 = Vec3::zero();
            let mut b_basis = Vec3::zero();
            let mut b_basis_r4 = Vec3::zero();
            let inv_4pi = 1.0 / (4.0 * std::f64::consts::PI);
            for j in 0..3 {
                if j == body {
                    continue;
                }
                // Use (r_i - r_j) to match the electrostatic field direction.
                let r = system.state.pos[body] - system.state.pos[j];
                let (inv_r3, inv_r4) = softened_inv_r3_r4(r.norm_sq(), epsilon);
                let qj = system.bodies[j].charge;
                e_basis = e_basis + r * (qj * inv_r3);
                e_basis_r4 = e_basis_r4 + r * (qj * inv_r4);

                // Magnetic basis uses v_j × (r_i - r_j).
                let vj_cross_r = system.state.vel[j].cross(r);
                b_basis = b_basis + vj_cross_r * (qj * inv_r3 * inv_4pi);
                b_basis_r4 = b_basis_r4 + vj_cross_r * (qj * inv_r4 * inv_4pi);
            }
            elec = e_basis * q_over_m;
            elec_r4 = e_basis_r4 * q_over_m;
            mag = system.state.vel[body].cross(b_basis) * q_over_m;
            mag_r4 = system.state.vel[body].cross(b_basis_r4) * q_over_m;
        }
    }

    // Physical-field Lorentz terms (in acceleration units; includes constants).
    let lorentz_e = elec * cfg.constants.k_e;
    let lorentz_vxb = mag * cfg.constants.mu_0;

    const R_GATE: f64 = 0.5;
    let gate_close = (min_pair_distance(&system.state.pos) < R_GATE) as u8 as f64;
    let gate_far = 1.0 - gate_close;

    let grav_close = grav * gate_close;
    let grav_far = grav * gate_far;
    let elec_close = elec * gate_close;
    let elec_far = elec * gate_far;
    let mag_close = mag * gate_close;
    let mag_far = mag * gate_far;

    let mut out = Vec::with_capacity(feature_names.len());
    for name in feature_names {
        let v = match name.as_str() {
            "grav_x" => grav.x,
            "grav_y" => grav.y,
            "grav_z" => grav.z,
            "grav_r4_x" => grav_r4.x,
            "grav_r4_y" => grav_r4.y,
            "grav_r4_z" => grav_r4.z,
            "elec_x" => elec.x,
            "elec_y" => elec.y,
            "elec_z" => elec.z,
            "elec_r4_x" => elec_r4.x,
            "elec_r4_y" => elec_r4.y,
            "elec_r4_z" => elec_r4.z,
            "lorentz_e_x" => lorentz_e.x,
            "lorentz_e_y" => lorentz_e.y,
            "lorentz_e_z" => lorentz_e.z,
            "mag_x" => mag.x,
            "mag_y" => mag.y,
            "mag_z" => mag.z,
            "mag_r4_x" => mag_r4.x,
            "mag_r4_y" => mag_r4.y,
            "mag_r4_z" => mag_r4.z,
            "lorentz_vxb_x" => lorentz_vxb.x,
            "lorentz_vxb_y" => lorentz_vxb.y,
            "lorentz_vxb_z" => lorentz_vxb.z,
            "gate_close" => gate_close,
            "gate_far" => gate_far,
            "grav_close_x" => grav_close.x,
            "grav_close_y" => grav_close.y,
            "grav_close_z" => grav_close.z,
            "grav_far_x" => grav_far.x,
            "grav_far_y" => grav_far.y,
            "grav_far_z" => grav_far.z,
            "elec_close_x" => elec_close.x,
            "elec_close_y" => elec_close.y,
            "elec_close_z" => elec_close.z,
            "elec_far_x" => elec_far.x,
            "elec_far_y" => elec_far.y,
            "elec_far_z" => elec_far.z,
            "mag_close_x" => mag_close.x,
            "mag_close_y" => mag_close.y,
            "mag_close_z" => mag_close.z,
            "mag_far_x" => mag_far.x,
            "mag_far_y" => mag_far.y,
            "mag_far_z" => mag_far.z,
            _ => 0.0,
        };
        out.push(v);
    }
    out
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

fn append_manual_vector_candidate_from_equation_text(
    vector_candidates: &mut Vec<CandidateSummary>,
    equation_text: &str,
    dataset_x: &Dataset,
    dataset_y: &Dataset,
    dataset_z: &Dataset,
    feature_names: &[String],
    result: &threebody_core::sim::SimResult,
    cfg: &Config,
    regime: &str,
    rollout_integrator: RolloutIntegrator,
    notes: Vec<String>,
) {
    let Some(model) = vector_model_from_equation_text(equation_text) else {
        return;
    };

    let allowed: std::collections::HashSet<&str> = feature_names.iter().map(|s| s.as_str()).collect();
    let mut sanitized = model;
    for eq in [&mut sanitized.eq_x, &mut sanitized.eq_y, &mut sanitized.eq_z] {
        eq.terms
            .retain(|t| t.coeff.is_finite() && allowed.contains(t.feature.as_str()));
    }

    let mse_x = threebody_discover::equation::score_equation(&sanitized.eq_x, dataset_x);
    let mse_y = threebody_discover::equation::score_equation(&sanitized.eq_y, dataset_y);
    let mse_z = threebody_discover::equation::score_equation(&sanitized.eq_z, dataset_z);
    let mse = (mse_x + mse_y + mse_z) / 3.0;
    let complexity = sanitized.eq_x.complexity() + sanitized.eq_y.complexity() + sanitized.eq_z.complexity();
    let (rmse, divergence_time) = rollout_metrics(&sanitized, feature_names, result, cfg, rollout_integrator);

    let mut flags = Vec::new();
    flags.extend(stability_flags_for(&sanitized.eq_x, regime));
    flags.extend(stability_flags_for(&sanitized.eq_y, regime));
    flags.extend(stability_flags_for(&sanitized.eq_z, regime));
    flags.sort();
    flags.dedup();

    let id = vector_candidates.len();
    vector_candidates.push(CandidateSummary {
        id,
        equation: sanitized.eq_x.clone(),
        equation_text: format_vector_model(&sanitized),
        metrics: CandidateMetrics {
            mse,
            complexity,
            rollout_rmse: Some(rmse),
            divergence_time,
            stability_flags: flags,
        },
        notes,
    });
}

fn build_sim_summary(
    result: &threebody_core::sim::SimResult,
    cfg: &Config,
    rollout_integrator: RolloutIntegrator,
    requested_steps: Option<usize>,
    requested_dt: Option<f64>,
) -> SimulationSummary {
    fn mean_abs_accel(
        result: &threebody_core::sim::SimResult,
        force_cfg: &ForceConfig,
    ) -> Option<f64> {
        if result.steps.is_empty() {
            return None;
        }
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for step in &result.steps {
            let acc = compute_accel(&step.system, force_cfg);
            for a in &acc {
                sum += a.norm();
                count += 1;
            }
        }
        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        }
    }

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

    let grav_force_cfg = ForceConfig {
        g: cfg.constants.g,
        k_e: cfg.constants.k_e,
        mu_0: cfg.constants.mu_0,
        epsilon: cfg.softening,
        enable_gravity: true,
        enable_em: false,
    };
    let em_force_cfg = ForceConfig {
        g: cfg.constants.g,
        k_e: cfg.constants.k_e,
        mu_0: cfg.constants.mu_0,
        epsilon: cfg.softening,
        enable_gravity: false,
        enable_em: true,
    };

    let mean_abs_accel_grav = cfg.enable_gravity.then(|| mean_abs_accel(result, &grav_force_cfg)).flatten();
    let mean_abs_accel_em = cfg.enable_em.then(|| mean_abs_accel(result, &em_force_cfg)).flatten();
    let mean_abs_accel_ratio_em_over_grav = match (mean_abs_accel_em, mean_abs_accel_grav) {
        (Some(em), Some(grav)) if em.is_finite() && grav.is_finite() && grav > 0.0 => Some(em / grav),
        _ => None,
    };

    SimulationSummary {
        steps: result.steps.len(),
        requested_steps,
        requested_dt,
        terminated_early: result.terminated_early,
        termination_reason: result.termination_reason.clone(),
        encounter_step: result.encounter.map(|e| e.step),
        encounter_min_pair_dist: result.encounter.map(|e| e.min_pair_dist),
        encounter_action: result.encounter_action.map(|a| format!("{a:?}")),
        energy_start,
        energy_end,
        energy_drift,
        min_pair_dist: min_pair,
        max_speed,
        max_accel,
        mean_abs_accel_grav,
        mean_abs_accel_em,
        mean_abs_accel_ratio_em_over_grav,
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
enum FeatureLibraryKind {
    DefaultPhysics,
    ExtendedPhysics,
    EmFieldsLorentz,
}

fn feature_library_kind_label(kind: FeatureLibraryKind) -> &'static str {
    match kind {
        FeatureLibraryKind::DefaultPhysics => "default",
        FeatureLibraryKind::ExtendedPhysics => "extended",
        FeatureLibraryKind::EmFieldsLorentz => "em_fields",
    }
}

fn parse_feature_library_kind(raw: &str) -> Option<FeatureLibraryKind> {
    match raw.trim().to_lowercase().as_str() {
        "default" | "default_physics" | "default-physics" => Some(FeatureLibraryKind::DefaultPhysics),
        "extended" | "extended_physics" | "extended-physics" => Some(FeatureLibraryKind::ExtendedPhysics),
        "em_fields" | "em-fields" | "em_fields_lorentz" | "em-fields-lorentz" | "fields" | "lorentz" => {
            Some(FeatureLibraryKind::EmFieldsLorentz)
        }
        _ => None,
    }
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
    feature_library: String,
    llm_mode: String,
    model: String,
    openai_key_file: Option<PathBuf>,
    require_llm: bool,
) -> anyhow::Result<()> {
    fs::create_dir_all(&out_dir)?;
    let mut next_ic: Option<InitialConditionSpec> = None;
    let mut next_manual_equation_text: Option<String> = None;
    let llm_mode = parse_llm_mode(&llm_mode)?;
    let llm_client = select_llm_client(llm_mode, &model, openai_key_file.as_deref(), require_llm)?;
    let mut current_fitness = parse_fitness_heuristic(&fitness)?;
    let mut current_rollout = parse_rollout_integrator(&rollout_integrator)?;
    let mut current_solver = solver_settings;
    let mut evaluation_iterations: Vec<FactoryEvaluationIteration> = Vec::new();
    let incumbent_bucket = BucketKey { steps, dt };
    let incumbent = load_incumbent_for_bucket(std::path::Path::new("results"), &incumbent_bucket);
    let feature_library_trimmed = feature_library.trim();
    let feature_library_lower = feature_library_trimmed.to_lowercase();
    let feature_library_pinned = feature_library_lower != "auto";
    let mut current_library = if feature_library_pinned {
        parse_feature_library_kind(&feature_library_lower).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown --feature-library={feature_library_trimmed} (expected auto|default|extended|em_fields)"
            )
        })?
    } else {
        let mut kind = FeatureLibraryKind::DefaultPhysics;
        if let Some(rec) = incumbent.as_ref() {
            let incumbent_features = feature_names_for_equation_text(&rec.equation_text);
            if incumbent_features.iter().any(|f| f.starts_with("lorentz_")) {
                kind = FeatureLibraryKind::EmFieldsLorentz;
            } else if incumbent_features.len() > FeatureLibrary::default_physics().features.len() {
                kind = FeatureLibraryKind::ExtendedPhysics;
            }
        }
        kind
    };
    let mut equation_ga_parent: Option<String> = incumbent.as_ref().map(|r| r.equation_text.clone());
    let mut previous_iteration_elites: Vec<(String, CandidateMetrics)> = Vec::new();

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

        let mut cfg = build_config(config.clone(), &mode, None, em, no_em, no_gravity)?;
        if matches!(cfg.integrator.kind, IntegratorKind::Rk45) && cfg.integrator.adaptive {
            cfg.integrator.max_rejects = cfg.integrator.max_rejects.max(64);
        }
        let regime = if cfg.enable_em { "em_quasistatic" } else { "gravity_only" };
        let ic_bounds = default_ic_bounds();
        let mut ic_notes = vec!["avoid close encounters".to_string(), "prefer bounded motion".to_string()];
        if let Some(rec) = incumbent.as_ref() {
            ic_notes.extend(incumbent_prompt_notes(rec));
        }
        if !previous_iteration_elites.is_empty() {
            ic_notes.push("PREV_ITER_ELITE_EQUATIONS (use these to pick ICs that distinguish models):".to_string());
            for (i, (eq, m)) in previous_iteration_elites.iter().take(3).enumerate() {
                let eq = truncate_to_chars(&single_line(eq), 300);
                ic_notes.push(format!(
                    "ELITE_{}: eq={} | rollout_rmse={} | mse={:.6e} | complexity={}",
                    i + 1,
                    eq,
                    format_opt_f64(m.rollout_rmse),
                    m.mse,
                    m.complexity
                ));
            }
            ic_notes.push(
                "IC_GOAL: propose initial conditions where these elite equations diverge in rollout so we can rank them."
                    .to_string(),
            );
        }

        #[derive(serde::Serialize)]
        struct SimAttemptLog {
            attempt: usize,
            ic_request: IcRequest,
            initial_conditions: InitialConditionSpec,
            produced_steps: usize,
            terminated_early: bool,
            termination_reason: Option<String>,
            encounter_step: Option<usize>,
            encounter_min_pair_dist: Option<f64>,
            encounter_action: Option<String>,
            min_pair_dist: Option<f64>,
            max_accel: Option<f64>,
        }

        fn sim_min_pair_dist(result: &threebody_core::sim::SimResult) -> Option<f64> {
            let mut min_pair: Option<f64> = None;
            for step in &result.steps {
                min_pair = Some(match min_pair {
                    Some(v) => v.min(step.regime.min_pair_dist),
                    None => step.regime.min_pair_dist,
                });
            }
            min_pair
        }

        fn sim_max_accel(result: &threebody_core::sim::SimResult) -> Option<f64> {
            let mut max_accel: Option<f64> = None;
            for step in &result.steps {
                max_accel = Some(match max_accel {
                    Some(v) => v.max(step.regime.max_accel),
                    None => step.regime.max_accel,
                });
            }
            max_accel
        }

        fn sim_is_usable(result: &threebody_core::sim::SimResult, requested_steps: usize) -> bool {
            !result.terminated_early && result.steps.len() == requested_steps + 1
        }

        fn pick_initial_conditions(
            llm_client: Option<&Box<dyn LlmClient>>,
            require_llm: bool,
            preset: &str,
            regime: &str,
            ic_bounds: &IcBounds,
            base_notes: &[String],
            seed: u64,
            next_ic: &mut Option<InitialConditionSpec>,
        ) -> anyhow::Result<(IcRequest, InitialConditionSpec, System, Option<String>, Option<String>)> {
            let max_ic_attempts = if require_llm { 3 } else { 1 };
            let mut ic_prompt = None;
            let mut ic_response = None;
            let mut last_err: Option<String> = None;

            for ic_attempt in 0..max_ic_attempts {
                let mut attempt_notes = base_notes.to_vec();
                if let Some(err) = last_err.as_ref() {
                    attempt_notes.push(format!("previous_ic_validation_error={}", single_line(err)));
                }
                let ic_request = IcRequest {
                    bounds: ic_bounds.clone(),
                    regime: regime.to_string(),
                    notes: attempt_notes,
                    seed: Some(seed + ic_attempt as u64),
                };

                let candidate = if let Some(spec) = next_ic.take() {
                    spec
                } else if let Some(client) = llm_client {
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
                            initial_conditions_from_preset(preset)?
                        }
                    }
                } else if require_llm {
                    anyhow::bail!("LLM required but no client is configured");
                } else {
                    initial_conditions_from_preset(preset)?
                };

                let candidate = force_nonplanar_ic(candidate, ic_bounds, seed + ic_attempt as u64);
                match system_from_ic(&candidate, ic_bounds) {
                    Ok(system) => return Ok((ic_request, candidate, system, ic_prompt, ic_response)),
                    Err(err) => {
                        if !require_llm {
                            eprintln!("IC validation failed ({err}); falling back to preset.");
                            let spec = force_nonplanar_ic(initial_conditions_from_preset(preset)?, ic_bounds, seed);
                            let system = system_from_ic(&spec, ic_bounds).unwrap_or_else(|_| {
                                // Presets are assumed safe; if barycentric/min-dist checks fail for any reason,
                                // keep going with the raw preset system.
                                preset_system(preset).unwrap_or_else(|_| {
                                    let bodies = [Body::new(1.0, 0.0), Body::new(1.0, 0.0), Body::new(1.0, 0.0)];
                                    let pos = [Vec3::zero(); 3];
                                    let vel = [Vec3::zero(); 3];
                                    System::new(bodies, State::new(pos, vel))
                                })
                            });
                            let ic_request = IcRequest {
                                bounds: ic_bounds.clone(),
                                regime: regime.to_string(),
                                notes: base_notes.to_vec(),
                                seed: Some(seed),
                            };
                            return Ok((ic_request, spec, system, ic_prompt, ic_response));
                        }
                        last_err = Some(err.to_string());
                        eprintln!("IC validation failed ({err}); retrying with LLM.");
                        continue;
                    }
                }
            }
            let err = last_err.unwrap_or_else(|| "unknown ic validation error".to_string());
            anyhow::bail!("IC validation failed after {max_ic_attempts} attempts: {err}");
        }

        let llm_client_ref = llm_client.as_ref();
        let mut sim_attempt_logs: Vec<SimAttemptLog> = Vec::new();
        let mut sim_failure_note: Option<String> = None;
        let mut ic_prompt: Option<String> = None;
        let mut ic_response: Option<String> = None;
        let max_sim_attempts: usize = 3;

        let mut final_ic_request: Option<IcRequest> = None;
        let mut final_ic_spec: Option<InitialConditionSpec> = None;
        let mut final_result: Option<threebody_core::sim::SimResult> = None;

        for sim_attempt in 1..=max_sim_attempts {
            let mut attempt_notes = ic_notes.clone();
            if let Some(note) = sim_failure_note.as_ref() {
                attempt_notes.push(note.clone());
            }
            let attempt_seed = seed + iter as u64 + (sim_attempt as u64 - 1) * 1000;
            let (ic_request, ic_spec, system, attempt_prompt, attempt_response) = pick_initial_conditions(
                llm_client_ref,
                require_llm,
                &preset,
                regime,
                &ic_bounds,
                &attempt_notes,
                attempt_seed,
                &mut next_ic,
            )?;
            ic_prompt = attempt_prompt.or(ic_prompt);
            ic_response = attempt_response.or(ic_response);

            let options = SimOptions { steps, dt };
            let result = simulate_with_cfg(system, &cfg, options);
            let produced_steps = result.steps.len();
            let usable = sim_is_usable(&result, steps);

            sim_attempt_logs.push(SimAttemptLog {
                attempt: sim_attempt,
                ic_request: ic_request.clone(),
                initial_conditions: ic_spec.clone(),
                produced_steps,
                terminated_early: result.terminated_early,
                termination_reason: result.termination_reason.clone(),
                encounter_step: result.encounter.map(|e| e.step),
                encounter_min_pair_dist: result.encounter.map(|e| e.min_pair_dist),
                encounter_action: result.encounter_action.map(|a| format!("{a:?}")),
                min_pair_dist: sim_min_pair_dist(&result),
                max_accel: sim_max_accel(&result),
            });

            if usable || sim_attempt == max_sim_attempts {
                final_ic_request = Some(ic_request);
                final_ic_spec = Some(ic_spec);
                final_result = Some(result);
                break;
            }

            sim_failure_note = Some(format!(
                "previous_sim_failure: produced_steps={}, terminated_early={}, termination_reason={}",
                produced_steps,
                result.terminated_early,
                result.termination_reason.clone().unwrap_or_else(|| "n/a".to_string())
            ));
        }

        let ic_request = final_ic_request.ok_or_else(|| anyhow::anyhow!("missing ic_request"))?;
        let ic_spec = final_ic_spec.ok_or_else(|| anyhow::anyhow!("missing ic_spec"))?;
        let result = final_result.ok_or_else(|| anyhow::anyhow!("missing simulation result"))?;

        fs::write(
            run_dir.join("sim_attempts.json"),
            serde_json::to_string_pretty(&sim_attempt_logs)?,
        )?;

        let ic_request_path = run_dir.join("ic_request.json");
        fs::write(&ic_request_path, serde_json::to_string_pretty(&ic_request)?)?;
        let ic_spec_path = run_dir.join("initial_conditions.json");
        fs::write(&ic_spec_path, serde_json::to_string_pretty(&ic_spec)?)?;
        let cfg_path = run_dir.join("config.json");
        fs::write(&cfg_path, serde_json::to_string_pretty(&cfg)?)?;

        let traj_path = run_dir.join("traj.csv");
        let mut csv_file = fs::File::create(&traj_path)?;
        write_csv(&mut csv_file, &result.steps, &cfg)?;
        let header = threebody_core::output::csv::csv_header(&cfg);
        let sidecar = build_sidecar(&cfg, &header, &result, Some(steps), Some(dt));
        let sidecar_path = run_dir.join("traj.json");
        let mut sidecar_file = fs::File::create(&sidecar_path)?;
        write_sidecar(&mut sidecar_file, &sidecar)?;

        let library = match current_library {
            FeatureLibraryKind::DefaultPhysics => FeatureLibrary::default_physics(),
            FeatureLibraryKind::ExtendedPhysics => FeatureLibrary::extended_physics(),
            FeatureLibraryKind::EmFieldsLorentz => FeatureLibrary::em_fields_lorentz(),
        };
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
        let mut vector_candidates = build_vector_candidates(
            &topk_x.entries,
            &topk_y.entries,
            &topk_z.entries,
            &vector_data.feature_names,
            &result,
            &cfg,
            regime,
            current_rollout,
        );

        if let Some(eq_text) = next_manual_equation_text.take() {
            append_manual_vector_candidate_from_equation_text(
                &mut vector_candidates,
                &eq_text,
                &dataset_x,
                &dataset_y,
                &dataset_z,
                &vector_data.feature_names,
                &result,
                &cfg,
                regime,
                current_rollout,
                vec![
                    "source=llm".to_string(),
                    "kind=carried_next_manual_equation_text".to_string(),
                ],
            );
        }

        // Equation-GA (lightweight): treat equations themselves as a population across iterations.
        // We carry forward the current best (and recent elites) and try a few small, axis-consistent
        // mutations. Everything is still scored numerically; the LLM is just another proposal source.
        {
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            for c in &vector_candidates {
                if let Some(norm) =
                    normalize_equation_text_for_features(&c.equation_text, &vector_data.feature_names)
                {
                    seen.insert(norm);
                }
            }

            let mut parents: Vec<(String, String)> = Vec::new();
            if let Some(eq) = equation_ga_parent.as_deref() {
                parents.push(("equation_ga_parent".to_string(), eq.to_string()));
            }
            for (i, (eq, _m)) in previous_iteration_elites.iter().take(2).enumerate() {
                parents.push((format!("equation_ga_elite_{}", i + 1), eq.clone()));
            }
            if parents.is_empty() {
                if let Some(best_seed) = vector_candidates.iter().min_by(|a, b| {
                    metrics_sort_key(&a.metrics)
                        .partial_cmp(&metrics_sort_key(&b.metrics))
                        .unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    parents.push((
                        "equation_ga_seed_from_current".to_string(),
                        best_seed.equation_text.clone(),
                    ));
                }
            }

            let mut parent_used = 0usize;
            for (parent_kind, parent_eq) in parents {
                if parent_used >= 2 {
                    break;
                }
                let Some(parent_norm) =
                    normalize_equation_text_for_features(&parent_eq, &vector_data.feature_names)
                else {
                    continue;
                };

                if !seen.contains(&parent_norm) {
                    append_manual_vector_candidate_from_equation_text(
                        &mut vector_candidates,
                        &parent_norm,
                        &dataset_x,
                        &dataset_y,
                        &dataset_z,
                        &vector_data.feature_names,
                        &result,
                        &cfg,
                        regime,
                        current_rollout,
                        vec![
                            "source=equation_ga".to_string(),
                            format!("kind={parent_kind}"),
                        ],
                    );
                    seen.insert(parent_norm.clone());
                }

                let mutant_seed = seed + iter as u64 + (parent_used as u64) * 10_000;
                for mutant in propose_equation_mutants(
                    &parent_norm,
                    &vector_data.feature_names,
                    regime,
                    mutant_seed,
                ) {
                    if !seen.insert(mutant.clone()) {
                        continue;
                    }
                    append_manual_vector_candidate_from_equation_text(
                        &mut vector_candidates,
                        &mutant,
                        &dataset_x,
                        &dataset_y,
                        &dataset_z,
                        &vector_data.feature_names,
                        &result,
                        &cfg,
                        regime,
                        current_rollout,
                        vec![
                            "source=equation_ga".to_string(),
                            "kind=equation_ga_mutant".to_string(),
                            format!("parent={}", truncate_to_chars(&single_line(&parent_norm), 120)),
                        ],
                    );
                }
                parent_used += 1;
            }
        }
        let grid_topk = grid_search(
            &vector_candidates
                .iter()
                .map(|c| c.equation.clone())
                .collect::<Vec<_>>(),
            &dataset_x,
        );
        let mut trace_written = false;

        let sim_summary = build_sim_summary(
            &result,
            &cfg,
            current_rollout,
            Some(steps),
            Some(dt),
        );
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
        judge_input
            .notes
            .push(format!("feature_library={}", feature_library_kind_label(current_library)));
        let judge_input_path = run_dir.join("judge_input.json");
        fs::write(&judge_input_path, serde_json::to_string_pretty(&judge_input)?)?;
        let mut judge_prompt = None;
        let mut judge_response = None;
        let judge = if let Some(client) = llm_client.as_ref() {
            match client.judge_candidates(&judge_input) {
                Ok(result) => {
                    judge_prompt = Some(result.prompt);
                    judge_response = Some(result.response);
                    let mut resp = result.value;
                    sanitize_judge_recommendations(&mut resp.recommendations, &ic_bounds);
                    match resp.validate(&judge_input) {
                        Ok(()) => Some(resp),
                        Err(msg) => {
                            eprintln!("LLM judge response failed validation: {msg}");
                            None
                        }
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

        // If the judge proposed a concrete equation, evaluate it immediately on this run's data.
        // This lets the LLM "write math" while numeric scoring still decides whether it helps.
        if let Some(j) = judge.as_ref() {
            if let Some(eq_text) = j.recommendations.next_manual_equation_text.as_deref() {
                append_manual_vector_candidate_from_equation_text(
                    &mut vector_candidates,
                    eq_text,
                    &dataset_x,
                    &dataset_y,
                    &dataset_z,
                    &vector_data.feature_names,
                    &result,
                    &cfg,
                    regime,
                    current_rollout,
                    vec![
                        "source=llm".to_string(),
                        "kind=judge_next_manual_equation_text_same_iter".to_string(),
                    ],
                );
            }
        }

        // Write a rollout trace for the best (numerically) candidate after all candidate injection.
        if let Some(best) = vector_candidates.iter().min_by(|a, b| {
            metrics_sort_key(&a.metrics)
                .partial_cmp(&metrics_sort_key(&b.metrics))
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            if let Some(best_model) = vector_model_from_equation_text(&best.equation_text) {
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
            feature_library: String,
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
            feature_library: feature_library_kind_label(current_library).to_string(),
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
        md.push_str(&format!(
            "- Feature library: {}\n",
            feature_library_kind_label(current_library)
        ));
        md.push_str(&format!("- Steps: {}\n", sim_summary.steps));
        md.push_str(&format!("- Energy drift: {:?}\n", sim_summary.energy_drift));
        md.push_str(&format!("- Min pair dist: {:?}\n", sim_summary.min_pair_dist));
        md.push_str(&format!(
            "- mean_abs_accel_grav: {}\n",
            format_opt_f64(sim_summary.mean_abs_accel_grav)
        ));
        md.push_str(&format!(
            "- mean_abs_accel_em: {}\n",
            format_opt_f64(sim_summary.mean_abs_accel_em)
        ));
        md.push_str(&format!(
            "- mean(|a_em|)/mean(|a_grav|): {}\n",
            format_opt_f64(sim_summary.mean_abs_accel_ratio_em_over_grav)
        ));
        if let Some(best_vec) = vector_candidates.iter().min_by(|a, b| {
            metrics_sort_key(&a.metrics)
                .partial_cmp(&metrics_sort_key(&b.metrics))
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
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
            metrics_sort_key(&a.metrics)
                .partial_cmp(&metrics_sort_key(&b.metrics))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update equation-GA state for the next iteration: keep a small elite set and a parent.
        // We deduplicate by a normalized equation string so repeated identical models don't crowd out diversity.
        {
            let mut elite_map: std::collections::HashMap<String, CandidateMetrics> = std::collections::HashMap::new();
            for c in &vector_candidates {
                let Some(norm) =
                    normalize_equation_text_for_features(&c.equation_text, &vector_data.feature_names)
                else {
                    continue;
                };
                let m = c.metrics.clone();
                let key_new = metrics_sort_key(&m);
                let replace = match elite_map.get(&norm) {
                    Some(existing) => key_new < metrics_sort_key(existing),
                    None => true,
                };
                if replace {
                    elite_map.insert(norm, m);
                }
            }
            let mut elites: Vec<(String, CandidateMetrics)> = elite_map.into_iter().collect();
            elites.sort_by(|a, b| {
                metrics_sort_key(&a.1)
                    .partial_cmp(&metrics_sort_key(&b.1))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            equation_ga_parent = elites.first().map(|(eq, _)| eq.clone());
            previous_iteration_elites = elites.into_iter().take(3).collect();
        }

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
            if !feature_library_pinned {
                if let Some(kind) = j
                    .recommendations
                    .next_feature_library
                    .as_deref()
                    .and_then(parse_feature_library_kind)
                {
                    current_library = kind;
                }
            }
            if let Some(next) = j.recommendations.next_initial_conditions {
                next_ic = Some(next);
            }
            if let Some(eq) = j.recommendations.next_manual_equation_text {
                let trimmed = eq.trim();
                if !trimmed.is_empty() {
                    next_manual_equation_text = Some(trimmed.to_string());
                }
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
    eval_notes.push(format!("feature_library={}", feature_library_trimmed));
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
        let feature_names = feature_names_for_equation_text(&prior.best.candidate.equation_text);
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

    let benchmark_eval: Option<BenchmarkEvalV1> = current_best
        .as_ref()
        .and_then(|best| compute_benchmark_eval_v1(out_dir.as_path(), best, incumbent_bucket).ok());
    if let Some(b) = benchmark_eval.as_ref() {
        let _ = fs::write(out_dir.join("benchmark_eval.json"), serde_json::to_string_pretty(b)?);
    }

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

    let eval_prompt_path = out_dir.join("evaluation_prompt.txt");
    let eval_md_path = out_dir.join("evaluation.md");
    let eval_llm_md_path = out_dir.join("evaluation_llm.md");
    let exec_md = render_executive_summary_md(
        &eval_input,
        current_best.as_ref(),
        prior_best.as_ref(),
        best_ic.as_ref(),
        incumbent_on_best_run.as_ref(),
        benchmark_eval.as_ref(),
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

    let eval_tex_path = out_dir.join("evaluation.tex");
    let eval_tex = render_evaluation_tex(
        &eval_input,
        current_best.as_ref(),
        prior_best.as_ref(),
        best_ic.as_ref(),
        EvaluationTexOptions {
            factory_dir: Some(out_dir.as_path()),
            benchmark: benchmark_eval.as_ref(),
            incumbent_on_best_run: incumbent_on_best_run.as_ref(),
        },
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

    update_best_results_index_best_effort(std::path::Path::new("results"));

    println!("Factory evaluation written -> {}", eval_md_path.display());
    Ok(())
}

fn benchmark_rollout_integrator_for_regime(regime: &str) -> RolloutIntegrator {
    match regime {
        "gravity_only" => RolloutIntegrator::Leapfrog,
        _ => RolloutIntegrator::Euler,
    }
}

fn benchmark_suite_v1(regime: &str, bounds: &IcBounds) -> Vec<(String, InitialConditionSpec)> {
    let mut cases: Vec<(String, InitialConditionSpec)> = Vec::new();
    match regime {
        "gravity_only" => {
            cases.push((
                "equilateral_tilt".to_string(),
                InitialConditionSpec {
                    bodies: vec![
                        threebody_discover::BodyInit {
                            mass: 1.0,
                            charge: 0.0,
                            pos: [0.577_350_269_189_625_8, 0.0, 0.2],
                            vel: [0.0, 1.0, 0.05],
                        },
                        threebody_discover::BodyInit {
                            mass: 1.0,
                            charge: 0.0,
                            pos: [-0.288_675_134_594_812_9, 0.5, -0.1],
                            vel: [-0.866_025_403_784_438_6, -0.5, -0.025],
                        },
                        threebody_discover::BodyInit {
                            mass: 1.0,
                            charge: 0.0,
                            pos: [-0.288_675_134_594_812_9, -0.5, -0.1],
                            vel: [0.866_025_403_784_438_6, -0.5, -0.025],
                        },
                    ],
                    barycentric: true,
                    notes: "benchmark_suite_v1 gravity_only equilateral_tilt".to_string(),
                },
            ));
            cases.push((
                "two_body_perturber_3d".to_string(),
                InitialConditionSpec {
                    bodies: vec![
                        threebody_discover::BodyInit {
                            mass: 1.0,
                            charge: 0.0,
                            pos: [-0.5, 0.0, 0.1],
                            vel: [0.0, 0.7, 0.05],
                        },
                        threebody_discover::BodyInit {
                            mass: 1.0,
                            charge: 0.0,
                            pos: [0.5, 0.0, -0.1],
                            vel: [0.0, -0.7, -0.05],
                        },
                        threebody_discover::BodyInit {
                            mass: 0.1,
                            charge: 0.0,
                            pos: [0.0, 0.3, 0.0],
                            vel: [0.0, 0.0, 0.0],
                        },
                    ],
                    barycentric: true,
                    notes: "benchmark_suite_v1 gravity_only two_body_perturber_3d".to_string(),
                },
            ));
            cases.push((
                "skew_triangle_3d".to_string(),
                InitialConditionSpec {
                    bodies: vec![
                        threebody_discover::BodyInit {
                            mass: 1.5,
                            charge: 0.0,
                            pos: [-0.7, 0.0, 0.15],
                            vel: [0.0, -0.3, 0.02],
                        },
                        threebody_discover::BodyInit {
                            mass: 1.0,
                            charge: 0.0,
                            pos: [0.7, 0.0, -0.05],
                            vel: [0.0, 0.45, -0.01],
                        },
                        threebody_discover::BodyInit {
                            mass: 0.8,
                            charge: 0.0,
                            pos: [0.0, 1.0, -0.1],
                            vel: [0.0, -0.1875, 0.0],
                        },
                    ],
                    barycentric: true,
                    notes: "benchmark_suite_v1 gravity_only skew_triangle_3d".to_string(),
                },
            ));
        }
        "em_quasistatic" => {
            // A small, deterministic suite intended to make EM effects non-trivial while staying within bounds.
            cases.push((
                "em_like_charges_fast".to_string(),
                InitialConditionSpec {
                    bodies: vec![
                        threebody_discover::BodyInit {
                            mass: 1.0,
                            charge: 1.0,
                            pos: [-0.8, 0.0, 0.2],
                            vel: [0.0, 1.0, 0.1],
                        },
                        threebody_discover::BodyInit {
                            mass: 1.0,
                            charge: 1.0,
                            pos: [0.8, 0.0, -0.2],
                            vel: [0.0, -1.0, -0.1],
                        },
                        threebody_discover::BodyInit {
                            mass: 0.8,
                            charge: 1.0,
                            pos: [0.0, 1.0, 0.0],
                            vel: [0.0, 0.0, 0.0],
                        },
                    ],
                    barycentric: true,
                    notes: "benchmark_suite_v1 em_quasistatic em_like_charges_fast".to_string(),
                },
            ));
            cases.push((
                "em_alternating_signs".to_string(),
                InitialConditionSpec {
                    bodies: vec![
                        threebody_discover::BodyInit {
                            mass: 1.0,
                            charge: 1.0,
                            pos: [-0.6, 0.0, 0.15],
                            vel: [0.0, 0.9, 0.05],
                        },
                        threebody_discover::BodyInit {
                            mass: 1.0,
                            charge: -1.0,
                            pos: [0.6, 0.0, -0.15],
                            vel: [0.0, -0.9, -0.05],
                        },
                        threebody_discover::BodyInit {
                            mass: 1.2,
                            charge: 1.0,
                            pos: [0.0, 1.0, 0.0],
                            vel: [0.0, 0.0, 0.0],
                        },
                    ],
                    barycentric: true,
                    notes: "benchmark_suite_v1 em_quasistatic em_alternating_signs".to_string(),
                },
            ));
            cases.push((
                "em_high_ang_momentum_3d".to_string(),
                InitialConditionSpec {
                    bodies: vec![
                        threebody_discover::BodyInit {
                            mass: 1.0,
                            charge: 0.6,
                            pos: [-0.9, 0.0, 0.25],
                            vel: [0.0, 1.2, 0.0],
                        },
                        threebody_discover::BodyInit {
                            mass: 1.0,
                            charge: -0.6,
                            pos: [0.9, 0.0, -0.25],
                            vel: [0.0, -1.2, 0.0],
                        },
                        threebody_discover::BodyInit {
                            mass: 0.5,
                            charge: 0.3,
                            pos: [0.0, 0.9, 0.0],
                            vel: [0.0, 0.0, 0.0],
                        },
                    ],
                    barycentric: true,
                    notes: "benchmark_suite_v1 em_quasistatic em_high_ang_momentum_3d".to_string(),
                },
            ));
        }
        _ => {}
    }

    // Final safety pass: ensure 3D and clamp within bounds.
    for (i, (_name, spec)) in cases.iter_mut().enumerate() {
        *spec = force_nonplanar_ic(spec.clone(), bounds, 1 + i as u64);
    }
    cases
}

fn compute_benchmark_eval_v1(
    factory_dir: &std::path::Path,
    best: &BestFactoryCandidate,
    bucket: BucketKey,
) -> anyhow::Result<BenchmarkEvalV1> {
    let bounds = default_ic_bounds();
    let suite_id = "benchmark_suite_v1".to_string();
    let rollout_integrator = benchmark_rollout_integrator_for_regime(&best.regime);
    let rollout_integrator_label = rollout_integrator_label(rollout_integrator).to_string();

    let run_dir = factory_dir.join(&best.run_id);
    let cfg_json = fs::read_to_string(run_dir.join("config.json"))?;
    let mut cfg: Config = serde_json::from_str(&cfg_json)?;
    if matches!(cfg.integrator.kind, IntegratorKind::Rk45) && cfg.integrator.adaptive {
        cfg.integrator.max_rejects = cfg.integrator.max_rejects.max(64);
    }
    cfg.validate().map_err(anyhow::Error::msg)?;

    let model = vector_model_from_equation_text(&best.candidate.equation_text)
        .ok_or_else(|| anyhow::anyhow!("failed to parse best equation into a model"))?;
    let feature_names = feature_names_for_equation_text(&best.candidate.equation_text);

    let suite = benchmark_suite_v1(&best.regime, &bounds);
    let mut cases: Vec<BenchmarkCaseEval> = Vec::new();
    let mut rmses: Vec<f64> = Vec::new();
    let mut divergence_min = f64::INFINITY;
    let mut notes: Vec<String> = Vec::new();

    for (case_name, ic_spec) in suite.into_iter() {
        let system = system_from_ic(&ic_spec, &bounds)?;
        let result = simulate_with_cfg(system, &cfg, SimOptions { steps: bucket.steps, dt: bucket.dt });
        let sim_summary = build_sim_summary(
            &result,
            &cfg,
            rollout_integrator,
            Some(bucket.steps),
            Some(bucket.dt),
        );

        let usable = !result.terminated_early && result.steps.len() == bucket.steps + 1;
        let (rollout_rmse, divergence_time) = if usable {
            let (rmse, div) = rollout_metrics(&model, &feature_names, &result, &cfg, rollout_integrator);
            if rmse.is_finite() {
                rmses.push(rmse);
            }
            divergence_min = divergence_min.min(div.unwrap_or(f64::INFINITY));
            (Some(rmse), div)
        } else {
            notes.push(format!(
                "benchmark_case_unusable name={} terminated_early={} steps={}",
                case_name,
                result.terminated_early,
                result.steps.len()
            ));
            (None, None)
        };

        cases.push(BenchmarkCaseEval {
            name: case_name,
            initial_conditions: ic_spec,
            simulation: sim_summary,
            rollout_rmse,
            divergence_time,
        });
    }

    let expected_cases = cases.len();
    let all_cases_ok = rmses.len() == expected_cases && expected_cases > 0;
    let rmse_mean = all_cases_ok.then(|| rmses.iter().sum::<f64>() / rmses.len() as f64);
    let rmse_max = all_cases_ok.then(|| rmses.iter().copied().fold(0.0, f64::max));
    let divergence_time_min = all_cases_ok
        .then(|| divergence_min)
        .and_then(|v| v.is_finite().then_some(v))
        .or_else(|| all_cases_ok.then_some(f64::INFINITY).and_then(|_| None));

    let aggregate = BenchmarkAggregate {
        suite_id: suite_id.clone(),
        rollout_integrator: rollout_integrator_label.clone(),
        cases: expected_cases,
        rmse_mean,
        rmse_max,
        divergence_time_min,
    };

    Ok(BenchmarkEvalV1 {
        version: "v1".to_string(),
        suite_id,
        bucket,
        regime: best.regime.clone(),
        rollout_integrator: rollout_integrator_label,
        candidate: BenchmarkCandidateRef {
            run_id: best.run_id.clone(),
            candidate_id: best.candidate.id,
            equation_text: best.candidate.equation_text.clone(),
            complexity: best.candidate.metrics.complexity,
        },
        cases,
        aggregate,
        notes,
    })
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

fn best_record_sort_key(record: &BestRecord) -> (f64, f64, usize) {
    let rollout_rmse = record
        .benchmark
        .as_ref()
        .and_then(|b| b.rmse_mean)
        .or(record.metrics.rollout_rmse)
        .unwrap_or(f64::INFINITY);
    let rollout_rmse = if rollout_rmse.is_finite() { rollout_rmse } else { f64::INFINITY };
    let mse = if record.metrics.mse.is_finite() { record.metrics.mse } else { f64::INFINITY };
    (rollout_rmse, mse, record.metrics.complexity)
}

fn record_effective_rollout_rmse(record: &BestRecord) -> Option<f64> {
    record
        .benchmark
        .as_ref()
        .and_then(|b| b.rmse_mean)
        .or(record.metrics.rollout_rmse)
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
struct BenchmarkAggregate {
    suite_id: String,
    rollout_integrator: String,
    cases: usize,
    rmse_mean: Option<f64>,
    rmse_max: Option<f64>,
    divergence_time_min: Option<f64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
struct BenchmarkCandidateRef {
    run_id: String,
    candidate_id: usize,
    equation_text: String,
    complexity: usize,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
struct BenchmarkCaseEval {
    name: String,
    initial_conditions: InitialConditionSpec,
    simulation: SimulationSummary,
    rollout_rmse: Option<f64>,
    divergence_time: Option<f64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
struct BenchmarkEvalV1 {
    version: String,
    suite_id: String,
    bucket: BucketKey,
    regime: String,
    rollout_integrator: String,
    candidate: BenchmarkCandidateRef,
    cases: Vec<BenchmarkCaseEval>,
    aggregate: BenchmarkAggregate,
    notes: Vec<String>,
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
    #[serde(default)]
    benchmark: Option<BenchmarkAggregate>,
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
    let best_run_id = best.run_id.clone();

    // Avoid apples-to-oranges comparisons: if the oracle simulation in the best run terminated
    // early or did not reach the requested horizon, skip this attempt entirely. Otherwise a
    // short/truncated rollout can look artificially "better" (lower error) and pollute the
    // global best index.
    if let Some(iter) = input.iterations.iter().find(|it| it.run_id == best_run_id) {
        if let Some(sim) = iter.simulation.as_ref() {
            if sim.terminated_early {
                return None;
            }
            let expected_len = bucket.steps.checked_add(1)?;
            if sim.steps != expected_len {
                return None;
            }
            if let Some(req_steps) = sim.requested_steps {
                if req_steps != bucket.steps {
                    return None;
                }
            }
            if let Some(req_dt) = sim.requested_dt {
                if (req_dt - bucket.dt).abs() > 1e-12 {
                    return None;
                }
            }
        }
    }

    let factory_dir = path.parent()?.to_path_buf();
    let run_dir = factory_dir
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .to_path_buf();

    let ic: Option<InitialConditionSpec> = fs::read_to_string(factory_dir.join(&best.run_id).join("initial_conditions.json"))
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok());

    let benchmark: Option<BenchmarkAggregate> = fs::read_to_string(factory_dir.join("benchmark_eval.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<BenchmarkEvalV1>(&raw).ok())
        .and_then(|bm| {
            if bm.bucket.matches(&bucket) && bm.regime == best.regime {
                Some(bm.aggregate)
            } else {
                None
            }
        });

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
        benchmark,
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
        let key = best_record_sort_key(&record);
        let existing_idx = buckets
            .iter()
            .position(|e| e.bucket.matches(&record.bucket));
        if let Some(idx) = existing_idx {
            let existing_key = best_record_sort_key(&buckets[idx]);
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
            "Selection rule: minimize (benchmark_rmse_mean when present, else rollout_rmse, then mse, complexity)."
                .to_string(),
            "Buckets are grouped by (steps, dt).".to_string(),
            "Only */factory/evaluation_input.json files are considered.".to_string(),
            "Runs are skipped if the oracle simulation terminated early or did not reach the requested horizon (steps+1)."
                .to_string(),
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

fn render_best_results_md(index: &BestResultsIndexV1, progress: &[BucketProgress]) -> String {
    let mut md = String::new();
    md.push_str("# Best Results (auto-updated)\n\n");
    md.push_str(&format!("Last updated: `{}`\n\n", index.updated_at_utc));

    md.push_str("## Executive Summary\n\n");
    md.push_str("- This file is the single top-level place to see the best discovered equations so far.\n");
    md.push_str("- Comparator: minimize `benchmark_rmse_mean` (when present), else `rollout_rmse`, then `mse`, then `complexity`.\n");
    md.push_str("- Buckets: grouped by `(steps, dt)` to avoid apples-to-oranges comparisons.\n");
    md.push_str("- Safety: runs are skipped if the oracle simulation terminated early or did not reach `steps+1` samples.\n");
    if index.buckets.is_empty() {
        md.push_str("\nNo runs found yet under `results/**/factory/evaluation_input.json`.\n");
        return md;
    }
    md.push_str("\nTracked buckets:\n");
    for rec in &index.buckets {
        md.push_str(&format!("- steps={}, dt={}\n", rec.bucket.steps, rec.bucket.dt));
    }

    let highlight = index
        .buckets
        .iter()
        .max_by(|a, b| a.bucket.steps.cmp(&b.bucket.steps).then_with(|| {
            a.bucket
                .dt
                .partial_cmp(&b.bucket.dt)
                .unwrap_or(std::cmp::Ordering::Equal)
        }));
    if let Some(best) = highlight {
        md.push_str("\n## Most Realistic / Longest-Horizon Bucket (Largest steps)\n\n");
        md.push_str(&format!(
            "- Bucket: steps={}, dt={}\n",
            best.bucket.steps, best.bucket.dt
        ));
        if let Some(bm) = best.benchmark.as_ref() {
            md.push_str(&format!(
                "- Benchmark: rmse_mean={}, rmse_max={}, rollout_integrator={}, suite_id={}\n",
                format_metric_opt(bm.rmse_mean),
                format_metric_opt(bm.rmse_max),
                bm.rollout_integrator,
                bm.suite_id
            ));
        }
        md.push_str(&format!(
            "- Run metrics: rollout_rmse={}, mse={:.6e}, complexity={}, divergence_time={}\n",
            format_metric_opt(best.metrics.rollout_rmse),
            best.metrics.mse,
            best.metrics.complexity,
            format_metric_opt(best.metrics.divergence_time),
        ));
        md.push_str("- Best equation (exact):\n");
        md.push_str("```text\n");
        md.push_str(&best.equation_text);
        md.push_str("\n```\n");
        md.push_str(&render_expanded_math_md(&best.equation_text));
        md.push_str(&format!("- Evidence: `{}/RESULTS.md`\n", best.run_dir));
        md.push_str(&format!("- Raw: `{}`\n", best.eval_input_path));
        md.push_str("\n");
    }

    md.push_str("\n## Current Incumbents\n\n");
    for rec in &index.buckets {
        md.push_str(&format!("### steps={}, dt={}\n\n", rec.bucket.steps, rec.bucket.dt));
        if let Some(bm) = rec.benchmark.as_ref() {
            md.push_str(&format!(
                "- Benchmark: rmse_mean={}, rmse_max={}, rollout_integrator={}, suite_id={}\n",
                format_metric_opt(bm.rmse_mean),
                format_metric_opt(bm.rmse_max),
                bm.rollout_integrator,
                bm.suite_id
            ));
        }
        md.push_str(&format!(
            "- Run metrics: rollout_rmse={}, mse={:.6e}, complexity={}, divergence_time={}\n",
            format_metric_opt(rec.metrics.rollout_rmse),
            rec.metrics.mse,
            rec.metrics.complexity,
            format_metric_opt(rec.metrics.divergence_time),
        ));
        md.push_str("- Best equation (exact):\n");
        md.push_str("```text\n");
        md.push_str(&rec.equation_text);
        md.push_str("\n```\n\n");

        if let Some(p) = progress.iter().find(|p| p.bucket.matches(&rec.bucket)) {
            let mut alts: Vec<&BestRecord> = p
                .top_unique
                .iter()
                .filter(|r| r.equation_text.trim() != rec.equation_text.trim())
                .collect();
            alts.truncate(3);
            if !alts.is_empty() {
                md.push_str("- Other equations seen in this bucket (best per unique equation; different ICs):\n");
                for alt in alts {
                    let eq = truncate_to_chars(&single_line(&alt.equation_text), 240);
                    md.push_str(&format!(
                        "  - score={}, rollout_rmse={}, mse={:.6e}, complexity={}: `{}` (evidence: `{}/RESULTS.md`)\n",
                        format_metric_opt(record_effective_rollout_rmse(alt)),
                        format_metric_opt(alt.metrics.rollout_rmse),
                        alt.metrics.mse,
                        alt.metrics.complexity,
                        eq,
                        alt.run_dir
                    ));
                }
                md.push('\n');
            }
        }

        md.push_str(&format!("- Evidence: `{}/RESULTS.md`\n", rec.run_dir));
        md.push_str(&format!("- Raw: `{}`\n\n", rec.eval_input_path));
    }

    md.push_str("## Did we improve (over local history)?\n\n");
    if progress.is_empty() {
        md.push_str("- n/a (no comparable history found yet)\n\n");
    } else {
        md.push_str("- Per bucket, compares the first recorded attempt vs current best.\n");
        md.push_str("- Note: if initial conditions differ between runs, this is suggestive, not definitive.\n\n");
        for p in progress {
            let first = record_effective_rollout_rmse(&p.first).unwrap_or(f64::INFINITY);
            let best = record_effective_rollout_rmse(&p.best).unwrap_or(f64::INFINITY);
            let improvement = if first.is_finite() && best.is_finite() && first > 0.0 {
                let delta = first - best;
                if delta.abs() <= 1e-12 {
                    "no change".to_string()
                } else {
                    format!("{:+.6} ({:+.1}%)", delta, (delta / first) * 100.0)
                }
            } else {
                "n/a".to_string()
            };
            md.push_str(&format!(
                "- steps={}, dt={}: runs={}, first_score={}, best_score={}, improvement={}\n",
                p.bucket.steps,
                p.bucket.dt,
                p.attempts,
                format_metric_opt(record_effective_rollout_rmse(&p.first)),
                format_metric_opt(record_effective_rollout_rmse(&p.best)),
                improvement
            ));
        }
        md.push('\n');
    }

    md.push_str("## Next Experiment (for humans and LLMs)\n\n");
    md.push_str("- Goal: reduce `rollout_rmse` without increasing `complexity`.\n");
    md.push_str("- If `mse` is tiny but `rollout_rmse` is large, the model fits accelerations but rolls out poorly (instability).\n");
    md.push_str("- If EM is enabled but the best equation stays gravity-only, check the EM/gravity signal ratio in `RESULTS.md`.\n");
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
    top_unique: Vec<BestRecord>,
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

        // Also compute a small "equation hall of fame" for this bucket: best attempt per unique equation.
        // This makes it easy for humans (and downstream tools) to see what different math was actually tried.
        let mut unique: std::collections::HashMap<String, BestRecord> = std::collections::HashMap::new();
        for mut rec in bucket_records.clone() {
            let feature_names = feature_names_for_equation_text(&rec.equation_text);
            let norm = normalize_equation_text_for_features(&rec.equation_text, &feature_names)
                .unwrap_or_else(|| single_line(&rec.equation_text));
            rec.equation_text = norm.clone();
            let key_new = best_record_sort_key(&rec);
            let replace = match unique.get(&norm) {
                Some(existing) => key_new < best_record_sort_key(existing),
                None => true,
            };
            if replace {
                unique.insert(norm, rec);
            }
        }
        let mut top_unique: Vec<BestRecord> = unique.into_values().collect();
        top_unique.sort_by(|a, b| {
            best_record_sort_key(a)
                .partial_cmp(&best_record_sort_key(b))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.eval_input_path.cmp(&b.eval_input_path))
        });
        top_unique.truncate(5);

        out.push(BucketProgress {
            bucket: best.bucket.clone(),
            attempts: bucket_records.len(),
            first,
            best: best.clone(),
            top_unique,
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
    tex.push_str(
        "Comparator: minimize benchmark\\_rmse\\_mean (when available), else rollout\\_rmse, then mse, then complexity (lower is better).\\\\\n",
    );
    tex.push_str("Buckets are grouped by (steps, dt) to avoid apples-to-oranges comparisons.\\\\\n\n");
    tex.push_str("Safety: runs are skipped if the oracle simulation terminated early or did not reach the requested horizon.\\\\\n\n");
    tex.push_str(
        "For each bucket, we also embed small excerpts from the run’s evaluation logs (executive summary + optional LLM narrative) to aid interpretation without chasing files.\\\\\n\n",
    );

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
    tex.push_str("When a held-out benchmark is available, the table uses benchmark\\_rmse\\_mean; otherwise it uses rollout\\_rmse.\\\\\n");
    tex.push_str("\\begin{longtable}{lrrrr}\n");
    tex.push_str("\\toprule\n");
    tex.push_str("Bucket & Runs & First score & Best score & Improvement\\\\\n");
    tex.push_str("\\midrule\n");
    for p in progress {
        let first = record_effective_rollout_rmse(&p.first).unwrap_or(f64::INFINITY);
        let best = record_effective_rollout_rmse(&p.best).unwrap_or(f64::INFINITY);
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
            format_opt_f64(record_effective_rollout_rmse(&p.first)),
            format_opt_f64(record_effective_rollout_rmse(&p.best)),
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

        if let Some(p) = progress.iter().find(|p| p.bucket.matches(&rec.bucket)) {
            let mut alts: Vec<&BestRecord> = p
                .top_unique
                .iter()
                .filter(|r| r.equation_text.trim() != rec.equation_text.trim())
                .collect();
            alts.truncate(2);
            if !alts.is_empty() {
                tex.push_str("\\paragraph{Other equations seen (same bucket; different ICs)}\n");
                tex.push_str("\\begin{itemize}\n");
                for alt in alts {
                    let eq = truncate_to_chars(&single_line(&alt.equation_text), 220);
                    tex.push_str(&format!(
                        "  \\item score={}, rollout\\_rmse={}, mse={:.6e}, complexity={}: \\texttt{{{}}}.\\\\\n",
                        format_opt_f64(record_effective_rollout_rmse(alt)),
                        format_opt_f64(alt.metrics.rollout_rmse),
                        alt.metrics.mse,
                        alt.metrics.complexity,
                        escape_latex(&eq),
                    ));
                }
                tex.push_str("\\end{itemize}\n\n");
            }
        }

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

        tex.push_str("\\paragraph{Run Logs (Embedded Excerpts)}\n");
        let factory_dir = PathBuf::from(&rec.factory_dir);
        let best_iter_dir = factory_dir.join(&rec.run_id);
        append_log_excerpt_tex(
            &mut tex,
            "evaluation.md (deterministic executive summary)",
            &factory_dir.join("evaluation.md"),
            8000,
        );
        append_log_excerpt_tex(
            &mut tex,
            "evaluation_llm.md (optional narrative; not used for numeric selection)",
            &factory_dir.join("evaluation_llm.md"),
            8000,
        );
        append_log_excerpt_tex(
            &mut tex,
            "evaluation_error.txt (present only if LLM evaluation failed and fallback was used)",
            &factory_dir.join("evaluation_error.txt"),
            2000,
        );
        append_log_excerpt_tex(
            &mut tex,
            "run_###/report.md (per-iteration summary for the best run_id)",
            &best_iter_dir.join("report.md"),
            3000,
        );
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

fn append_log_excerpt_tex(tex: &mut String, label: &str, path: &std::path::Path, max_chars: usize) {
    tex.push_str(&format!("\\subparagraph{{{}}}\\\\\n", escape_latex(label)));
    tex.push_str(&format!(
        "Source: \\texttt{{{}}}.\\\\\n",
        escape_latex(&path.display().to_string())
    ));
    let raw = match fs::read_to_string(path) {
        Ok(v) => v,
        Err(_) => {
            tex.push_str("Missing.\\\\\n\n");
            return;
        }
    };
    let excerpt = safe_verbatim_excerpt(&raw, max_chars, 110);
    tex.push_str("\\begin{verbatim}\n");
    tex.push_str(&excerpt);
    tex.push_str("\n\\end{verbatim}\n\n");
}

fn safe_verbatim_excerpt(input: &str, max_chars: usize, wrap_width: usize) -> String {
    let truncated = truncate_to_chars(input, max_chars);
    let sanitized = truncated
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace("\\end{verbatim}", "\\\\end{verbatim}")
        .replace("\\begin{verbatim}", "\\\\begin{verbatim}");
    let wrapped = wrap_long_lines(&sanitized, wrap_width);
    ensure_verbatim_safe_lines(&wrapped)
}

fn wrap_long_lines(input: &str, width: usize) -> String {
    if width == 0 {
        return input.to_string();
    }
    let mut out = String::new();
    for (li, line) in input.lines().enumerate() {
        if li > 0 {
            out.push('\n');
        }
        out.push_str(&wrap_one_line(line, width));
    }
    out
}

fn ensure_verbatim_safe_lines(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for (li, line) in input.lines().enumerate() {
        if li > 0 {
            out.push('\n');
        }
        let trimmed = line.trim_start_matches(|c: char| c.is_whitespace());
        if trimmed.starts_with("\\end{verbatim}") || trimmed.starts_with("\\begin{verbatim}") {
            let prefix_len = line.len() - trimmed.len();
            let (prefix, rest) = line.split_at(prefix_len);
            out.push_str(prefix);
            out.push('\\');
            out.push_str(rest);
        } else {
            out.push_str(line);
        }
    }
    out
}

fn wrap_one_line(mut line: &str, width: usize) -> String {
    if width == 0 {
        return line.to_string();
    }
    let mut out = String::new();
    loop {
        let len = line.chars().count();
        if len <= width {
            out.push_str(line);
            break;
        }

        let mut break_byte: Option<usize> = None;
        for (ci, (byte, ch)) in line.char_indices().enumerate() {
            if ci >= width {
                break;
            }
            if ch.is_whitespace() {
                break_byte = Some(byte);
            }
        }

        let (head, tail) = match break_byte {
            Some(b) if b > 0 => (&line[..b], line[b..].trim_start_matches(|c: char| c.is_whitespace())),
            _ => {
                let b = line
                    .char_indices()
                    .nth(width)
                    .map(|(i, _)| i)
                    .unwrap_or(line.len());
                (&line[..b], &line[b..])
            }
        };
        out.push_str(head);
        out.push('\n');
        line = tail;
        if line.is_empty() {
            break;
        }
    }
    out
}

fn write_best_results(results_root: &std::path::Path, index: &BestResultsIndexV1) -> anyhow::Result<()> {
    fs::create_dir_all(results_root)?;
    let json_path = results_root.join("best_results.json");
    let md_path = results_root.join("best_results.md");
    let progress = compute_bucket_progress(results_root, index).unwrap_or_default();
    fs::write(&json_path, serde_json::to_string_pretty(index)?)?;
    fs::write(&md_path, render_best_results_md(index, &progress))?;
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
    out.push_str("...");
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
    let benchmark = record.benchmark.as_ref().and_then(|b| b.rmse_mean);
    vec![
        format!("INCUMBENT_BEST_EQUATION: {eq}"),
        format!(
            "INCUMBENT_BENCHMARK_RMSE_MEAN: {}",
            format_metric_opt(benchmark)
        ),
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

fn feature_names_for_equation_text(equation_text: &str) -> Vec<String> {
    let default = FeatureLibrary::default_physics();
    let default_set: std::collections::HashSet<&str> = default.features.iter().map(|s| s.as_str()).collect();
    let [tx, ty, tz] = parse_vector_equation_terms(equation_text);
    let mut uses_extended = false;
    for (_, f) in tx.iter().chain(ty.iter()).chain(tz.iter()) {
        if !default_set.contains(f.as_str()) {
            uses_extended = true;
            break;
        }
    }
    if uses_extended {
        FeatureLibrary::extended_physics().features
    } else {
        default.features
    }
}

fn normalize_equation_text_for_features(equation_text: &str, feature_names: &[String]) -> Option<String> {
    let mut model = vector_model_from_equation_text(equation_text)?;
    let allowed: std::collections::HashSet<&str> = feature_names.iter().map(|s| s.as_str()).collect();
    for eq in [&mut model.eq_x, &mut model.eq_y, &mut model.eq_z] {
        eq.terms
            .retain(|t| t.coeff.is_finite() && allowed.contains(t.feature.as_str()));
        eq.terms.sort_by(|a, b| a.feature.cmp(&b.feature));
    }
    Some(format_vector_model(&model))
}

fn feature_pool_for_axis(feature_names: &[String], axis: char, regime: &str) -> Vec<String> {
    fn axis_ok(name: &str, axis: char) -> bool {
        name == "gate_close" || name == "gate_far" || name.ends_with(&format!("_{axis}"))
    }
    fn regime_ok(name: &str, regime: &str) -> bool {
        if regime == "gravity_only" {
            name.starts_with("grav") || name.starts_with("gate")
        } else {
            true
        }
    }

    feature_names
        .iter()
        .map(|s| s.as_str())
        .filter(|name| axis_ok(name, axis) && regime_ok(name, regime))
        .map(|s| s.to_string())
        .collect()
}

fn format_terms_for_equation_text(mut terms: Vec<(f64, String)>) -> String {
    terms.retain(|(c, f)| c.is_finite() && !f.trim().is_empty());
    if terms.is_empty() {
        return "0".to_string();
    }
    terms.sort_by(|a, b| a.1.cmp(&b.1));
    terms
        .into_iter()
        .map(|(c, f)| format!("{:+.6}*{f}", c))
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_vector_equation_text(terms: [Vec<(f64, String)>; 3]) -> String {
    let [tx, ty, tz] = terms;
    format!(
        "ax={} ; ay={} ; az={}",
        format_terms_for_equation_text(tx),
        format_terms_for_equation_text(ty),
        format_terms_for_equation_text(tz)
    )
}

fn propose_equation_mutants(
    parent_equation_text: &str,
    feature_names: &[String],
    regime: &str,
    seed: u64,
) -> Vec<String> {
    use std::collections::{HashMap, HashSet};

    fn to_map(terms: Vec<(f64, String)>) -> HashMap<String, f64> {
        let mut map: HashMap<String, f64> = HashMap::new();
        for (coeff, feature) in terms {
            if !coeff.is_finite() || feature.trim().is_empty() {
                continue;
            }
            *map.entry(feature).or_insert(0.0) += coeff;
        }
        map.retain(|_, v| v.is_finite() && v.abs() > 1e-12);
        map
    }

    fn map_to_terms(map: HashMap<String, f64>) -> Vec<(f64, String)> {
        let mut out: Vec<(f64, String)> = map
            .into_iter()
            .filter(|(_f, c)| c.is_finite() && c.abs() > 1e-12)
            .map(|(f, c)| (c, f))
            .collect();
        out.sort_by(|a, b| a.1.cmp(&b.1));
        out
    }

    fn choose_index(rng: &mut threebody_discover::ga::Lcg, len: usize) -> Option<usize> {
        if len == 0 {
            None
        } else {
            Some(rng.gen_range_usize(0, len - 1))
        }
    }

    fn mutate_axis(
        map: &mut HashMap<String, f64>,
        pool: &[String],
        rng: &mut threebody_discover::ga::Lcg,
        max_terms: usize,
    ) {
        let r = rng.gen_range_f64(0.0, 1.0);
        if r < 0.45 {
            // Coefficient jitter.
            let keys: Vec<String> = map.keys().cloned().collect();
            if let Some(idx) = choose_index(rng, keys.len()) {
                let k = &keys[idx];
                let factor = 1.0 + rng.gen_range_f64(-0.15, 0.15);
                if let Some(v) = map.get_mut(k) {
                    *v *= factor;
                    if !v.is_finite() || v.abs() <= 1e-8 {
                        map.remove(k);
                    }
                }
            }
        } else if r < 0.80 {
            // Add a term.
            if !pool.is_empty() {
                for _ in 0..8 {
                    let idx = rng.gen_range_usize(0, pool.len() - 1);
                    let feat = &pool[idx];
                    if map.contains_key(feat) {
                        continue;
                    }
                    let coeff = rng.gen_range_f64(-1.5, 1.5);
                    if coeff.is_finite() && coeff.abs() > 1e-8 {
                        map.insert(feat.clone(), coeff);
                    }
                    break;
                }
            }
        } else {
            // Remove a term.
            let keys: Vec<String> = map.keys().cloned().collect();
            if let Some(idx) = choose_index(rng, keys.len()) {
                map.remove(&keys[idx]);
            }
        }

        // Keep models small.
        while map.len() > max_terms {
            let keys: Vec<String> = map.keys().cloned().collect();
            if let Some(idx) = choose_index(rng, keys.len()) {
                map.remove(&keys[idx]);
            } else {
                break;
            }
        }
    }

    let pool_x = feature_pool_for_axis(feature_names, 'x', regime);
    let pool_y = feature_pool_for_axis(feature_names, 'y', regime);
    let pool_z = feature_pool_for_axis(feature_names, 'z', regime);

    let base_terms = parse_vector_equation_terms(parent_equation_text);
    let base = [
        to_map(base_terms[0].clone()),
        to_map(base_terms[1].clone()),
        to_map(base_terms[2].clone()),
    ];

    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    // Deterministic gravity-axis completion variant.
    if regime == "gravity_only" && feature_names.iter().any(|f| f == "grav_z") {
        let mut tx = base_terms[0].clone();
        let mut ty = base_terms[1].clone();
        let mut tz = base_terms[2].clone();
        let has_grav_z = tz.iter().any(|(_c, f)| f == "grav_z");
        if !has_grav_z {
            tz.push((1.0, "grav_z".to_string()));
        }
        // Keep x/y grav terms if present; otherwise add the canonical ones.
        if !tx.iter().any(|(_c, f)| f == "grav_x") && feature_names.iter().any(|f| f == "grav_x") {
            tx.push((1.0, "grav_x".to_string()));
        }
        if !ty.iter().any(|(_c, f)| f == "grav_y") && feature_names.iter().any(|f| f == "grav_y") {
            ty.push((1.0, "grav_y".to_string()));
        }
        let eq = format_vector_equation_text([tx, ty, tz]);
        if let Some(norm) = normalize_equation_text_for_features(&eq, feature_names) {
            seen.insert(norm.clone());
            out.push(norm);
        }
    }

    // Random small mutations around the parent.
    let mut rng = threebody_discover::ga::Lcg::new(seed);
    let max_terms = 4usize;
    let target = 3usize;
    let mut attempts = 0usize;
    while out.len() < target && attempts < 30 {
        attempts += 1;
        let mut mx = base[0].clone();
        let mut my = base[1].clone();
        let mut mz = base[2].clone();
        mutate_axis(&mut mx, &pool_x, &mut rng, max_terms);
        mutate_axis(&mut my, &pool_y, &mut rng, max_terms);
        mutate_axis(&mut mz, &pool_z, &mut rng, max_terms);

        let eq = format_vector_equation_text([map_to_terms(mx), map_to_terms(my), map_to_terms(mz)]);
        let Some(norm) = normalize_equation_text_for_features(&eq, feature_names) else {
            continue;
        };
        if !seen.insert(norm.clone()) {
            continue;
        }
        out.push(norm);
    }

    out
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
            encounter: sidecar.encounter.map(|e| threebody_core::sim::EncounterEvent {
                step: e.step,
                min_pair_dist: e.min_pair_dist,
                epsilon_before: e.epsilon_before,
                epsilon_after: e.epsilon_after,
                substeps_used: e.substeps_used,
            }),
            encounter_action: sidecar.encounter_action,
            warnings: sidecar.warnings,
            terminated_early: sidecar.terminated_early,
            termination_reason: sidecar.termination_reason,
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

#[derive(Clone, Copy, Debug, Default)]
struct EvaluationTexOptions<'a> {
    /// Root directory containing `evaluation.*` and `run_###/` artifacts. When present, the TeX
    /// report embeds small excerpts from these files as an appendix.
    factory_dir: Option<&'a std::path::Path>,
    /// Optional held-out benchmark evidence for this run.
    benchmark: Option<&'a BenchmarkEvalV1>,
    /// Optional apples-to-apples re-score of the prior best equation on this run's oracle.
    incumbent_on_best_run: Option<&'a IncumbentEvalOnBestRun>,
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
    opts: EvaluationTexOptions<'_>,
) -> String {
    let footnote_paths = |paths: &[String]| -> String {
        if paths.is_empty() {
            return String::new();
        }
        let mut out = String::new();
        out.push_str("\\footnote{");
        for (i, p) in paths.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("\\texttt{{{}}}", escape_latex(p)));
        }
        out.push('}');
        out
    };

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

    tex.push_str("\\section*{Candidate Equations Tried (Top 3 per Iteration)}\n");
    tex.push_str(
        "For each iteration we show the three best-scoring candidates under the numeric comparator (rollout\\_rmse, then mse, then complexity).\\\\\n\n",
    );
    for iter in &eval_input.iterations {
        tex.push_str(&format!(
            "\\subsection*{{Iteration {} (\\texttt{{{}}})}}\n",
            iter.iteration,
            escape_latex(&iter.run_id)
        ));
        if iter.top_candidates.is_empty() {
            tex.push_str("No candidates recorded.\\\\\n\n");
            continue;
        }
        for cand in &iter.top_candidates {
            let m = &cand.metrics;
            tex.push_str(&format!(
                "\\noindent\\textbf{{Candidate id={}}}: rollout\\_rmse={}, divergence\\_time={}, mse={:.6e}, complexity={}, flags=\\texttt{{{}}}.\\\\\n",
                cand.id,
                format_opt_f64(m.rollout_rmse),
                format_opt_f64(m.divergence_time),
                m.mse,
                m.complexity,
                escape_latex(&m.stability_flags.join(", "))
            ));
            tex.push_str("\\begin{verbatim}\n");
            tex.push_str(&cand.equation_text);
            tex.push_str("\n\\end{verbatim}\n\n");
        }

        if let Some(j) = iter.judge.as_ref() {
            let mut steering: Vec<String> = Vec::new();
            if let Some(v) = j.recommendations.next_feature_library.as_ref() {
                steering.push(format!("next_feature_library={v}"));
            }
            if let Some(v) = j.recommendations.next_rollout_integrator.as_ref() {
                steering.push(format!("next_rollout_integrator={v}"));
            }
            if let Some(v) = j.recommendations.next_discovery_solver.as_ref() {
                steering.push(format!("next_discovery_solver={v}"));
            }
            if let Some(v) = j.recommendations.next_normalize {
                steering.push(format!("next_normalize={v}"));
            }
            if let Some(v) = j.recommendations.next_stls_threshold {
                steering.push(format!("next_stls_threshold={v}"));
            }
            if let Some(v) = j.recommendations.next_ridge_lambda {
                steering.push(format!("next_ridge_lambda={v:.3e}"));
            }
            if let Some(v) = j.recommendations.next_lasso_alpha {
                steering.push(format!("next_lasso_alpha={v:.6}"));
            }
            if !steering.is_empty() {
                tex.push_str(&format!(
                    "\\noindent\\textbf{{LLM steering}}: \\texttt{{{}}}.\\\\\n\n",
                    escape_latex(&steering.join(", "))
                ));
            }

            if !j.summary.trim().is_empty() {
                let summary = truncate_to_chars(&single_line(&j.summary), 600);
                tex.push_str(&format!(
                    "\\noindent\\textbf{{LLM summary}}: \\texttt{{{}}}.\\\\\n\n",
                    escape_latex(&summary)
                ));
            }

            if !j.recommendations.next_search_directions.is_empty() {
                tex.push_str("\\noindent\\textbf{LLM suggested next experiments:}\\\\\n");
                tex.push_str("\\begin{itemize}\n");
                for d in &j.recommendations.next_search_directions {
                    let d = truncate_to_chars(&single_line(d), 400);
                    tex.push_str(&format!("  \\item \\texttt{{{}}}\n", escape_latex(&d)));
                }
                tex.push_str("\\end{itemize}\n\n");
            }

            if !j.recommendations.notes.trim().is_empty() {
                let notes = truncate_to_chars(&single_line(&j.recommendations.notes), 600);
                tex.push_str(&format!(
                    "\\noindent\\textbf{{LLM notes}}: \\texttt{{{}}}.\\\\\n\n",
                    escape_latex(&notes)
                ));
            }

            if let Some(eq) = j.recommendations.next_manual_equation_text.as_ref() {
                let trimmed = eq.trim();
                if !trimmed.is_empty() {
                    tex.push_str("\\noindent\\textbf{LLM proposed next equation (hypothesis):}\\\\\n");
                    tex.push_str("\\begin{verbatim}\n");
                    tex.push_str(trimmed);
                    tex.push_str("\n\\end{verbatim}\n\n");
                }
            }
        }
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

    tex.push_str("\\section*{Evidence Appendix (Embedded Excerpts)}\n");
    if let Some(factory_dir) = opts.factory_dir {
        append_log_excerpt_tex(
            &mut tex,
            "evaluation.md (deterministic executive summary)",
            &factory_dir.join("evaluation.md"),
            8000,
        );
        append_log_excerpt_tex(
            &mut tex,
            "evaluation_llm.md (optional narrative; not used for numeric selection)",
            &factory_dir.join("evaluation_llm.md"),
            8000,
        );
        append_log_excerpt_tex(
            &mut tex,
            "evaluation_prompt.txt (prompt used to generate evaluation_llm.md when enabled)",
            &factory_dir.join("evaluation_prompt.txt"),
            4000,
        );
        append_log_excerpt_tex(
            &mut tex,
            "evaluation_error.txt (present only if LLM evaluation failed and fallback was used)",
            &factory_dir.join("evaluation_error.txt"),
            2000,
        );
        append_log_excerpt_tex(
            &mut tex,
            "evaluation_history.json (best-vs-prior summary over local history)",
            &factory_dir.join("evaluation_history.json"),
            4000,
        );
        append_log_excerpt_tex(
            &mut tex,
            "benchmark_eval.json (held-out suite; present only when computed)",
            &factory_dir.join("benchmark_eval.json"),
            6000,
        );

        if let Some(best) = current_best {
            let run_dir = factory_dir.join(&best.run_id);
            append_log_excerpt_tex(
                &mut tex,
                "run_###/report.md (per-iteration summary for the best run)",
                &run_dir.join("report.md"),
                4000,
            );
            append_log_excerpt_tex(
                &mut tex,
                "run_###/discovery.json (solver metadata + candidates for the best run)",
                &run_dir.join("discovery.json"),
                6000,
            );
            append_log_excerpt_tex(
                &mut tex,
                "run_###/rollout_trace.json (error curve for the best run)",
                &run_dir.join("rollout_trace.json"),
                5000,
            );
            append_log_excerpt_tex(
                &mut tex,
                "run_###/judge_prompt.txt (LLM judge prompt, best run)",
                &run_dir.join("judge_prompt.txt"),
                4000,
            );
            append_log_excerpt_tex(
                &mut tex,
                "run_###/judge_response.txt (LLM judge response, best run)",
                &run_dir.join("judge_response.txt"),
                4000,
            );
            append_log_excerpt_tex(
                &mut tex,
                "run_###/ic_prompt.txt (LLM IC prompt, best run)",
                &run_dir.join("ic_prompt.txt"),
                4000,
            );
            append_log_excerpt_tex(
                &mut tex,
                "run_###/ic_response.txt (LLM IC response, best run)",
                &run_dir.join("ic_response.txt"),
                4000,
            );
            append_log_excerpt_tex(
                &mut tex,
                "run_###/config.json (simulation config, best run)",
                &run_dir.join("config.json"),
                3000,
            );
            append_log_excerpt_tex(
                &mut tex,
                "run_###/initial_conditions.json (ICs, best run)",
                &run_dir.join("initial_conditions.json"),
                3000,
            );
        }
    } else {
        tex.push_str("No run directory provided; evidence embedding disabled.\\\\\n\n");
    }

    tex.push_str("\\section*{End-of-Run Summary (Checklist + Footnotes)}\n");
    tex.push_str(
        "This final section is designed as a single scrollable snapshot. Footnotes point to raw artifacts on disk for auditability.\\\\\n\n",
    );
    tex.push_str("\\begin{itemize}\n");
    if let Some(best) = current_best {
        let m = &best.candidate.metrics;
        let mut best_artifacts: Vec<String> = Vec::new();
        best_artifacts.push(format!("{}/report.md", best.run_id));
        best_artifacts.push(format!("{}/report.json", best.run_id));
        best_artifacts.push(format!("{}/discovery.json", best.run_id));
        best_artifacts.push(format!("{}/rollout_trace.json", best.run_id));
        best_artifacts.push(format!("{}/traj.csv", best.run_id));
        best_artifacts.push(format!("{}/traj.json", best.run_id));
        best_artifacts.push(format!("{}/config.json", best.run_id));
        best_artifacts.push(format!("{}/initial_conditions.json", best.run_id));
        tex.push_str(&format!(
            "  \\item Best run: \\texttt{{{}}} (regime: \\texttt{{{}}}). {} \n",
            escape_latex(&best.run_id),
            escape_latex(&best.regime),
            footnote_paths(&best_artifacts)
        ));
        tex.push_str(&format!(
            "  \\item Best metrics: rollout\\_rmse={}, divergence\\_time={}, mse={:.6e}, complexity={}. \n",
            format_opt_f64(m.rollout_rmse),
            format_opt_f64(m.divergence_time),
            m.mse,
            m.complexity
        ));
    } else {
        tex.push_str("  \\item Best run: n/a (no candidates recorded). \n");
    }

    let mut run_artifacts: Vec<String> = Vec::new();
    run_artifacts.push("evaluation.md".to_string());
    run_artifacts.push("evaluation_input.json".to_string());
    if let Some(factory_dir) = opts.factory_dir {
        for name in [
            "evaluation_history.json",
            "benchmark_eval.json",
            "evaluation_llm.md",
            "evaluation_prompt.txt",
            "evaluation_error.txt",
        ] {
            if factory_dir.join(name).exists() {
                run_artifacts.push(name.to_string());
            }
        }
    }
    tex.push_str(&format!(
        "  \\item Core artifacts: \\texttt{{evaluation.md}} (deterministic), \\texttt{{evaluation.tex}} (this PDF), plus machine-readable JSON.{} \n",
        footnote_paths(&run_artifacts)
    ));
    tex.push_str("\\end{itemize}\n\n");

    tex.push_str("\\subsection*{Run Notes (key=value)}\n");
    if eval_input.notes.is_empty() {
        tex.push_str("No notes.\\\\\n\n");
    } else {
        tex.push_str("\\begin{itemize}\n");
        for note in &eval_input.notes {
            tex.push_str(&format!("  \\item \\texttt{{{}}}\n", escape_latex(note)));
        }
        tex.push_str("\\end{itemize}\n\n");
    }

    tex.push_str("\\subsection*{Per-Iteration Best Candidates (Condensed)}\n");
    tex.push_str("\\small\n");
    tex.push_str("\\begin{longtable}{lrrrrp{6cm}}\n");
    tex.push_str("\\toprule\n");
    tex.push_str("Run (integrator) & rollout\\_rmse & divergence\\_time & mse & complexity & equation (1-line)\\\\\n");
    tex.push_str("\\midrule\n");
    for iter in &eval_input.iterations {
        let Some(best_iter) = best_candidate_in_iteration(iter) else {
            continue;
        };
        let integrator = iter
            .simulation
            .as_ref()
            .map(|s| s.rollout_integrator.as_str())
            .unwrap_or("n/a");
        let label = format!("{} ({integrator})", iter.run_id);
        let eq_short = truncate_to_chars(&single_line(&best_iter.equation_text), 160);
        tex.push_str(&format!(
            "\\texttt{{{}}} & {} & {} & {:.6e} & {} & \\texttt{{{}}}\\\\\n",
            escape_latex(&label),
            format_opt_f64(best_iter.metrics.rollout_rmse),
            format_opt_f64(best_iter.metrics.divergence_time),
            best_iter.metrics.mse,
            best_iter.metrics.complexity,
            escape_latex(&eq_short)
        ));
    }
    tex.push_str("\\bottomrule\n");
    tex.push_str("\\end{longtable}\n");
    tex.push_str("\\normalsize\n\n");

    if let Some(inc) = opts.incumbent_on_best_run {
        if let Some(best) = current_best {
            let curr_roll = best.candidate.metrics.rollout_rmse.unwrap_or(f64::INFINITY);
            let delta = curr_roll - inc.rollout_rmse;
            tex.push_str("\\subsection*{Evidence: Prior Best Re-Scored on This Run}\n");
            tex.push_str(
                "This is an apples-to-apples check: we roll out the prior best equation on the current run’s oracle trajectory.\\\\\n",
            );
            tex.push_str(&format!(
                "Prior best rollout\\_rmse on best-run oracle: {:.6} (divergence\\_time={}).\\\\\n",
                inc.rollout_rmse,
                format_opt_f64(inc.divergence_time)
            ));
            tex.push_str(&format!(
                "Current best rollout\\_rmse on best-run oracle: {}.\\\\\n",
                format_opt_f64(best.candidate.metrics.rollout_rmse)
            ));
            if delta.is_finite() {
                tex.push_str(&format!("Delta (current - prior): {delta:+.6}.\\\\\n\n"));
            } else {
                tex.push_str("Delta (current - prior): n/a.\\\\\n\n");
            }
        }
    }

    if let Some(bm) = opts.benchmark {
        tex.push_str("\\subsection*{Held-out Benchmark (Fixed IC Suite)}\n");
        tex.push_str("This is an apples-to-apples check across runs in the same (steps, dt) bucket.\\\\\n");
        tex.push_str(&format!(
            "suite\\_id=\\texttt{{{}}}, cases={}, rollout\\_integrator=\\texttt{{{}}}.{}\\\\\n",
            escape_latex(&bm.aggregate.suite_id),
            bm.aggregate.cases,
            escape_latex(&bm.aggregate.rollout_integrator),
            footnote_paths(&vec!["benchmark_eval.json".to_string()])
        ));
        tex.push_str(&format!(
            "aggregate: rmse\\_mean={}, rmse\\_max={}, divergence\\_time\\_min={}.\\\\\n\n",
            format_opt_f64(bm.aggregate.rmse_mean),
            format_opt_f64(bm.aggregate.rmse_max),
            format_opt_f64(bm.aggregate.divergence_time_min)
        ));
        tex.push_str("\\begin{tabular}{lrr}\n");
        tex.push_str("\\toprule\n");
        tex.push_str("Case & rollout\\_rmse & divergence\\_time\\\\\n");
        tex.push_str("\\midrule\n");
        for c in &bm.cases {
            tex.push_str(&format!(
                "\\texttt{{{}}} & {} & {}\\\\\n",
                escape_latex(&c.name),
                format_opt_f64(c.rollout_rmse),
                format_opt_f64(c.divergence_time)
            ));
        }
        tex.push_str("\\bottomrule\n");
        tex.push_str("\\end{tabular}\n\n");
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
    benchmark: Option<&BenchmarkEvalV1>,
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
                    md.push_str(&format!("- produced_samples={}\n", sim.steps));
                    if let Some(req) = sim.requested_steps {
                        let expected = req.saturating_add(1);
                        md.push_str(&format!("- requested_steps={} (expected_samples={})\n", req, expected));
                    }
                    if let Some(dt) = sim.requested_dt {
                        md.push_str(&format!("- requested_dt={dt}\n"));
                    }
                    md.push_str(&format!("- terminated_early={}\n", sim.terminated_early));
                    if let Some(reason) = sim.termination_reason.as_ref() {
                        md.push_str(&format!("- termination_reason={}\n", reason));
                    }
                    if sim.encounter_step.is_some()
                        || sim.encounter_min_pair_dist.is_some()
                        || sim.encounter_action.is_some()
                    {
                        md.push_str("- encounter:\n");
                        if let Some(step) = sim.encounter_step {
                            md.push_str(&format!("  - step={step}\n"));
                        }
                        if let Some(d) = sim.encounter_min_pair_dist {
                            md.push_str(&format!("  - min_pair_dist={d}\n"));
                        }
                        if let Some(action) = sim.encounter_action.as_ref() {
                            md.push_str(&format!("  - action={action}\n"));
                        }
                    }
                    md.push_str(&format!("- min_pair_dist={}\n", format_opt_f64(sim.min_pair_dist)));
                    md.push_str(&format!("- energy_drift={}\n", format_opt_f64(sim.energy_drift)));
                    md.push_str(&format!(
                        "- mean_abs_accel_grav={}\n",
                        format_opt_f64(sim.mean_abs_accel_grav)
                    ));
                    md.push_str(&format!(
                        "- mean_abs_accel_em={}\n",
                        format_opt_f64(sim.mean_abs_accel_em)
                    ));
                    md.push_str(&format!(
                        "- mean(|a_em|)/mean(|a_grav|)={}\n",
                        format_opt_f64(sim.mean_abs_accel_ratio_em_over_grav)
                    ));
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

    md.push_str("\n## Held-out Benchmark (Fixed IC Suite)\n");
    if let Some(bm) = benchmark {
        md.push_str("- This is an apples-to-apples check across runs in the same bucket.\n");
        md.push_str(&format!(
            "- suite_id={}, cases={}, rollout_integrator={}\n",
            bm.aggregate.suite_id, bm.aggregate.cases, bm.aggregate.rollout_integrator
        ));
        md.push_str(&format!(
            "- rmse_mean={}, rmse_max={}, divergence_time_min={}\n",
            format_opt_f64(bm.aggregate.rmse_mean),
            format_opt_f64(bm.aggregate.rmse_max),
            format_opt_f64(bm.aggregate.divergence_time_min)
        ));
        md.push_str("- per-case:\n");
        for c in &bm.cases {
            md.push_str(&format!(
                "  - {}: rollout_rmse={}, divergence_time={}\n",
                c.name,
                format_opt_f64(c.rollout_rmse),
                format_opt_f64(c.divergence_time)
            ));
        }
        if !bm.notes.is_empty() {
            md.push_str("- notes:\n");
            for n in &bm.notes {
                md.push_str(&format!("  - {}\n", n));
            }
        }
        md.push_str("- full details: `benchmark_eval.json`\n");
    } else {
        md.push_str("- n/a (benchmark not computed for this run)\n");
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
    fn force_nonplanar_ic_adds_z_when_planar() {
        let bounds = default_ic_bounds();
        let planar = initial_conditions_from_preset("two-body").unwrap();
        assert!(ic_is_nearly_planar(&planar));
        let forced = force_nonplanar_ic(planar, &bounds, 123);
        assert!(!ic_is_nearly_planar(&forced));
        assert!(forced.notes.contains("forced_nonplanar_z=true"));
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
    fn sim_summary_includes_em_signal_ratio_when_em_enabled() {
        let mut cfg = Config::default();
        cfg.enable_gravity = true;
        cfg.enable_em = true;
        cfg.integrator.kind = IntegratorKind::Leapfrog;
        cfg.integrator.adaptive = false;

        let bodies = [Body::new(1.0, 1.0), Body::new(1.0, -1.0), Body::new(1.0, 1.0)];
        let pos = [
            Vec3::new(-0.6, 0.0, 0.0),
            Vec3::new(0.6, 0.0, 0.0),
            Vec3::new(0.0, 0.9, 0.0),
        ];
        let vel = [
            Vec3::new(0.0, 0.2, 0.0),
            Vec3::new(0.0, -0.2, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        let system = System::new(bodies, State::new(pos, vel));
        let result = simulate_with_cfg(system, &cfg, SimOptions { steps: 2, dt: 0.01 });
        let sim = build_sim_summary(&result, &cfg, RolloutIntegrator::Euler, Some(5), Some(0.01));

        let grav = sim.mean_abs_accel_grav.expect("grav mean");
        let em = sim.mean_abs_accel_em.expect("em mean");
        let ratio = sim
            .mean_abs_accel_ratio_em_over_grav
            .expect("ratio mean");
        assert!(grav.is_finite() && grav > 0.0);
        assert!(em.is_finite() && em > 0.0);
        assert!(ratio.is_finite() && ratio > 0.0);
    }

    #[test]
    fn extended_feature_gates_are_consistent() {
        let mut cfg = Config::default();
        cfg.enable_gravity = true;
        cfg.enable_em = true;

        // Force a "close" configuration so gate_close=1.
        let bodies = [Body::new(1.0, 0.1), Body::new(1.0, -0.1), Body::new(1.0, 0.1)];
        let pos = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.3, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let vel = [Vec3::zero(); 3];
        let system = System::new(bodies, State::new(pos, vel));

        let feature_names = vec![
            "grav_x".to_string(),
            "grav_r4_x".to_string(),
            "gate_close".to_string(),
            "gate_far".to_string(),
            "grav_close_x".to_string(),
            "grav_far_x".to_string(),
        ];
        let feats = compute_feature_vector(&system, 0, &cfg, &feature_names);
        let idx = |name: &str| feature_names.iter().position(|f| f == name).unwrap();

        let gate_close = feats[idx("gate_close")];
        let gate_far = feats[idx("gate_far")];
        assert_eq!(gate_close, 1.0);
        assert_eq!(gate_far, 0.0);

        let grav_x = feats[idx("grav_x")];
        let grav_close_x = feats[idx("grav_close_x")];
        let grav_far_x = feats[idx("grav_far_x")];
        assert!((grav_x - grav_close_x).abs() < 1e-12);
        assert_eq!(grav_far_x, 0.0);

        let grav_r4_x = feats[idx("grav_r4_x")];
        assert!(grav_r4_x.is_finite());
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
    fn sanitize_judge_recommendations_drops_invalid_next_initial_conditions() {
        let bounds = default_ic_bounds();
        let mut rec = threebody_discover::JudgeRecommendations {
            next_initial_conditions: Some(InitialConditionSpec {
                bodies: vec![
                    threebody_discover::judge::BodyInit {
                        mass: 1.0,
                        charge: 0.0,
                        pos: [0.0, 0.0, 0.0],
                        vel: [0.0, 0.0, 0.0],
                    },
                    threebody_discover::judge::BodyInit {
                        mass: 1.0,
                        charge: 0.0,
                        pos: [1.0, 0.0, 0.0],
                        vel: [0.0, 0.0, 0.0],
                    },
                ],
                barycentric: true,
                notes: "invalid: only two bodies".to_string(),
            }),
            next_rollout_integrator: None,
            next_ga_heuristic: None,
            next_discovery_solver: None,
            next_normalize: None,
            next_feature_library: None,
            next_stls_threshold: None,
            next_ridge_lambda: None,
            next_lasso_alpha: None,
            next_manual_equation_text: None,
            next_search_directions: vec![],
            notes: String::new(),
        };
        sanitize_judge_recommendations(&mut rec, &bounds);
        assert!(rec.next_initial_conditions.is_none());
    }

    #[test]
    fn sanitize_judge_recommendations_drops_invalid_enum_and_numeric_recommendations() {
        let bounds = default_ic_bounds();
        let mut rec = threebody_discover::JudgeRecommendations {
            next_initial_conditions: None,
            next_rollout_integrator: Some("rk4".to_string()),
            next_ga_heuristic: Some("fitness".to_string()),
            next_discovery_solver: Some("svd".to_string()),
            next_normalize: Some(true),
            next_feature_library: Some("mega".to_string()),
            next_stls_threshold: Some(-1.0),
            next_ridge_lambda: Some(-1e-3),
            next_lasso_alpha: Some(0.0),
            next_manual_equation_text: Some("ax=0 ; ay=0 ; az=0".to_string()),
            next_search_directions: vec!["ok".to_string()],
            notes: String::new(),
        };

        sanitize_judge_recommendations(&mut rec, &bounds);
        assert!(rec.next_rollout_integrator.is_none());
        assert!(rec.next_ga_heuristic.is_none());
        assert!(rec.next_discovery_solver.is_none());
        assert!(rec.next_feature_library.is_none());
        assert!(rec.next_stls_threshold.is_none());
        assert!(rec.next_ridge_lambda.is_none());
        assert!(rec.next_lasso_alpha.is_none());
        // Keep unrelated fields.
        assert_eq!(rec.next_normalize, Some(true));
        assert!(rec.next_manual_equation_text.is_some());
        assert_eq!(rec.next_search_directions, vec!["ok".to_string()]);
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
    fn safe_verbatim_excerpt_does_not_emit_verbatim_terminator_lines() {
        let input = "\\end{verbatim}\n\\begin{verbatim}\nnormal";
        let out = safe_verbatim_excerpt(input, 10_000, 1);
        for line in out.lines() {
            let trimmed = line.trim_start_matches(|c: char| c.is_whitespace());
            assert!(
                !trimmed.starts_with("\\end{verbatim}") && !trimmed.starts_with("\\begin{verbatim}"),
                "unsafe verbatim terminator emitted: {trimmed:?}"
            );
        }
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
        let tex = render_evaluation_tex(
            &eval_input,
            Some(&best),
            None,
            Some(&ic),
            EvaluationTexOptions::default(),
        );
        for needle in [
            "\\section*{Executive Summary}",
            "\\section*{Best Equation (Exact)}",
            "\\section*{Candidate Equations Tried (Top 3 per Iteration)}",
            "\\section*{Initial Conditions (Best Run)}",
            "\\section*{Comparison to Prior Attempts}",
            "\\section*{End-of-Run Summary (Checklist + Footnotes)}",
            "\\section*{Evidence Appendix (Embedded Excerpts)}",
        ] {
            assert!(tex.contains(needle), "missing {needle}");
        }
        assert!(tex.contains("ax=+1*grav_x"));
    }

    #[test]
    fn render_evaluation_tex_includes_llm_summary_and_search_directions() {
        let judge = FactoryEvaluationIterationJudge {
            summary: "judge_summary_marker".to_string(),
            ranking: vec![0],
            recommendations: JudgeRecommendationsLite {
                next_rollout_integrator: None,
                next_ga_heuristic: None,
                next_discovery_solver: None,
                next_normalize: None,
                next_feature_library: None,
                next_stls_threshold: None,
                next_ridge_lambda: None,
                next_lasso_alpha: None,
                next_manual_equation_text: None,
                next_search_directions: vec!["ablation: remove gate_close".to_string()],
                notes: "judge_notes_marker".to_string(),
            },
        };

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
                judge: Some(judge),
            }],
        };

        let best = best_candidate_from_eval_input(&eval_input).unwrap();
        let tex = render_evaluation_tex(
            &eval_input,
            Some(&best),
            None,
            None,
            EvaluationTexOptions::default(),
        );
        assert!(tex.contains("LLM summary"));
        assert!(tex.contains(&escape_latex("judge_summary_marker")));
        assert!(tex.contains("LLM suggested next experiments"));
        assert!(tex.contains(&escape_latex("ablation: remove gate_close")));
        assert!(tex.contains("LLM notes"));
        assert!(tex.contains(&escape_latex("judge_notes_marker")));
    }

    #[test]
    fn render_evaluation_tex_embeds_evidence_excerpts_when_factory_dir_present() {
        let root = unique_temp_path("eval_tex_embed", "dir");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("evaluation.md"), "EXEC_MD_MARKER: hello\n").unwrap();
        fs::write(root.join("evaluation_llm.md"), "LLM_MD_MARKER: world\n").unwrap();
        fs::write(root.join("evaluation_prompt.txt"), "PROMPT_MARKER: p\n").unwrap();
        fs::write(root.join("evaluation_history.json"), "HISTORY_MARKER: h\n").unwrap();

        let run_dir = root.join("run_001");
        fs::create_dir_all(&run_dir).unwrap();
        fs::write(run_dir.join("report.md"), "REPORT_MARKER: r\n").unwrap();
        fs::write(run_dir.join("discovery.json"), "DISCOVERY_MARKER: d\n").unwrap();
        fs::write(run_dir.join("rollout_trace.json"), "TRACE_MARKER: t\n").unwrap();
        fs::write(run_dir.join("judge_prompt.txt"), "JUDGE_PROMPT_MARKER: jp\n").unwrap();
        fs::write(run_dir.join("judge_response.txt"), "JUDGE_RESPONSE_MARKER: jr\n").unwrap();
        fs::write(run_dir.join("ic_prompt.txt"), "IC_PROMPT_MARKER: ip\n").unwrap();
        fs::write(run_dir.join("ic_response.txt"), "IC_RESPONSE_MARKER: ir\n").unwrap();
        fs::write(run_dir.join("config.json"), "{}\n").unwrap();
        fs::write(run_dir.join("initial_conditions.json"), "{}\n").unwrap();

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
        let tex = render_evaluation_tex(
            &eval_input,
            Some(&best),
            None,
            None,
            EvaluationTexOptions {
                factory_dir: Some(root.as_path()),
                benchmark: None,
                incumbent_on_best_run: None,
            },
        );
        for marker in [
            "EXEC_MD_MARKER",
            "LLM_MD_MARKER",
            "PROMPT_MARKER",
            "HISTORY_MARKER",
            "REPORT_MARKER",
            "DISCOVERY_MARKER",
            "TRACE_MARKER",
            "JUDGE_PROMPT_MARKER",
            "JUDGE_RESPONSE_MARKER",
            "IC_PROMPT_MARKER",
            "IC_RESPONSE_MARKER",
        ] {
            assert!(tex.contains(marker), "missing excerpt marker: {marker}");
        }
        assert!(tex.contains("\\footnote{"), "expected at least one footnote");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn normalize_equation_text_for_features_filters_unknown_terms() {
        let features = FeatureLibrary::default_physics().features;
        let raw = "ax=+1.000000*grav_x +2.000000*foo_x ; ay=+1.000000*grav_y ; az=0";
        let norm = normalize_equation_text_for_features(raw, &features).expect("normalize");
        assert!(norm.contains("grav_x"));
        assert!(norm.contains("grav_y"));
        assert!(!norm.contains("foo_x"));
    }

    #[test]
    fn propose_equation_mutants_is_deterministic_and_axis_consistent() {
        let features = FeatureLibrary::default_physics().features;
        let parent = "ax=+1.000000*grav_x ; ay=+1.000000*grav_y ; az=0";
        let a = propose_equation_mutants(parent, &features, "gravity_only", 123);
        let b = propose_equation_mutants(parent, &features, "gravity_only", 123);
        assert_eq!(a, b);
        for eq in &a {
            let [tx, ty, tz] = parse_vector_equation_terms(eq);
            for (_c, f) in &tx {
                assert!(f == "gate_close" || f == "gate_far" || f.ends_with("_x"));
            }
            for (_c, f) in &ty {
                assert!(f == "gate_close" || f == "gate_far" || f.ends_with("_y"));
            }
            for (_c, f) in &tz {
                assert!(f == "gate_close" || f == "gate_far" || f.ends_with("_z"));
            }
        }
    }

    #[test]
    fn propose_equation_mutants_includes_grav_z_completion_in_gravity_only() {
        let features = FeatureLibrary::default_physics().features;
        let parent = "ax=+1.000000*grav_x ; ay=+1.000000*grav_y ; az=0";
        let muts = propose_equation_mutants(parent, &features, "gravity_only", 7);
        assert!(
            muts.iter().any(|s| s.contains("az=") && s.contains("grav_z")),
            "expected a mutant that includes grav_z in az"
        );
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

    fn mk_eval_input_with_sim(
        steps: usize,
        dt: f64,
        produced_steps: usize,
        terminated_early: bool,
        rollout_rmse: f64,
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
                simulation: Some(SimulationSummary {
                    steps: produced_steps,
                    requested_steps: Some(steps),
                    requested_dt: Some(dt),
                    terminated_early,
                    termination_reason: None,
                    encounter_step: None,
                    encounter_min_pair_dist: None,
                    encounter_action: None,
                    energy_start: None,
                    energy_end: None,
                    energy_drift: None,
                    min_pair_dist: None,
                    max_speed: None,
                    max_accel: None,
                    mean_abs_accel_grav: None,
                    mean_abs_accel_em: None,
                    mean_abs_accel_ratio_em_over_grav: None,
                    dt_min: None,
                    dt_max: None,
                    dt_avg: None,
                    warnings: vec![],
                    rollout_integrator: "euler".to_string(),
                }),
                top_candidates: vec![FactoryEvaluationCandidate {
                    id: 0,
                    equation_text: equation_text.to_string(),
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
    fn scan_best_results_skips_terminated_or_short_horizon_oracles() {
        let root = unique_temp_path("best_results_skip_bad_oracle", "dir");
        if root.exists() {
            let _ = fs::remove_dir_all(&root);
        }
        fs::create_dir_all(&root).unwrap();

        let bad_term = root.join("bad_term").join("factory");
        let bad_short = root.join("bad_short").join("factory");
        let good = root.join("good").join("factory");
        fs::create_dir_all(&bad_term).unwrap();
        fs::create_dir_all(&bad_short).unwrap();
        fs::create_dir_all(&good).unwrap();

        // Best metrics but terminated early -> must be skipped.
        fs::write(
            bad_term.join("evaluation_input.json"),
            serde_json::to_string_pretty(&mk_eval_input_with_sim(100, 0.01, 101, true, 0.01, "a=bad_term")).unwrap(),
        )
        .unwrap();

        // Best metrics but short horizon (produced_steps != steps+1) -> must be skipped.
        fs::write(
            bad_short.join("evaluation_input.json"),
            serde_json::to_string_pretty(&mk_eval_input_with_sim(100, 0.01, 50, false, 0.02, "a=bad_short")).unwrap(),
        )
        .unwrap();

        // Worse metrics but valid oracle -> should be selected.
        fs::write(
            good.join("evaluation_input.json"),
            serde_json::to_string_pretty(&mk_eval_input_with_sim(100, 0.01, 101, false, 0.2, "a=good")).unwrap(),
        )
        .unwrap();

        let index = scan_best_results(&root).unwrap();
        assert_eq!(index.buckets.len(), 1);
        assert!(index.buckets[0].eval_input_path.contains("good"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn scan_best_results_prefers_lower_benchmark_when_available() {
        let root = unique_temp_path("best_results_benchmark_choose", "dir");
        if root.exists() {
            let _ = fs::remove_dir_all(&root);
        }
        fs::create_dir_all(&root).unwrap();

        let run1_factory = root.join("run1").join("factory");
        let run2_factory = root.join("run2").join("factory");
        fs::create_dir_all(&run1_factory).unwrap();
        fs::create_dir_all(&run2_factory).unwrap();

        // run1: looks better on within-run rollout_rmse, but worse on benchmark.
        fs::write(
            run1_factory.join("evaluation_input.json"),
            serde_json::to_string_pretty(&mk_eval_input_with_sim(100, 0.01, 101, false, 0.01, "a=run1")).unwrap(),
        )
        .unwrap();
        let bm1 = BenchmarkEvalV1 {
            version: "v1".to_string(),
            suite_id: "benchmark_suite_v1".to_string(),
            bucket: BucketKey { steps: 100, dt: 0.01 },
            regime: "gravity_only".to_string(),
            rollout_integrator: "leapfrog".to_string(),
            candidate: BenchmarkCandidateRef {
                run_id: "run_001".to_string(),
                candidate_id: 0,
                equation_text: "a=run1".to_string(),
                complexity: 1,
            },
            cases: vec![],
            aggregate: BenchmarkAggregate {
                suite_id: "benchmark_suite_v1".to_string(),
                rollout_integrator: "leapfrog".to_string(),
                cases: 3,
                rmse_mean: Some(0.9),
                rmse_max: Some(1.0),
                divergence_time_min: None,
            },
            notes: vec![],
        };
        fs::write(
            run1_factory.join("benchmark_eval.json"),
            serde_json::to_string_pretty(&bm1).unwrap(),
        )
        .unwrap();

        // run2: worse within-run metric, but better benchmark -> should win.
        fs::write(
            run2_factory.join("evaluation_input.json"),
            serde_json::to_string_pretty(&mk_eval_input_with_sim(100, 0.01, 101, false, 0.2, "a=run2")).unwrap(),
        )
        .unwrap();
        let bm2 = BenchmarkEvalV1 {
            version: "v1".to_string(),
            suite_id: "benchmark_suite_v1".to_string(),
            bucket: BucketKey { steps: 100, dt: 0.01 },
            regime: "gravity_only".to_string(),
            rollout_integrator: "leapfrog".to_string(),
            candidate: BenchmarkCandidateRef {
                run_id: "run_001".to_string(),
                candidate_id: 0,
                equation_text: "a=run2".to_string(),
                complexity: 1,
            },
            cases: vec![],
            aggregate: BenchmarkAggregate {
                suite_id: "benchmark_suite_v1".to_string(),
                rollout_integrator: "leapfrog".to_string(),
                cases: 3,
                rmse_mean: Some(0.1),
                rmse_max: Some(0.2),
                divergence_time_min: None,
            },
            notes: vec![],
        };
        fs::write(
            run2_factory.join("benchmark_eval.json"),
            serde_json::to_string_pretty(&bm2).unwrap(),
        )
        .unwrap();

        let index = scan_best_results(&root).unwrap();
        assert_eq!(index.buckets.len(), 1);
        assert!(index.buckets[0].eval_input_path.contains("run2"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn executive_summary_includes_termination_metadata_when_present() {
        let mut eval_input = mk_eval_input_with_sim(100, 0.01, 50, true, 0.1, "ax=+1*grav_x ; ay=0 ; az=0");
        if let Some(sim) = eval_input.iterations[0].simulation.as_mut() {
            sim.termination_reason = Some("max_rejects_exceeded".to_string());
            sim.encounter_step = Some(7);
            sim.encounter_min_pair_dist = Some(0.05);
            sim.encounter_action = Some("StopAndReport".to_string());
        }
        let best = best_candidate_from_eval_input(&eval_input);
        let md = render_executive_summary_md(&eval_input, best.as_ref(), None, None, None, None);
        assert!(md.contains("terminated_early=true"));
        assert!(md.contains("termination_reason=max_rejects_exceeded"));
        assert!(md.contains("requested_steps=100"));
        assert!(md.contains("requested_dt=0.01"));
        assert!(md.contains("encounter:"));
        assert!(md.contains("step=7"));
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
                benchmark: None,
            }],
            notes: vec![],
        };
        let md = render_best_results_md(&index, &[]);
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
                benchmark: None,
            }],
            notes: vec![],
        };
        let tex = render_findings_tex(&index, &[]);
        assert!(tex.contains("\\section*{Executive Summary}"));
        assert!(tex.contains("\\section*{Best Results by Bucket}"));
        assert!(tex.contains("Run Logs (Embedded Excerpts)"));
        assert!(tex.contains("ax=+1.000000*grav_x"));
    }

    #[test]
    fn render_findings_tex_embeds_log_excerpt_when_present() {
        let root = unique_temp_path("findings_logs", "dir");
        if root.exists() {
            let _ = fs::remove_dir_all(&root);
        }
        let factory_dir = root.join("factory");
        let best_run_dir = factory_dir.join("run_001");
        fs::create_dir_all(&best_run_dir).unwrap();
        fs::write(factory_dir.join("evaluation.md"), "EVAL_LOG_MARKER: hello\n").unwrap();
        fs::write(best_run_dir.join("report.md"), "REPORT_LOG_MARKER: world\n").unwrap();

        let index = BestResultsIndexV1 {
            version: "v1".to_string(),
            updated_at_utc: "unix_seconds=0".to_string(),
            buckets: vec![BestRecord {
                bucket: BucketKey { steps: 200, dt: 0.01 },
                run_dir: root.to_string_lossy().to_string(),
                factory_dir: factory_dir.to_string_lossy().to_string(),
                eval_input_path: factory_dir
                    .join("evaluation_input.json")
                    .to_string_lossy()
                    .to_string(),
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
                benchmark: None,
            }],
            notes: vec![],
        };

        let tex = render_findings_tex(&index, &[]);
        assert!(tex.contains("EVAL_LOG_MARKER: hello"));
        assert!(tex.contains("REPORT_LOG_MARKER: world"));

        let _ = fs::remove_dir_all(&root);
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
    fn lorentz_feature_components_match_core_fields() {
        let mut cfg = Config::default();
        cfg.enable_gravity = false;
        cfg.enable_em = true;
        cfg.softening = 0.01;
        cfg.constants.k_e = 2.5;
        cfg.constants.mu_0 = 0.7;

        let bodies = [
            Body::new(2.0, 1.5),
            Body::new(3.0, -0.7),
            Body::new(1.0, 0.0),
        ];
        let pos = [
            Vec3::new(-0.8, 0.1, 0.0),
            Vec3::new(0.9, -0.2, 0.0),
            Vec3::new(0.1, 1.2, 0.0),
        ];
        let vel = [
            Vec3::new(0.05, 0.2, 0.0),
            Vec3::new(-0.1, 0.15, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        let system = System::new(bodies, State::new(pos, vel));

        let force_cfg = threebody_core::forces::ForceConfig {
            g: cfg.constants.g,
            k_e: cfg.constants.k_e,
            mu_0: cfg.constants.mu_0,
            epsilon: cfg.softening,
            enable_gravity: cfg.enable_gravity,
            enable_em: cfg.enable_em,
        };
        let fields = threebody_core::forces::compute_fields(&system, &force_cfg);
        let accel = threebody_core::forces::compute_accel(&system, &force_cfg);

        let feature_names = FeatureLibrary::em_fields_lorentz().features;
        let idx = |name: &str| feature_names.iter().position(|n| n == name).unwrap();

        for i in 0..3 {
            let feats = compute_feature_vector(&system, i, &cfg, &feature_names);
            let got_e = Vec3::new(
                feats[idx("lorentz_e_x")],
                feats[idx("lorentz_e_y")],
                feats[idx("lorentz_e_z")],
            );
            let got_vxb = Vec3::new(
                feats[idx("lorentz_vxb_x")],
                feats[idx("lorentz_vxb_y")],
                feats[idx("lorentz_vxb_z")],
            );

            let qi = system.bodies[i].charge;
            let mi = system.bodies[i].mass;
            let expected_e = if qi == 0.0 || mi == 0.0 {
                Vec3::zero()
            } else {
                fields.e[i] * (qi / mi)
            };
            let expected_vxb = if qi == 0.0 || mi == 0.0 {
                Vec3::zero()
            } else {
                system.state.vel[i].cross(fields.b[i]) * (qi / mi)
            };

            assert!(
                got_e.approx_eq(expected_e, 1e-10, 1e-10),
                "body {i}: got_e={got_e:?} expected_e={expected_e:?}"
            );
            assert!(
                got_vxb.approx_eq(expected_vxb, 1e-10, 1e-10),
                "body {i}: got_vxb={got_vxb:?} expected_vxb={expected_vxb:?}"
            );
            assert!(
                (got_e + got_vxb).approx_eq(accel[i], 1e-10, 1e-10),
                "body {i}: got_total={:?} expected_total={:?}",
                got_e + got_vxb,
                accel[i]
            );
        }
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
            next_feature_library: None,
            next_stls_threshold: Some(0.2),
            next_ridge_lambda: Some(1e-6),
            next_lasso_alpha: None,
            next_manual_equation_text: None,
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
