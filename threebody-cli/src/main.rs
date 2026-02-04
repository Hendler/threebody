use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

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
use threebody_discover::{grid_search, run_search, Dataset, FitnessHeuristic};

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
        /// Fitness heuristic: mse | mse_parsimony.
        #[arg(long, default_value = "mse")]
        fitness: String,
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
        /// Fitness heuristic: mse | mse_parsimony.
        #[arg(long, default_value = "mse")]
        fitness: String,
        /// Rollout integrator for evaluation: euler | leapfrog.
        #[arg(long, default_value = "euler")]
        rollout_integrator: String,
        /// LLM mode: off, mock, openai.
        #[arg(long, default_value = "mock")]
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
            fitness,
        } => {
            run_discovery(runs, population, seed, out, llm, model, fitness)?;
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
            fitness,
            rollout_integrator,
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
                fitness,
                rollout_integrator,
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
    let cfg = build_config(config_path, &mode, integrator_override, no_em, no_gravity)?;
    let system = preset_system(&preset)?;
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
    fitness: String,
) -> anyhow::Result<()> {
    let dataset = discovery_dataset();
    let library = FeatureLibrary::default_physics();
    let fitness = parse_fitness_heuristic(&fitness)?;
    let cfg = DiscoveryConfig {
        runs,
        population,
        seed,
        fitness,
        ..DiscoveryConfig::default()
    };
    let llm_client = if llm { Some(OpenAIClient::from_env(&model)?) } else { None };

    let topk = run_search(&dataset, &library, &cfg);
    let candidates: Vec<_> = topk.entries.iter().map(|e| e.equation.clone()).collect();
    let grid_topk = grid_search(&candidates, &dataset);

    let judge = if let Some(client) = llm_client.as_ref() {
        let judge_input =
            build_judge_input_from_entries(&dataset, &library, &topk.entries, None, "toy_dataset", "x");
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

fn build_judge_input_from_entries(
    dataset: &Dataset,
    library: &FeatureLibrary,
    entries: &[threebody_discover::EquationScore],
    simulation: Option<SimulationSummary>,
    regime: &str,
    target_description: &str,
) -> JudgeInput {
    let candidates = entries
        .iter()
        .enumerate()
        .map(|(id, e)| CandidateSummary {
            id,
            equation: e.equation.clone(),
            equation_text: e.equation.format(),
            metrics: CandidateMetrics {
                mse: e.score,
                complexity: e.equation.complexity(),
                rollout_rmse: None,
                divergence_time: None,
                stability_flags: stability_flags_for(&e.equation, regime),
            },
            notes: vec![],
        })
        .collect::<Vec<_>>();
    let dataset_summary =
        build_dataset_summary(&dataset.feature_names, library, dataset.samples.len(), target_description);
    build_judge_input_from_candidates(dataset_summary, candidates, simulation, regime)
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
    let eps = cfg.softening.max(1e-9);
    for step in &result.steps {
        let system = &step.system;
        let acc = compute_accel(system, &force_cfg);
        for body in 0..3 {
            let features = compute_feature_vector(system, body, eps, feature_names);
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
    eps: f64,
    feature_names: &[String],
) -> Vec<f64> {
    let v = system.state.vel[body];
    let v_mag = v.norm();
    let mut r_inv2 = 0.0;
    let mut r_inv3 = 0.0;
    let mut v_cross_r = 0.0;
    let mut v_cross_v_cross_r = 0.0;
    for j in 0..3 {
        if j == body {
            continue;
        }
        let r = system.state.pos[j] - system.state.pos[body];
        let r2 = r.norm_sq() + eps * eps;
        let r_norm = r2.sqrt();
        r_inv2 += 1.0 / r2;
        r_inv3 += 1.0 / (r2 * r_norm);
        v_cross_r += v.cross(r).norm() / (r2 * r_norm);
        v_cross_v_cross_r += v.cross(v.cross(r)).norm() / (r2 * r_norm);
    }
    feature_names
        .iter()
        .map(|name| match name.as_str() {
            "r_inv2" => r_inv2,
            "r_inv3" => r_inv3,
            "v" => v_mag,
            "v_cross_r" => v_cross_r,
            "v_cross_v_cross_r" => v_cross_v_cross_r,
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
    let eps = cfg.softening.max(1e-9);
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    let mut t = 0.0;
    let threshold = (0.5 * min_pair_distance(&result.steps[0].system.state.pos)).max(0.1);
    let mut divergence_time = None;
    for i in 0..(result.steps.len().saturating_sub(1)) {
        let dt = result.steps[i].dt;
        system = rollout_step(&system, &feature_dataset, model, eps, dt, rollout_integrator);
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
    let eps = cfg.softening.max(1e-9);
    let mut trace = Vec::new();
    let mut t = 0.0;
    for i in 0..(result.steps.len().saturating_sub(1)) {
        let dt = result.steps[i].dt;
        system = rollout_step(&system, &feature_dataset, model, eps, dt, rollout_integrator);
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
    eps: f64,
) -> [Vec3; 3] {
    let mut acc = [Vec3::zero(); 3];
    for body in 0..3 {
        let features = compute_feature_vector(system, body, eps, &feature_dataset.feature_names);
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
    eps: f64,
    dt: f64,
    integrator: RolloutIntegrator,
) -> System {
    let acc = predict_accel(system, feature_dataset, model, eps);
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
            let acc_new = predict_accel(&interim, feature_dataset, model, eps);
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

fn select_llm_client(mode: LlmMode, model: &str) -> anyhow::Result<Option<Box<dyn LlmClient>>> {
    match mode {
        LlmMode::Off => Ok(None),
        LlmMode::Mock => Ok(Some(Box::new(MockLlm))),
        LlmMode::OpenAI => Ok(Some(Box::new(OpenAIClient::from_env(model)?))),
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
    no_em: bool,
    no_gravity: bool,
    runs: usize,
    population: usize,
    seed: u64,
    fitness: String,
    rollout_integrator: String,
    llm_mode: String,
    model: String,
) -> anyhow::Result<()> {
    fs::create_dir_all(&out_dir)?;
    let mut next_ic: Option<InitialConditionSpec> = None;
    let llm_mode = parse_llm_mode(&llm_mode)?;
    let llm_client = select_llm_client(llm_mode, &model)?;
    let mut current_fitness = parse_fitness_heuristic(&fitness)?;
    let mut current_rollout = parse_rollout_integrator(&rollout_integrator)?;

    for iter in 0..max_iters {
        let run_id = format!("run_{:03}", iter + 1);
        let run_dir = out_dir.join(&run_id);
        fs::create_dir_all(&run_dir)?;

        let cfg = build_config(config.clone(), &mode, None, no_em, no_gravity)?;
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
        let vector_data = build_vector_dataset(&result, &cfg, &library.features);
        let [dataset_x, dataset_y, dataset_z] = component_datasets(&vector_data);
        let disc_cfg = DiscoveryConfig {
            runs,
            population,
            seed: seed + iter as u64,
            fitness: current_fitness,
            ..DiscoveryConfig::default()
        };
        let topk_x = run_search(&dataset_x, &library, &disc_cfg);
        let topk_y = run_search(&dataset_y, &library, &disc_cfg);
        let topk_z = run_search(&dataset_z, &library, &disc_cfg);
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
            .push(format!("ga_fitness={}", current_fitness.as_str()));
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
        let discovery_json = serde_json::to_string_pretty(&serde_json::json!({
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
        barycentric: false,
        notes: format!("preset:{preset}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use threebody_discover::Equation;

    #[test]
    fn vector_dataset_includes_all_bodies() {
        let cfg = Config::default();
        let system = preset_system("two-body").unwrap();
        let result = simulate_with_cfg(system, &cfg, SimOptions { steps: 2, dt: 0.01 });
        let library = FeatureLibrary::default_physics();
        let vec_data = build_vector_dataset(&result, &cfg, &library.features);
        let expected = result.steps.len() * 3;
        assert_eq!(vec_data.samples.len(), expected);
        assert_eq!(vec_data.targets.len(), expected);
    }

    #[test]
    fn rollout_metrics_are_finite() {
        let cfg = Config::default();
        let system = preset_system("two-body").unwrap();
        let result = simulate_with_cfg(system, &cfg, SimOptions { steps: 3, dt: 0.01 });
        let library = FeatureLibrary::default_physics();
        let vec_data = build_vector_dataset(&result, &cfg, &library.features);
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
}
