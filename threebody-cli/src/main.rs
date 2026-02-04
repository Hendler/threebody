use std::fs;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use threebody_core::config::{Config, IntegratorKind};
use threebody_core::integrators::{boris::Boris, leapfrog::Leapfrog, rk45::Rk45, Integrator};
use threebody_core::math::vec3::Vec3;
use threebody_core::output::csv::write_csv;
use threebody_core::output::sidecar::{build_sidecar, write_sidecar};
use threebody_core::sim::{simulate, SimOptions};
use threebody_core::state::{Body, State, System};

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
        #[arg(long, default_value = "config.json")]
        config: PathBuf,
        /// Output CSV path.
        #[arg(long, default_value = "traj.csv")]
        output: PathBuf,
        /// Steps to simulate.
        #[arg(long, default_value_t = 100)]
        steps: usize,
        /// Timestep.
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
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
        #[arg(long, default_value = "config.json")]
        config: PathBuf,
        #[arg(long, default_value = "traj.csv")]
        output: PathBuf,
        #[arg(long, default_value_t = 100)]
        steps: usize,
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
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
                preset,
                integrator,
                no_em,
                no_gravity,
                format,
                dry_run,
                summary,
            )?;
        }
    }
    Ok(())
}

fn run_simulation(
    config_path: PathBuf,
    output: PathBuf,
    steps: usize,
    dt: f64,
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
    let cfg_json = fs::read_to_string(&config_path)?;
    let mut cfg: Config = serde_json::from_str(&cfg_json)?;
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
    cfg.validate().map_err(anyhow::Error::msg)?;

    let system = preset_system(&preset)?;
    let options = SimOptions { steps, dt };
    let (integrator, encounter_integrator): (Box<dyn Integrator>, Option<Box<dyn Integrator>>) =
        match cfg.integrator.kind {
            IntegratorKind::Leapfrog => (Box::new(Leapfrog), None),
            IntegratorKind::Rk45 => (Box::new(Rk45), None),
            IntegratorKind::Boris => (Box::new(Boris), None),
            IntegratorKind::ImplicitMidpoint => (Box::new(Rk45), None),
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
