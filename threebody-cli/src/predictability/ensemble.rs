use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use threebody_core::config::Config;
use threebody_core::output::sidecar::{build_sidecar, write_sidecar};
use threebody_core::sim::SimOptions;
use threebody_discover::ga::Lcg;
use threebody_discover::judge::{IcBounds, InitialConditionSpec};

use crate::predictability::PREDICTABILITY_VERSION;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EnsembleManifest {
    pub version: String,
    pub n: usize,
    pub seed: u64,
    pub sigma_pos: f64,
    pub sigma_vel: f64,
    pub steps: usize,
    pub dt: f64,
    pub mode: String,
    pub config: Config,
    pub members: Vec<EnsembleMember>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EnsembleMember {
    pub index: usize,
    pub dir: String,
    pub traj_csv: String,
    pub sidecar_json: String,
    pub ic_json: String,
    pub terminated_early: bool,
    pub termination_reason: Option<String>,
    pub warnings: Vec<String>,
    pub sim_stats: threebody_core::sim::SimStats,
}

struct GaussianRng {
    rng: Lcg,
    spare: Option<f64>,
}

impl GaussianRng {
    fn new(seed: u64) -> Self {
        Self {
            rng: Lcg::new(seed),
            spare: None,
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.rng.next_f64()
    }

    fn next_normal(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        // Box–Muller.
        let mut u1 = self.next_f64();
        let u2 = self.next_f64();
        if u1 <= 0.0 {
            u1 = 1e-12;
        }
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        let z0 = r * theta.cos();
        let z1 = r * theta.sin();
        self.spare = Some(z1);
        z0
    }
}

pub(crate) fn run_ensemble(
    config: Option<PathBuf>,
    ic: PathBuf,
    out_dir: PathBuf,
    n: usize,
    sigma_pos: f64,
    sigma_vel: f64,
    seed: u64,
    steps: usize,
    dt: f64,
    mode: String,
    em: bool,
    no_em: bool,
    em_model: Option<String>,
    no_gravity: bool,
) -> anyhow::Result<()> {
    if n == 0 {
        anyhow::bail!("--n must be >= 1");
    }
    if sigma_pos < 0.0 || sigma_vel < 0.0 {
        anyhow::bail!("sigma values must be >= 0");
    }
    if dt <= 0.0 || !dt.is_finite() {
        anyhow::bail!("--dt must be finite and > 0");
    }

    let cfg = crate::build_config(config, &mode, None, em, no_em, em_model, no_gravity)?;

    if !ic.exists() {
        anyhow::bail!("ic JSON not found: {}", ic.display());
    }
    let ic_json = fs::read_to_string(&ic)?;
    let base_ic: InitialConditionSpec = serde_json::from_str(&ic_json)?;
    let bounds = crate::default_ic_bounds();

    let out_dir = out_dir;
    ensure_dir_empty_or_create(&out_dir)?;

    let mut gauss = GaussianRng::new(seed);
    let mut members = Vec::with_capacity(n);

    for idx in 0..n {
        let member_dir = out_dir.join(member_dir_name(idx, n));
        fs::create_dir_all(&member_dir)?;

        let (member_ic, system) =
            sample_perturbed_system(&base_ic, &bounds, sigma_pos, sigma_vel, &mut gauss, idx)?;
        let options = SimOptions { steps, dt };
        let result = crate::simulate_with_cfg(system, &cfg, options);

        let csv_path = member_dir.join("traj.csv");
        let mut csv_file = fs::File::create(&csv_path)?;
        threebody_core::output::csv::write_csv(&mut csv_file, &result.steps, &cfg)?;

        let header = threebody_core::output::csv::csv_header(&cfg);
        let sidecar = build_sidecar(&cfg, &header, &result, Some(steps), Some(dt));
        let sidecar_path = member_dir.join("traj.json");
        let mut sidecar_file = fs::File::create(&sidecar_path)?;
        write_sidecar(&mut sidecar_file, &sidecar)?;

        let ic_out_path = member_dir.join("ic.json");
        fs::write(&ic_out_path, serde_json::to_string_pretty(&member_ic)?)?;

        members.push(EnsembleMember {
            index: idx,
            dir: rel_str(&member_dir, &out_dir),
            traj_csv: rel_str(&csv_path, &out_dir),
            sidecar_json: rel_str(&sidecar_path, &out_dir),
            ic_json: rel_str(&ic_out_path, &out_dir),
            terminated_early: sidecar.terminated_early,
            termination_reason: sidecar.termination_reason.clone(),
            warnings: sidecar.warnings.clone(),
            sim_stats: sidecar.sim_stats,
        });
    }

    let manifest = EnsembleManifest {
        version: PREDICTABILITY_VERSION.to_string(),
        n,
        seed,
        sigma_pos,
        sigma_vel,
        steps,
        dt,
        mode,
        config: cfg,
        members,
    };
    let manifest_path = out_dir.join("ensemble_manifest.json");
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    eprintln!("wrote manifest: {}", manifest_path.display());
    Ok(())
}

fn ensure_dir_empty_or_create(dir: &Path) -> anyhow::Result<()> {
    if dir.exists() {
        let mut has_entries = false;
        for entry in fs::read_dir(dir)? {
            let _ = entry?;
            has_entries = true;
            break;
        }
        if has_entries {
            anyhow::bail!(
                "output directory is not empty: {} (choose a new --out-dir)",
                dir.display()
            );
        }
        return Ok(());
    }
    fs::create_dir_all(dir)?;
    Ok(())
}

fn member_dir_name(idx: usize, n: usize) -> String {
    let width = (n.saturating_sub(1)).to_string().len().max(1);
    format!("member_{:0width$}", idx, width = width)
}

fn rel_str(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
}

fn clamp(v: f64, min: f64, max: f64) -> f64 {
    if !v.is_finite() {
        min
    } else {
        v.min(max).max(min)
    }
}

fn perturb_ic_once(
    base: &InitialConditionSpec,
    bounds: &IcBounds,
    sigma_pos: f64,
    sigma_vel: f64,
    rng: &mut GaussianRng,
) -> InitialConditionSpec {
    let mut spec = base.clone();
    for b in &mut spec.bodies {
        for k in 0..3 {
            let dp = sigma_pos * rng.next_normal();
            let dv = sigma_vel * rng.next_normal();
            b.pos[k] = clamp(b.pos[k] + dp, bounds.pos_min, bounds.pos_max);
            b.vel[k] = clamp(b.vel[k] + dv, bounds.vel_min, bounds.vel_max);
        }
    }
    spec.barycentric = true;
    spec
}

fn sample_perturbed_system(
    base: &InitialConditionSpec,
    bounds: &IcBounds,
    sigma_pos: f64,
    sigma_vel: f64,
    rng: &mut GaussianRng,
    member_idx: usize,
) -> anyhow::Result<(InitialConditionSpec, threebody_core::state::System)> {
    let attempts = 64usize;
    for attempt in 0..attempts {
        let mut spec = perturb_ic_once(base, bounds, sigma_pos, sigma_vel, rng);
        if spec.notes.trim().is_empty() {
            spec.notes = format!(
                "predictability_ensemble=true member={} attempt={}",
                member_idx, attempt
            );
        } else {
            spec.notes = format!(
                "{} | predictability_ensemble=true member={} attempt={}",
                spec.notes.trim(),
                member_idx,
                attempt
            );
        }
        match crate::system_from_ic(&spec, bounds) {
            Ok(system) => return Ok((spec, system)),
            Err(_) => continue,
        }
    }
    anyhow::bail!(
        "failed to sample a valid perturbed IC after {} attempts (try smaller sigmas)",
        attempts
    )
}

#[cfg(test)]
mod tests {
    use super::{EnsembleManifest, member_dir_name, run_ensemble};
    use std::fs;
    use std::path::PathBuf;

    fn temp_dir(prefix: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("threebody_{prefix}_{pid}_{nanos}"));
        p
    }

    #[test]
    fn member_dir_name_zero_pads() {
        assert_eq!(member_dir_name(0, 1), "member_0");
        assert_eq!(member_dir_name(3, 10), "member_3");
        assert_eq!(member_dir_name(3, 11), "member_03");
    }

    #[test]
    fn ensemble_writes_manifest_and_members() {
        let out_dir = temp_dir("ensemble_out");
        let ic_path = out_dir.join("ic.json");
        fs::create_dir_all(&out_dir).unwrap();
        // A safe IC with reasonable separation.
        let ic = serde_json::json!({
            "bodies": [
                {"mass": 1.0, "charge": 0.0, "pos": [-0.5, 0.0, 0.1], "vel": [0.0, 0.2, 0.0]},
                {"mass": 1.0, "charge": 0.0, "pos": [0.5, 0.0, -0.1], "vel": [0.0, -0.2, 0.0]},
                {"mass": 1.0, "charge": 0.0, "pos": [0.0, 1.0, 0.0], "vel": [0.0, 0.0, 0.0]}
            ],
            "barycentric": true,
            "notes": "test"
        });
        fs::write(&ic_path, serde_json::to_string_pretty(&ic).unwrap()).unwrap();

        let ensemble_dir = out_dir.join("ens");
        run_ensemble(
            None,
            ic_path.clone(),
            ensemble_dir.clone(),
            2,
            1e-4,
            1e-4,
            123,
            3,
            0.01,
            "truth".to_string(),
            false,
            false,
            None,
            false,
        )
        .unwrap();

        let manifest_path = ensemble_dir.join("ensemble_manifest.json");
        assert!(manifest_path.exists());
        let raw = fs::read_to_string(&manifest_path).unwrap();
        let manifest: EnsembleManifest = serde_json::from_str(&raw).unwrap();
        assert_eq!(manifest.n, 2);
        assert_eq!(manifest.members.len(), 2);
        for m in &manifest.members {
            assert!(ensemble_dir.join(&m.traj_csv).exists());
            assert!(ensemble_dir.join(&m.sidecar_json).exists());
            assert!(ensemble_dir.join(&m.ic_json).exists());
        }

        // Best-effort cleanup.
        let _ = fs::remove_dir_all(&out_dir);
    }
}
