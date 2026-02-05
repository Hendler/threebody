use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::{self, Write};

use serde::{Deserialize, Serialize};

use crate::config::{CloseEncounterAction, Config};
use crate::regime::RegimeDiagnostics;
use crate::sim::{SimResult, SimStats};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SidecarInitialState {
    pub mass: [f64; 3],
    pub charge: [f64; 3],
    pub pos: [[f64; 3]; 3],
    pub vel: [[f64; 3]; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Sidecar {
    pub config: Config,
    pub header_hash: String,
    pub warnings: Vec<String>,
    pub last_regime: Option<RegimeDiagnostics>,
    pub sim_stats: SimStats,
    pub initial_state: Option<SidecarInitialState>,
    /// Requested number of integration steps (not counting the initial state at step 0).
    /// Optional for backward compatibility with older sidecars.
    #[serde(default)]
    pub requested_steps: Option<usize>,
    /// Requested base dt (even when adaptive integrators are used).
    /// Optional for backward compatibility with older sidecars.
    #[serde(default)]
    pub requested_dt: Option<f64>,
    /// Whether the simulator terminated before completing the requested horizon.
    #[serde(default)]
    pub terminated_early: bool,
    /// Machine-readable termination reason (e.g., `max_rejects_exceeded`).
    #[serde(default)]
    pub termination_reason: Option<String>,
    #[serde(default)]
    pub encounter: Option<SidecarEncounterEvent>,
    #[serde(default)]
    pub encounter_action: Option<CloseEncounterAction>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SidecarEncounterEvent {
    pub step: usize,
    pub min_pair_dist: f64,
}

pub fn build_sidecar(
    cfg: &Config,
    header: &[String],
    result: &SimResult,
    requested_steps: Option<usize>,
    requested_dt: Option<f64>,
) -> Sidecar {
    let header_hash = hash_header(header);
    let last_regime = result.steps.last().map(|s| s.regime);
    let initial_state = result.steps.first().map(|s| {
        let mut mass = [0.0; 3];
        let mut charge = [0.0; 3];
        let mut pos = [[0.0; 3]; 3];
        let mut vel = [[0.0; 3]; 3];
        for i in 0..3 {
            mass[i] = s.system.bodies[i].mass;
            charge[i] = s.system.bodies[i].charge;
            let p = s.system.state.pos[i];
            let v = s.system.state.vel[i];
            pos[i] = [p.x, p.y, p.z];
            vel[i] = [v.x, v.y, v.z];
        }
        SidecarInitialState {
            mass,
            charge,
            pos,
            vel,
        }
    });
    Sidecar {
        config: *cfg,
        header_hash,
        warnings: result.warnings.clone(),
        last_regime,
        sim_stats: result.stats,
        initial_state,
        requested_steps,
        requested_dt,
        terminated_early: result.terminated_early,
        termination_reason: result.termination_reason.clone(),
        encounter: result.encounter.map(|e| SidecarEncounterEvent {
            step: e.step,
            min_pair_dist: e.min_pair_dist,
        }),
        encounter_action: result.encounter_action,
    }
}

pub fn write_sidecar<W: Write>(mut writer: W, sidecar: &Sidecar) -> io::Result<()> {
    let json = serde_json::to_string_pretty(sidecar).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    writer.write_all(json.as_bytes())
}

fn hash_header(header: &[String]) -> String {
    let mut hasher = DefaultHasher::new();
    header.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::{build_sidecar, hash_header, Sidecar};
    use crate::config::Config;
    use crate::sim::{EncounterEvent, SimResult, SimStats, SimStep};
    use crate::state::{Body, State, System};
    use crate::math::vec3::Vec3;
    use crate::diagnostics::Diagnostics;
    use crate::regime::RegimeDiagnostics;

    fn dummy_result() -> SimResult {
        let system = System::new(
            [Body::new(1.0, 0.0), Body::new(0.0, 0.0), Body::new(0.0, 0.0)],
            State::new([Vec3::zero(); 3], [Vec3::zero(); 3]),
        );
        let step = SimStep {
            system,
            diagnostics: Diagnostics {
                linear_momentum: Vec3::zero(),
                angular_momentum: Vec3::zero(),
                energy_proxy: 0.0,
            },
            regime: RegimeDiagnostics {
                min_pair_dist: 0.0,
                max_speed: 0.0,
                max_accel: 0.0,
                dt_ratio: 0.0,
            },
            t: 0.0,
            dt: 0.1,
        };
        SimResult {
            steps: vec![step],
            encounter: Some(EncounterEvent {
                step: 0,
                min_pair_dist: 0.05,
            }),
            encounter_action: Some(crate::config::CloseEncounterAction::StopAndReport),
            warnings: vec!["warn".to_string()],
            terminated_early: true,
            termination_reason: Some("max_rejects_exceeded".to_string()),
            stats: SimStats {
                accepted_steps: 1,
                rejected_steps: 0,
                dt_min: Some(0.1),
                dt_max: Some(0.1),
                dt_avg: Some(0.1),
            },
        }
    }

    #[test]
    fn sidecar_roundtrip_and_hash() {
        let cfg = Config::default();
        let header = vec!["a".to_string(), "b".to_string()];
        let result = dummy_result();
        let sidecar = build_sidecar(&cfg, &header, &result, Some(5), Some(0.01));
        assert_eq!(sidecar.header_hash, hash_header(&header));
        let json = serde_json::to_string(&sidecar).unwrap();
        let decoded: Sidecar = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, sidecar);
    }

    #[test]
    fn sidecar_deserializes_missing_new_fields_with_defaults() {
        let cfg = Config::default();
        let legacy = serde_json::json!({
            "config": cfg,
            "header_hash": "deadbeef",
            "warnings": [],
            "last_regime": null,
            "sim_stats": {
                "accepted_steps": 0,
                "rejected_steps": 0,
                "dt_min": null,
                "dt_max": null,
                "dt_avg": null
            },
            "initial_state": null
        });
        let decoded: Sidecar = serde_json::from_value(legacy).expect("should decode legacy sidecar");
        assert!(decoded.requested_steps.is_none());
        assert!(decoded.requested_dt.is_none());
        assert!(!decoded.terminated_early);
        assert!(decoded.termination_reason.is_none());
        assert!(decoded.encounter.is_none());
        assert!(decoded.encounter_action.is_none());
    }
}
