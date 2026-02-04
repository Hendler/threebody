use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::{self, Write};

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::regime::RegimeDiagnostics;
use crate::sim::SimResult;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Sidecar {
    pub config: Config,
    pub header_hash: String,
    pub warnings: Vec<String>,
    pub last_regime: Option<RegimeDiagnostics>,
}

pub fn build_sidecar(cfg: &Config, header: &[String], result: &SimResult) -> Sidecar {
    let header_hash = hash_header(header);
    let last_regime = result.steps.last().map(|s| s.regime);
    Sidecar {
        config: *cfg,
        header_hash,
        warnings: result.warnings.clone(),
        last_regime,
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
    use crate::sim::{SimResult, SimStep};
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
            encounter: None,
            encounter_action: None,
            warnings: vec!["warn".to_string()],
            terminated_early: false,
            termination_reason: None,
        }
    }

    #[test]
    fn sidecar_roundtrip_and_hash() {
        let cfg = Config::default();
        let header = vec!["a".to_string(), "b".to_string()];
        let result = dummy_result();
        let sidecar = build_sidecar(&cfg, &header, &result);
        assert_eq!(sidecar.header_hash, hash_header(&header));
        let json = serde_json::to_string(&sidecar).unwrap();
        let decoded: Sidecar = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, sidecar);
    }
}
