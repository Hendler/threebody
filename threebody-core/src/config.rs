use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Constants {
    pub g: f64,
    pub k_e: f64,
    pub mu_0: f64,
}

impl Default for Constants {
    fn default() -> Self {
        Self {
            g: 1.0,
            k_e: 1.0,
            mu_0: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum IntegratorKind {
    Leapfrog,
    Rk45,
    Boris,
    ImplicitMidpoint,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct IntegratorConfig {
    pub kind: IntegratorKind,
    pub dt: f64,
    pub rtol: f64,
    pub atol: f64,
    pub dt_min: f64,
    pub dt_max: f64,
}

impl Default for IntegratorConfig {
    fn default() -> Self {
        Self {
            kind: IntegratorKind::Leapfrog,
            dt: 0.01,
            rtol: 1e-9,
            atol: 1e-12,
            dt_min: 1e-6,
            dt_max: 0.1,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct OutputConfig {
    pub include_fields: bool,
    pub include_potentials: bool,
    pub include_diagnostics: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            include_fields: true,
            include_potentials: true,
            include_diagnostics: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub constants: Constants,
    pub integrator: IntegratorConfig,
    pub softening: f64,
    pub enable_gravity: bool,
    pub enable_em: bool,
    pub output: OutputConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            constants: Constants::default(),
            integrator: IntegratorConfig::default(),
            softening: 0.0,
            enable_gravity: true,
            enable_em: true,
            output: OutputConfig::default(),
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if !self.constants.g.is_finite()
            || !self.constants.k_e.is_finite()
            || !self.constants.mu_0.is_finite()
        {
            return Err("constants must be finite".to_string());
        }
        if self.softening < 0.0 {
            return Err("softening must be >= 0".to_string());
        }
        if self.integrator.dt <= 0.0 {
            return Err("dt must be > 0".to_string());
        }
        if self.integrator.dt_min <= 0.0 || self.integrator.dt_max <= 0.0 {
            return Err("dt_min and dt_max must be > 0".to_string());
        }
        if self.integrator.dt_min > self.integrator.dt_max {
            return Err("dt_min must be <= dt_max".to_string());
        }
        if self.integrator.rtol < 0.0 || self.integrator.atol < 0.0 {
            return Err("tolerances must be >= 0".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, Constants, IntegratorConfig, IntegratorKind, OutputConfig};
    use serde_json;

    #[test]
    fn json_roundtrip() {
        let cfg = Config::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let decoded: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, decoded);
    }

    #[test]
    fn defaults_are_valid() {
        let cfg = Config::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn invalid_configs_are_rejected() {
        let mut cfg = Config {
            constants: Constants::default(),
            integrator: IntegratorConfig {
                kind: IntegratorKind::Rk45,
                dt: -1.0,
                ..IntegratorConfig::default()
            },
            softening: -0.1,
            enable_gravity: true,
            enable_em: false,
            output: OutputConfig::default(),
        };
        assert!(cfg.validate().is_err());

        cfg.integrator.dt = 0.1;
        cfg.softening = 0.0;
        cfg.integrator.dt_min = 1.0;
        cfg.integrator.dt_max = 0.5;
        assert!(cfg.validate().is_err());
    }
}
