use crate::equation::{Equation, Term};
use crate::judge::FeatureDescription;

#[derive(Clone, Debug)]
pub struct FeatureLibrary {
    pub features: Vec<String>,
}

impl FeatureLibrary {
    pub fn default_physics() -> Self {
        Self {
            features: vec![
                "grav_x".to_string(),
                "grav_y".to_string(),
                "grav_z".to_string(),
                "elec_x".to_string(),
                "elec_y".to_string(),
                "elec_z".to_string(),
                "mag_x".to_string(),
                "mag_y".to_string(),
                "mag_z".to_string(),
            ],
        }
    }

    /// A minimal EM library that learns Lorentz acceleration directly from physical fields.
    ///
    /// Features are in acceleration units (already include physical constants from the simulator):
    /// - `lorentz_e_*` = (q_i/m_i) * E_i,*
    /// - `lorentz_vxb_*` = (q_i/m_i) * (v_i × B_i),*
    ///
    /// Gravity is included via the existing `grav_*` basis (no G).
    pub fn em_fields_lorentz() -> Self {
        let mut features = vec![
            "grav_x".to_string(),
            "grav_y".to_string(),
            "grav_z".to_string(),
            "lorentz_e_x".to_string(),
            "lorentz_e_y".to_string(),
            "lorentz_e_z".to_string(),
            "lorentz_vxb_x".to_string(),
            "lorentz_vxb_y".to_string(),
            "lorentz_vxb_z".to_string(),
        ];
        features.sort();
        features.dedup();
        Self { features }
    }

    pub fn extended_physics() -> Self {
        let mut features = Self::default_physics().features;

        for axis in ["x", "y", "z"] {
            features.push(format!("grav_r4_{axis}"));
            features.push(format!("elec_r4_{axis}"));
            features.push(format!("mag_r4_{axis}"));
        }

        features.push("gate_close".to_string());
        features.push("gate_far".to_string());
        for axis in ["x", "y", "z"] {
            features.push(format!("grav_close_{axis}"));
            features.push(format!("grav_far_{axis}"));
            features.push(format!("elec_close_{axis}"));
            features.push(format!("elec_far_{axis}"));
            features.push(format!("mag_close_{axis}"));
            features.push(format!("mag_far_{axis}"));
        }

        features.sort();
        features.dedup();
        Self { features }
    }

    pub fn random_equation(&self, rng: &mut crate::ga::Lcg, max_terms: usize) -> Equation {
        let mut terms = Vec::new();
        let term_count = rng.gen_range_usize(1, max_terms.max(1));
        for _ in 0..term_count {
            let feature = self.features[rng.gen_range_usize(0, self.features.len() - 1)].clone();
            let coeff = rng.gen_range_f64(-2.0, 2.0);
            terms.push(Term { feature, coeff });
        }
        Equation { terms }
    }

    pub fn feature_descriptions(&self) -> Vec<FeatureDescription> {
        self.features
            .iter()
            .map(|name| {
                if let Some(axis) = name.strip_prefix("lorentz_e_") {
                    return FeatureDescription {
                        name: name.clone(),
                        description: format!(
                            "Lorentz electric term {axis}-component: (q_i/m_i) E_i,{axis} (includes k_e and softening)"
                        ),
                        tags: vec!["em".to_string(), "fields".to_string(), "lorentz".to_string()],
                    };
                }
                if let Some(axis) = name.strip_prefix("lorentz_vxb_") {
                    return FeatureDescription {
                        name: name.clone(),
                        description: format!(
                            "Lorentz magnetic term {axis}-component: (q_i/m_i) (v_i×B_i)_{axis} (includes μ0 and softening)"
                        ),
                        tags: vec!["em".to_string(), "fields".to_string(), "lorentz".to_string()],
                    };
                }
                if let Some(axis) = name.strip_prefix("grav_r4_") {
                    return FeatureDescription {
                        name: name.clone(),
                        description: format!(
                            "gravity r4 basis {axis}-component: Σ m_j (r_j - r_i)_{axis} / |r_j-r_i|^4 (no G)"
                        ),
                        tags: vec!["gravity".to_string(), "distance_scaling".to_string()],
                    };
                }
                if let Some(axis) = name.strip_prefix("elec_r4_") {
                    return FeatureDescription {
                        name: name.clone(),
                        description: format!(
                            "electric r4 basis {axis}-component: (q_i/m_i) Σ q_j (r_i - r_j)_{axis} / |r_i-r_j|^4 (no k_e)"
                        ),
                        tags: vec!["em".to_string(), "electric".to_string(), "distance_scaling".to_string()],
                    };
                }
                if let Some(axis) = name.strip_prefix("mag_r4_") {
                    return FeatureDescription {
                        name: name.clone(),
                        description: format!(
                            "magnetic r4 basis {axis}-component: (q_i/m_i) (v_i×B_r4)_{{{axis}}} where B_r4=(1/4π)Σ q_j(v_j×(r_i-r_j))/|r|^4 (no μ0)"
                        ),
                        tags: vec!["em".to_string(), "magnetic".to_string(), "distance_scaling".to_string()],
                    };
                }
                if name == "gate_close" {
                    return FeatureDescription {
                        name: name.clone(),
                        description: "binary gate: 1 when min_pair_dist < r_gate, else 0 (used for piecewise models)".to_string(),
                        tags: vec!["gate".to_string()],
                    };
                }
                if name == "gate_far" {
                    return FeatureDescription {
                        name: name.clone(),
                        description: "binary gate: 1 when min_pair_dist >= r_gate, else 0 (used for piecewise models)".to_string(),
                        tags: vec!["gate".to_string()],
                    };
                }
                for (prefix, tag) in [("grav", "gravity"), ("elec", "electric"), ("mag", "magnetic")] {
                    if let Some(axis) = name.strip_prefix(&format!("{prefix}_close_")) {
                        return FeatureDescription {
                            name: name.clone(),
                            description: format!("gated ({prefix}) close {axis}-component: gate_close * {prefix}_{axis}"),
                            tags: vec![tag.to_string(), "gate".to_string()],
                        };
                    }
                    if let Some(axis) = name.strip_prefix(&format!("{prefix}_far_")) {
                        return FeatureDescription {
                            name: name.clone(),
                            description: format!("gated ({prefix}) far {axis}-component: gate_far * {prefix}_{axis}"),
                            tags: vec![tag.to_string(), "gate".to_string()],
                        };
                    }
                }

                match name.as_str() {
                "grav_x" => FeatureDescription {
                    name: name.clone(),
                    description: "gravitational basis x-component: Σ m_j (r_j - r_i)_x / |r_j-r_i|^3 (no G)".to_string(),
                    tags: vec!["gravity".to_string(), "vector_component".to_string()],
                },
                "grav_y" => FeatureDescription {
                    name: name.clone(),
                    description: "gravitational basis y-component: Σ m_j (r_j - r_i)_y / |r_j-r_i|^3 (no G)".to_string(),
                    tags: vec!["gravity".to_string(), "vector_component".to_string()],
                },
                "grav_z" => FeatureDescription {
                    name: name.clone(),
                    description: "gravitational basis z-component: Σ m_j (r_j - r_i)_z / |r_j-r_i|^3 (no G)".to_string(),
                    tags: vec!["gravity".to_string(), "vector_component".to_string()],
                },
                "elec_x" => FeatureDescription {
                    name: name.clone(),
                    description: "electric basis x-component: (q_i/m_i) Σ q_j (r_i - r_j)_x / |r_i-r_j|^3 (no k_e)".to_string(),
                    tags: vec!["em".to_string(), "electric".to_string(), "vector_component".to_string()],
                },
                "elec_y" => FeatureDescription {
                    name: name.clone(),
                    description: "electric basis y-component: (q_i/m_i) Σ q_j (r_i - r_j)_y / |r_i-r_j|^3 (no k_e)".to_string(),
                    tags: vec!["em".to_string(), "electric".to_string(), "vector_component".to_string()],
                },
                "elec_z" => FeatureDescription {
                    name: name.clone(),
                    description: "electric basis z-component: (q_i/m_i) Σ q_j (r_i - r_j)_z / |r_i-r_j|^3 (no k_e)".to_string(),
                    tags: vec!["em".to_string(), "electric".to_string(), "vector_component".to_string()],
                },
                "mag_x" => FeatureDescription {
                    name: name.clone(),
                    description: "magnetic basis x-component: (q_i/m_i) (v_i × B_basis)_x where B_basis=(1/4π)Σ q_j (v_j×(r_i-r_j))/|r|^3 (no μ0)".to_string(),
                    tags: vec!["em".to_string(), "magnetic".to_string(), "vector_component".to_string()],
                },
                "mag_y" => FeatureDescription {
                    name: name.clone(),
                    description: "magnetic basis y-component: (q_i/m_i) (v_i × B_basis)_y (no μ0)".to_string(),
                    tags: vec!["em".to_string(), "magnetic".to_string(), "vector_component".to_string()],
                },
                "mag_z" => FeatureDescription {
                    name: name.clone(),
                    description: "magnetic basis z-component: (q_i/m_i) (v_i × B_basis)_z (no μ0)".to_string(),
                    tags: vec!["em".to_string(), "magnetic".to_string(), "vector_component".to_string()],
                },
                _ => FeatureDescription {
                    name: name.clone(),
                    description: "feature".to_string(),
                    tags: vec![],
                },
            }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::FeatureLibrary;
    use crate::ga::Lcg;

    #[test]
    fn library_builds_random_equation() {
        let library = FeatureLibrary::default_physics();
        let mut rng = Lcg::new(123);
        let eq = library.random_equation(&mut rng, 3);
        assert!(!eq.terms.is_empty());
    }

    #[test]
    fn extended_library_includes_expected_features() {
        let lib = FeatureLibrary::extended_physics();
        for f in [
            "grav_x",
            "elec_y",
            "mag_z",
            "grav_r4_x",
            "elec_r4_y",
            "mag_r4_z",
            "gate_close",
            "gate_far",
            "grav_close_x",
            "grav_far_x",
        ] {
            assert!(lib.features.iter().any(|name| name == f), "missing {f}");
        }
    }

    #[test]
    fn em_fields_library_includes_expected_features() {
        let lib = FeatureLibrary::em_fields_lorentz();
        for f in [
            "grav_x",
            "grav_y",
            "grav_z",
            "lorentz_e_x",
            "lorentz_e_y",
            "lorentz_e_z",
            "lorentz_vxb_x",
            "lorentz_vxb_y",
            "lorentz_vxb_z",
        ] {
            assert!(lib.features.iter().any(|name| name == f), "missing {f}");
        }
    }
}
