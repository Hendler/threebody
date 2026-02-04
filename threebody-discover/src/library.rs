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
            .map(|name| match name.as_str() {
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
}
