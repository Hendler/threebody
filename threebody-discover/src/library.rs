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
                "r_inv2".to_string(),
                "r_inv3".to_string(),
                "v".to_string(),
                "v_cross_r".to_string(),
                "v_cross_v_cross_r".to_string(),
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
                "r_inv2" => FeatureDescription {
                    name: name.clone(),
                    description: "sum of inverse-square distances (proxy for gravity magnitude)".to_string(),
                    tags: vec!["distance".to_string(), "gravity".to_string()],
                },
                "r_inv3" => FeatureDescription {
                    name: name.clone(),
                    description: "sum of inverse-cube distances (directional gravity proxy)".to_string(),
                    tags: vec!["distance".to_string(), "gravity".to_string()],
                },
                "v" => FeatureDescription {
                    name: name.clone(),
                    description: "speed magnitude of the target body".to_string(),
                    tags: vec!["velocity".to_string()],
                },
                "v_cross_r" => FeatureDescription {
                    name: name.clone(),
                    description: "magnitude of v x r scaled by distance (magnetic-like proxy)".to_string(),
                    tags: vec!["velocity".to_string(), "cross".to_string(), "em".to_string()],
                },
                "v_cross_v_cross_r" => FeatureDescription {
                    name: name.clone(),
                    description: "magnitude of v x (v x r) scaled by distance (Lorentz-like proxy)".to_string(),
                    tags: vec!["velocity".to_string(), "cross".to_string(), "em".to_string()],
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
