use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Term {
    pub feature: String,
    pub coeff: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Equation {
    pub terms: Vec<Term>,
}

impl Equation {
    pub fn predict(&self, dataset: &Dataset, sample: &[f64]) -> f64 {
        let mut sum = 0.0;
        for term in &self.terms {
            if let Some(&idx) = dataset.index.get(&term.feature) {
                if let Some(value) = sample.get(idx) {
                    sum += term.coeff * value;
                }
            }
        }
        sum
    }

    pub fn complexity(&self) -> usize {
        self.terms.len()
    }

    pub fn format(&self) -> String {
        if self.terms.is_empty() {
            return "0".to_string();
        }
        let mut parts = Vec::new();
        for term in &self.terms {
            parts.push(format!("{:+.6}*{}", term.coeff, term.feature));
        }
        parts.join(" ")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Dataset {
    pub feature_names: Vec<String>,
    pub samples: Vec<Vec<f64>>,
    pub targets: Vec<f64>,
    #[serde(skip)]
    pub index: HashMap<String, usize>,
}

impl Dataset {
    pub fn new(feature_names: Vec<String>, samples: Vec<Vec<f64>>, targets: Vec<f64>) -> Self {
        let mut index = HashMap::new();
        for (i, name) in feature_names.iter().enumerate() {
            index.insert(name.clone(), i);
        }
        Self {
            feature_names,
            samples,
            targets,
            index,
        }
    }

    pub fn with_index(mut self) -> Self {
        let mut index = HashMap::new();
        for (i, name) in self.feature_names.iter().enumerate() {
            index.insert(name.clone(), i);
        }
        self.index = index;
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EquationScore {
    pub equation: Equation,
    pub score: f64,
}

pub fn score_equation(eq: &Equation, dataset: &Dataset) -> f64 {
    let mut mse = 0.0;
    let n = dataset.samples.len().max(1) as f64;
    for (i, sample) in dataset.samples.iter().enumerate() {
        let pred = eq.predict(dataset, sample);
        let target = dataset.targets[i];
        let err = pred - target;
        mse += err * err;
    }
    mse / n
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TopK {
    pub k: usize,
    pub entries: Vec<EquationScore>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            entries: Vec::new(),
        }
    }

    pub fn update(&mut self, candidate: EquationScore) {
        self.entries.push(candidate);
        self.entries.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        if self.entries.len() > self.k {
            self.entries.truncate(self.k);
        }
    }

    pub fn best(&self) -> Option<&EquationScore> {
        self.entries.first()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equation_predicts_linear_combination() {
        let dataset = Dataset::new(
            vec!["x".to_string(), "y".to_string()],
            vec![vec![1.0, 2.0]],
            vec![5.0],
        );
        let eq = Equation {
            terms: vec![Term {
                feature: "x".to_string(),
                coeff: 1.0,
            }, Term {
                feature: "y".to_string(),
                coeff: 2.0,
            }],
        };
        let pred = eq.predict(&dataset, &dataset.samples[0]);
        assert!((pred - 5.0).abs() < 1e-12);
    }

    #[test]
    fn score_equation_is_mse() {
        let dataset = Dataset::new(
            vec!["x".to_string()],
            vec![vec![1.0], vec![2.0]],
            vec![1.0, 2.0],
        );
        let eq = Equation {
            terms: vec![Term {
                feature: "x".to_string(),
                coeff: 1.0,
            }],
        };
        let score = score_equation(&eq, &dataset);
        assert!(score < 1e-12);
    }

    #[test]
    fn topk_keeps_best_three() {
        let mut topk = TopK::new(3);
        for i in 0..5 {
            topk.update(EquationScore {
                equation: Equation { terms: vec![] },
                score: (5 - i) as f64,
            });
        }
        assert_eq!(topk.entries.len(), 3);
        assert!(topk.entries[0].score <= topk.entries[1].score);
    }
}
