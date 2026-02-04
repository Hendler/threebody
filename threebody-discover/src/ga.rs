use crate::equation::{score_equation_with, Equation, EquationScore, FitnessHeuristic, TopK};
use crate::library::FeatureLibrary;
use crate::Dataset;

#[derive(Clone, Debug)]
pub struct DiscoveryConfig {
    pub runs: usize,
    pub population: usize,
    pub max_terms: usize,
    pub mutation_rate: f64,
    pub seed: u64,
    pub fitness: FitnessHeuristic,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            runs: 50,
            population: 20,
            max_terms: 4,
            mutation_rate: 0.3,
            seed: 42,
            fitness: FitnessHeuristic::Mse,
        }
    }
}

pub fn run_search(
    dataset: &Dataset,
    library: &FeatureLibrary,
    cfg: &DiscoveryConfig,
) -> TopK {
    let mut rng = Lcg::new(cfg.seed);
    let mut population: Vec<Equation> = (0..cfg.population)
        .map(|_| library.random_equation(&mut rng, cfg.max_terms))
        .collect();
    let mut topk = TopK::new(3);

    for _ in 0..cfg.runs {
        let mut scored: Vec<EquationScore> = population
            .iter()
            .cloned()
            .map(|eq| EquationScore {
                score: score_equation_with(&eq, dataset, cfg.fitness),
                equation: eq,
            })
            .collect();
        scored.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        for s in &scored {
            topk.update(s.clone());
        }
        let survivors = scored.iter().take((cfg.population / 2).max(1)).cloned().collect::<Vec<_>>();
        population = survivors
            .iter()
            .map(|s| s.equation.clone())
            .collect();
        while population.len() < cfg.population {
            let parent = &survivors[rng.gen_range_usize(0, survivors.len() - 1)].equation;
            let child = mutate(parent, library, cfg, &mut rng);
            population.push(child);
        }
    }
    topk
}

fn mutate(eq: &Equation, library: &FeatureLibrary, cfg: &DiscoveryConfig, rng: &mut Lcg) -> Equation {
    let mut terms = eq.terms.clone();
    if terms.is_empty() {
        return library.random_equation(rng, cfg.max_terms);
    }
    let roll = rng.next_f64();
    if roll < cfg.mutation_rate {
        let idx = rng.gen_range_usize(0, terms.len() - 1);
        terms[idx].coeff += rng.gen_range_f64(-0.5, 0.5);
    } else if roll < cfg.mutation_rate * 1.5 && terms.len() < cfg.max_terms {
        terms.push(library.random_equation(rng, 1).terms[0].clone());
    } else if terms.len() > 1 {
        terms.remove(rng.gen_range_usize(0, terms.len() - 1));
    }
    Equation { terms }
}

#[derive(Clone, Debug)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    pub fn next_f64(&mut self) -> f64 {
        let v = self.next_u64();
        (v as f64) / (u64::MAX as f64)
    }

    pub fn gen_range_f64(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next_f64()
    }

    pub fn gen_range_usize(&mut self, min: usize, max: usize) -> usize {
        if min == max {
            return min;
        }
        let span = max - min + 1;
        min + (self.next_u64() as usize % span)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equation::Dataset;
    use crate::library::FeatureLibrary;

    fn dataset() -> Dataset {
        Dataset::new(
            vec!["x".to_string()],
            vec![vec![1.0], vec![2.0], vec![3.0]],
            vec![1.0, 2.0, 3.0],
        )
    }

    #[test]
    fn search_returns_top3() {
        let data = dataset();
        let lib = FeatureLibrary::default_physics();
        let cfg = DiscoveryConfig {
            runs: 5,
            population: 6,
            max_terms: 2,
            mutation_rate: 0.5,
            seed: 1,
            fitness: FitnessHeuristic::Mse,
        };
        let topk = run_search(&data, &lib, &cfg);
        assert_eq!(topk.entries.len(), 3);
    }

    #[test]
    fn more_runs_not_worse_best() {
        let data = dataset();
        let lib = FeatureLibrary::default_physics();
        let cfg1 = DiscoveryConfig {
            runs: 2,
            population: 6,
            max_terms: 2,
            mutation_rate: 0.5,
            seed: 2,
            fitness: FitnessHeuristic::Mse,
        };
        let cfg2 = DiscoveryConfig {
            runs: 6,
            population: 6,
            max_terms: 2,
            mutation_rate: 0.5,
            seed: 2,
            fitness: FitnessHeuristic::Mse,
        };
        let top1 = run_search(&data, &lib, &cfg1);
        let top2 = run_search(&data, &lib, &cfg2);
        let best1 = top1.best().unwrap().score;
        let best2 = top2.best().unwrap().score;
        assert!(best2 <= best1 + 1e-12);
    }
}
