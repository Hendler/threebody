use crate::equation::{score_equation, Equation, EquationScore, TopK};
use crate::llm::LlmClient;
use crate::Dataset;

pub fn grid_search(
    equations: &[Equation],
    dataset: &Dataset,
    llm: Option<&dyn LlmClient>,
) -> TopK {
    let mut topk = TopK::new(3);
    for eq in equations {
        topk.update(EquationScore {
            equation: eq.clone(),
            score: score_equation(eq, dataset),
        });
    }
    if let Some(client) = llm {
        if let Ok(ranked) = client.rank_equations(&topk.entries) {
            topk.entries = ranked;
        }
    }
    topk
}

#[cfg(test)]
mod tests {
    use super::grid_search;
    use crate::equation::{Dataset, Equation, Term};
    use crate::llm::MockLlm;

    #[test]
    fn grid_search_returns_top3() {
        let dataset = Dataset::new(
            vec!["x".to_string()],
            vec![vec![1.0], vec![2.0]],
            vec![1.0, 2.0],
        );
        let equations = vec![
            Equation { terms: vec![Term { feature: "x".to_string(), coeff: 1.0 }] },
            Equation { terms: vec![Term { feature: "x".to_string(), coeff: 2.0 }] },
            Equation { terms: vec![Term { feature: "x".to_string(), coeff: 0.5 }] },
            Equation { terms: vec![Term { feature: "x".to_string(), coeff: 3.0 }] },
        ];
        let topk = grid_search(&equations, &dataset, Some(&MockLlm));
        assert_eq!(topk.entries.len(), 3);
    }
}
