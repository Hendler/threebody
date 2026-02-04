use crate::equation::Equation;
use serde::{Deserialize, Serialize};

pub const RUBRIC_VERSION: &str = "v1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoreScale {
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RubricWeights {
    pub fidelity: f64,
    pub parsimony: f64,
    pub physical_plausibility: f64,
    pub regime_consistency: f64,
    pub stability_risk: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Rubric {
    pub version: String,
    pub scale: ScoreScale,
    pub weights: RubricWeights,
    pub notes: Vec<String>,
}

impl Rubric {
    pub fn default_rubric() -> Self {
        Self {
            version: RUBRIC_VERSION.to_string(),
            scale: ScoreScale { min: 0.0, max: 5.0 },
            weights: RubricWeights {
                fidelity: 0.45,
                parsimony: 0.20,
                physical_plausibility: 0.15,
                regime_consistency: 0.10,
                stability_risk: 0.10,
            },
            notes: vec![
                "Fidelity is primary; do not up-rank higher-error models unless evidence shows improved stability or regime validity."
                    .to_string(),
                "Use only provided evidence; if unknown, score low and state uncertainty.".to_string(),
                "Separate observation (metrics) from inference; avoid new physical claims.".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidateMetrics {
    pub mse: f64,
    pub complexity: usize,
    pub rollout_rmse: Option<f64>,
    pub divergence_time: Option<f64>,
    pub stability_flags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidateSummary {
    pub id: usize,
    pub equation: Equation,
    pub equation_text: String,
    pub metrics: CandidateMetrics,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetSummary {
    pub n_samples: usize,
    pub target_description: String,
    pub feature_names: Vec<String>,
    pub feature_descriptions: Vec<FeatureDescription>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeatureDescription {
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimulationSummary {
    pub steps: usize,
    pub energy_start: Option<f64>,
    pub energy_end: Option<f64>,
    pub energy_drift: Option<f64>,
    pub min_pair_dist: Option<f64>,
    pub max_speed: Option<f64>,
    pub max_accel: Option<f64>,
    pub dt_min: Option<f64>,
    pub dt_max: Option<f64>,
    pub dt_avg: Option<f64>,
    pub warnings: Vec<String>,
    pub rollout_integrator: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IcBounds {
    pub mass_min: f64,
    pub mass_max: f64,
    pub charge_min: f64,
    pub charge_max: f64,
    pub pos_min: f64,
    pub pos_max: f64,
    pub vel_min: f64,
    pub vel_max: f64,
    pub min_pair_dist: f64,
    pub recommend_barycentric: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IcRequest {
    pub bounds: IcBounds,
    pub regime: String,
    pub notes: Vec<String>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BodyInit {
    pub mass: f64,
    pub charge: f64,
    pub pos: [f64; 3],
    pub vel: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InitialConditionSpec {
    pub bodies: Vec<BodyInit>,
    pub barycentric: bool,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JudgeInput {
    pub rubric: Rubric,
    pub regime: String,
    pub dataset: DatasetSummary,
    pub simulation: Option<SimulationSummary>,
    pub candidates: Vec<CandidateSummary>,
    pub ic_bounds: IcBounds,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoreComponents {
    pub fidelity: f64,
    pub parsimony: f64,
    pub physical_plausibility: f64,
    pub regime_consistency: f64,
    pub stability_risk: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JudgeScore {
    pub id: usize,
    pub total: f64,
    pub components: ScoreComponents,
    pub rationale: String,
    pub flags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JudgeRecommendations {
    pub next_initial_conditions: Option<InitialConditionSpec>,
    pub next_rollout_integrator: Option<String>,
    pub next_ga_heuristic: Option<String>,
    pub next_discovery_solver: Option<String>,
    pub next_normalize: Option<bool>,
    pub next_stls_threshold: Option<f64>,
    pub next_ridge_lambda: Option<f64>,
    pub next_lasso_alpha: Option<f64>,
    pub next_search_directions: Vec<String>,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JudgeResponse {
    pub version: String,
    pub ranking: Vec<usize>,
    pub scores: Vec<JudgeScore>,
    pub recommendations: JudgeRecommendations,
    pub summary: String,
}

impl JudgeResponse {
    pub fn validate(&self, input: &JudgeInput) -> Result<(), String> {
        if self.version != input.rubric.version {
            return Err("rubric version mismatch".to_string());
        }
        let scale = &input.rubric.scale;
        let weight_sum = input.rubric.weights.fidelity
            + input.rubric.weights.parsimony
            + input.rubric.weights.physical_plausibility
            + input.rubric.weights.regime_consistency
            + input.rubric.weights.stability_risk;
        let min_total = scale.min * weight_sum;
        let max_total = scale.max * weight_sum;
        for score in &self.scores {
            if !input.candidates.iter().any(|c| c.id == score.id) {
                return Err(format!("unknown candidate id: {}", score.id));
            }
            for val in [
                score.components.fidelity,
                score.components.parsimony,
                score.components.physical_plausibility,
                score.components.regime_consistency,
                score.components.stability_risk,
            ] {
                if !val.is_finite() || val < scale.min || val > scale.max {
                    return Err("component score out of range".to_string());
                }
            }
            if !score.total.is_finite() || score.total < min_total || score.total > max_total {
                return Err("total score out of range".to_string());
            }
        }
        if !self.ranking.is_empty() {
            let mut seen = std::collections::HashSet::new();
            for id in &self.ranking {
                if !input.candidates.iter().any(|c| c.id == *id) {
                    return Err("ranking contains unknown id".to_string());
                }
                if !seen.insert(*id) {
                    return Err("ranking contains duplicate id".to_string());
                }
            }
        }

        if let Some(name) = self.recommendations.next_rollout_integrator.as_deref() {
            if name != "euler" && name != "leapfrog" {
                return Err("invalid next_rollout_integrator".to_string());
            }
        }
        if let Some(name) = self.recommendations.next_ga_heuristic.as_deref() {
            if name != "mse" && name != "mse_parsimony" {
                return Err("invalid next_ga_heuristic".to_string());
            }
        }
        if let Some(name) = self.recommendations.next_discovery_solver.as_deref() {
            if name != "stls" && name != "lasso" && name != "ga" {
                return Err("invalid next_discovery_solver".to_string());
            }
        }
        for val in [
            self.recommendations.next_stls_threshold,
            self.recommendations.next_ridge_lambda,
            self.recommendations.next_lasso_alpha,
        ] {
            if let Some(v) = val {
                if !v.is_finite() || v < 0.0 {
                    return Err("invalid solver hyperparameter".to_string());
                }
            }
        }
        Ok(())
    }
}

pub fn build_ic_prompt(request: &IcRequest) -> String {
    let b = &request.bounds;
    let mut prompt = String::new();
    prompt.push_str("You are selecting initial conditions for a 3-body simulation.\n");
    prompt.push_str("Return ONLY valid JSON that matches the schema below. No extra text.\n");
    prompt.push_str("Constraints:\n");
    prompt.push_str(&format!(
        "- mass in [{:.3}, {:.3}]\n- charge in [{:.3}, {:.3}]\n- position components in [{:.3}, {:.3}]\n- velocity components in [{:.3}, {:.3}]\n- minimum pair distance >= {:.3}\n",
        b.mass_min, b.mass_max, b.charge_min, b.charge_max, b.pos_min, b.pos_max, b.vel_min, b.vel_max, b.min_pair_dist
    ));
    if b.recommend_barycentric {
        prompt.push_str("- barycentric = true is recommended.\n");
    }
    prompt.push_str(&format!("Regime: {}\n", request.regime));
    if !request.notes.is_empty() {
        prompt.push_str("Notes:\n");
        for note in &request.notes {
            prompt.push_str(&format!("- {}\n", note));
        }
    }
    prompt.push_str("Schema:\n");
    prompt.push_str("{\"bodies\":[{\"mass\":1.0,\"charge\":0.0,\"pos\":[0,0,0],\"vel\":[0,0,0]}, ... x3],\"barycentric\":true,\"notes\":\"...\"}\n");
    prompt.push_str("Output JSON only.\n");
    prompt
}

pub fn build_judge_prompt(input: &JudgeInput) -> String {
    let mut prompt = String::new();
    prompt.push_str("You are an academic reviewer evaluating candidate equations for a dynamical system.\n");
    prompt.push_str("Your role is advisory and supplemental. The primary numeric ranking is based on MSE.\n");
    prompt.push_str("Use ONLY the evidence provided. If evidence is missing, score low and note uncertainty.\n");
    prompt.push_str("Return JSON only. Do not include Markdown or extra text.\n\n");

    prompt.push_str("Rubric (0-5 for each component):\n");
    prompt.push_str(&format!(
        "Weights: fidelity={:.2}, parsimony={:.2}, physical_plausibility={:.2}, regime_consistency={:.2}, stability_risk={:.2}\n",
        input.rubric.weights.fidelity,
        input.rubric.weights.parsimony,
        input.rubric.weights.physical_plausibility,
        input.rubric.weights.regime_consistency,
        input.rubric.weights.stability_risk
    ));
    for note in &input.rubric.notes {
        prompt.push_str(&format!("- {}\n", note));
    }
    prompt.push_str("- Compute total = sum(weight_i * component_i).\n");
    prompt.push_str("- If MSE values are within 5%, you may break ties using parsimony and plausibility.\n");
    prompt.push_str("\nDataset summary:\n");
    prompt.push_str(&format!(
        "samples={}, target={}\n",
        input.dataset.n_samples, input.dataset.target_description
    ));
    prompt.push_str("Features:\n");
    for fd in &input.dataset.feature_descriptions {
        prompt.push_str(&format!(
            "- {}: {} (tags: {})\n",
            fd.name,
            fd.description,
            fd.tags.join(", ")
        ));
    }
    prompt.push_str(&format!("\nRegime: {}\n", input.regime));
    if let Some(sim) = &input.simulation {
        prompt.push_str("Simulation summary:\n");
        prompt.push_str(&format!(
            "steps={}, energy_start={:?}, energy_end={:?}, energy_drift={:?}, min_pair_dist={:?}, max_speed={:?}, max_accel={:?}, dt_min={:?}, dt_max={:?}, dt_avg={:?}\n",
            sim.steps,
            sim.energy_start,
            sim.energy_end,
            sim.energy_drift,
            sim.min_pair_dist,
            sim.max_speed,
            sim.max_accel,
            sim.dt_min,
            sim.dt_max,
            sim.dt_avg
        ));
        prompt.push_str(&format!("rollout_integrator={}\n", sim.rollout_integrator));
        if !sim.warnings.is_empty() {
            prompt.push_str(&format!("warnings: {}\n", sim.warnings.join("; ")));
        }
    }
    prompt.push_str("\nCandidates:\n");
    for c in &input.candidates {
        prompt.push_str(&format!(
            "id={}, eq=\"{}\", mse={:.6}, complexity={}, rollout_rmse={:?}, divergence_time={:?}, flags={}\n",
            c.id,
            c.equation_text,
            c.metrics.mse,
            c.metrics.complexity,
            c.metrics.rollout_rmse,
            c.metrics.divergence_time,
            c.metrics.stability_flags.join(", ")
        ));
        if !c.notes.is_empty() {
            prompt.push_str(&format!("notes: {}\n", c.notes.join("; ")));
        }
    }
    prompt.push_str("\nInitial condition bounds for next run:\n");
    let b = &input.ic_bounds;
    prompt.push_str(&format!(
        "mass=[{:.3},{:.3}], charge=[{:.3},{:.3}], pos=[{:.3},{:.3}], vel=[{:.3},{:.3}], min_pair_dist>={:.3}, barycentric_recommended={}\n",
        b.mass_min, b.mass_max, b.charge_min, b.charge_max, b.pos_min, b.pos_max, b.vel_min, b.vel_max, b.min_pair_dist, b.recommend_barycentric
    ));

    if !input.notes.is_empty() {
        prompt.push_str("\nAdditional notes:\n");
        for note in &input.notes {
            prompt.push_str(&format!("- {}\n", note));
        }
    }

    prompt.push_str("\nOutput JSON schema:\n");
    prompt.push_str("Allowed next_rollout_integrator: \"euler\" or \"leapfrog\".\n");
    prompt.push_str("Allowed next_ga_heuristic: \"mse\" or \"mse_parsimony\".\n");
    prompt.push_str("Allowed next_discovery_solver: \"stls\", \"lasso\", or \"ga\".\n");
    prompt.push_str("{\n");
    prompt.push_str("  \"version\": \"v1\",\n");
    prompt.push_str("  \"ranking\": [0,1,2],\n");
    prompt.push_str("  \"scores\": [\n");
    prompt.push_str("    {\"id\":0,\"total\":3.5,\"components\":{\"fidelity\":3,\"parsimony\":4,\"physical_plausibility\":3,\"regime_consistency\":4,\"stability_risk\":3},\"rationale\":\"...\",\"flags\":[\"...\"]}\n");
    prompt.push_str("  ],\n");
    prompt.push_str("  \"recommendations\": {\n");
    prompt.push_str("    \"next_initial_conditions\": {\"bodies\":[{\"mass\":1.0,\"charge\":0.0,\"pos\":[0,0,0],\"vel\":[0,0,0]}],\"barycentric\":true,\"notes\":\"...\"},\n");
    prompt.push_str("    \"next_rollout_integrator\": \"euler\",\n");
    prompt.push_str("    \"next_ga_heuristic\": \"mse\",\n");
    prompt.push_str("    \"next_discovery_solver\": \"stls\",\n");
    prompt.push_str("    \"next_normalize\": true,\n");
    prompt.push_str("    \"next_stls_threshold\": 0.1,\n");
    prompt.push_str("    \"next_ridge_lambda\": 1e-8,\n");
    prompt.push_str("    \"next_lasso_alpha\": 0.01,\n");
    prompt.push_str("    \"next_search_directions\": [\"...\"],\n");
    prompt.push_str("    \"notes\": \"...\"\n");
    prompt.push_str("  },\n");
    prompt.push_str("  \"summary\": \"...\"\n");
    prompt.push_str("}\n");
    prompt
}

pub const FACTORY_EVALUATION_VERSION: &str = "v1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JudgeRecommendationsLite {
    pub next_rollout_integrator: Option<String>,
    pub next_ga_heuristic: Option<String>,
    pub next_discovery_solver: Option<String>,
    pub next_normalize: Option<bool>,
    pub next_stls_threshold: Option<f64>,
    pub next_ridge_lambda: Option<f64>,
    pub next_lasso_alpha: Option<f64>,
    pub next_search_directions: Vec<String>,
    pub notes: String,
}

impl From<&JudgeRecommendations> for JudgeRecommendationsLite {
    fn from(value: &JudgeRecommendations) -> Self {
        Self {
            next_rollout_integrator: value.next_rollout_integrator.clone(),
            next_ga_heuristic: value.next_ga_heuristic.clone(),
            next_discovery_solver: value.next_discovery_solver.clone(),
            next_normalize: value.next_normalize,
            next_stls_threshold: value.next_stls_threshold,
            next_ridge_lambda: value.next_ridge_lambda,
            next_lasso_alpha: value.next_lasso_alpha,
            next_search_directions: value.next_search_directions.clone(),
            notes: value.notes.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscoverySolverSummary {
    pub name: String,
    pub normalize: bool,
    pub fitness_heuristic: String,
    pub stls: Option<StlsSolverSummary>,
    pub lasso: Option<LassoSolverSummary>,
    pub ga: Option<GaSolverSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StlsSolverSummary {
    pub auto_thresholds: bool,
    pub thresholds: Vec<f64>,
    pub ridge_lambda: f64,
    pub max_iter: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LassoSolverSummary {
    pub auto_alphas: bool,
    pub alphas: Vec<f64>,
    pub max_iter: usize,
    pub tol: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GaSolverSummary {
    pub runs: usize,
    pub population: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FactoryEvaluationCandidate {
    pub id: usize,
    pub equation_text: String,
    pub metrics: CandidateMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FactoryEvaluationIterationJudge {
    pub summary: String,
    pub ranking: Vec<usize>,
    pub recommendations: JudgeRecommendationsLite,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FactoryEvaluationIteration {
    pub iteration: usize,
    pub run_id: String,
    pub regime: String,
    pub solver: DiscoverySolverSummary,
    pub simulation: Option<SimulationSummary>,
    pub top_candidates: Vec<FactoryEvaluationCandidate>,
    pub judge: Option<FactoryEvaluationIterationJudge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FactoryEvaluationInput {
    pub version: String,
    pub notes: Vec<String>,
    pub iterations: Vec<FactoryEvaluationIteration>,
}

pub fn build_factory_evaluation_prompt(input: &FactoryEvaluationInput) -> String {
    let payload = serde_json::to_string_pretty(input).unwrap_or_else(|_| "{}".to_string());
    let mut prompt = String::new();
    prompt.push_str("You are writing a short evaluation of an automated equation-discovery run.\n");
    prompt.push_str("Audience: a motivated high school physics student (minimal calculus).\n");
    prompt.push_str("Return Markdown ONLY (no JSON). Be concrete and avoid jargon.\n");
    prompt.push_str("Do NOT invent data or claim physical laws beyond the provided evidence.\n\n");

    prompt.push_str("Explain these metrics clearly:\n");
    prompt.push_str("- mse: training fit error (lower is better)\n");
    prompt.push_str("- rollout_rmse: forward-simulation error vs the oracle trajectory (lower is better)\n");
    prompt.push_str("- divergence_time: time until the learned model noticeably diverges (higher is better)\n");
    prompt.push_str("- complexity: rough proxy for equation length (lower is simpler)\n\n");

    prompt.push_str("Required sections (use these headings):\n");
    prompt.push_str("## What was run\n");
    prompt.push_str("## Best result (plain English)\n");
    prompt.push_str("## How good is it?\n");
    prompt.push_str("## What the numbers mean\n");
    prompt.push_str("## Next steps (easy)\n");
    prompt.push_str("## Next steps (more advanced)\n");
    prompt.push_str("## How to report improvements\n\n");

    prompt.push_str("When describing best results:\n");
    prompt.push_str("- Prefer models with low mse AND low rollout_rmse.\n");
    prompt.push_str("- If two models have similar errors, prefer lower complexity.\n");
    prompt.push_str("- Call out any stability_flags and what they might mean.\n");
    prompt.push_str("- Reference artifact paths like `run_003/report.md` when useful.\n\n");

    prompt.push_str("Data (JSON):\n```json\n");
    prompt.push_str(&payload);
    prompt.push_str("\n```\n");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn judge_response_validation_rejects_out_of_range() {
        let rubric = Rubric::default_rubric();
        let input = JudgeInput {
            rubric: rubric.clone(),
            regime: "gravity_only".to_string(),
            dataset: DatasetSummary {
                n_samples: 1,
                target_description: "a_mag".to_string(),
                feature_names: vec!["r_inv2".to_string()],
                feature_descriptions: vec![FeatureDescription {
                    name: "r_inv2".to_string(),
                    description: "sum 1/r^2".to_string(),
                    tags: vec!["distance".to_string()],
                }],
            },
            simulation: None,
            candidates: vec![CandidateSummary {
                id: 0,
                equation: Equation { terms: vec![] },
                equation_text: "0".to_string(),
                metrics: CandidateMetrics {
                    mse: 1.0,
                    complexity: 0,
                    rollout_rmse: None,
                    divergence_time: None,
                    stability_flags: vec![],
                },
                notes: vec![],
            }],
            ic_bounds: IcBounds {
                mass_min: 0.1,
                mass_max: 10.0,
                charge_min: -1.0,
                charge_max: 1.0,
                pos_min: -1.0,
                pos_max: 1.0,
                vel_min: -1.0,
                vel_max: 1.0,
                min_pair_dist: 0.2,
                recommend_barycentric: true,
            },
            notes: vec![],
        };
        let bad = JudgeResponse {
            version: rubric.version.clone(),
            ranking: vec![0],
            scores: vec![JudgeScore {
                id: 0,
                total: 99.0,
                components: ScoreComponents {
                    fidelity: 99.0,
                    parsimony: 0.0,
                    physical_plausibility: 0.0,
                    regime_consistency: 0.0,
                    stability_risk: 0.0,
                },
                rationale: "bad".to_string(),
                flags: vec![],
            }],
            recommendations: JudgeRecommendations {
                next_initial_conditions: None,
                next_rollout_integrator: None,
                next_ga_heuristic: None,
                next_discovery_solver: None,
                next_normalize: None,
                next_stls_threshold: None,
                next_ridge_lambda: None,
                next_lasso_alpha: None,
                next_search_directions: vec![],
                notes: String::new(),
            },
            summary: String::new(),
        };
        assert!(bad.validate(&input).is_err());
    }

    #[test]
    fn ic_prompt_contains_bounds() {
        let request = IcRequest {
            bounds: IcBounds {
                mass_min: 0.1,
                mass_max: 10.0,
                charge_min: -1.0,
                charge_max: 1.0,
                pos_min: -1.0,
                pos_max: 1.0,
                vel_min: -1.0,
                vel_max: 1.0,
                min_pair_dist: 0.2,
                recommend_barycentric: true,
            },
            regime: "gravity_only".to_string(),
            notes: vec!["avoid collisions".to_string()],
            seed: Some(1),
        };
        let prompt = build_ic_prompt(&request);
        assert!(prompt.contains("minimum pair distance"));
        assert!(prompt.contains("barycentric"));
    }

    #[test]
    fn judge_prompt_mentions_discovery_solver_recommendations() {
        let rubric = Rubric::default_rubric();
        let input = JudgeInput {
            rubric: rubric.clone(),
            regime: "gravity_only".to_string(),
            dataset: DatasetSummary {
                n_samples: 1,
                target_description: "a_x".to_string(),
                feature_names: vec!["grav_x".to_string()],
                feature_descriptions: vec![FeatureDescription {
                    name: "grav_x".to_string(),
                    description: "gravity basis x-component".to_string(),
                    tags: vec!["gravity".to_string()],
                }],
            },
            simulation: None,
            candidates: vec![CandidateSummary {
                id: 0,
                equation: Equation { terms: vec![] },
                equation_text: "0".to_string(),
                metrics: CandidateMetrics {
                    mse: 1.0,
                    complexity: 0,
                    rollout_rmse: None,
                    divergence_time: None,
                    stability_flags: vec![],
                },
                notes: vec![],
            }],
            ic_bounds: IcBounds {
                mass_min: 0.1,
                mass_max: 10.0,
                charge_min: -1.0,
                charge_max: 1.0,
                pos_min: -1.0,
                pos_max: 1.0,
                vel_min: -1.0,
                vel_max: 1.0,
                min_pair_dist: 0.2,
                recommend_barycentric: true,
            },
            notes: vec![],
        };
        let prompt = build_judge_prompt(&input);
        assert!(prompt.contains("next_discovery_solver"));
        assert!(prompt.contains("\"stls\""));
        assert!(prompt.contains("\"lasso\""));
        assert!(prompt.contains("\"ga\""));
    }

    #[test]
    fn factory_evaluation_prompt_mentions_audience_and_metrics() {
        let input = FactoryEvaluationInput {
            version: FACTORY_EVALUATION_VERSION.to_string(),
            notes: vec!["steps=5".to_string()],
            iterations: vec![FactoryEvaluationIteration {
                iteration: 1,
                run_id: "run_001".to_string(),
                regime: "gravity_only".to_string(),
                solver: DiscoverySolverSummary {
                    name: "stls".to_string(),
                    normalize: true,
                    fitness_heuristic: "mse".to_string(),
                    stls: Some(StlsSolverSummary {
                        auto_thresholds: true,
                        thresholds: vec![],
                        ridge_lambda: 1e-8,
                        max_iter: 25,
                    }),
                    lasso: None,
                    ga: None,
                },
                simulation: None,
                top_candidates: vec![FactoryEvaluationCandidate {
                    id: 0,
                    equation_text: "a = -G*m*r_hat/r^2".to_string(),
                    metrics: CandidateMetrics {
                        mse: 1.0,
                        complexity: 3,
                        rollout_rmse: Some(0.1),
                        divergence_time: Some(1.0),
                        stability_flags: vec![],
                    },
                }],
                judge: None,
            }],
        };
        let prompt = build_factory_evaluation_prompt(&input);
        assert!(prompt.to_lowercase().contains("high school"));
        for needle in ["mse", "rollout_rmse", "divergence_time", "complexity"] {
            assert!(prompt.contains(needle), "missing {needle}");
        }
    }
}
