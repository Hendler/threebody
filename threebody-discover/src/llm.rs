use crate::judge::{
    build_ic_prompt, build_judge_prompt, IcRequest, InitialConditionSpec, JudgeInput, JudgeRecommendations,
    JudgeResponse, JudgeScore, ScoreComponents,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug)]
pub struct LlmError(pub String);

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for LlmError {}

pub trait LlmClient {
    fn propose_initial_conditions(&self, request: &IcRequest) -> Result<LlmResult<InitialConditionSpec>, LlmError>;
    fn judge_candidates(&self, input: &JudgeInput) -> Result<LlmResult<JudgeResponse>, LlmError>;
}

#[derive(Debug, Clone)]
pub struct LlmResult<T> {
    pub value: T,
    pub prompt: String,
    pub response: String,
}

#[derive(Clone, Debug)]
pub struct MockLlm;

impl LlmClient for MockLlm {
    fn propose_initial_conditions(&self, request: &IcRequest) -> Result<LlmResult<InitialConditionSpec>, LlmError> {
        let prompt = build_ic_prompt(request);
        let seed = request.seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        });
        let offset = (seed % 10) as f64 * 0.01;
        let value = InitialConditionSpec {
            bodies: vec![
                crate::judge::BodyInit { mass: 1.0, charge: 0.0, pos: [-0.5 - offset, 0.0, 0.0], vel: [0.0, 0.7, 0.0] },
                crate::judge::BodyInit { mass: 1.0, charge: 0.0, pos: [0.5 + offset, 0.0, 0.0], vel: [0.0, -0.7, 0.0] },
                crate::judge::BodyInit { mass: 0.1, charge: 0.0, pos: [0.0, 0.3, 0.0], vel: [0.0, 0.0, 0.0] },
            ],
            barycentric: true,
            notes: "mock two-body-plus-perturber".to_string(),
        };
        let response = serde_json::to_string(&value).map_err(|e| LlmError(e.to_string()))?;
        Ok(LlmResult { value, prompt, response })
    }

    fn judge_candidates(&self, input: &JudgeInput) -> Result<LlmResult<JudgeResponse>, LlmError> {
        let prompt = build_judge_prompt(input);
        let mut scores: Vec<JudgeScore> = Vec::new();
        for c in &input.candidates {
            let mse = c.metrics.mse.max(1e-12);
            let fidelity = (5.0 / (1.0 + mse)).min(5.0);
            let parsimony = (5.0 / (1.0 + c.metrics.complexity as f64)).min(5.0);
            let physical = 3.0;
            let regime = 3.0;
            let stability = if c.metrics.stability_flags.is_empty() { 4.0 } else { 2.0 };
            let total = input.rubric.weights.fidelity * fidelity
                + input.rubric.weights.parsimony * parsimony
                + input.rubric.weights.physical_plausibility * physical
                + input.rubric.weights.regime_consistency * regime
                + input.rubric.weights.stability_risk * stability;
            scores.push(JudgeScore {
                id: c.id,
                total,
                components: ScoreComponents {
                    fidelity,
                    parsimony,
                    physical_plausibility: physical,
                    regime_consistency: regime,
                    stability_risk: stability,
                },
                rationale: "mock judge: weighted score from mse and complexity".to_string(),
                flags: vec![],
            });
        }
        scores.sort_by(|a, b| a.total.partial_cmp(&b.total).unwrap());
        let ranking = scores.iter().map(|s| s.id).collect();
        let value = JudgeResponse {
            version: input.rubric.version.clone(),
            ranking,
            scores,
            recommendations: JudgeRecommendations {
                next_initial_conditions: None,
                next_search_directions: vec!["expand library with explicit r-hat terms".to_string()],
                notes: "mock judge".to_string(),
            },
            summary: "mock judge summary".to_string(),
        };
        let response = serde_json::to_string(&value).map_err(|e| LlmError(e.to_string()))?;
        Ok(LlmResult { value, prompt, response })
    }
}

#[derive(Clone, Debug)]
pub struct OpenAIClient {
    pub api_key: String,
    pub model: String,
    pub base_url: String,
}

impl OpenAIClient {
    pub fn from_env(model: &str) -> Result<Self, LlmError> {
        let api_key = env::var("OPENAI_API_KEY").map_err(|_| LlmError("OPENAI_API_KEY missing".to_string()))?;
        Ok(Self {
            api_key,
            model: model.to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
        })
    }

    fn request(&self, prompt: &str) -> Result<String, LlmError> {
        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            input: &'a str,
        }
        #[derive(Deserialize)]
        struct ContentItem {
            text: Option<String>,
        }
        #[derive(Deserialize)]
        struct OutputItem {
            content: Option<Vec<ContentItem>>,
        }
        #[derive(Deserialize)]
        struct Resp {
            output: Option<Vec<OutputItem>>,
        }

        let url = format!("{}/responses", self.base_url);
        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&Req {
                model: &self.model,
                input: prompt,
            })
            .send()
            .map_err(|e| LlmError(e.to_string()))?;
        let body: Resp = resp.json().map_err(|e| LlmError(e.to_string()))?;
        let text = body
            .output
            .and_then(|mut out| out.pop())
            .and_then(|o| o.content)
            .and_then(|mut c| c.pop())
            .and_then(|c| c.text)
            .ok_or_else(|| LlmError("missing response text".to_string()))?;
        Ok(text)
    }
}

impl LlmClient for OpenAIClient {
    fn propose_initial_conditions(&self, request: &IcRequest) -> Result<LlmResult<InitialConditionSpec>, LlmError> {
        let prompt = build_ic_prompt(request);
        let text = self.request(&prompt)?;
        let value = parse_json(&text)?;
        Ok(LlmResult {
            value,
            prompt,
            response: text,
        })
    }

    fn judge_candidates(&self, input: &JudgeInput) -> Result<LlmResult<JudgeResponse>, LlmError> {
        let prompt = build_judge_prompt(input);
        let text = self.request(&prompt)?;
        let value = parse_json(&text)?;
        Ok(LlmResult {
            value,
            prompt,
            response: text,
        })
    }
}

fn parse_json<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, LlmError> {
    if let Ok(value) = serde_json::from_str::<T>(text) {
        return Ok(value);
    }
    let start = text.find('{').ok_or_else(|| LlmError("missing json object".to_string()))?;
    let end = text.rfind('}').ok_or_else(|| LlmError("missing json object end".to_string()))?;
    let slice = &text[start..=end];
    serde_json::from_str(slice).map_err(|e| LlmError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::{LlmClient, MockLlm};
    use crate::judge::{CandidateMetrics, CandidateSummary, DatasetSummary, FeatureDescription, IcBounds, IcRequest, JudgeInput, Rubric};
    use crate::equation::Equation;

    #[test]
    fn mock_llm_proposes_ic() {
        let llm = MockLlm;
        let req = IcRequest {
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
            notes: vec![],
            seed: Some(1),
        };
        let ic = llm.propose_initial_conditions(&req).unwrap();
        assert_eq!(ic.value.bodies.len(), 3);
    }

    #[test]
    fn mock_llm_judges() {
        let llm = MockLlm;
        let input = JudgeInput {
            rubric: Rubric::default_rubric(),
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
            candidates: vec![
                CandidateSummary {
                    id: 0,
                    equation: Equation { terms: vec![] },
                    metrics: CandidateMetrics {
                        mse: 1.0,
                        complexity: 0,
                        rollout_rmse: None,
                        divergence_time: None,
                        stability_flags: vec![],
                    },
                    notes: vec![],
                },
                CandidateSummary {
                    id: 1,
                    equation: Equation { terms: vec![] },
                    metrics: CandidateMetrics {
                        mse: 2.0,
                        complexity: 2,
                        rollout_rmse: None,
                        divergence_time: None,
                        stability_flags: vec![],
                    },
                    notes: vec![],
                },
            ],
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
        let resp = llm.judge_candidates(&input).unwrap();
        assert!(!resp.value.ranking.is_empty());
    }
}
