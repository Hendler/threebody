use crate::judge::{
    build_factory_evaluation_prompt, build_ic_prompt, build_judge_prompt, FactoryEvaluationInput, IcRequest,
    InitialConditionSpec, JudgeInput, JudgeRecommendations, JudgeResponse, JudgeScore, ScoreComponents,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::Mutex;

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
    fn explain_factory_evaluation(&self, input: &FactoryEvaluationInput) -> Result<LlmResult<String>, LlmError>;
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
        scores.sort_by(|a, b| b.total.partial_cmp(&a.total).unwrap());
        let ranking = scores.iter().map(|s| s.id).collect();
        let value = JudgeResponse {
            version: input.rubric.version.clone(),
            ranking,
            scores,
            recommendations: JudgeRecommendations {
                next_initial_conditions: None,
                next_rollout_integrator: Some("leapfrog".to_string()),
                next_ga_heuristic: Some("mse_parsimony".to_string()),
                next_discovery_solver: Some("stls".to_string()),
                next_normalize: Some(true),
                next_stls_threshold: None,
                next_ridge_lambda: None,
                next_lasso_alpha: None,
                next_search_directions: vec!["expand library with explicit r-hat terms".to_string()],
                notes: "mock judge".to_string(),
            },
            summary: "mock judge summary".to_string(),
        };
        let response = serde_json::to_string(&value).map_err(|e| LlmError(e.to_string()))?;
        Ok(LlmResult { value, prompt, response })
    }

    fn explain_factory_evaluation(&self, input: &FactoryEvaluationInput) -> Result<LlmResult<String>, LlmError> {
        let prompt = build_factory_evaluation_prompt(input);
        let response = build_mock_factory_evaluation_md(input);
        Ok(LlmResult {
            value: response.clone(),
            prompt,
            response,
        })
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
        Self::from_env_or_file(model, None)
    }

    pub fn from_env_or_file(model: &str, key_file: Option<&Path>) -> Result<Self, LlmError> {
        let default_path = Path::new(".openai_key");
        let key_path = key_file.or_else(|| default_path.exists().then_some(default_path));
        let api_key = if let Some(path) = key_path {
            let raw = fs::read_to_string(path)
                .map_err(|e| LlmError(format!("failed to read OpenAI key file {}: {}", path.display(), e)))?;
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Err(LlmError(format!("OpenAI key file {} is empty", path.display())));
            }
            trimmed.to_string()
        } else {
            env::var("OPENAI_API_KEY").map_err(|_| LlmError("OPENAI_API_KEY missing".to_string()))?
        };
        let base_url = env::var("OPENAI_BASE_URL")
            .or_else(|_| env::var("THREEBODY_OPENAI_BASE_URL"))
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        Ok(Self {
            api_key,
            model: model.to_string(),
            base_url,
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
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| LlmError(e.to_string()))?;
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

    fn explain_factory_evaluation(&self, input: &FactoryEvaluationInput) -> Result<LlmResult<String>, LlmError> {
        let prompt = build_factory_evaluation_prompt(input);
        let text = self.request(&prompt)?;
        Ok(LlmResult {
            value: text.clone(),
            prompt,
            response: text,
        })
    }
}

#[derive(Debug)]
pub struct AutoLlmClient {
    primary: Option<OpenAIClient>,
    fallback: MockLlm,
    state: Mutex<AutoLlmState>,
}

#[derive(Debug, Clone)]
struct AutoLlmState {
    primary_enabled: bool,
    warned: bool,
    last_error: Option<String>,
}

impl AutoLlmClient {
    pub fn from_env_or_file(model: &str, key_file: Option<&Path>) -> Self {
        let primary = OpenAIClient::from_env_or_file(model, key_file).ok();
        let primary_enabled = primary.is_some();
        Self {
            primary,
            fallback: MockLlm,
            state: Mutex::new(AutoLlmState {
                primary_enabled,
                warned: false,
                last_error: None,
            }),
        }
    }

    fn note_primary_failure(&self, err: &LlmError) {
        let mut state = self.state.lock().unwrap();
        state.primary_enabled = false;
        state.last_error = Some(err.to_string());
        if !state.warned {
            eprintln!(
                "OpenAI LLM disabled for remainder of run (falling back to mock): {}",
                err
            );
            state.warned = true;
        }
    }

    fn try_primary<T>(
        &self,
        f: impl FnOnce(&OpenAIClient) -> Result<LlmResult<T>, LlmError>,
    ) -> Option<Result<LlmResult<T>, LlmError>> {
        let state = self.state.lock().unwrap();
        if !state.primary_enabled {
            return None;
        }
        let primary = self.primary.as_ref()?;
        drop(state);
        Some(f(primary))
    }
}

impl LlmClient for AutoLlmClient {
    fn propose_initial_conditions(&self, request: &IcRequest) -> Result<LlmResult<InitialConditionSpec>, LlmError> {
        if let Some(result) = self.try_primary(|p| p.propose_initial_conditions(request)) {
            match result {
                Ok(v) => return Ok(v),
                Err(err) => self.note_primary_failure(&err),
            }
        }
        self.fallback.propose_initial_conditions(request)
    }

    fn judge_candidates(&self, input: &JudgeInput) -> Result<LlmResult<JudgeResponse>, LlmError> {
        if let Some(result) = self.try_primary(|p| p.judge_candidates(input)) {
            match result {
                Ok(v) => return Ok(v),
                Err(err) => self.note_primary_failure(&err),
            }
        }
        self.fallback.judge_candidates(input)
    }

    fn explain_factory_evaluation(&self, input: &FactoryEvaluationInput) -> Result<LlmResult<String>, LlmError> {
        if let Some(result) = self.try_primary(|p| p.explain_factory_evaluation(input)) {
            match result {
                Ok(v) => return Ok(v),
                Err(err) => self.note_primary_failure(&err),
            }
        }
        self.fallback.explain_factory_evaluation(input)
    }
}

fn build_mock_factory_evaluation_md(input: &FactoryEvaluationInput) -> String {
    fn format_opt(v: Option<f64>) -> String {
        match v {
            Some(x) if x.is_finite() => format!("{x:.6}"),
            Some(_) => "nan".to_string(),
            None => "n/a".to_string(),
        }
    }

    let n_iters = input.iterations.len();
    let mut best_run: Option<(&str, &str, &crate::judge::FactoryEvaluationCandidate)> = None;
    for iter in &input.iterations {
        for cand in &iter.top_candidates {
            match best_run {
                None => best_run = Some((&iter.run_id, &iter.regime, cand)),
                Some((_rid, _regime, best)) => {
                    let cand_roll = cand.metrics.rollout_rmse.unwrap_or(f64::INFINITY);
                    let best_roll = best.metrics.rollout_rmse.unwrap_or(f64::INFINITY);
                    let cand_mse = if cand.metrics.mse.is_finite() { cand.metrics.mse } else { f64::INFINITY };
                    let best_mse = if best.metrics.mse.is_finite() { best.metrics.mse } else { f64::INFINITY };
                    let better = cand_roll < best_roll
                        || (cand_roll == best_roll && cand_mse < best_mse)
                        || (cand_roll == best_roll
                            && cand_mse == best_mse
                            && cand.metrics.complexity < best.metrics.complexity);
                    if better {
                        best_run = Some((&iter.run_id, &iter.regime, cand));
                    }
                }
            }
        }
    }

    let mut md = String::new();
    md.push_str("# Factory Evaluation\n\n");
    md.push_str(&format!("- Iterations: {}\n", n_iters));
    if !input.notes.is_empty() {
        md.push_str("- Notes:\n");
        for note in &input.notes {
            md.push_str(&format!("  - {}\n", note));
        }
    }
    md.push_str("\n## What was run\n");
    md.push_str("This run repeatedly:\n");
    md.push_str("1) picks initial conditions (sometimes via an LLM),\n");
    md.push_str("2) simulates a trajectory (the “oracle”),\n");
    md.push_str("3) fits simple equations to predict acceleration, and\n");
    md.push_str("4) optionally uses an LLM judge to interpret results and suggest next settings.\n");

    md.push_str("\n## Best result (plain English)\n");
    if let Some((run_id, regime, cand)) = best_run {
        md.push_str(&format!("- Best run: `{}` (regime: `{}`)\n", run_id, regime));
        md.push_str(&format!("- Equation (vector form): {}\n", cand.equation_text));
        md.push_str(&format!(
            "- Metrics: mse={:.6e}, rollout_rmse={}, divergence_time={}, complexity={}\n",
            cand.metrics.mse,
            format_opt(cand.metrics.rollout_rmse),
            format_opt(cand.metrics.divergence_time),
            cand.metrics.complexity
        ));
        if !cand.metrics.stability_flags.is_empty() {
            md.push_str(&format!(
                "- Stability flags: {}\n",
                cand.metrics.stability_flags.join(", ")
            ));
        }
        md.push_str(&format!(
            "\nFor details, open `{}/report.md` and `{}/discovery.json`.\n",
            run_id, run_id
        ));
    } else {
        md.push_str("No candidate equations were recorded.\n");
    }

    md.push_str("\n## How good is it?\n");
    md.push_str("A model can “fit” the training data (low `mse`) but still drift when you roll it forward.\n");
    md.push_str("The most useful quick check is `rollout_rmse` (lower is better) and `divergence_time` (higher is better).\n");

    md.push_str("\n## What the numbers mean\n");
    md.push_str("- `mse`: average squared error on the training samples (lower is better).\n");
    md.push_str("- `rollout_rmse`: how far the learned model’s simulated trajectory drifts from the oracle (lower is better).\n");
    md.push_str("- `divergence_time`: how long the learned model stays close before it diverges (higher is better).\n");
    md.push_str("- `complexity`: roughly how long the equation is (lower usually generalizes better).\n");

    md.push_str("\n## Next steps (easy)\n");
    md.push_str("- Run more iterations and compare `evaluation.md` across runs.\n");
    md.push_str("- Look at `run_###/report.md` and check whether improvements reduce `rollout_rmse`.\n");
    md.push_str("- Try switching the rollout integrator between `euler` and `leapfrog`.\n");

    md.push_str("\n## Next steps (more advanced)\n");
    md.push_str("- Try `--solver lasso` vs `--solver stls` and compare stability.\n");
    md.push_str("- Tune sparsity: increase STLS threshold (or LASSO alpha) to simplify equations.\n");
    md.push_str("- Expand or refine the feature library (new physics-inspired terms), then re-run discovery.\n");
    md.push_str("- Validate on new initial conditions (generalization), not just the same trajectory.\n");

    md.push_str("\n## How to report improvements\n");
    md.push_str("If you’re not a math/physics expert, the most helpful report is:\n");
    md.push_str("- which run directory (`run_###`) looks best,\n");
    md.push_str("- the equation text,\n");
    md.push_str("- the metrics (`mse`, `rollout_rmse`, `divergence_time`, `complexity`), and\n");
    md.push_str("- anything notable in `simulation.warnings` or stability flags.\n");
    md
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
    use super::{AutoLlmClient, LlmClient, MockLlm, OpenAIClient};
    use crate::judge::{
        CandidateMetrics, CandidateSummary, DatasetSummary, FactoryEvaluationCandidate, FactoryEvaluationInput,
        FactoryEvaluationIteration, FeatureDescription, IcBounds, IcRequest, JudgeInput, Rubric,
    };
    use crate::equation::Equation;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

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
                    equation_text: "0".to_string(),
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
                    equation_text: "0".to_string(),
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

    #[test]
    fn openai_client_reads_key_file() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut path = std::env::temp_dir();
        path.push(format!("openai_key_test_{now}.txt"));
        fs::write(&path, "sk-test-key\n").unwrap();

        let client = OpenAIClient::from_env_or_file("gpt-5.2", Some(&path)).unwrap();
        assert_eq!(client.api_key, "sk-test-key");

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn auto_llm_falls_back_when_primary_fails() {
        let primary = OpenAIClient {
            api_key: "sk-test".to_string(),
            model: "gpt-test".to_string(),
            base_url: "http://127.0.0.1:1".to_string(),
        };
        let llm = AutoLlmClient {
            primary: Some(primary),
            fallback: MockLlm,
            state: std::sync::Mutex::new(super::AutoLlmState {
                primary_enabled: true,
                warned: false,
                last_error: None,
            }),
        };
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

        let state = llm.state.lock().unwrap();
        assert!(!state.primary_enabled);
        assert!(state.last_error.as_ref().is_some());
    }

    #[test]
    fn mock_llm_writes_factory_evaluation_markdown() {
        let llm = MockLlm;
        let input = FactoryEvaluationInput {
            version: crate::judge::FACTORY_EVALUATION_VERSION.to_string(),
            notes: vec!["steps=5".to_string(), "dt=0.01".to_string()],
            iterations: vec![
                FactoryEvaluationIteration {
                    iteration: 1,
                    run_id: "run_001".to_string(),
                    regime: "gravity_only".to_string(),
                    solver: crate::judge::DiscoverySolverSummary {
                        name: "stls".to_string(),
                        normalize: true,
                        fitness_heuristic: "mse".to_string(),
                        stls: None,
                        lasso: None,
                        ga: None,
                    },
                    simulation: None,
                    top_candidates: vec![FactoryEvaluationCandidate {
                        id: 0,
                        equation_text: "a = bad".to_string(),
                        metrics: CandidateMetrics {
                            mse: 1.0,
                            complexity: 10,
                            rollout_rmse: Some(0.9),
                            divergence_time: Some(1.0),
                            stability_flags: vec![],
                        },
                    }],
                    judge: None,
                },
                FactoryEvaluationIteration {
                    iteration: 2,
                    run_id: "run_002".to_string(),
                    regime: "gravity_only".to_string(),
                    solver: crate::judge::DiscoverySolverSummary {
                        name: "stls".to_string(),
                        normalize: true,
                        fitness_heuristic: "mse".to_string(),
                        stls: None,
                        lasso: None,
                        ga: None,
                    },
                    simulation: None,
                    top_candidates: vec![FactoryEvaluationCandidate {
                        id: 0,
                        equation_text: "a = good".to_string(),
                        metrics: CandidateMetrics {
                            mse: 1.0,
                            complexity: 1,
                            rollout_rmse: Some(0.1),
                            divergence_time: Some(1.0),
                            stability_flags: vec![],
                        },
                    }],
                    judge: None,
                },
            ],
        };
        let out = llm.explain_factory_evaluation(&input).unwrap();
        assert!(out.value.contains("# Factory Evaluation"));
        assert!(out.value.contains("`run_002`"));
        assert!(out.value.to_lowercase().contains("next steps"));
    }
}
