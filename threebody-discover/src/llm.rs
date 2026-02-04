use crate::equation::{EquationScore, TopK};
use serde::{Deserialize, Serialize};
use std::env;
use std::fmt;

#[derive(Debug)]
pub struct LlmError(pub String);

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for LlmError {}

pub trait LlmClient {
    fn rank_equations(&self, entries: &[EquationScore]) -> Result<Vec<EquationScore>, LlmError>;
    fn interpret_results(&self, topk: &TopK) -> Result<String, LlmError>;
}

#[derive(Clone, Debug)]
pub struct MockLlm;

impl LlmClient for MockLlm {
    fn rank_equations(&self, entries: &[EquationScore]) -> Result<Vec<EquationScore>, LlmError> {
        let mut ranked = entries.to_vec();
        ranked.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        Ok(ranked)
    }

    fn interpret_results(&self, topk: &TopK) -> Result<String, LlmError> {
        Ok(format!("Top {} equations reviewed.", topk.entries.len()))
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
    fn rank_equations(&self, entries: &[EquationScore]) -> Result<Vec<EquationScore>, LlmError> {
        let mut prompt = String::from(
            "Rank the equations by best (lowest score). Return a JSON array of indices in order.",
        );
        for (i, eq) in entries.iter().enumerate() {
            prompt.push_str(&format!("\n{i}: score={}", eq.score));
        }
        let text = self.request(&prompt)?;
        let indices: Vec<usize> = serde_json::from_str(&text).map_err(|e| LlmError(e.to_string()))?;
        let mut ranked = Vec::new();
        for idx in indices {
            if let Some(eq) = entries.get(idx) {
                ranked.push(eq.clone());
            }
        }
        if ranked.is_empty() {
            return Err(LlmError("empty ranking".to_string()));
        }
        Ok(ranked)
    }

    fn interpret_results(&self, topk: &TopK) -> Result<String, LlmError> {
        let mut prompt = String::from("Summarize the top equations and suggest next search directions.");
        for (i, eq) in topk.entries.iter().enumerate() {
            prompt.push_str(&format!("\n{i}: score={}", eq.score));
        }
        self.request(&prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::{LlmClient, MockLlm};
    use crate::equation::{Equation, EquationScore, TopK};

    #[test]
    fn mock_llm_ranks_by_score() {
        let llm = MockLlm;
        let entries = vec![
            EquationScore { equation: Equation { terms: vec![] }, score: 2.0 },
            EquationScore { equation: Equation { terms: vec![] }, score: 1.0 },
        ];
        let ranked = llm.rank_equations(&entries).unwrap();
        assert!(ranked[0].score <= ranked[1].score);
    }

    #[test]
    fn mock_llm_interprets() {
        let llm = MockLlm;
        let mut topk = TopK::new(3);
        topk.update(EquationScore { equation: Equation { terms: vec![] }, score: 1.0 });
        let text = llm.interpret_results(&topk).unwrap();
        assert!(text.contains("Top"));
    }
}
