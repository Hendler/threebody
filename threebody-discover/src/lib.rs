//! Discovery crate: sparse equation search and LLM-assisted ranking.

pub mod equation;
pub mod library;
pub mod ga;
pub mod llm;
pub mod grid;
pub mod judge;

pub use equation::{Dataset, Equation, EquationScore, TopK};
pub use equation::FitnessHeuristic;
pub use ga::{run_search, DiscoveryConfig};
pub use grid::grid_search;
pub use judge::{
    BodyInit, CandidateMetrics, CandidateSummary, DatasetSummary, FeatureDescription, IcBounds, IcRequest,
    InitialConditionSpec, JudgeInput, JudgeRecommendations, JudgeResponse, Rubric, ScoreComponents,
    SimulationSummary, RUBRIC_VERSION,
};
