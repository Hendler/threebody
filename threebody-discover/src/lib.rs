//! Discovery crate: sparse equation search and LLM-assisted ranking.

pub mod equation;
pub mod library;
pub mod ga;
pub mod llm;
pub mod grid;
pub mod judge;
pub mod sparse;

pub use equation::{Dataset, Equation, EquationScore, TopK};
pub use equation::FitnessHeuristic;
pub use ga::{run_search, DiscoveryConfig};
pub use grid::grid_search;
pub use sparse::{lasso_path_search, stls_path_search, LassoConfig, StlsConfig};
pub use judge::{
    BodyInit, CandidateMetrics, CandidateSummary, DatasetSummary, FeatureDescription, IcBounds, IcRequest,
    InitialConditionSpec, JudgeInput, JudgeRecommendations, JudgeResponse, Rubric, ScoreComponents,
    SimulationSummary, RUBRIC_VERSION,
    DiscoverySolverSummary, FactoryEvaluationCandidate, FactoryEvaluationInput, FactoryEvaluationIteration,
    FactoryEvaluationIterationJudge, FACTORY_EVALUATION_VERSION, GaSolverSummary, JudgeRecommendationsLite,
    LassoSolverSummary, StlsSolverSummary,
};
