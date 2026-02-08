//! Discovery crate: sparse equation search and LLM-assisted ranking.

pub mod equation;
pub mod ga;
pub mod grid;
pub mod judge;
pub mod library;
pub mod llm;
pub mod sparse;

pub use equation::FitnessHeuristic;
pub use equation::{Dataset, Equation, EquationScore, TopK};
pub use ga::{DiscoveryConfig, run_search};
pub use grid::grid_search;
pub use judge::{
    BodyInit, CandidateMetrics, CandidateSummary, DatasetSummary, DiscoverySolverSummary,
    FACTORY_EVALUATION_VERSION, FactoryEvaluationCandidate, FactoryEvaluationInput,
    FactoryEvaluationIteration, FactoryEvaluationIterationJudge, FeatureDescription,
    GaSolverSummary, IcBounds, IcRequest, InitialConditionSpec, JudgeInput, JudgeRecommendations,
    JudgeRecommendationsLite, JudgeResponse, LassoSolverSummary, RUBRIC_VERSION, Rubric,
    ScoreComponents, SimulationSummary, StlsSolverSummary,
};
pub use sparse::{LassoConfig, StlsConfig, lasso_path_search, stls_path_search};
