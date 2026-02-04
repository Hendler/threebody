//! Discovery crate: sparse equation search and LLM-assisted ranking.

pub mod equation;
pub mod library;
pub mod ga;
pub mod llm;
pub mod grid;

pub use equation::{Dataset, Equation, EquationScore, TopK};
pub use ga::{run_search, DiscoveryConfig};
pub use grid::grid_search;
