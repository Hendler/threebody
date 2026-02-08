//! Core library for three-body simulation and discovery tooling.

pub mod analysis;
pub mod benchmarks;
pub mod config;
pub mod diagnostics;
pub mod forces;
pub mod frames;
pub mod integrators;
pub mod math;
pub mod output;
pub mod physics;
pub mod prelude;
pub mod regime;
pub mod sim;
pub mod state;

// Re-export common items via the prelude to keep imports DRY.
pub use crate::config::Config;
pub use crate::integrators::{boris::Boris, leapfrog::Leapfrog, rk45::Rk45};
pub use crate::prelude::*;
pub use crate::sim::{SimOptions, SimResult, simulate};

#[cfg(test)]
mod tests {
    #[test]
    fn prelude_imports_compile() {
        let _ = crate::prelude::Body::new(1.0, 0.0);
        let _ = crate::prelude::Vec3::zero();
    }
}
