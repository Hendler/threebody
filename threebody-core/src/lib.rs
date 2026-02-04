//! Core library for three-body simulation and discovery tooling.

pub mod prelude;
pub mod physics;
pub mod math;
pub mod state;
pub mod frames;
pub mod forces;
pub mod regime;
pub mod diagnostics;
pub mod config;
pub mod integrators;
pub mod sim;
pub mod output;
pub mod benchmarks;

// Re-export common items via the prelude to keep imports DRY.
pub use crate::prelude::*;
pub use crate::config::Config;
pub use crate::integrators::{boris::Boris, leapfrog::Leapfrog, rk45::Rk45};
pub use crate::sim::{simulate, SimOptions, SimResult};

#[cfg(test)]
mod tests {
    #[test]
    fn prelude_imports_compile() {
        let _ = crate::prelude::Body::new(1.0, 0.0);
        let _ = crate::prelude::Vec3::zero();
    }
}
