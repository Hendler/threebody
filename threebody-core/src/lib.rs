//! Core library for three-body simulation and discovery tooling.

pub mod prelude;
pub mod physics;
pub mod math;
pub mod state;
pub mod frames;
pub mod forces;
pub mod regime;
pub mod diagnostics;

// Re-export common items via the prelude to keep imports DRY.
pub use crate::prelude::*;

#[cfg(test)]
mod tests {
    #[test]
    fn prelude_imports_compile() {
        let _ = crate::prelude::Body::new(1.0, 0.0);
        let _ = crate::prelude::Vec3::zero();
    }
}
