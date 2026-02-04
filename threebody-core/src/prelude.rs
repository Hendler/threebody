//! Common re-exports for consumers of `threebody-core`.
//!
//! # Examples
//! ```
//! use threebody_core::prelude::*;
//! let _b = Body::new(1.0, 0.0);
//! let _v = Vec3::zero();
//! ```

pub use crate::math::vec3::Vec3;
pub use crate::state::{Body, State, System};
