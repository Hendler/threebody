//! Physics model definitions and validity matrix.
//!
//! Equations (ASCII):
//! Gravity:
//!   a_i(g) = sum_{j != i} G * m_j * (r_j - r_i) / |r_j - r_i|^3
//!
//! Electrostatics:
//!   E_i = sum_{j != i} k_e * q_j * (r_i - r_j) / |r_i - r_j|^3
//!
//! Magnetostatics (quasi-static Biot-Savart style):
//!   B_i = sum_{j != i} (mu_0 / 4pi) * q_j * (v_j x (r_i - r_j)) / |r_i - r_j|^3
//!
//! Lorentz acceleration:
//!   a_i(em) = (q_i / m_i) * (E_i + v_i x B_i)
//!
//! Total:
//!   a_i = a_i(g) + a_i(em)
//!
//! Potentials:
//!   phi_i = sum_{j != i} k_e * q_j / |r_i - r_j|
//!   A_i   = sum_{j != i} (mu_0 / 4pi) * q_j * v_j / |r_i - r_j|
//!
//! # Examples
//! ```
//! use threebody_core::physics::gravity_accel_single;
//! let g = 1.0;
//! let m = 2.0;
//! let r_i = [1.0, 0.0, 0.0];
//! let r_j = [3.0, 0.0, 0.0];
//! let a = gravity_accel_single(g, m, r_i, r_j);
//! // Distance is 2.0, so acceleration magnitude is G*m / r^2 = 2.0 / 4.0 = 0.5.
//! assert!((a[0] - 0.5).abs() < 1e-12);
//! assert_eq!(a[1], 0.0);
//! assert_eq!(a[2], 0.0);
//! ```
//!
//! ```
//! use threebody_core::physics::model_validity;
//! let validity = model_validity();
//! assert!(validity.regimes.contains(&"gravity_only"));
//! assert!(validity.regimes.contains(&"em_quasistatic"));
//! assert!(validity.non_claims.contains(&"no_retardation"));
//! assert!(validity.non_claims.contains(&"no_radiation_reaction"));
//! assert!(validity.non_claims.contains(&"no_self_fields"));
//! ```

/// Machine-readable model validity matrix for supported regimes and non-claims.
#[derive(Debug, Clone, Copy)]
pub struct ModelValidity {
    pub regimes: &'static [&'static str],
    pub non_claims: &'static [&'static str],
}

/// Returns the supported regimes and explicit non-claims.
pub const fn model_validity() -> ModelValidity {
    ModelValidity {
        regimes: &["gravity_only", "em_quasistatic"],
        non_claims: &["no_retardation", "no_radiation_reaction", "no_self_fields"],
    }
}

/// Compute gravitational acceleration on body i from body j.
/// Uses the formula: a_i = G * m_j * (r_j - r_i) / |r_j - r_i|^3.
pub fn gravity_accel_single(g: f64, m_j: f64, r_i: [f64; 3], r_j: [f64; 3]) -> [f64; 3] {
    let dx = r_j[0] - r_i[0];
    let dy = r_j[1] - r_i[1];
    let dz = r_j[2] - r_i[2];
    let r2 = dx * dx + dy * dy + dz * dz;
    let r = r2.sqrt();
    let inv_r3 = if r == 0.0 { 0.0 } else { 1.0 / (r * r * r) };
    let scale = g * m_j * inv_r3;
    [scale * dx, scale * dy, scale * dz]
}
