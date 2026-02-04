/// Returns true if `a` and `b` are approximately equal within relative/absolute tolerances.
pub fn approx_eq(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    let diff = (a - b).abs();
    if diff <= abs {
        return true;
    }
    diff <= rel * a.abs().max(b.abs())
}

/// Clamps `value` to the inclusive range [`min`, `max`].
pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::{approx_eq, clamp};

    #[test]
    fn approx_eq_relative_and_absolute() {
        assert!(approx_eq(1.0, 1.0 + 1e-9, 1e-8, 0.0));
        assert!(approx_eq(0.0, 1e-9, 1e-8, 1e-8));
        assert!(!approx_eq(1.0, 1.1, 1e-3, 1e-6));
    }

    #[test]
    fn clamp_edges() {
        assert_eq!(clamp(0.5, 0.0, 1.0), 0.5);
        assert_eq!(clamp(-1.0, 0.0, 1.0), 0.0);
        assert_eq!(clamp(2.0, 0.0, 1.0), 1.0);
    }
}
