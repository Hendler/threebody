/// Compute accuracy per millisecond.
pub fn accuracy_per_ms(accuracy: f64, millis: f64) -> f64 {
    if millis <= 0.0 {
        return 0.0;
    }
    accuracy / millis
}

/// Return true if regression exceeds tolerance (relative drop).
pub fn regression_exceeds(baseline: f64, current: f64, tolerance: f64) -> bool {
    if baseline <= 0.0 {
        return false;
    }
    let drop = (baseline - current) / baseline;
    drop > tolerance
}

#[cfg(test)]
mod tests {
    use super::{accuracy_per_ms, regression_exceeds};

    #[test]
    fn accuracy_per_ms_handles_zero() {
        assert_eq!(accuracy_per_ms(1.0, 0.0), 0.0);
    }

    #[test]
    fn mocked_regression_case_fails() {
        let baseline = 10.0;
        let current = 8.0;
        let tolerance = 0.1;
        assert!(regression_exceeds(baseline, current, tolerance));
    }
}
