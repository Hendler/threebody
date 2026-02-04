use crate::equation::{score_equation_with, Equation, EquationScore, FitnessHeuristic, Term, TopK};
use crate::Dataset;

#[derive(Clone, Debug)]
pub struct StlsConfig {
    pub thresholds: Vec<f64>,
    pub ridge_lambda: f64,
    pub max_iter: usize,
    pub normalize: bool,
}

impl Default for StlsConfig {
    fn default() -> Self {
        Self {
            thresholds: Vec::new(),
            ridge_lambda: 1e-8,
            max_iter: 25,
            normalize: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LassoConfig {
    pub alphas: Vec<f64>,
    pub max_iter: usize,
    pub tol: f64,
    pub normalize: bool,
}

impl Default for LassoConfig {
    fn default() -> Self {
        Self {
            alphas: Vec::new(),
            max_iter: 2000,
            tol: 1e-6,
            normalize: true,
        }
    }
}

pub fn stls_path_search(_dataset: &Dataset, _cfg: &StlsConfig, _fitness: FitnessHeuristic) -> TopK {
    let p = _dataset.feature_names.len();
    let mut topk = TopK::new(3);
    if p == 0 || _dataset.samples.is_empty() {
        return topk;
    }
    if _dataset.samples.len() != _dataset.targets.len() {
        return topk;
    }

    let scales = if _cfg.normalize {
        rms_scales(&_dataset.samples, p)
    } else {
        vec![1.0; p]
    };

    let thresholds = if _cfg.thresholds.is_empty() {
        auto_stls_thresholds(_dataset, &scales, _cfg.ridge_lambda)
    } else {
        _cfg.thresholds.clone()
    };

    let mut any = false;
    for &tau in &thresholds {
        if !tau.is_finite() || tau < 0.0 {
            continue;
        }
        let beta = stls_fit_beta(
            &_dataset.samples,
            &_dataset.targets,
            &scales,
            _cfg.ridge_lambda,
            tau,
            _cfg.max_iter,
        )
        .unwrap_or_else(|| vec![0.0; p]);
        let mut xi = vec![0.0; p];
        for j in 0..p {
            xi[j] = beta[j] / scales[j];
        }
        push_candidate(&mut topk, &xi, _dataset, _fitness);
        any = true;
    }

    if !any {
        push_candidate(&mut topk, &vec![0.0; p], _dataset, _fitness);
    }
    topk
}

pub fn lasso_path_search(_dataset: &Dataset, _cfg: &LassoConfig, _fitness: FitnessHeuristic) -> TopK {
    let p = _dataset.feature_names.len();
    let mut topk = TopK::new(3);
    if p == 0 || _dataset.samples.is_empty() {
        return topk;
    }
    if _dataset.samples.len() != _dataset.targets.len() {
        return topk;
    }

    let scales = if _cfg.normalize {
        rms_scales(&_dataset.samples, p)
    } else {
        vec![1.0; p]
    };

    let mut alphas = if _cfg.alphas.is_empty() {
        auto_lasso_alphas(_dataset, &scales)
    } else {
        _cfg.alphas.clone()
    };
    alphas.retain(|a| a.is_finite() && *a >= 0.0);
    alphas.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let mut warm: Option<Vec<f64>> = None;
    let mut any = false;
    for &alpha in &alphas {
        let beta = lasso_fit_beta(
            &_dataset.samples,
            &_dataset.targets,
            &scales,
            alpha,
            _cfg.max_iter,
            _cfg.tol,
            warm.as_deref(),
        )
        .unwrap_or_else(|| vec![0.0; p]);
        warm = Some(beta.clone());
        let mut xi = vec![0.0; p];
        for j in 0..p {
            xi[j] = beta[j] / scales[j];
        }
        push_candidate(&mut topk, &xi, _dataset, _fitness);
        any = true;
    }
    if !any {
        push_candidate(&mut topk, &vec![0.0; p], _dataset, _fitness);
    }
    topk
}

fn rms_scales(samples: &[Vec<f64>], p: usize) -> Vec<f64> {
    if p == 0 {
        return Vec::new();
    }
    let mut sum_sq = vec![0.0; p];
    let n = samples.len().max(1) as f64;
    for row in samples {
        for j in 0..p {
            let v = row.get(j).copied().unwrap_or(0.0);
            sum_sq[j] += v * v;
        }
    }
    sum_sq
        .into_iter()
        .map(|ss| {
            let rms = (ss / n).sqrt();
            if rms == 0.0 { 1.0 } else { rms }
        })
        .collect()
}

fn ridge_solve_scaled(
    samples: &[Vec<f64>],
    targets: &[f64],
    scales: &[f64],
    active: Option<&[usize]>,
    lambda: f64,
) -> Option<Vec<f64>> {
    let p = scales.len();
    if p == 0 {
        return Some(Vec::new());
    }
    if samples.len() != targets.len() {
        return None;
    }

    let indices: Vec<usize> = match active {
        Some(s) => s.to_vec(),
        None => (0..p).collect(),
    };
    let m = indices.len();
    if m == 0 {
        return Some(vec![0.0; p]);
    }

    let mut a = vec![0.0f64; m * m];
    let mut b = vec![0.0f64; m];
    for (i, row) in samples.iter().enumerate() {
        let y = targets[i];
        if !y.is_finite() {
            return None;
        }
        for aj in 0..m {
            let j = indices[aj];
            let sj = scales[j];
            let xj = row.get(j).copied().unwrap_or(0.0) / sj;
            if !xj.is_finite() {
                return None;
            }
            b[aj] += xj * y;
            for ak in 0..m {
                let k = indices[ak];
                let sk = scales[k];
                let xk = row.get(k).copied().unwrap_or(0.0) / sk;
                a[aj * m + ak] += xj * xk;
            }
        }
    }
    for i in 0..m {
        a[i * m + i] += lambda;
    }

    let beta_active = cholesky_solve(&a, &b, m).or_else(|| gaussian_solve(a, b, m))?;
    let mut beta = vec![0.0; p];
    for (ai, &j) in indices.iter().enumerate() {
        beta[j] = beta_active[ai];
    }
    Some(beta)
}

fn stls_fit_beta(
    samples: &[Vec<f64>],
    targets: &[f64],
    scales: &[f64],
    ridge_lambda: f64,
    threshold: f64,
    max_iter: usize,
) -> Option<Vec<f64>> {
    let p = scales.len();
    if p == 0 {
        return Some(Vec::new());
    }

    let mut beta = ridge_solve_scaled(samples, targets, scales, None, ridge_lambda)?;
    let mut prev_active: Vec<usize> = Vec::new();

    for _ in 0..max_iter.max(1) {
        let mut active = Vec::new();
        for j in 0..p {
            if beta[j].abs() >= threshold {
                active.push(j);
            }
        }
        if active.is_empty() {
            return Some(vec![0.0; p]);
        }
        let next = ridge_solve_scaled(samples, targets, scales, Some(&active), ridge_lambda)?;

        let mut max_delta = 0.0f64;
        for j in 0..p {
            max_delta = max_delta.max((next[j] - beta[j]).abs());
        }
        let same_support = active == prev_active;
        beta = next;
        if same_support && max_delta < 1e-10 {
            break;
        }
        prev_active = active;
    }
    Some(beta)
}

fn auto_stls_thresholds(dataset: &Dataset, scales: &[f64], ridge_lambda: f64) -> Vec<f64> {
    let p = scales.len();
    let beta0 = ridge_solve_scaled(&dataset.samples, &dataset.targets, scales, None, ridge_lambda)
        .unwrap_or_else(|| vec![0.0; p]);
    let mut bmax = 0.0f64;
    for &v in &beta0 {
        bmax = bmax.max(v.abs());
    }
    if bmax == 0.0 {
        return vec![0.0];
    }
    let mut thresholds = Vec::new();
    thresholds.push(0.0);
    for k in 1..=12 {
        let exp = -(k as f64) / 3.0;
        thresholds.push(bmax * 10f64.powf(exp));
    }
    thresholds
}

fn soft_threshold(z: f64, alpha: f64) -> f64 {
    if z > alpha {
        z - alpha
    } else if z < -alpha {
        z + alpha
    } else {
        0.0
    }
}

fn lasso_fit_beta(
    samples: &[Vec<f64>],
    targets: &[f64],
    scales: &[f64],
    alpha: f64,
    max_iter: usize,
    tol: f64,
    warm_start: Option<&[f64]>,
) -> Option<Vec<f64>> {
    if alpha < 0.0 || !alpha.is_finite() {
        return None;
    }
    let p = scales.len();
    if p == 0 {
        return Some(Vec::new());
    }
    let n = samples.len();
    if n != targets.len() {
        return None;
    }
    for &y in targets {
        if !y.is_finite() {
            return None;
        }
    }

    let mut cols = vec![vec![0.0f64; n]; p];
    let mut col_norm2 = vec![0.0f64; p];
    for (i, row) in samples.iter().enumerate() {
        for j in 0..p {
            let sj = scales[j];
            let v = row.get(j).copied().unwrap_or(0.0) / sj;
            if !v.is_finite() {
                return None;
            }
            cols[j][i] = v;
            col_norm2[j] += v * v;
        }
    }

    let mut beta = vec![0.0f64; p];
    if let Some(init) = warm_start {
        if init.len() == p {
            beta.copy_from_slice(init);
        }
    }

    // r = y - X beta
    let mut r = targets.to_vec();
    for j in 0..p {
        let bj = beta[j];
        if bj == 0.0 {
            continue;
        }
        for i in 0..n {
            r[i] -= cols[j][i] * bj;
        }
    }

    for _ in 0..max_iter.max(1) {
        let mut max_change = 0.0f64;
        for j in 0..p {
            let norm2 = col_norm2[j];
            if norm2 == 0.0 {
                beta[j] = 0.0;
                continue;
            }

            let old = beta[j];
            if old != 0.0 {
                for i in 0..n {
                    r[i] += cols[j][i] * old;
                }
            }

            let mut rho = 0.0f64;
            for i in 0..n {
                rho += cols[j][i] * r[i];
            }
            let new = soft_threshold(rho, alpha) / norm2;
            beta[j] = new;
            if new != 0.0 {
                for i in 0..n {
                    r[i] -= cols[j][i] * new;
                }
            }
            max_change = max_change.max((new - old).abs());
        }
        if max_change < tol {
            break;
        }
    }

    Some(beta)
}

fn auto_lasso_alphas(dataset: &Dataset, scales: &[f64]) -> Vec<f64> {
    let p = scales.len();
    let n = dataset.samples.len();
    if p == 0 || n == 0 {
        return vec![0.0];
    }

    let mut alpha_max = 0.0f64;
    for j in 0..p {
        let mut dot = 0.0f64;
        let sj = scales[j];
        for i in 0..n {
            let x = dataset.samples[i].get(j).copied().unwrap_or(0.0) / sj;
            dot += x * dataset.targets[i];
        }
        alpha_max = alpha_max.max(dot.abs());
    }
    if alpha_max == 0.0 {
        return vec![0.0];
    }
    let mut alphas = Vec::new();
    for k in 0..=12 {
        let exp = -(k as f64) / 3.0;
        alphas.push(alpha_max * 10f64.powf(exp));
    }
    alphas
}

fn cholesky_solve(a: &[f64], b: &[f64], p: usize) -> Option<Vec<f64>> {
    debug_assert_eq!(a.len(), p * p);
    debug_assert_eq!(b.len(), p);

    let mut l = vec![0.0f64; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut sum = a[i * p + j];
            for k in 0..j {
                sum -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return None;
                }
                l[i * p + j] = sum.sqrt();
            } else {
                let diag = l[j * p + j];
                if diag == 0.0 {
                    return None;
                }
                let v = sum / diag;
                if !v.is_finite() {
                    return None;
                }
                l[i * p + j] = v;
            }
        }
    }

    // Forward solve: L z = b
    let mut z = vec![0.0f64; p];
    for i in 0..p {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * p + k] * z[k];
        }
        let diag = l[i * p + i];
        if diag == 0.0 {
            return None;
        }
        z[i] = sum / diag;
    }

    // Backward solve: L^T x = z
    let mut x = vec![0.0f64; p];
    for ii in 0..p {
        let i = p - 1 - ii;
        let mut sum = z[i];
        for k in (i + 1)..p {
            sum -= l[k * p + i] * x[k];
        }
        let diag = l[i * p + i];
        if diag == 0.0 {
            return None;
        }
        x[i] = sum / diag;
    }
    Some(x)
}

fn gaussian_solve(mut a: Vec<f64>, mut b: Vec<f64>, p: usize) -> Option<Vec<f64>> {
    debug_assert_eq!(a.len(), p * p);
    debug_assert_eq!(b.len(), p);

    for k in 0..p {
        let mut pivot_row = k;
        let mut pivot_val = a[k * p + k].abs();
        for i in (k + 1)..p {
            let v = a[i * p + k].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = i;
            }
        }
        if !pivot_val.is_finite() || pivot_val == 0.0 {
            return None;
        }
        if pivot_row != k {
            for j in 0..p {
                a.swap(k * p + j, pivot_row * p + j);
            }
            b.swap(k, pivot_row);
        }

        let pivot = a[k * p + k];
        if pivot == 0.0 {
            return None;
        }
        for i in (k + 1)..p {
            let factor = a[i * p + k] / pivot;
            if factor == 0.0 {
                continue;
            }
            for j in k..p {
                a[i * p + j] -= factor * a[k * p + j];
            }
            b[i] -= factor * b[k];
        }
    }

    let mut x = vec![0.0f64; p];
    for ii in 0..p {
        let i = p - 1 - ii;
        let mut sum = b[i];
        for j in (i + 1)..p {
            sum -= a[i * p + j] * x[j];
        }
        let diag = a[i * p + i];
        if diag == 0.0 || !diag.is_finite() {
            return None;
        }
        x[i] = sum / diag;
    }
    Some(x)
}

fn equation_from_coeffs(coeffs: &[f64], feature_names: &[String]) -> Equation {
    let mut terms = Vec::new();
    for (j, &c) in coeffs.iter().enumerate() {
        if !c.is_finite() {
            continue;
        }
        let c = if c.abs() < 1e-12 { 0.0 } else { c };
        if c != 0.0 {
            let feature = feature_names
                .get(j)
                .cloned()
                .unwrap_or_else(|| format!("f{j}"));
            terms.push(Term { feature, coeff: c });
        }
    }
    Equation { terms }
}

fn push_candidate(topk: &mut TopK, coeffs: &[f64], dataset: &Dataset, fitness: FitnessHeuristic) {
    let eq = equation_from_coeffs(coeffs, &dataset.feature_names);
    let score = score_equation_with(&eq, dataset, fitness);
    topk.update(EquationScore { equation: eq, score });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_scales_are_nonzero_and_roundtrip_prediction_matches() {
        // X has a zero column; scale for that column should fall back to 1.0.
        let x = vec![vec![3.0, 0.0], vec![4.0, 0.0]];
        let scales = rms_scales(&x, 2);
        assert!(scales[0] > 0.0);
        assert_eq!(scales[1], 1.0);

        // If X' = X / scales, then coefficients denormalize as xi = beta / scales.
        let beta = vec![2.0, 5.0];
        let xi = vec![beta[0] / scales[0], beta[1] / scales[1]];
        let pred_orig = x[0][0] * xi[0] + x[0][1] * xi[1];
        let pred_norm = (x[0][0] / scales[0]) * beta[0] + (x[0][1] / scales[1]) * beta[1];
        assert!((pred_orig - pred_norm).abs() < 1e-12);
    }

    #[test]
    fn ridge_solve_recovers_exact_linear_coeffs() {
        // X = [[1,0],[0,1],[1,1]], y = [1,2,3] => xi=[1,2]
        let x = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let y = vec![1.0, 2.0, 3.0];
        let scales = vec![1.0, 1.0];
        let xi = ridge_solve_scaled(&x, &y, &scales, None, 0.0).expect("solve");
        assert_eq!(xi.len(), 2);
        assert!((xi[0] - 1.0).abs() < 1e-10);
        assert!((xi[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn stls_path_search_recovers_sparse_coeffs_on_identity_design() {
        // X = I, y = X xi_true. STLS should recover exact sparse xi.
        let p = 5;
        let mut samples = Vec::new();
        for i in 0..p {
            let mut row = vec![0.0; p];
            row[i] = 1.0;
            samples.push(row);
        }
        let xi_true = vec![0.0, 1.5, 0.0, -2.0, 0.0];
        let targets = xi_true.clone();
        let feature_names = (0..p).map(|i| format!("x{i}")).collect::<Vec<_>>();
        let dataset = Dataset::new(feature_names, samples, targets);

        let cfg = StlsConfig {
            thresholds: vec![0.1, 0.6],
            ridge_lambda: 1e-12,
            max_iter: 10,
            normalize: true,
        };
        let topk = stls_path_search(&dataset, &cfg, FitnessHeuristic::Mse);
        let best = topk.best().expect("best");
        assert!(best.score < 1e-12);
        assert_eq!(best.equation.terms.len(), 2);
        assert_eq!(best.equation.terms[0].feature, "x1");
        assert!((best.equation.terms[0].coeff - 1.5).abs() < 1e-10);
        assert_eq!(best.equation.terms[1].feature, "x3");
        assert!((best.equation.terms[1].coeff - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn stls_handles_collinear_columns() {
        let dataset = Dataset::new(
            vec!["x".to_string(), "x_dup".to_string()],
            vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]],
            vec![1.0, 2.0, 3.0],
        );
        let cfg = StlsConfig {
            thresholds: vec![0.0],
            ridge_lambda: 1e-12,
            max_iter: 10,
            normalize: true,
        };
        let topk = stls_path_search(&dataset, &cfg, FitnessHeuristic::Mse);
        let best = topk.best().expect("best");
        assert!(best.score < 1e-12);
        for t in &best.equation.terms {
            assert!(t.coeff.is_finite());
        }
    }

    #[test]
    fn lasso_identity_design_matches_soft_threshold() {
        // X = I, LASSO solution is elementwise soft-thresholding of y at alpha.
        let p = 3;
        let mut samples = Vec::new();
        for i in 0..p {
            let mut row = vec![0.0; p];
            row[i] = 1.0;
            samples.push(row);
        }
        let targets = vec![3.0, -0.5, 0.2];
        let feature_names = (0..p).map(|i| format!("x{i}")).collect::<Vec<_>>();
        let dataset = Dataset::new(feature_names, samples, targets);

        let alpha = 0.3;
        let cfg = LassoConfig {
            alphas: vec![alpha],
            max_iter: 50,
            tol: 1e-12,
            normalize: false,
        };
        let topk = lasso_path_search(&dataset, &cfg, FitnessHeuristic::Mse);
        let best = topk.best().expect("best");
        assert_eq!(best.equation.terms.len(), 2);
        assert_eq!(best.equation.terms[0].feature, "x0");
        assert!((best.equation.terms[0].coeff - 2.7).abs() < 1e-10);
        assert_eq!(best.equation.terms[1].feature, "x1");
        assert!((best.equation.terms[1].coeff - (-0.2)).abs() < 1e-10);
    }

    #[test]
    fn lasso_nonzeros_do_not_increase_with_alpha() {
        let p = 3;
        let mut samples = Vec::new();
        for i in 0..p {
            let mut row = vec![0.0; p];
            row[i] = 1.0;
            samples.push(row);
        }
        let targets = vec![3.0, 2.0, 1.0];
        let scales = vec![1.0; p];
        let mut prev_nnz = usize::MAX;
        for &alpha in &[0.0, 0.5, 1.5, 2.5] {
            let beta = lasso_fit_beta(&samples, &targets, &scales, alpha, 100, 1e-12, None).expect("solve");
            let nnz = beta.iter().filter(|&&v| v.abs() > 1e-9).count();
            assert!(nnz <= prev_nnz);
            prev_nnz = nnz;
        }
    }
}
