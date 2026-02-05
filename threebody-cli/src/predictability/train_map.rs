use std::fs;
use std::io::{self, BufRead};
use std::path::PathBuf;

use serde::Serialize;
use threebody_discover::equation::{score_equation, Equation, EquationScore, FitnessHeuristic};
use threebody_discover::sparse::{stls_path_search, StlsConfig};
use threebody_discover::Dataset;

use super::extract::EncounterRecord;
use super::features::{PairId, PairMetrics};
use crate::predictability::PREDICTABILITY_VERSION;

#[derive(Debug, Clone, Serialize)]
struct StlsSettings {
    ridge_lambda: f64,
    max_iter: usize,
    normalize: bool,
}

#[derive(Debug, Clone, Serialize)]
struct TargetCandidate {
    equation: Equation,
    train_score: f64,
    test_mse: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct TargetModel {
    target: String,
    top3: Vec<TargetCandidate>,
}

#[derive(Debug, Clone, Serialize)]
struct EncounterMapModel {
    version: String,
    n_samples: usize,
    n_train: usize,
    n_test: usize,
    feature_names: Vec<String>,
    stls: StlsSettings,
    targets: Vec<TargetModel>,
    notes: Vec<String>,
}

pub(crate) fn run_train_map(encounters: PathBuf, out: PathBuf, seed: u64) -> anyhow::Result<()> {
    if !encounters.exists() {
        anyhow::bail!("encounters JSONL not found: {}", encounters.display());
    }
    let file = fs::File::open(&encounters)?;
    let reader = io::BufReader::new(file);
    let mut records = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let rec: EncounterRecord = serde_json::from_str(line).map_err(|e| {
            anyhow::anyhow!(
                "invalid JSONL at {}:{}: {}",
                encounters.display(),
                line_no + 1,
                e
            )
        })?;
        records.push(rec);
    }

    let (feature_names, x, y_eps, y_h) = build_dataset_rows(&records);
    if x.is_empty() {
        anyhow::bail!(
            "no usable encounter rows found in {} (try different windows or a more chaotic trajectory)",
            encounters.display()
        );
    }

    let (train_idx, test_idx) = train_test_split(x.len(), seed, 0.8);
    let (x_train, y_eps_train) = subset_xy(&x, &y_eps, &train_idx);
    let (x_test, y_eps_test) = subset_xy(&x, &y_eps, &test_idx);
    let (x_train2, y_h_train) = subset_xy(&x, &y_h, &train_idx);
    let (x_test2, y_h_test) = subset_xy(&x, &y_h, &test_idx);
    debug_assert_eq!(x_train, x_train2);
    debug_assert_eq!(x_test, x_test2);

    let cfg = StlsConfig {
        thresholds: Vec::new(),
        ridge_lambda: 1e-8,
        max_iter: 25,
        normalize: true,
    };

    let dataset_eps_train = Dataset::new(feature_names.clone(), x_train, y_eps_train);
    let dataset_h_train = Dataset::new(feature_names.clone(), x_train2, y_h_train);

    let top_eps = stls_path_search(&dataset_eps_train, &cfg, FitnessHeuristic::Mse);
    let top_h = stls_path_search(&dataset_h_train, &cfg, FitnessHeuristic::Mse);

    let dataset_eps_test = Dataset::new(feature_names.clone(), x_test, y_eps_test);
    let dataset_h_test = Dataset::new(feature_names.clone(), x_test2, y_h_test);

    let eps_models = topk_with_test(&top_eps.entries, &dataset_eps_test);
    let h_models = topk_with_test(&top_h.entries, &dataset_h_test);

    let model = EncounterMapModel {
        version: PREDICTABILITY_VERSION.to_string(),
        n_samples: x.len(),
        n_train: train_idx.len(),
        n_test: test_idx.len(),
        feature_names,
        stls: StlsSettings {
            ridge_lambda: cfg.ridge_lambda,
            max_iter: cfg.max_iter,
            normalize: cfg.normalize,
        },
        targets: vec![
            TargetModel {
                target: "delta_event_pair_specific_energy".to_string(),
                top3: eps_models,
            },
            TargetModel {
                target: "delta_event_pair_h".to_string(),
                top3: h_models,
            },
        ],
        notes: vec![
            "This is a lightweight, sparse linear baseline for encounter mapping.".to_string(),
            "Interpret with care: three-body scattering maps are generally nonlinear.".to_string(),
        ],
    };

    fs::write(&out, serde_json::to_string_pretty(&model)?)?;
    eprintln!("wrote model: {}", out.display());
    Ok(())
}

fn topk_with_test(top3: &[EquationScore], test: &Dataset) -> Vec<TargetCandidate> {
    let mut out = Vec::new();
    for es in top3 {
        let test_mse = if test.samples.is_empty() {
            None
        } else {
            Some(score_equation(&es.equation, test))
        };
        out.push(TargetCandidate {
            equation: es.equation.clone(),
            train_score: es.score,
            test_mse,
        });
    }
    out
}

fn pair_key(pair: PairId) -> String {
    format!("{}{}", pair.i, pair.j)
}

fn build_dataset_rows(records: &[EncounterRecord]) -> (Vec<String>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let pairs = [PairId::new(0, 1), PairId::new(0, 2), PairId::new(1, 2)];
    let mut feature_names = Vec::new();
    feature_names.push("const".to_string());
    feature_names.push("event_d".to_string());
    feature_names.push("event_inv_d".to_string());
    feature_names.push("event_inv_d2".to_string());
    feature_names.push("event_is_pre_energy_pair".to_string());
    for p in &pairs {
        let suffix = pair_key(*p);
        feature_names.push(format!("pre_r_{suffix}"));
        feature_names.push(format!("pre_speed_{suffix}"));
        feature_names.push(format!("pre_cos_{suffix}"));
        feature_names.push(format!("pre_eps_{suffix}"));
        feature_names.push(format!("pre_h_{suffix}"));
    }

    let mut x = Vec::new();
    let mut y_eps = Vec::new();
    let mut y_h = Vec::new();
    for r in records {
        let d = r.event_min_pair_dist;
        if !d.is_finite() || d <= 0.0 {
            continue;
        }
        let inv_d = 1.0 / d;
        let inv_d2 = inv_d * inv_d;
        let mut row = Vec::with_capacity(feature_names.len());
        row.push(1.0);
        row.push(d);
        row.push(inv_d);
        row.push(inv_d2);
        row.push(if r.labels.event_pair_equals_pre_energy_pair { 1.0 } else { 0.0 });

        for p in &pairs {
            let Some(m) = find_pair_metrics(&r.pre.pairs, *p) else {
                row.clear();
                break;
            };
            row.push(m.r);
            row.push(m.speed);
            row.push(m.cos_approach);
            row.push(m.specific_energy);
            row.push(m.h);
        }
        if row.is_empty() {
            continue;
        }

        if row.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let t1 = r.labels.delta_event_pair_specific_energy;
        let t2 = r.labels.delta_event_pair_h;
        if !t1.is_finite() || !t2.is_finite() {
            continue;
        }
        x.push(row);
        y_eps.push(t1);
        y_h.push(t2);
    }
    (feature_names, x, y_eps, y_h)
}

fn find_pair_metrics(pairs: &[PairMetrics], pair: PairId) -> Option<&PairMetrics> {
    pairs.iter().find(|m| m.pair == pair)
}

fn train_test_split(n: usize, seed: u64, train_frac: f64) -> (Vec<usize>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = threebody_discover::ga::Lcg::new(seed);
    // Fisher–Yates shuffle.
    for i in (1..n).rev() {
        let j = rng.gen_range_usize(0, i);
        idx.swap(i, j);
    }
    let n_train = ((train_frac.clamp(0.0, 1.0)) * (n as f64)).round() as usize;
    let n_train = n_train.clamp(1, n.saturating_sub(1).max(1));
    let train = idx[..n_train].to_vec();
    let test = idx[n_train..].to_vec();
    (train, test)
}

fn subset_xy(x: &[Vec<f64>], y: &[f64], idx: &[usize]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut xs = Vec::with_capacity(idx.len());
    let mut ys = Vec::with_capacity(idx.len());
    for &i in idx {
        if let (Some(row), Some(t)) = (x.get(i), y.get(i)) {
            xs.push(row.clone());
            ys.push(*t);
        }
    }
    (xs, ys)
}

#[cfg(test)]
mod tests {
    use super::run_train_map;
    use crate::predictability::extract::{EncounterLabels, EncounterRecord, Snapshot};
    use crate::predictability::features::{PairId, PairMetrics};
    use std::fs;
    use std::path::PathBuf;

    fn temp_dir(prefix: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("threebody_{prefix}_{pid}_{nanos}"));
        p
    }

    fn snapshot(step: usize, t: f64, pair: PairId, eps: f64) -> Snapshot {
        let pairs = vec![
            PairMetrics {
                pair: PairId::new(0, 1),
                r: 1.0,
                speed: 0.1,
                cos_approach: -0.1,
                specific_energy: eps,
                h: 0.01,
                e: None,
                a: None,
            },
            PairMetrics {
                pair: PairId::new(0, 2),
                r: 2.0,
                speed: 0.2,
                cos_approach: 0.2,
                specific_energy: eps + 0.1,
                h: 0.02,
                e: None,
                a: None,
            },
            PairMetrics {
                pair: PairId::new(1, 2),
                r: 3.0,
                speed: 0.3,
                cos_approach: 0.3,
                specific_energy: eps + 0.2,
                h: 0.03,
                e: None,
                a: None,
            },
        ];
        Snapshot {
            step,
            t,
            dt: 0.01,
            min_pair: pair,
            min_pair_dist: 0.5,
            energy_pair: pair,
            energy_pair_energy: eps,
            pairs,
        }
    }

    #[test]
    fn train_map_writes_model_json() {
        let dir = temp_dir("train_map");
        fs::create_dir_all(&dir).unwrap();
        let encounters_path = dir.join("encounters.jsonl");
        let out_path = dir.join("model.json");

        // Write a handful of synthetic encounter records.
        let mut lines = Vec::new();
        for i in 0..8 {
            let pair = PairId::new(0, 1);
            let pre = snapshot(i, i as f64 * 0.1, pair, -1.0);
            let post = snapshot(i + 1, i as f64 * 0.1 + 0.05, pair, -0.9);
            let rec = EncounterRecord {
                version: "v1".to_string(),
                event_step: i,
                event_t: i as f64 * 0.1,
                event_pair: pair,
                event_min_pair_dist: 0.25 + 0.01 * i as f64,
                pre,
                post,
                labels: EncounterLabels {
                    binary_pair_swapped: false,
                    pre_energy_pair: pair,
                    post_energy_pair: pair,
                    event_pair_equals_pre_energy_pair: true,
                    event_pair_equals_post_energy_pair: true,
                    delta_event_pair_specific_energy: 0.1 * i as f64,
                    delta_event_pair_h: 0.01 * i as f64,
                },
            };
            lines.push(serde_json::to_string(&rec).unwrap());
        }
        fs::write(&encounters_path, lines.join("\n")).unwrap();

        run_train_map(encounters_path, out_path.clone(), 42).unwrap();
        assert!(out_path.exists());
        let raw = fs::read_to_string(&out_path).unwrap();
        assert!(raw.contains("\"targets\""));

        let _ = fs::remove_dir_all(&dir);
    }
}
