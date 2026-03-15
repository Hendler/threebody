use std::process::Command;
use std::{env, fs};

fn start_openai_chat_stub() -> Option<(
    std::net::SocketAddr,
    std::thread::JoinHandle<()>,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
)> {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    fn respond(mut stream: TcpStream, content: &str) {
        let body = serde_json::json!({
            "choices": [
                {
                    "message": {
                        "content": content
                    }
                }
            ]
        })
        .to_string();
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = stream.write_all(resp.as_bytes());
    }

    fn extract_prompt(body: &str) -> Option<String> {
        let v: serde_json::Value = serde_json::from_str(body).ok()?;
        v.get("messages")?
            .get(0)?
            .get("content")?
            .as_str()
            .map(|s| s.to_string())
    }

    fn handle_conn(mut stream: TcpStream) {
        // Read headers.
        let mut buf = Vec::new();
        let mut tmp = [0u8; 1024];
        while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
            match stream.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
                Err(_) => return,
            }
            if buf.len() > 1024 * 1024 {
                return;
            }
        }
        let header_end = match buf.windows(4).position(|w| w == b"\r\n\r\n") {
            Some(i) => i + 4,
            None => return,
        };
        let headers = String::from_utf8_lossy(&buf[..header_end]);
        let mut content_len: usize = 0;
        for line in headers.lines() {
            let lower = line.to_ascii_lowercase();
            if let Some(v) = lower.strip_prefix("content-length:") {
                content_len = v.trim().parse::<usize>().unwrap_or(0);
            }
        }

        // Read body.
        let mut body_bytes = buf[header_end..].to_vec();
        while body_bytes.len() < content_len {
            match stream.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => body_bytes.extend_from_slice(&tmp[..n]),
                Err(_) => break,
            }
        }
        let body = String::from_utf8_lossy(&body_bytes);
        let prompt = extract_prompt(&body).unwrap_or_default();

        let ic_json = r#"{"bodies":[{"mass":1.0,"charge":0.0,"pos":[-0.5,0.0,0.0],"vel":[0.0,0.7,0.0]},{"mass":1.0,"charge":0.0,"pos":[0.5,0.0,0.0],"vel":[0.0,-0.7,0.0]},{"mass":0.1,"charge":0.0,"pos":[0.0,0.3,0.0],"vel":[0.0,0.0,0.0]}],"barycentric":true,"notes":"stub"}"#;
        let judge_json = r#"{
  "version":"v1",
  "ranking":[0],
  "scores":[
    {
      "id":0,
      "total":"auto",
      "components":{
        "fidelity":"auto",
        "parsimony":"auto",
        "physical_plausibility":"auto",
        "regime_consistency":"auto",
        "stability_risk":"auto"
      },
      "rationale":"stub judge"
    }
  ],
  "recommendations":{
    "next_initial_conditions":null,
    "next_rollout_integrator":null,
    "next_ga_heuristic":null,
    "next_discovery_solver":null,
    "next_normalize":null,
    "next_stls_threshold":"auto",
    "next_ridge_lambda":"auto",
    "next_lasso_alpha":"auto"
  },
  "summary":"stub judge ok"
}"#;
        let eval_md = r#"# Factory Evaluation (stub)

## What was run
- stub

## Best result (plain English)
- stub

## How good is it?
- stub

## What the numbers mean
- stub

## Next steps (easy)
- stub

## Next steps (more advanced)
- stub

## How to report improvements
- stub
"#;

        if prompt.contains("selecting initial conditions") {
            respond(stream, ic_json);
        } else if prompt.contains("academic reviewer evaluating candidate equations") {
            respond(stream, judge_json);
        } else if prompt
            .contains("writing a short evaluation of an automated equation-discovery run")
        {
            respond(stream, eval_md);
        } else {
            respond(stream, "{}");
        }
    }

    let listener = match TcpListener::bind("127.0.0.1:0") {
        Ok(v) => v,
        Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => return None,
        Err(err) => panic!("bind stub server: {err}"),
    };
    listener.set_nonblocking(true).expect("set nonblocking");
    let addr = listener.local_addr().expect("local addr");
    let running = Arc::new(AtomicBool::new(true));
    let running_thread = running.clone();

    let handle = thread::spawn(move || {
        while running_thread.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((stream, _)) => handle_conn(stream),
                Err(err)
                    if matches!(
                        err.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                    ) =>
                {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(_) => break,
            }
        }
    });

    Some((addr, handle, running))
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    env::temp_dir().join(format!("{prefix}_{}_{}", std::process::id(), nanos))
}

#[test]
fn help_includes_commands() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let output = Command::new(exe).arg("--help").output().expect("run help");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("example-config"));
    assert!(stdout.contains("example-ic"));
    assert!(stdout.contains("simulate"));
    assert!(stdout.contains("run"));
    assert!(stdout.contains("factory"));
    assert!(stdout.contains("quickstart"));
}

#[test]
fn example_config_is_valid_json() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let output = Command::new(exe)
        .arg("example-config")
        .output()
        .expect("run example-config");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let _cfg: serde_json::Value = serde_json::from_str(&stdout).expect("valid json");
}

#[test]
fn example_ic_is_valid_json() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let output = Command::new(exe)
        .args(["example-ic", "--preset", "three-body"])
        .output()
        .expect("run example-ic");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let value: serde_json::Value = serde_json::from_str(&stdout).expect("valid json");
    assert!(value.get("bodies").is_some());
}

#[test]
fn dry_run_does_not_write_output() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir = env::temp_dir();
    let output_path = tmp_dir.join(format!("threebody_dryrun_{}.csv", std::process::id()));
    if output_path.exists() {
        let _ = fs::remove_file(&output_path);
    }
    let output = Command::new(exe)
        .args([
            "simulate",
            "--dry-run",
            "--output",
            output_path.to_str().unwrap(),
            "--steps",
            "1",
            "--dt",
            "0.01",
        ])
        .output()
        .expect("run dry-run");
    assert!(output.status.success());
    assert!(!output_path.exists());
}

#[test]
fn discover_writes_top3() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir = env::temp_dir().join(format!("threebody_discover_{}", std::process::id()));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    // First produce a trajectory + sidecar, then run discovery on it using defaults (traj.csv/traj.json).
    let sim = Command::new(exe)
        .current_dir(&tmp_dir)
        .args(["simulate", "--steps", "5", "--dt", "0.01"])
        .output()
        .expect("run simulate");
    assert!(sim.status.success(), "simulate failed: {:?}", sim);

    let out_path = tmp_dir.join(format!("threebody_discover_{}.json", std::process::id()));
    if out_path.exists() {
        let _ = fs::remove_file(&out_path);
    }
    let output = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "discover",
            "--solver",
            "stls",
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run discover");
    assert!(output.status.success());
    let json = fs::read_to_string(&out_path).expect("output json");
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let solver = value.get("solver").expect("solver metadata");
    assert_eq!(solver.get("name").and_then(|v| v.as_str()), Some("stls"));
    assert!(value.get("top3").is_some());
    assert!(value.get("component_top3").is_some());
    assert!(value.get("grid_top3").is_some());
}

#[test]
fn discover_solver_ga_still_runs() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir = env::temp_dir().join(format!("threebody_discover_ga_{}", std::process::id()));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let sim = Command::new(exe)
        .current_dir(&tmp_dir)
        .args(["simulate", "--steps", "5", "--dt", "0.01"])
        .output()
        .expect("run simulate");
    assert!(sim.status.success(), "simulate failed: {:?}", sim);

    let out_path = tmp_dir.join("out.json");
    let output = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "discover",
            "--solver",
            "ga",
            "--runs",
            "2",
            "--population",
            "4",
            "--seed",
            "1",
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run discover ga");
    assert!(output.status.success());
    let json = fs::read_to_string(&out_path).expect("output json");
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let solver = value.get("solver").expect("solver metadata");
    assert_eq!(solver.get("name").and_then(|v| v.as_str()), Some("ga"));
    assert!(value.get("top3").is_some());
}

#[test]
fn factory_runs_once_with_mock_llm() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir = env::temp_dir().join(format!("threebody_factory_{}", std::process::id()));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    let output = Command::new(exe)
        .args([
            "factory",
            "--out-dir",
            tmp_dir.to_str().unwrap(),
            "--max-iters",
            "1",
            "--auto",
            "--steps",
            "5",
            "--dt",
            "0.01",
            "--llm-mode",
            "mock",
        ])
        .output()
        .expect("run factory");
    assert!(output.status.success());
    let report = tmp_dir.join("run_001").join("report.json");
    assert!(report.exists());
    let json = fs::read_to_string(&report).expect("report json");
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let solver = value.get("solver").expect("solver metadata");
    assert_eq!(solver.get("name").and_then(|v| v.as_str()), Some("stls"));
    let trace = tmp_dir.join("run_001").join("rollout_trace.json");
    assert!(trace.exists());
    let eval_md = tmp_dir.join("evaluation.md");
    assert!(eval_md.exists());
    assert!(tmp_dir.join("evaluation_llm.md").exists());
    assert!(tmp_dir.join("evaluation_prompt.txt").exists());
    assert!(tmp_dir.join("evaluation_input.json").exists());
    assert!(tmp_dir.join("evaluation_history.json").exists());
    assert!(tmp_dir.join("evaluation.tex").exists());
}

#[test]
fn factory_archive_seed_recursively_merges_prior_runs() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let root = unique_temp_dir("threebody_factory_seeded");
    let prior_out = root.join("prior").join("nested").join("run_a");
    let seeded_out = root.join("seeded_run");
    if root.exists() {
        let _ = fs::remove_dir_all(&root);
    }
    fs::create_dir_all(prior_out.parent().unwrap()).expect("create prior parent");

    let prior = Command::new(exe)
        .args([
            "factory",
            "--out-dir",
            prior_out.to_str().unwrap(),
            "--max-iters",
            "1",
            "--auto",
            "--steps",
            "5",
            "--dt",
            "0.01",
            "--seed",
            "123",
            "--llm-mode",
            "mock",
        ])
        .output()
        .expect("run prior factory");
    assert!(
        prior.status.success(),
        "prior run failed: {}",
        String::from_utf8_lossy(&prior.stderr)
    );
    assert!(prior_out.join("equation_search_archive.json").exists());

    let seeded = Command::new(exe)
        .args([
            "factory",
            "--out-dir",
            seeded_out.to_str().unwrap(),
            "--max-iters",
            "1",
            "--auto",
            "--steps",
            "5",
            "--dt",
            "0.01",
            "--seed",
            "999",
            "--llm-mode",
            "mock",
            "--equation-archive-seed",
            root.join("prior").to_str().unwrap(),
        ])
        .output()
        .expect("run seeded factory");
    assert!(
        seeded.status.success(),
        "seeded run failed: {}",
        String::from_utf8_lossy(&seeded.stderr)
    );

    let seeded_meta = seeded_out.join("equation_search_archive_seeded_from.json");
    assert!(seeded_meta.exists(), "expected seed-merge metadata file");
    let seeded_json = fs::read_to_string(&seeded_meta).expect("read seeded metadata");
    let seeded_value: serde_json::Value = serde_json::from_str(&seeded_json).unwrap();
    let seed_paths = seeded_value
        .get("seed_paths")
        .and_then(|v| v.as_array())
        .expect("seed_paths array");
    assert!(
        !seed_paths.is_empty(),
        "expected at least one resolved seed archive path"
    );
    let seeded_count = seeded_value
        .get("archive_node_count_after_seed")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert!(seeded_count > 0, "expected merged archive nodes");

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn factory_injects_equation_ga_candidates_and_uses_elite_equations_for_ic_notes() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir = env::temp_dir().join(format!("threebody_factory_eq_ga_{}", std::process::id()));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }

    let output = Command::new(exe)
        .args([
            "factory",
            "--out-dir",
            tmp_dir.to_str().unwrap(),
            "--max-iters",
            "2",
            "--auto",
            "--steps",
            "5",
            "--dt",
            "0.01",
            "--llm-mode",
            "mock",
        ])
        .output()
        .expect("run factory 2 iters");
    assert!(
        output.status.success(),
        "factory failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let sim_attempts = fs::read_to_string(tmp_dir.join("run_002").join("sim_attempts.json"))
        .expect("sim_attempts.json");
    let attempts_v: serde_json::Value = serde_json::from_str(&sim_attempts).unwrap();
    let notes = attempts_v
        .get(0)
        .and_then(|v| v.get("ic_request"))
        .and_then(|v| v.get("notes"))
        .and_then(|v| v.as_array())
        .expect("ic_request.notes");
    assert!(
        notes.iter().any(|n| {
            n.as_str()
                .unwrap_or_default()
                .contains("PREV_ITER_ELITE_EQUATIONS")
        }),
        "expected elite-equation notes in IC request"
    );

    let discovery =
        fs::read_to_string(tmp_dir.join("run_002").join("discovery.json")).expect("discovery.json");
    let discovery_v: serde_json::Value = serde_json::from_str(&discovery).unwrap();
    let vector_candidates = discovery_v
        .get("vector_candidates")
        .and_then(|v| v.as_array())
        .expect("vector_candidates array");
    let has_mutant = vector_candidates.iter().any(|c| {
        c.get("notes")
            .and_then(|v| v.as_array())
            .map_or(false, |arr| {
                arr.iter().any(|n| {
                    n.as_str()
                        .unwrap_or_default()
                        .contains("kind=equation_ga_mutant")
                })
            })
    });
    assert!(
        has_mutant,
        "expected at least one equation_ga_mutant candidate"
    );

    let _ = fs::remove_dir_all(&tmp_dir);
}

#[test]
fn conflicting_em_flags_fail() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir = env::temp_dir().join(format!("threebody_em_conflict_{}", std::process::id()));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let output = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "simulate",
            "--em",
            "--no-em",
            "--steps",
            "1",
            "--dt",
            "0.01",
            "--dry-run",
        ])
        .output()
        .expect("run conflicting flags");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.to_lowercase().contains("conflicting"));
}

#[test]
fn quick_start_flow_runs() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir = env::temp_dir().join(format!("threebody_quickstart_{}", std::process::id()));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let config_path = tmp_dir.join("config.json");
    let ic_path = tmp_dir.join("ic.json");

    let out = Command::new(exe)
        .current_dir(&tmp_dir)
        .args(["example-config", "--out", config_path.to_str().unwrap()])
        .output()
        .expect("example-config");
    assert!(out.status.success());

    let out = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "example-ic",
            "--preset",
            "three-body",
            "--out",
            ic_path.to_str().unwrap(),
        ])
        .output()
        .expect("example-ic");
    assert!(out.status.success());

    let traj_csv = tmp_dir.join("traj.csv");
    let traj_json = tmp_dir.join("traj.json");
    let out = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "simulate",
            "--config",
            config_path.to_str().unwrap(),
            "--ic",
            ic_path.to_str().unwrap(),
            "--output",
            traj_csv.to_str().unwrap(),
            "--steps",
            "5",
            "--dt",
            "0.01",
        ])
        .output()
        .expect("simulate");
    assert!(out.status.success(), "simulate failed: {:?}", out);
    assert!(traj_csv.exists());
    assert!(traj_json.exists());

    let discover_out = tmp_dir.join("top_equations.json");
    let out = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "discover",
            "--solver",
            "stls",
            "--out",
            discover_out.to_str().unwrap(),
        ])
        .output()
        .expect("discover");
    assert!(out.status.success(), "discover failed: {:?}", out);
    let json = fs::read_to_string(&discover_out).expect("discovery output json");
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(value.get("top3").is_some());
    assert!(value.get("judge").is_some());
    let solver = value.get("solver").expect("solver metadata");
    assert_eq!(solver.get("name").and_then(|v| v.as_str()), Some("stls"));

    let factory_dir = tmp_dir.join("factory_out");
    let out = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "experiment",
            "--out-dir",
            factory_dir.to_str().unwrap(),
            "--max-iters",
            "1",
            "--auto",
            "--steps",
            "5",
            "--dt",
            "0.01",
            "--runs",
            "2",
            "--population",
            "4",
            "--llm-mode",
            "mock",
        ])
        .output()
        .expect("experiment");
    assert!(out.status.success(), "experiment failed: {:?}", out);
    assert!(factory_dir.join("run_001").join("report.json").exists());
    assert!(factory_dir.join("evaluation.md").exists());
    assert!(factory_dir.join("evaluation_llm.md").exists());
    assert!(factory_dir.join("evaluation_input.json").exists());
    assert!(factory_dir.join("evaluation_history.json").exists());
    assert!(factory_dir.join("evaluation.tex").exists());
}

#[test]
fn bench_em_writes_suite_and_results() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_root = env::temp_dir().join(format!("threebody_bench_em_{}", std::process::id()));
    if tmp_root.exists() {
        let _ = fs::remove_dir_all(&tmp_root);
    }
    fs::create_dir_all(&tmp_root).expect("create temp dir");

    let results_dir = tmp_root.join("results");
    let out_dir = results_dir.join("bench_out");

    let output = Command::new(exe)
        .args([
            "bench-em",
            "--results-dir",
            results_dir.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--steps",
            "2",
            "--dt",
            "0.01",
            "--max-cases",
            "1",
            "--heldout",
            "0",
            "--no-pdf",
        ])
        .output()
        .expect("run bench-em");
    assert!(
        output.status.success(),
        "bench-em failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(out_dir.join("RESULTS.md").exists());
    assert!(out_dir.join("suite.json").exists());
    assert!(out_dir
        .join("factory")
        .join("evaluation_input.json")
        .exists());
    assert!(out_dir
        .join("factory")
        .join("run_001")
        .join("report.md")
        .exists());
}

#[test]
fn quickstart_command_runs() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir = env::temp_dir().join(format!("threebody_quickstart_cmd_{}", std::process::id()));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let out = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "quickstart",
            "--out-dir",
            tmp_dir.to_str().unwrap(),
            "--steps",
            "5",
            "--max-iters",
            "1",
        ])
        .output()
        .expect("quickstart");
    assert!(out.status.success(), "quickstart failed: {:?}", out);

    assert!(tmp_dir.join("config.json").exists());
    assert!(tmp_dir.join("RESULTS.md").exists());

    let factory_dir = tmp_dir.join("factory");
    assert!(factory_dir.join("run_001").join("report.json").exists());
    assert!(factory_dir.join("evaluation.md").exists());
    assert!(factory_dir.join("evaluation.tex").exists());
    assert!(factory_dir.join("evaluation_history.json").exists());
}

#[test]
fn quickstart_autoresearch_writes_score_artifacts() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir = env::temp_dir().join(format!(
        "threebody_quickstart_autoresearch_{}",
        std::process::id()
    ));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let out = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "quickstart",
            "--profile",
            "autoresearch",
            "--out-dir",
            tmp_dir.to_str().unwrap(),
            "--steps",
            "5",
            "--max-iters",
            "1",
        ])
        .output()
        .expect("quickstart autoresearch");
    assert!(
        out.status.success(),
        "quickstart autoresearch failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(tmp_dir.join("autoresearch_score.json").exists());
    assert!(tmp_dir.join("autoresearch_score.tsv").exists());
    assert!(tmp_dir.join("autoresearch_score.txt").exists());
    assert!(tmp_dir.join("AUTORESEARCH_PROGRAM.md").exists());

    let json = fs::read_to_string(tmp_dir.join("autoresearch_score.json"))
        .expect("read autoresearch score json");
    let value: serde_json::Value = serde_json::from_str(&json).expect("parse autoresearch score");
    assert_eq!(
        value.get("profile").and_then(|v| v.as_str()),
        Some("autoresearch")
    );
    assert_eq!(
        value.get("score_direction").and_then(|v| v.as_str()),
        Some("lower_is_better")
    );
}

#[test]
fn quickstart_command_runs_with_require_llm_against_stub() {
    use std::net::TcpStream;
    use std::sync::atomic::Ordering;

    let Some((addr, handle, running)) = start_openai_chat_stub() else {
        eprintln!(
            "skipping require-llm stub test (loopback sockets not permitted in this environment)"
        );
        return;
    };

    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir = env::temp_dir().join(format!(
        "threebody_quickstart_cmd_llm_{}",
        std::process::id()
    ));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let base_url = format!("http://{}/v1", addr);
    let out = Command::new(exe)
        .current_dir(&tmp_dir)
        .env("OPENAI_API_KEY", "sk-test")
        .env("OPENAI_BASE_URL", base_url)
        .env("OPENAI_API_STYLE", "chat")
        .args([
            "quickstart",
            "--out-dir",
            tmp_dir.to_str().unwrap(),
            "--steps",
            "5",
            "--max-iters",
            "1",
            "--require-llm",
        ])
        .output()
        .expect("quickstart require-llm");

    // Shut down stub server thread.
    running.store(false, Ordering::SeqCst);
    let _ = TcpStream::connect(addr);
    handle.join().expect("join stub server");

    assert!(
        out.status.success(),
        "quickstart require-llm failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(tmp_dir.join("RESULTS.md").exists());
    assert!(tmp_dir.join("factory").join("evaluation_llm.md").exists());
}

#[test]
fn discover_accepts_explicit_input_and_sidecar_paths() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir = env::temp_dir().join(format!("threebody_discover_paths_{}", std::process::id()));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    // Produce non-default filenames, then call `discover --input/--sidecar`.
    let input_csv = tmp_dir.join("run.csv");
    let sim = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "simulate",
            "--output",
            input_csv.to_str().unwrap(),
            "--steps",
            "5",
            "--dt",
            "0.01",
        ])
        .output()
        .expect("run simulate");
    assert!(sim.status.success(), "simulate failed: {:?}", sim);
    let sidecar_json = tmp_dir.join("run.json");
    assert!(input_csv.exists());
    assert!(sidecar_json.exists());

    let out_path = tmp_dir.join("out.json");
    let out = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "discover",
            "--solver",
            "stls",
            "--input",
            input_csv.to_str().unwrap(),
            "--sidecar",
            sidecar_json.to_str().unwrap(),
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("discover with explicit paths");
    assert!(out.status.success(), "discover failed: {:?}", out);
    let json = fs::read_to_string(&out_path).expect("output json");
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(value.get("top3").is_some());
    assert_eq!(
        value.get("input_csv").and_then(|v| v.as_str()),
        Some(input_csv.to_str().unwrap())
    );
    assert_eq!(
        value.get("sidecar_json").and_then(|v| v.as_str()),
        Some(sidecar_json.to_str().unwrap())
    );
}

#[test]
fn run_alias_works() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir = env::temp_dir().join(format!("threebody_run_alias_{}", std::process::id()));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let out = Command::new(exe)
        .current_dir(&tmp_dir)
        .args(["run", "--steps", "1", "--dt", "0.01", "--dry-run"])
        .output()
        .expect("run alias");
    assert!(out.status.success(), "run failed: {:?}", out);
}

#[test]
fn predictability_takens_writes_report_with_sensitivity_fields() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir = env::temp_dir().join(format!("threebody_takens_{}", std::process::id()));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let input_csv = tmp_dir.join("traj.csv");
    let sim = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "simulate",
            "--output",
            input_csv.to_str().unwrap(),
            "--steps",
            "80",
            "--dt",
            "0.01",
        ])
        .output()
        .expect("run simulate");
    assert!(sim.status.success(), "simulate failed: {:?}", sim);
    assert!(input_csv.exists());

    let report_path = tmp_dir.join("takens_report.json");
    let out = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "predictability",
            "takens",
            "--input",
            input_csv.to_str().unwrap(),
            "--column",
            "min_pair_dist",
            "--sensors",
            "a1_x,v1_x",
            "--out",
            report_path.to_str().unwrap(),
            "--tau",
            "1,2",
            "--m",
            "3,4",
            "--k",
            "6,10",
            "--lambda",
            "1e-8,1e-6",
            "--model",
            "both",
            "--split-mode",
            "chronological",
            "--sensitivity-weight",
            "0.1",
            "--train-frac",
            "0.7",
            "--val-frac",
            "0.15",
        ])
        .output()
        .expect("run predictability takens");

    assert!(
        out.status.success(),
        "takens failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(report_path.exists());

    let json = fs::read_to_string(&report_path).expect("takens report json");
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid takens report");
    assert_eq!(
        value.get("column").and_then(|v| v.as_str()),
        Some("min_pair_dist")
    );
    let sensors = value
        .get("sensors")
        .and_then(|v| v.as_array())
        .expect("sensors in report");
    assert!(sensors.iter().any(|s| s.as_str() == Some("min_pair_dist")));
    assert!(sensors.iter().any(|s| s.as_str() == Some("a1_x")));
    assert!(sensors.iter().any(|s| s.as_str() == Some("v1_x")));
    let best = value.get("best").expect("best row in report");
    let model = best
        .get("config")
        .and_then(|v| v.get("model"))
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    assert!(model == "linear" || model == "rational");
    let arch = best.get("architecture").expect("architecture metadata in best row");
    assert!(arch.get("model_family").and_then(|v| v.as_str()).is_some());
    assert!(arch
        .get("feature_dim")
        .and_then(|v| v.as_u64())
        .map_or(false, |v| v > 0));
    assert!(arch
        .get("parameter_count")
        .and_then(|v| v.as_u64())
        .map_or(false, |v| v > 0));
    assert!(best.get("holdout_mse").and_then(|v| v.as_f64()).is_some());
    assert!(best
        .get("holdout_sensitivity")
        .and_then(|v| v.get("median_rel_error"))
        .and_then(|v| v.as_f64())
        .is_some());
}

#[test]
fn predictability_report_summarizes_channel_efficacy() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir = env::temp_dir().join(format!("threebody_efficacy_report_{}", std::process::id()));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let report_raw = tmp_dir.join("takens_a1x.json");
    let report_derived = tmp_dir.join("takens_min_pair.json");

    fs::write(
        &report_raw,
        r#"{
  "column":"a1_x",
  "best":{
    "config":{"model":"linear"},
    "holdout_mse":0.0005,
    "holdout_baseline_mse":0.1,
    "holdout_relative_improvement":0.995,
    "holdout_sensitivity":{"median_rel_error":0.02}
  }
}"#,
    )
    .expect("write raw report");

    fs::write(
        &report_derived,
        r#"{
  "column":"min_pair_dist",
  "best":{
    "config":{"model":"linear"},
    "holdout_mse":0.2,
    "holdout_baseline_mse":0.1,
    "holdout_relative_improvement":-1.0,
    "holdout_sensitivity":{"median_rel_error":1.0}
  }
}"#,
    )
    .expect("write derived report");

    let out_json = tmp_dir.join("efficacy_report.json");
    let out_md = tmp_dir.join("efficacy_report.md");
    let reports_csv = format!("{},{}", report_raw.display(), report_derived.display());

    let out = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "predictability",
            "report",
            "--reports",
            &reports_csv,
            "--out",
            out_json.to_str().unwrap(),
            "--markdown-out",
            out_md.to_str().unwrap(),
            "--improvement-threshold",
            "0.0",
            "--max-sensitivity-median",
            "0.1",
            "--bootstrap-resamples",
            "256",
            "--bootstrap-ci",
            "0.95",
            "--bootstrap-seed",
            "7",
        ])
        .output()
        .expect("run predictability report");

    assert!(
        out.status.success(),
        "report failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out_json.exists());
    assert!(out_md.exists());

    let json = fs::read_to_string(&out_json).expect("read efficacy report");
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid efficacy json");
    let agg = value.get("aggregate").expect("aggregate");
    assert_eq!(agg.get("n_channels").and_then(|v| v.as_u64()), Some(2));
    assert_eq!(agg.get("n_effective").and_then(|v| v.as_u64()), Some(1));
    assert_eq!(
        agg.get("claim_status").and_then(|v| v.as_str()),
        Some("information_helpful_in_some_channels")
    );
    assert!(agg
        .get("median_relative_improvement_all_ci_low")
        .and_then(|v| v.as_f64())
        .is_some());
    assert!(agg
        .get("median_relative_improvement_all_ci_high")
        .and_then(|v| v.as_f64())
        .is_some());
}

#[test]
fn predictability_compare_reports_effect_size_and_non_regression() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir =
        env::temp_dir().join(format!("threebody_efficacy_compare_{}", std::process::id()));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let before = tmp_dir.join("before.json");
    let after = tmp_dir.join("after.json");
    fs::write(
        &before,
        r#"{
  "aggregate": {
    "n_effective": 1,
    "effective_rate": 0.5,
    "median_relative_improvement_all": 0.2,
    "median_relative_improvement_raw": 0.9,
    "median_relative_improvement_derived": -0.5,
    "info_value_delta_raw_minus_derived": 1.4
  },
  "channels": [
    {"column":"a1_x","holdout_relative_improvement":0.9,"effective":true},
    {"column":"min_pair_dist","holdout_relative_improvement":-0.5,"effective":false}
  ]
}"#,
    )
    .expect("write before");
    fs::write(
        &after,
        r#"{
  "aggregate": {
    "n_effective": 2,
    "effective_rate": 1.0,
    "median_relative_improvement_all": 0.95,
    "median_relative_improvement_raw": 0.97,
    "median_relative_improvement_derived": 0.92,
    "info_value_delta_raw_minus_derived": 0.05
  },
  "channels": [
    {"column":"a1_x","holdout_relative_improvement":0.95,"effective":true},
    {"column":"min_pair_dist","holdout_relative_improvement":0.92,"effective":true}
  ]
}"#,
    )
    .expect("write after");

    let out_json = tmp_dir.join("compare.json");
    let out_md = tmp_dir.join("compare.md");
    let out = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "predictability",
            "compare",
            "--before",
            before.to_str().unwrap(),
            "--after",
            after.to_str().unwrap(),
            "--out",
            out_json.to_str().unwrap(),
            "--markdown-out",
            out_md.to_str().unwrap(),
            "--non-regression-tol",
            "0.0",
        ])
        .output()
        .expect("run predictability compare");

    assert!(
        out.status.success(),
        "compare failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out_json.exists());
    assert!(out_md.exists());

    let json = fs::read_to_string(&out_json).expect("read compare");
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid compare");
    let agg = value.get("aggregate").expect("aggregate");
    assert!(
        agg.get("median_relative_improvement_all_delta")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
            > 0.0
    );
    let flags = value.get("flags").expect("flags");
    assert_eq!(
        flags
            .get("overall_non_regression")
            .and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn predictability_context_window_reports_minimum_effective_context() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir = env::temp_dir().join(format!("threebody_context_window_{}", std::process::id()));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let input_csv = tmp_dir.join("traj.csv");
    let sim = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "simulate",
            "--output",
            input_csv.to_str().unwrap(),
            "--steps",
            "120",
            "--dt",
            "0.01",
            "--preset",
            "two-body",
        ])
        .output()
        .expect("run simulate");
    assert!(
        sim.status.success(),
        "simulate failed: {}",
        String::from_utf8_lossy(&sim.stderr)
    );
    assert!(input_csv.exists());

    let out_json = tmp_dir.join("context_window.json");
    let out_md = tmp_dir.join("context_window.md");
    let out = Command::new(exe)
        .current_dir(&tmp_dir)
        .args([
            "predictability",
            "context-window",
            "--input",
            input_csv.to_str().unwrap(),
            "--column",
            "r1_x",
            "--sensors",
            "r1_x,v1_x",
            "--out",
            out_json.to_str().unwrap(),
            "--markdown-out",
            out_md.to_str().unwrap(),
            "--tau",
            "1",
            "--m",
            "1,2,3",
            "--k",
            "6",
            "--lambda",
            "1e-8,1e-6",
            "--model",
            "all",
            "--split-mode",
            "chronological",
            "--improvement-threshold",
            "0.0",
            "--max-sensitivity-median",
            "10.0",
        ])
        .output()
        .expect("run context-window");
    assert!(
        out.status.success(),
        "context-window failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out_json.exists());
    assert!(out_md.exists());

    let json = fs::read_to_string(&out_json).expect("read context window report");
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid context report");
    let points = value
        .get("points")
        .and_then(|v| v.as_array())
        .expect("points array");
    assert_eq!(points.len(), 3);
    let summary = value.get("summary").expect("summary");
    assert_eq!(summary.get("n_points").and_then(|v| v.as_u64()), Some(3));
    assert!(summary.get("best_m").and_then(|v| v.as_u64()).is_some());
}
