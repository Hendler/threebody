use std::process::Command;
use std::{env, fs};

fn start_openai_chat_stub() -> (std::net::SocketAddr, std::thread::JoinHandle<()>, std::sync::Arc<std::sync::atomic::AtomicBool>) {
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
        } else if prompt.contains("writing a short evaluation of an automated equation-discovery run") {
            respond(stream, eval_md);
        } else {
            respond(stream, "{}");
        }
    }

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub server");
    listener
        .set_nonblocking(true)
        .expect("set nonblocking");
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

    (addr, handle, running)
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
fn conflicting_em_flags_fail() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let tmp_dir = env::temp_dir().join(format!("threebody_em_conflict_{}", std::process::id()));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let output = Command::new(exe)
        .current_dir(&tmp_dir)
        .args(["simulate", "--em", "--no-em", "--steps", "1", "--dt", "0.01", "--dry-run"])
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
    assert!(out_dir.join("factory").join("evaluation_input.json").exists());
    assert!(out_dir.join("factory").join("run_001").join("report.md").exists());
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
fn quickstart_command_runs_with_require_llm_against_stub() {
    use std::net::TcpStream;
    use std::sync::atomic::Ordering;

    let (addr, handle, running) = start_openai_chat_stub();

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
