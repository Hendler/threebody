use std::process::Command;
use std::{env, fs};

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
