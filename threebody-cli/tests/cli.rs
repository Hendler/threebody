use std::process::Command;
use std::{env, fs};

#[test]
fn help_includes_commands() {
    let exe = env!("CARGO_BIN_EXE_threebody-cli");
    let output = Command::new(exe).arg("--help").output().expect("run help");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("example-config"));
    assert!(stdout.contains("simulate"));
    assert!(stdout.contains("run"));
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
    let tmp_dir = env::temp_dir();
    let out_path = tmp_dir.join(format!("threebody_discover_{}.json", std::process::id()));
    if out_path.exists() {
        let _ = fs::remove_file(&out_path);
    }
    let output = Command::new(exe)
        .args([
            "discover",
            "--runs",
            "2",
            "--population",
            "4",
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run discover");
    assert!(output.status.success());
    let json = fs::read_to_string(&out_path).expect("output json");
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(value.get("top3").is_some());
    assert!(value.get("grid_top3").is_some());
}
