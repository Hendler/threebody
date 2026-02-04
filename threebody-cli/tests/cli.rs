use std::process::Command;

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
