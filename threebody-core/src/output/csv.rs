use std::io::{self, Write};

use crate::config::Config;
use crate::diagnostics::{compute_diagnostics, Diagnostics};
use crate::forces::{compute_accel, compute_fields, ForceConfig};
use crate::math::vec3::Vec3;
use crate::regime::{compute_regime, RegimeDiagnostics};
use crate::sim::SimStep;

pub fn csv_header(cfg: &Config) -> Vec<String> {
    let mut header = Vec::new();
    header.push("step".to_string());
    header.push("t".to_string());
    header.push("dt".to_string());
    push_vec3_group(&mut header, "r", 3);
    push_vec3_group(&mut header, "v", 3);
    push_vec3_group(&mut header, "a", 3);

    if cfg.output.include_fields {
        push_vec3_group(&mut header, "e", 3);
        push_vec3_group(&mut header, "b", 3);
    }
    if cfg.output.include_potentials {
        for i in 1..=3 {
            header.push(format!("phi{}_{}", i, "scalar"));
        }
        push_vec3_group(&mut header, "A", 3);
    }
    if cfg.output.include_diagnostics {
        header.push("energy_proxy".to_string());
        header.push("P_x".to_string());
        header.push("P_y".to_string());
        header.push("P_z".to_string());
        header.push("L_x".to_string());
        header.push("L_y".to_string());
        header.push("L_z".to_string());
        header.push("min_pair_dist".to_string());
        header.push("max_speed".to_string());
        header.push("max_accel".to_string());
        header.push("dt_ratio".to_string());
    }
    header
}

fn push_vec3_group(header: &mut Vec<String>, prefix: &str, count: usize) {
    for i in 1..=count {
        header.push(format!("{}{}_x", prefix, i));
        header.push(format!("{}{}_y", prefix, i));
        header.push(format!("{}{}_z", prefix, i));
    }
}

pub fn write_csv<W: Write>(mut writer: W, steps: &[SimStep], cfg: &Config) -> io::Result<()> {
    let header = csv_header(cfg);
    writeln!(writer, "{}", header.join(","))?;

    let force_cfg = ForceConfig {
        g: cfg.constants.g,
        k_e: cfg.constants.k_e,
        mu_0: cfg.constants.mu_0,
        epsilon: cfg.softening,
        enable_gravity: cfg.enable_gravity,
        enable_em: cfg.enable_em,
    };

    for (idx, step) in steps.iter().enumerate() {
        let acc = compute_accel(&step.system, &force_cfg);
        let fields = compute_fields(&step.system, &force_cfg);
        let diagnostics = compute_diagnostics(&step.system, cfg.constants.g, cfg.constants.k_e, cfg.softening);
        let regime = compute_regime(&step.system, &acc, step.dt);

        let mut row = Vec::with_capacity(header.len());
        row.push(idx.to_string());
        row.push(step.t.to_string());
        row.push(step.dt.to_string());
        push_vec3_values(&mut row, &step.system.state.pos);
        push_vec3_values(&mut row, &step.system.state.vel);
        push_vec3_values(&mut row, &acc);
        if cfg.output.include_fields {
            push_vec3_values(&mut row, &fields.e);
            push_vec3_values(&mut row, &fields.b);
        }
        if cfg.output.include_potentials {
            for i in 0..3 {
                row.push(fields.phi[i].to_string());
            }
            push_vec3_values(&mut row, &fields.a);
        }
        if cfg.output.include_diagnostics {
            push_diagnostics(&mut row, &diagnostics, &regime);
        }
        writeln!(writer, "{}", row.join(","))?;
    }
    Ok(())
}

fn push_vec3_values(row: &mut Vec<String>, values: &[Vec3; 3]) {
    for v in values {
        row.push(v.x.to_string());
        row.push(v.y.to_string());
        row.push(v.z.to_string());
    }
}

fn push_diagnostics(row: &mut Vec<String>, diag: &Diagnostics, regime: &RegimeDiagnostics) {
    row.push(diag.energy_proxy.to_string());
    row.push(diag.linear_momentum.x.to_string());
    row.push(diag.linear_momentum.y.to_string());
    row.push(diag.linear_momentum.z.to_string());
    row.push(diag.angular_momentum.x.to_string());
    row.push(diag.angular_momentum.y.to_string());
    row.push(diag.angular_momentum.z.to_string());
    row.push(regime.min_pair_dist.to_string());
    row.push(regime.max_speed.to_string());
    row.push(regime.max_accel.to_string());
    row.push(regime.dt_ratio.to_string());
}

#[cfg(test)]
mod tests {
    use super::csv_header;
    use crate::config::{Config, OutputConfig};

    #[test]
    fn header_includes_expected_columns() {
        let cfg = Config::default();
        let header = csv_header(&cfg);
        assert!(header.contains(&"r1_x".to_string()));
        assert!(header.contains(&"v2_y".to_string()));
        assert!(header.contains(&"a3_z".to_string()));
        assert!(header.contains(&"dt".to_string()));
        assert!(header.contains(&"energy_proxy".to_string()));
    }

    #[test]
    fn header_excludes_fields_when_disabled() {
        let mut cfg = Config::default();
        cfg.output = OutputConfig {
            include_fields: false,
            include_potentials: false,
            include_diagnostics: false,
        };
        let header = csv_header(&cfg);
        assert!(!header.contains(&"e1_x".to_string()));
        assert!(!header.contains(&"phi1_scalar".to_string()));
        assert!(!header.contains(&"energy_proxy".to_string()));
    }
}
