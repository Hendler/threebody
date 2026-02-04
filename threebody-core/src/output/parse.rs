use std::collections::HashMap;

/// Parse a CSV header line into column names.
pub fn parse_header(line: &str) -> Vec<String> {
    line.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Build a header index map and ensure required columns are present.
pub fn require_columns(header: &[String], required: &[&str]) -> Result<HashMap<String, usize>, String> {
    let mut map = HashMap::new();
    for (idx, name) in header.iter().enumerate() {
        map.insert(name.clone(), idx);
    }
    for &req in required {
        if !map.contains_key(req) {
            return Err(format!("missing required column: {}", req));
        }
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::{parse_header, require_columns};

    #[test]
    fn parse_header_with_extra_columns() {
        let line = "step,t,r1_x,extra";
        let header = parse_header(line);
        let map = require_columns(&header, &["step", "r1_x"]).unwrap();
        assert_eq!(map.get("step"), Some(&0));
        assert_eq!(map.get("r1_x"), Some(&2));
    }

    #[test]
    fn missing_required_column_is_error() {
        let header = parse_header("step,t,r1_x");
        let err = require_columns(&header, &["missing"]).unwrap_err();
        assert!(err.contains("missing required column"));
    }
}
