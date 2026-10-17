// SPDX-License-Identifier: Apache-2.0
//! Spark's output masking (`SQLQueryTestHelper.replaceNotIncludedMsg`), applied to engine rows
//! before they are compared with a recorded golden. The recorded files are already masked, so
//! the engine's output must be masked the same way. Rules apply per line, in this order:
//!
//! 1. `#<digits>` becomes `#x` (expression ids), `plan_id=<digits>` becomes `plan_id=x`,
//!    `*(<digits>) ` becomes `*`, `Loop id: <digits>` becomes `Loop id: xxxx`, and
//!    `uuid(Some(..))` becomes `uuid(Some(x))`.
//! 2. The warehouse directory becomes `[not included in comparison]/{warehouse_dir}`, and
//!    `Location file:` becomes `Location `.
//! 3. `Created By`, `Created Time` and `Last Access` lines are cut after the label; `Owner<TAB>`
//!    and `Partition Statistics<TAB>` / `Statistics<TAB>` keep the label and mask the value.
//! 4. `transient_lastDdlTime=<digits>` is masked, and the JSON keys `created_time`,
//!    `last_access` and `created_by` get the marker in the key and the value `None`.

const MSG: &str = "[not included in comparison]";

/// Replace every run of `prefix` followed by digits with `prefix` + `with`.
fn mask_digits(line: &str, prefix: &str, with: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(i) = rest.find(prefix) {
        let (head, tail) = rest.split_at(i + prefix.len());
        out.push_str(head);
        let digits = tail.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 {
            out.push_str(with);
        }
        rest = &tail[digits..];
    }
    out.push_str(rest);
    out
}

/// Replace `*(<digits>) ` by `*`.
fn strip_codegen_ids(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(i) = rest.find("*(") {
        let tail = &rest[i + 2..];
        let digits = tail.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && tail[digits..].starts_with(") ") {
            out.push_str(&rest[..=i]);
            rest = &tail[digits + 2..];
        } else {
            out.push_str(&rest[..i + 2]);
            rest = tail;
        }
    }
    out.push_str(rest);
    out
}

/// `uuid(Some(<anything>))` -> `uuid(Some(x))`.
fn mask_uuid(line: &str) -> String {
    const OPEN: &str = "uuid(Some(";
    match line.find(OPEN).map(|i| i + OPEN.len()) {
        Some(start) => match line[start..].find("))") {
            Some(len) => format!("{}x{}", &line[..start], &line[start + len..]),
            None => line.to_owned(),
        },
        None => line.to_owned(),
    }
}

/// `"key":"value"` -> `"key [not included in comparison]":"None"`.
fn mask_json_key(line: &str, key: &str) -> String {
    let pat = format!("\"{key}\":\"");
    let Some(i) = line.find(&pat) else {
        return line.to_owned();
    };
    let vstart = i + pat.len();
    let Some(len) = line[vstart..].find('"') else {
        return line.to_owned();
    };
    format!(
        "{}\"{key} {MSG}\":\"None\"{}",
        &line[..i],
        &line[vstart + len + 1..]
    )
}

/// Mask one line of output; `warehouse` is the engine's warehouse directory (may be empty).
#[must_use]
pub fn replace_not_included_line(line: &str, warehouse: &str) -> String {
    let mut s = mask_digits(line, "#", "x");
    s = mask_digits(&s, "plan_id=", "x");
    s = mask_digits(&s, "Loop id: ", "xxxx");
    s = mask_digits(&s, "transient_lastDdlTime=", MSG);
    s = strip_codegen_ids(&s);
    s = mask_uuid(&s);
    let wh = warehouse.strip_prefix("file:").unwrap_or(warehouse);
    if !wh.is_empty() {
        s = s.replace(wh, &format!("{MSG}/{{warehouse_dir}}"));
        s = s.replace("Location file:", "Location ");
    }
    for label in ["Created By", "Created Time", "Last Access"] {
        if s.starts_with(label) {
            s = format!("{label} {MSG}");
        }
    }
    for label in ["Owner\t", "Partition Statistics\t", "Statistics\t"] {
        if let Some(rest) = s.strip_prefix(label) {
            s = if label == "Owner\t" {
                format!("{label}{MSG}")
            } else {
                let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
                format!("{label}{MSG}{}", &rest[digits..])
            };
        }
    }
    for key in ["created_time", "last_access", "created_by"] {
        s = mask_json_key(&s, key);
    }
    s
}

/// Mask every line of a rendered row (cells may contain newlines).
#[must_use]
pub fn replace_not_included(row: &str, warehouse: &str) -> String {
    row.split('\n')
        .map(|l| replace_not_included_line(l, warehouse))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::replace_not_included_line as m;

    #[test]
    fn expression_and_plan_ids() {
        assert_eq!(m("Project [a#123L, b#4]", ""), "Project [a#xL, b#x]");
        assert_eq!(m("Exchange (plan_id=77)", ""), "Exchange (plan_id=x)");
        assert_eq!(m("*(2) Project [a#1]", ""), "*Project [a#x]");
        assert_eq!(m("a * (b) ", ""), "a * (b) ");
        assert_eq!(m("Loop id: 1234", ""), "Loop id: xxxx");
        assert_eq!(m("uuid(Some(-123)) AS q#9", ""), "uuid(Some(x)) AS q#x");
    }

    #[test]
    fn warehouse_paths() {
        assert_eq!(
            m("Location file:/tmp/wh/t/c=1", "/tmp/wh"),
            "Location [not included in comparison]/{warehouse_dir}/t/c=1"
        );
        assert_eq!(
            m("InMemoryFileIndex(file:/tmp/wh/t)", "file:/tmp/wh"),
            "InMemoryFileIndex(file:[not included in comparison]/{warehouse_dir}/t)"
        );
    }

    #[test]
    fn describe_metadata() {
        assert_eq!(
            m("Created By\tSpark 4.2.0", ""),
            "Created By [not included in comparison]"
        );
        assert_eq!(
            m("Last Access\tUNKNOWN", ""),
            "Last Access [not included in comparison]"
        );
        assert_eq!(m("Owner\troot", ""), "Owner\t[not included in comparison]");
        assert_eq!(
            m("Statistics\t123 bytes, 3 rows", ""),
            "Statistics\t[not included in comparison] bytes, 3 rows"
        );
        assert_eq!(
            m("[numFiles=1, transient_lastDdlTime=1700000000]", ""),
            "[numFiles=1, transient_lastDdlTime=[not included in comparison]]"
        );
        assert_eq!(
            m(r#"{"created_time":"Mon","b":1}"#, ""),
            r#"{"created_time [not included in comparison]":"None","b":1}"#
        );
    }
}
