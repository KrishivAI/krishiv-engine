// SPDX-License-Identifier: Apache-2.0
//! `krishiv debug-bundle`: collects a redacted diagnostic archive.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use clap::{Args, ValueEnum};

use crate::archive::{BundleArchive, PathSub, Redacted};
use crate::common::{build_session, key_value, read_input, split_statements};

/// Bundle format version.
const BUNDLE_VERSION: u64 = 1;
/// Bundle format marker in `manifest.json`.
const BUNDLE_FORMAT: &str = "krishiv.debug-bundle";

/// Archive container.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum BundleFormat {
    /// Uncompressed ustar.
    Tar,
    /// Stored (uncompressed) zip.
    Zip,
}

/// Arguments of `krishiv debug-bundle`.
///
/// Per-statement failures are recorded in the bundle (as `error_class` JSON
/// documents) and the exit status stays 0; only I/O or setup failures (bad
/// flags, unreadable input, unregistered config) exit non-zero. This differs
/// from `krishiv sql`, which exits 1 on a query error.
#[derive(Args, Debug)]
pub(crate) struct BundleArgs {
    /// SQL to run (one or more `;`-separated statements).
    #[arg(
        short = 'e',
        long = "execute",
        value_name = "SQL",
        conflicts_with = "file"
    )]
    execute: Option<String>,
    /// Read SQL from a file (`-` is stdin). With neither `-e` nor `-f`, stdin is read.
    #[arg(short = 'f', long = "file", value_name = "PATH")]
    file: Option<PathBuf>,
    /// Register a Parquet file or directory as a temp view; repeatable.
    #[arg(long = "table", value_name = "NAME=PATH", value_parser = key_value)]
    table: Vec<(String, String)>,
    /// Session configuration entry; repeatable.
    #[arg(long = "conf", value_name = "KEY=VALUE", value_parser = key_value)]
    conf: Vec<(String, String)>,
    /// Archive container.
    #[arg(long, value_enum, default_value = "tar")]
    format: BundleFormat,
    /// Output archive path.
    #[arg(short = 'o', long = "output", value_name = "OUT")]
    output: PathBuf,
}

/// Runs the command: 0 when the bundle was written (even when individual
/// statements failed and were recorded in it), 1 on I/O or setup error.
pub(crate) fn run(args: &BundleArgs) -> i32 {
    match run_inner(args) {
        Ok(()) => 0,
        Err(msg) => {
            let _ = writeln!(std::io::stderr(), "{msg}");
            1
        }
    }
}

fn run_inner(args: &BundleArgs) -> Result<(), String> {
    let input = read_input(args.execute.as_ref(), args.file.as_ref())?;
    let session = build_session(&args.conf, &args.table)?;
    let table_paths: Vec<String> = args.table.iter().map(|(_, p)| p.clone()).collect();
    let statements = split_statements(&input);
    let mut groups = table_spellings(&table_paths);
    let inline: Vec<String> = direct_parquet_paths(
        &statements
            .iter()
            .map(|(_, s)| s.clone())
            .collect::<Vec<_>>(),
    );
    for p in inline {
        if table_paths.iter().any(|t| t == &p) {
            continue;
        }
        if groups.iter().any(|g| g.first() == Some(&p)) {
            continue;
        }
        groups.push(absolute_spellings(&p));
    }
    let subs = PathSub::for_table_spellings(&groups);

    let mut archive = BundleArchive::new();
    let mut names: Vec<String> = Vec::new();

    for (index, (_, stmt)) in statements.iter().enumerate() {
        let tag = format!("{:03}", index + 1);
        let (plan_text, profile_text) = run_statement(&session, stmt, &subs);
        let sql_redacted = Redacted::new_query(krishiv_sql::redact_sql(stmt).as_str(), &subs);
        // The SQL text itself may hold literals: mask them too.
        let sql_redacted = Redacted::new_query(sql_redacted.as_str(), &[]);
        archive.add(&format!("queries/{tag}.sql"), sql_redacted)?;
        archive.add(&format!("queries/{tag}.plan.json"), plan_text)?;
        archive.add(&format!("queries/{tag}.profile.json"), profile_text)?;
        names.push(format!("queries/{tag}.sql"));
        names.push(format!("queries/{tag}.plan.json"));
        names.push(format!("queries/{tag}.profile.json"));
    }

    let config_json = config_json(&session, &args.conf, &subs);
    archive.add("config.json", Redacted::vetted(&config_json, &subs))?;
    let environment_json = environment_json(&session);
    archive.add(
        "environment.json",
        Redacted::vetted(&environment_json, &subs),
    )?;
    let version = version_text();
    archive.add("version.txt", Redacted::vetted(&version, &subs))?;

    names.push("config.json".to_owned());
    names.push("environment.json".to_owned());
    names.push("version.txt".to_owned());
    names.push("manifest.json".to_owned());
    names.sort();
    let manifest = manifest_json(&names);
    archive.add("manifest.json", Redacted::vetted(&manifest, &[]))?;

    let bytes = match args.format {
        BundleFormat::Tar => archive.to_tar()?,
        BundleFormat::Zip => archive.to_zip()?,
    };
    write_atomically(&args.output, &bytes)
}

/// Every spelling of each `--table` path: as given plus the absolute form
/// plans print, so neither the path nor the working directory it reveals
/// survives.
fn table_spellings(paths: &[String]) -> Vec<Vec<String>> {
    paths.iter().map(|p| absolute_spellings(p)).collect()
}

/// Every spelling of one path: as given, the absolute form plans print
/// (normalized against the canonical working directory, so a symlinked cwd
/// such as macOS `/tmp` -> `/private/tmp` and `./`/`../` segments cannot leak
/// the real location), plus the fully canonicalized form when the target
/// exists.
fn absolute_spellings(p: &str) -> Vec<String> {
    let mut out = vec![p.to_owned()];
    let path = Path::new(p);
    let joined = if path.is_relative() {
        let Some(cwd) = std::env::current_dir().ok() else {
            return out;
        };
        let base = std::fs::canonicalize(&cwd).unwrap_or(cwd);
        base.join(path)
    } else {
        path.to_path_buf()
    };
    if let Some(s) = lexical_normalize(&joined).to_str()
        && !out.contains(&s.to_owned())
    {
        out.push(s.to_owned());
    }
    if let Ok(canonical) = std::fs::canonicalize(&joined)
        && let Some(s) = canonical.to_str()
        && !out.contains(&s.to_owned())
    {
        out.push(s.to_owned());
    }
    out
}

/// `path` with `.` dropped and `..` resolved lexically (no filesystem
/// access), so spellings compare equal regardless of redundant segments.
fn lexical_normalize(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            c => out.push(c.as_os_str()),
        }
    }
    out
}

/// Paths referenced inline as `parquet.'<path>'` (any quote style, any case)
/// in the SQL text. These bypass `--table`, but plans still print them, so
/// each gets its own `<path-N>` placeholder.
fn direct_parquet_paths(statements: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for stmt in statements {
        for p in parquet_paths_in(stmt) {
            if !out.contains(&p) {
                out.push(p);
            }
        }
    }
    out
}

/// Quoted paths after `parquet.` in one statement.
fn parquet_paths_in(stmt: &str) -> Vec<String> {
    let bytes = stmt.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let is_parquet = stmt
            .get(i..i + 7)
            .is_some_and(|s| s.eq_ignore_ascii_case("parquet"));
        if is_parquet {
            let mut j = i + 7;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if bytes.get(j) == Some(&b'.') {
                j += 1;
                while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                if let Some(&q) = bytes.get(j)
                    && (q == b'\'' || q == b'"' || q == b'`')
                {
                    let (path, next) = read_quoted(stmt, j, q);
                    if !path.is_empty() && !out.contains(&path) {
                        out.push(path);
                    }
                    i = next;
                    continue;
                }
            }
        }
        let ch = stmt[i..].chars().next().unwrap_or('\u{FFFD}');
        i += ch.len_utf8();
    }
    out
}

/// The quoted string opening at byte `open` (quote `q`): its contents and the
/// byte index just past the closing quote. Handles `` `` `` inside backticks,
/// `''` inside single quotes and `\` escapes inside single/double quotes;
/// unclosed quotes yield no path.
fn read_quoted(stmt: &str, open: usize, q: u8) -> (String, usize) {
    let bytes = stmt.as_bytes();
    let mut buf: Vec<u8> = Vec::new();
    let mut j = open + 1;
    while j < bytes.len() {
        let b = bytes[j];
        if q == b'`' {
            if b == b'`' {
                if bytes.get(j + 1) == Some(&b'`') {
                    buf.push(b'`');
                    j += 2;
                    continue;
                }
                return (String::from_utf8(buf).unwrap_or_default(), j + 1);
            }
            buf.push(b);
            j += 1;
        } else {
            if b == b'\\' {
                if let Some(&n) = bytes.get(j + 1) {
                    buf.push(n);
                    j += 2;
                } else {
                    j += 1;
                }
                continue;
            }
            if b == q {
                if q == b'\'' && bytes.get(j + 1) == Some(&b'\'') {
                    buf.push(b'\'');
                    j += 2;
                    continue;
                }
                return (String::from_utf8(buf).unwrap_or_default(), j + 1);
            }
            buf.push(b);
            j += 1;
        }
    }
    (String::new(), bytes.len())
}

/// Runs one statement, returning its redacted plan and profile documents.
/// A failure records its error class and redacted message instead.
fn run_statement(session: &krishiv::Session, stmt: &str, subs: &[PathSub]) -> (Redacted, Redacted) {
    let df = futures::executor::block_on(session.sql(stmt));
    let df = match df {
        Ok(df) => df,
        Err(e) => {
            let doc = error_json(&e, subs);
            return (Redacted::vetted(&doc, subs), Redacted::vetted(&doc, subs));
        }
    };
    let plan = futures::executor::block_on(df.explain_verbose());
    let plan_text = match plan {
        Ok(v) => scrub_json_value(&v.to_json(), subs),
        Err(e) => error_json(&e, subs),
    };
    let profile = futures::executor::block_on(df.explain_analyze());
    let profile_text = match profile {
        Ok(p) => scrub_json_value(&p.to_json(), subs),
        Err(e) => error_json(&e, subs),
    };
    (
        Redacted::vetted(&plan_text, subs),
        Redacted::vetted(&profile_text, subs),
    )
}

/// `{"error_class": ..., "message": ...}` with the message redacted.
fn error_json(error: &krishiv::Error, subs: &[PathSub]) -> String {
    let class = error.error_class().name();
    let message = Redacted::new_query(&error.to_string(), subs);
    let mut out = String::from("{\"error_class\":");
    push_json_string(&mut out, class);
    out.push_str(",\"message\":");
    push_json_string(&mut out, message.as_str());
    out.push('}');
    out
}

/// Every effective config entry as JSON: registered entries with a value
/// (set or default) plus unregistered `--conf` keys. Secret-classified keys
/// and URL values with credentials show a fixed marker.
fn config_json(
    session: &krishiv::Session,
    conf_args: &[(String, String)],
    subs: &[PathSub],
) -> String {
    let mut entries: BTreeMap<String, String> = BTreeMap::new();
    for entry in krishiv_common::config::REGISTRY {
        if let Some(value) = session.conf().get(entry.key) {
            entries.insert(
                entry.key.to_owned(),
                scrub_config_value(entry.key, &value, subs),
            );
        }
    }
    for (k, v) in conf_args {
        if krishiv_common::config::lookup(k).is_none() && !entries.contains_key(k) {
            entries.insert(k.clone(), scrub_config_value(k, v, subs));
        }
    }
    let mut out = String::from("{");
    for (i, (k, v)) in entries.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_json_string(&mut out, k);
        out.push(':');
        push_json_string(&mut out, v);
    }
    out.push('}');
    out
}

fn scrub_config_value(key: &str, value: &str, subs: &[PathSub]) -> String {
    if krishiv_common::secret::is_sensitive_key(key) {
        return krishiv_common::secret::REDACTED.to_owned();
    }
    if krishiv_common::secret::SecretUrl::parse(value).is_some() {
        return krishiv_common::secret::REDACTED.to_owned();
    }
    // Paths in values (e.g. spill dirs) must not leak real locations.
    Redacted::new(value, subs).as_str().to_owned()
}

/// Environment facts only: no hostname, user, home, env or working directory.
fn environment_json(session: &krishiv::Session) -> String {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get() as u64)
        .unwrap_or(1);
    let features = enabled_features();
    let memory = session
        .conf()
        .get("krishiv.memory.limit")
        .unwrap_or_else(|| "unset".to_owned());
    let mut out = String::from("{");
    push_json_string(&mut out, "os");
    out.push(':');
    push_json_string(&mut out, std::env::consts::OS);
    out.push_str(",\"arch\":");
    push_json_string(&mut out, std::env::consts::ARCH);
    out.push_str(",\"cpu_level\":");
    push_json_string(&mut out, cpu_level());
    out.push_str(",\"logical_cpus\":");
    out.push_str(&logical.to_string());
    out.push_str(",\"rust_target\":");
    push_json_string(&mut out, &rust_target());
    out.push_str(",\"features\":[");
    for (i, f) in features.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_json_string(&mut out, f);
    }
    out.push_str("],\"allocator\":");
    push_json_string(&mut out, allocator_name());
    out.push_str(",\"memory_limit\":");
    push_json_string(&mut out, &memory);
    out.push('}');
    out
}

fn cpu_level() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        "x86-64-v3"
    }
    #[cfg(all(target_arch = "aarch64", not(target_arch = "x86_64")))]
    {
        "neoverse-n1"
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "unknown"
    }
}

fn allocator_name() -> &'static str {
    #[cfg(feature = "mimalloc")]
    {
        "mimalloc"
    }
    #[cfg(not(feature = "mimalloc"))]
    {
        "system"
    }
}

fn rust_target() -> String {
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}

fn enabled_features() -> Vec<&'static str> {
    let mut out = Vec::new();
    if cfg!(feature = "mimalloc") {
        out.push("mimalloc");
    }
    if cfg!(feature = "ui") {
        out.push("ui");
    }
    if cfg!(feature = "kafka") {
        out.push("kafka");
    }
    if cfg!(feature = "iceberg") {
        out.push("iceberg");
    }
    if cfg!(feature = "delta") {
        out.push("delta");
    }
    if cfg!(feature = "cdc") {
        out.push("cdc");
    }
    if cfg!(feature = "connect") {
        out.push("connect");
    }
    if cfg!(feature = "flight-sql") {
        out.push("flight-sql");
    }
    if cfg!(feature = "mcp") {
        out.push("mcp");
    }
    if cfg!(feature = "python") {
        out.push("python");
    }
    if cfg!(feature = "wasm-udf") {
        out.push("wasm-udf");
    }
    if cfg!(feature = "celeborn") {
        out.push("celeborn");
    }
    if cfg!(feature = "full") {
        out.push("full");
    }
    out.sort_unstable();
    out
}

/// The `--version` output.
fn version_text() -> String {
    format!(
        "krishiv {}\nallocator={}\n",
        env!("CARGO_PKG_VERSION"),
        allocator_name()
    )
}

fn manifest_json(files: &[String]) -> String {
    let mut out = String::from("{\"format\":");
    push_json_string(&mut out, BUNDLE_FORMAT);
    out.push_str(",\"version\":");
    out.push_str(&BUNDLE_VERSION.to_string());
    out.push_str(",\"files\":[");
    for (i, f) in files.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_json_string(&mut out, f);
    }
    out.push_str("]}");
    out
}

/// Redacts every string in a JSON document (plan/profile). Wall-clock
/// fields (`*_ns`, `yields`, `stalls`) are zeroed so two runs over the same
/// input on the same machine are byte-identical; counts are kept.
fn scrub_json_value(text: &str, subs: &[PathSub]) -> String {
    let mut value: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => {
            // Not JSON (should not happen): mask as query text.
            return Redacted::new_query(text, subs).as_str().to_owned();
        }
    };
    normalize_numbers(&mut value);
    scrub_value(&mut value, subs);
    serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_owned())
}

/// Zeroes wall-clock measurements in a profile document.
fn normalize_numbers(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(items) => {
            for item in items {
                normalize_numbers(item);
            }
        }
        serde_json::Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                let clock = k.ends_with("_ns") || k == "yields" || k == "stalls";
                if clock && v.is_number() {
                    *v = serde_json::Value::from(0);
                } else {
                    normalize_numbers(v);
                }
            }
        }
        _ => {}
    }
}

fn scrub_value(value: &mut serde_json::Value, subs: &[PathSub]) {
    match value {
        serde_json::Value::String(s) => {
            *s = Redacted::new_query(s, subs).as_str().to_owned();
        }
        serde_json::Value::Array(items) => {
            for item in items {
                scrub_value(item, subs);
            }
        }
        serde_json::Value::Object(map) => {
            for (_, v) in map.iter_mut() {
                scrub_value(v, subs);
            }
        }
        _ => {}
    }
}

fn push_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Writes `bytes` to `path` via a temp file in the same directory, renamed
/// on success and removed on error.
fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
    if let Some(dir) = parent {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let dir: &Path = parent.unwrap_or_else(|| Path::new("."));
    let tmp_name = format!(
        ".{}.tmp.{}",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("bundle"),
        std::process::id()
    );
    let tmp = dir.join(tmp_name);
    let result = (|| -> Result<(), String> {
        std::fs::write(&tmp, bytes).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_json_redacts_paths_and_keeps_class() {
        let subs = PathSub::for_tables(&["/secret/dir".to_owned()]);
        let err: krishiv::Error = krishiv_common::error::TableOrViewNotFound {
            relation_name: "`/secret/dir/missing_table`".to_owned(),
            search_path: "[`a`, `/secret/dir/other`]".to_owned(),
        }
        .into();
        let doc = error_json(&err, &subs);
        assert!(doc.contains("TABLE_OR_VIEW_NOT_FOUND"), "{doc}");
        assert!(doc.contains("error_class"), "{doc}");
        assert!(!doc.contains("/secret/dir"), "{doc}");
        assert!(!doc.contains("secret"), "{doc}");
        assert!(doc.contains("<path-1>"), "{doc}");
        // A secret-bearing DivideByZero parameter is redacted too.
        let err: krishiv::Error = krishiv_common::error::DivideByZero {
            config: "\"/secret/dir\"".to_owned(),
        }
        .into();
        let doc = error_json(&err, &subs);
        assert!(doc.contains("DIVIDE_BY_ZERO"), "{doc}");
        assert!(!doc.contains("/secret/dir"), "{doc}");
    }

    #[test]
    fn relative_spellings_normalize_dot_segments() {
        assert_eq!(
            lexical_normalize(std::path::Path::new("/a/./b/../c")),
            std::path::PathBuf::from("/a/c")
        );
        assert_eq!(
            lexical_normalize(std::path::Path::new("/a/b/../../c")),
            std::path::PathBuf::from("/c")
        );
        // Absolute spellings never keep `.`/`..` segments, so a plan printing
        // the normalized path still matches the substitution.
        let spellings = absolute_spellings("./sub/../tbl.parquet");
        assert!(spellings.first() == Some(&"./sub/../tbl.parquet".to_owned()));
        assert!(spellings.len() >= 2, "{spellings:?}");
        assert!(
            spellings[1..]
                .iter()
                .all(|s| !s.split('/').any(|seg| seg == "." || seg == "..")),
            "{spellings:?}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn absolute_spellings_follow_symlinks() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().expect("tempdir");
        let real = dir.path().join("real-dir-9f3a");
        std::fs::create_dir(&real).expect("mkdir");
        let file = real.join("tbl-9f3a.parquet");
        std::fs::write(&file, b"parquet").expect("write");
        symlink(&real, dir.path().join("link-dir-9f3a")).expect("symlink");
        // A spelling through the symlink canonicalizes to the real location,
        // so a plan printing the canonical path still matches.
        let via_link = format!(
            "{}/link-dir-9f3a/tbl-9f3a.parquet",
            dir.path().to_str().expect("utf-8")
        );
        let spellings = absolute_spellings(&via_link);
        let canonical = std::fs::canonicalize(&file)
            .expect("canonicalize")
            .to_str()
            .expect("utf-8")
            .to_owned();
        assert!(spellings.contains(&via_link), "{spellings:?}");
        assert!(spellings.contains(&canonical), "{spellings:?}");
    }

    #[test]
    fn inline_parquet_paths_are_found_in_any_quote_or_case() {
        let stmts = vec![
            "select * from parquet.`/a/b.parquet`".to_owned(),
            "SELECT * FROM PARQUET.'/c/d.parquet'".to_owned(),
            "select * from parquet . \"/e/f.parquet\"".to_owned(),
        ];
        let paths = direct_parquet_paths(&stmts);
        assert_eq!(
            paths,
            vec![
                "/a/b.parquet".to_owned(),
                "/c/d.parquet".to_owned(),
                "/e/f.parquet".to_owned(),
            ]
        );
        assert!(direct_parquet_paths(&["select 1".to_owned()]).is_empty());
    }

    #[test]
    fn config_scrubs_secrets_and_urls() {
        let session = build_session(
            &[
                ("krishiv.exec.batchSize".to_owned(), "1024".to_owned()),
                ("my.app.token".to_owned(), "hunter2-value".to_owned()),
                (
                    "my.app.url".to_owned(),
                    "https://u:cred-value@h/x".to_owned(),
                ),
            ],
            &[],
        )
        .expect("session");
        let doc = config_json(
            &session,
            &[
                ("my.app.token".to_owned(), "hunter2-value".to_owned()),
                (
                    "my.app.url".to_owned(),
                    "https://u:cred-value@h/x".to_owned(),
                ),
            ],
            &[],
        );
        assert!(!doc.contains("hunter2-value"), "{doc}");
        assert!(!doc.contains("cred-value"), "{doc}");
        assert!(doc.contains("*********(redacted)"), "{doc}");
        assert!(doc.contains("krishiv.exec.batchSize"), "{doc}");
    }

    #[test]
    fn environment_holds_no_paths_or_users() {
        let session = build_session(&[], &[]).expect("session");
        let doc = environment_json(&session);
        assert!(doc.contains(std::env::consts::OS));
        assert!(doc.contains(std::env::consts::ARCH));
        for forbidden in ["HOME", "USER", "hostname", "/home/", "/root/"] {
            assert!(!doc.contains(forbidden), "{doc}");
        }
    }

    #[test]
    fn profile_numbers_are_deterministic() {
        let doc = scrub_json_value(
            r#"{"query":{"wall_time_ns":123,"workers":1,"stalls":7,"yields":9},"x":{"rows_in":5}}"#,
            &[],
        );
        assert!(doc.contains("\"wall_time_ns\":0"), "{doc}");
        assert!(doc.contains("\"stalls\":0"), "{doc}");
        assert!(doc.contains("\"yields\":0"), "{doc}");
        assert!(doc.contains("\"rows_in\":5"), "{doc}");
        assert!(doc.contains("\"workers\":1"), "{doc}");
        let fallback = scrub_json_value("not json 'secret'", &[]);
        assert!(!fallback.contains("secret"), "{fallback}");
    }
}
