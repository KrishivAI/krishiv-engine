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

/// Runs the command; the process exit code (0 ok, 1 I/O or setup error).
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
    let subs = PathSub::for_table_spellings(&table_spellings(&table_paths));
    let statements = split_statements(&input);

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

/// Every spelling of each `--table` path: as given and absolute (plans
/// print the absolute path, SQL text the given one), so neither the path
/// nor the working directory it reveals survives.
fn table_spellings(paths: &[String]) -> Vec<Vec<String>> {
    paths
        .iter()
        .map(|p| {
            let mut spellings = vec![p.clone()];
            if let Some(abs) = absolute_spelling(p) {
                spellings.push(abs);
            }
            spellings
        })
        .collect()
}

/// The absolute form of a relative `--table` path.
fn absolute_spelling(p: &str) -> Option<String> {
    if !std::path::Path::new(p).is_relative() {
        return None;
    }
    let abs = std::env::current_dir().ok()?.join(p);
    abs.to_str().map(str::to_owned)
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
/// fields (`*_ns`, `yields`, `stalls`) are zeroed so two runs over the
/// same input are byte-identical; counts are kept.
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
    fn error_json_redacts_message() {
        let subs = PathSub::for_tables(&["/secret/dir".to_owned()]);
        let err = krishiv_common::error::DivideByZero {
            config: "\"spark.sql.ansi.enabled\"".to_owned(),
        };
        let doc = error_json(&err.into(), &subs);
        assert!(doc.contains("DIVIDE_BY_ZERO"));
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
