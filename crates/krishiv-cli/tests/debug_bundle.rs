// SPDX-License-Identifier: Apache-2.0
//! `krishiv debug-bundle` end-to-end tests.
#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Output, Stdio};

fn krishiv(args: &[&str], stdin: Option<&str>) -> Output {
    krishiv_in(None, args, stdin)
}

fn krishiv_in(cwd: Option<&std::path::Path>, args: &[&str], stdin: Option<&str>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_krishiv"));
    child
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(dir) = cwd {
        child.current_dir(dir);
    }
    let mut child = child.spawn().expect("spawn krishiv");
    let mut pipe = child.stdin.take().expect("stdin");
    if let Some(s) = stdin {
        pipe.write_all(s.as_bytes()).expect("write stdin");
    }
    drop(pipe);
    child.wait_with_output().expect("wait")
}

fn err(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).expect("utf-8")
}

/// Minimal ustar reader (mirrors `src/archive.rs`).
fn read_tar(bytes: &[u8]) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut at = 0;
    while at + 512 <= bytes.len() {
        let h = &bytes[at..at + 512];
        at += 512;
        if h.iter().all(|&b| b == 0) {
            break;
        }
        let end = h[..100].iter().position(|&b| b == 0).unwrap_or(100);
        let name = std::str::from_utf8(&h[..end]).expect("name").to_owned();
        let size_raw = &h[124..136];
        let end = size_raw
            .iter()
            .position(|&b| b == 0 || b == b' ')
            .unwrap_or(12);
        let size = usize::try_from(
            u64::from_str_radix(
                std::str::from_utf8(&size_raw[..end]).expect("octal").trim(),
                8,
            )
            .expect("size"),
        )
        .expect("size");
        out.insert(name, bytes[at..at + size].to_vec());
        at += size.next_multiple_of(512);
    }
    out
}

/// Minimal stored-zip reader (local headers only).
fn read_zip(bytes: &[u8]) -> BTreeMap<String, Vec<u8>> {
    fn u16f(b: &[u8], at: usize) -> usize {
        u16::from_le_bytes([b[at], b[at + 1]]) as usize
    }
    let mut out = BTreeMap::new();
    let mut at = 0;
    while at + 30 <= bytes.len() {
        let sig = u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        if sig == 0x0605_4b50 || sig == 0x0201_4b50 {
            break;
        }
        assert_eq!(sig, 0x0403_4b50, "bad zip header at {at}");
        let comp = u32::from_le_bytes([
            bytes[at + 18],
            bytes[at + 19],
            bytes[at + 20],
            bytes[at + 21],
        ]) as usize;
        let name_len = u16f(bytes, at + 26);
        let extra_len = u16f(bytes, at + 28);
        let head = at + 30;
        let name = std::str::from_utf8(&bytes[head..head + name_len])
            .expect("zip name")
            .to_owned();
        let start = head + name_len + extra_len;
        out.insert(name, bytes[start..start + comp].to_vec());
        at = start + comp;
    }
    out
}

fn read_bundle(path: &std::path::Path) -> BTreeMap<String, Vec<u8>> {
    let bytes = std::fs::read(path).expect("read bundle");
    if path.extension().and_then(|s| s.to_str()) == Some("zip") {
        read_zip(&bytes)
    } else {
        read_tar(&bytes)
    }
}

const LINEITEM: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/tpch/q1-lineitem.parquet"
);

/// Stage a parquet file under a recognizable directory name.
fn staged_table(dir: &std::path::Path) -> (String, String) {
    let secret_dir = dir.join("recognizable-7f3a9c-dir");
    std::fs::create_dir_all(&secret_dir).expect("mkdir");
    let dest = secret_dir.join("sensitive-7f3a9c-data.parquet");
    std::fs::copy(LINEITEM, &dest).expect("copy parquet");
    ("t".to_owned(), dest.to_str().expect("utf-8").to_owned())
}

#[test]
fn redaction_hides_literals_secrets_and_paths() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (name, path) = staged_table(dir.path());
    let secret_value = "hunter2-secret-value-9f3a";
    let url_cred = "cred-pass-9f3a";
    let query = "select l_returnflag, count(*) as n from t \
        where l_returnflag = 'alice-secret-literal-9f3a' and l_quantity = 987654321 \
        and l_linestatus in ('AA110011', 'BB220022', 'CC330033') group by l_returnflag";
    let direct = format!("select count(*) as n from parquet.`{path}`");
    let failing = "select * from missing_table_xyz_9f3a";
    let out = dir.path().join("bundle.tar");
    let o = krishiv(
        &[
            "debug-bundle",
            "-e",
            &format!("{query}; {direct}; {failing}; select 1 / 0"),
            "--table",
            &format!("{name}={path}"),
            "--conf",
            &format!("my.app.token={secret_value}"),
            "--conf",
            &format!("my.app.url=https://user:{url_cred}@host-9f3a/x"),
            "--conf",
            "my.app.password=hunter2-secret-value-9f3a",
            "--format",
            "tar",
            "-o",
            out.to_str().expect("utf-8"),
        ],
        None,
    );
    assert!(o.status.success(), "{}", err(&o));
    let files = read_bundle(&out);
    let all: Vec<u8> = files.values().flatten().copied().collect();
    let all_text = String::from_utf8_lossy(&all);
    for (n, secret) in [
        "alice-secret-literal-9f3a",
        "987654321",
        "AA110011",
        "BB220022",
        "CC330033",
        "recognizable-7f3a9c-dir",
        "sensitive-7f3a9c-data",
        secret_value,
        url_cred,
        "host-9f3a",
    ]
    .into_iter()
    .enumerate()
    {
        // The message names the entry by position: the planted value and
        // the bundle text never go into a failure message or a log.
        assert!(
            !all_text.contains(secret),
            "the bundle leaks planted value #{n}"
        );
    }
    // Placeholders and the fixed marker survive.
    let sql2 = std::str::from_utf8(&files["queries/002.sql"]).expect("utf-8");
    let plan2 = std::str::from_utf8(&files["queries/002.plan.json"]).expect("utf-8");
    assert!(sql2.contains("<path-1>"), "{sql2}");
    assert!(plan2.contains("<path-1>"), "{plan2}");
    assert!(
        all_text.contains("*********(redacted)"),
        "no redaction marker"
    );
    // Failing statements record their error class, redacted.
    let plan3 = std::str::from_utf8(&files["queries/003.plan.json"]).expect("utf-8");
    assert!(plan3.contains("TABLE_OR_VIEW_NOT_FOUND"), "{plan3}");
    assert!(plan3.contains("error_class"), "{plan3}");
    let profile4 = std::str::from_utf8(&files["queries/004.profile.json"]).expect("utf-8");
    assert!(
        profile4.contains("DIVIDE_BY_ZERO") || profile4.contains("error_class"),
        "{profile4}"
    );
    // Manifest lists every file, sorted; version matches --version.
    let manifest = std::str::from_utf8(&files["manifest.json"]).expect("utf-8");
    let mut listed: Vec<String> = files.keys().cloned().collect();
    listed.sort();
    for f in &listed {
        assert!(manifest.contains(f), "{manifest}");
    }
    let version = std::str::from_utf8(&files["version.txt"]).expect("utf-8");
    assert!(version.starts_with("krishiv "), "{version}");
    assert!(version.contains("allocator="), "{version}");
    // environment.json has facts but no secrets or home paths.
    let env = std::str::from_utf8(&files["environment.json"]).expect("utf-8");
    assert!(env.contains(std::env::consts::OS), "{env}");
    assert!(!env.contains("HOME"), "{env}");
}

#[test]
fn direct_parquet_path_without_table_is_redacted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (_name, path) = staged_table(dir.path());
    let query = format!("select count(*) as n from parquet.`{path}`");
    let out = dir.path().join("direct.tar");
    let o = krishiv(
        &[
            "debug-bundle",
            "-e",
            &query,
            "-o",
            out.to_str().expect("utf-8"),
        ],
        None,
    );
    assert!(o.status.success(), "{}", err(&o));
    let files = read_bundle(&out);
    let all: Vec<u8> = files.values().flatten().copied().collect();
    let all_text = String::from_utf8_lossy(&all);
    for (n, secret) in ["recognizable-7f3a9c-dir", "sensitive-7f3a9c-data"]
        .into_iter()
        .enumerate()
    {
        // The message names the entry by position: the planted value and
        // the bundle text never go into a failure message or a log.
        assert!(
            !all_text.contains(secret),
            "the bundle leaks planted value #{n}"
        );
    }
    let plan = std::str::from_utf8(&files["queries/001.plan.json"]).expect("utf-8");
    assert!(plan.contains("<path-1>"), "{plan}");
}

#[test]
fn relative_table_spelling_is_redacted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (name, _abs) = staged_table(dir.path());
    let work = dir.path().join("work-subdir-9f3a");
    std::fs::create_dir(&work).expect("mkdir");
    // Relative spellings with `./`/`../` segments, resolved against the
    // process working directory (which may textually differ from its
    // canonical form when symlinked). The second statement scans the file
    // directly, so plans print the absolute path.
    let rel = "./../recognizable-7f3a9c-dir/./sensitive-7f3a9c-data.parquet";
    let out = dir.path().join("rel.tar");
    let o = krishiv_in(
        Some(&work),
        &[
            "debug-bundle",
            "-e",
            &format!("select count(*) as n from t; select count(*) as n from parquet.`{rel}`"),
            "--table",
            &format!("{name}={rel}"),
            "-o",
            out.to_str().expect("utf-8"),
        ],
        None,
    );
    assert!(o.status.success(), "{}", err(&o));
    let files = read_bundle(&out);
    let all: Vec<u8> = files.values().flatten().copied().collect();
    let all_text = String::from_utf8_lossy(&all);
    for (n, secret) in ["recognizable-7f3a9c-dir", "sensitive-7f3a9c-data"]
        .into_iter()
        .enumerate()
    {
        // The message names the entry by position: the planted value and
        // the bundle text never go into a failure message or a log.
        assert!(
            !all_text.contains(secret),
            "the bundle leaks planted value #{n}"
        );
    }
    let plan = std::str::from_utf8(&files["queries/002.plan.json"]).expect("utf-8");
    assert!(
        plan.contains("<path-"),
        "direct-scan plan must hide the path: {plan}"
    );
}

#[test]
fn zip_and_tar_holds_same_files_and_runs_are_deterministic() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (name, path) = staged_table(dir.path());
    let mk = |format: &str, out: &std::path::Path| {
        krishiv(
            &[
                "debug-bundle",
                "-e",
                "select 1 as a",
                "--table",
                &format!("{name}={path}"),
                "--format",
                format,
                "-o",
                out.to_str().expect("utf-8"),
            ],
            None,
        )
    };
    let tar1 = dir.path().join("a.tar");
    let tar2 = dir.path().join("b.tar");
    let zip1 = dir.path().join("a.zip");
    assert!(mk("tar", &tar1).status.success());
    assert!(mk("tar", &tar2).status.success());
    assert!(mk("zip", &zip1).status.success());
    assert_eq!(
        std::fs::read(&tar1).expect("read"),
        std::fs::read(&tar2).expect("read")
    );
    let tar_files = read_bundle(&tar1);
    let zip_files = read_bundle(&zip1);
    assert_eq!(
        tar_files.keys().collect::<Vec<_>>(),
        zip_files.keys().collect::<Vec<_>>()
    );
    for (k, v) in &tar_files {
        assert_eq!(v, &zip_files[k], "{k}");
    }
}

#[test]
fn empty_input_writes_only_metadata_and_bad_setup_fails_without_temp() {
    let dir = tempfile::tempdir().expect("tempdir");
    // Empty input: no queries, still a valid bundle.
    let out = dir.path().join("empty.tar");
    let o = krishiv(
        &[
            "debug-bundle",
            "-e",
            "  ;; ",
            "-o",
            out.to_str().expect("utf-8"),
        ],
        None,
    );
    assert!(o.status.success(), "{}", err(&o));
    let files = read_bundle(&out);
    assert!(
        files.keys().all(|k| !k.starts_with("queries/")),
        "{files:?}"
    );
    assert!(files.contains_key("manifest.json"));
    // A setup failure (bad table path is lazy, so use a bad conf value)
    // leaves no output and no temp file behind.
    let bad = dir.path().join("bad.tar");
    let o = krishiv(
        &[
            "debug-bundle",
            "--conf",
            "krishiv.exec.batchSize=lots",
            "-e",
            "select 1",
            "-o",
            bad.to_str().expect("utf-8"),
        ],
        None,
    );
    assert!(!o.status.success());
    assert!(!bad.exists());
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .expect("readdir")
        .filter_map(|e| e.ok().map(|e| e.file_name()))
        .filter(|n| n.to_string_lossy().contains(".tmp."))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");

    // Writing onto an existing directory fails and cleans its temp file.
    let existing = dir.path().join("is-a-dir");
    std::fs::create_dir(&existing).expect("mkdir");
    let o = krishiv(
        &[
            "debug-bundle",
            "-e",
            "select 1",
            "-o",
            existing.to_str().expect("utf-8"),
        ],
        None,
    );
    assert!(!o.status.success(), "{}", err(&o));
    assert!(existing.is_dir());
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .expect("readdir")
        .filter_map(|e| e.ok().map(|e| e.file_name()))
        .filter(|n| n.to_string_lossy().contains(".tmp."))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

/// Reads the bundle back with the system tools when they exist (ignored by
/// default: needs `tar`/`unzip` on `PATH`).
#[test]
#[ignore]
fn system_tools_read_both_formats_when_present() {
    let dir = tempfile::tempdir().expect("tempdir");
    // tar path
    let out = dir.path().join("sys2.tar");
    let o = krishiv(
        &[
            "debug-bundle",
            "-e",
            "select 1 as a",
            "-o",
            out.to_str().expect("utf-8"),
        ],
        None,
    );
    assert!(o.status.success(), "{}", err(&o));
    match Command::new("tar").arg("-tf").arg(&out).output() {
        Ok(listing) if listing.status.success() => {
            let text = String::from_utf8_lossy(&listing.stdout);
            assert!(text.contains("manifest.json"), "{text}");
        }
        _ => eprintln!("tar not available: skipping"),
    }
    let out = dir.path().join("sys2.zip");
    let o = krishiv(
        &[
            "debug-bundle",
            "-e",
            "select 1 as a",
            "--format",
            "zip",
            "-o",
            out.to_str().expect("utf-8"),
        ],
        None,
    );
    assert!(o.status.success(), "{}", err(&o));
    match Command::new("unzip").arg("-l").arg(&out).output() {
        Ok(listing) if listing.status.success() => {
            let text = String::from_utf8_lossy(&listing.stdout);
            assert!(text.contains("manifest.json"), "{text}");
        }
        _ => eprintln!("unzip not available: skipping"),
    }
}
