// SPDX-License-Identifier: Apache-2.0
//! Tests for the workflow, SPDX and banned-identifier lints.

use xtask::lint::{
    banned_idents_in, confined_idents_in, has_spdx_header, unsafe_allowed_in, unsafe_lines,
    unsafe_sites_per_crate, workflow_violations,
};

#[test]
fn pull_request_target_fails_the_lint() {
    let wf = "name: x\non:\n  pull_request_target:\n    types: [opened]\njobs: {}\n";
    let v = workflow_violations("bad.yml", wf);
    assert_eq!(v.len(), 1);
    assert!(v[0].contains("pull_request_target"));
}

#[test]
fn commented_mentions_are_ignored() {
    let wf = "# never use pull_request_target\non:\n  pull_request:\njobs:\n  a:\n    runs-on: ubuntu-latest\n";
    assert!(workflow_violations("ok.yml", wf).is_empty());
}

#[test]
fn any_self_hosted_label_fails() {
    for runs_on in [
        "[self-hosted, krishiv-heavy]",
        "[self-hosted]",
        "krishiv-heavy",
        "self-hosted",
    ] {
        for trig in ["pull_request", "schedule", "workflow_dispatch"] {
            let wf = format!("on:\n  {trig}:\njobs:\n  a:\n    runs-on: {runs_on}\n");
            let v = workflow_violations("bad.yml", &wf);
            assert_eq!(v.len(), 1, "{trig} {runs_on}: {v:?}");
            assert!(v[0].contains("self-hosted"));
        }
    }
    let multiline = "on:\n  schedule:\n    - cron: '0 2 * * *'\njobs:\n  a:\n    runs-on:\n      - self-hosted\n      - linux\n";
    assert_eq!(workflow_violations("bad.yml", multiline).len(), 1);
}

#[test]
fn hosted_workflows_may_use_pull_request() {
    let wf = "on:\n  pull_request:\n  push:\njobs:\n  a:\n    runs-on: ubuntu-latest\n";
    assert!(workflow_violations("ok.yml", wf).is_empty());
}

#[test]
fn privileged_outsider_triggers_are_banned() {
    for trig in ["workflow_run", "issue_comment", "pull_request_target"] {
        let wf = format!("on:\n  {trig}:\njobs:\n  a:\n    runs-on: ubuntu-latest\n");
        let v = workflow_violations("bad.yml", &wf);
        assert_eq!(v.len(), 1, "{trig}: {v:?}");
        assert!(v[0].contains(trig));
    }
    let inline = "on: [push, workflow_run]\njobs:\n  a:\n    runs-on: ubuntu-latest\n";
    assert_eq!(workflow_violations("bad.yml", inline).len(), 1);
}

#[test]
fn only_github_hosted_runner_labels_pass() {
    for ok in [
        "ubuntu-latest",
        "ubuntu-24.04-arm",
        "macos-15",
        "windows-latest",
    ] {
        let wf = format!("on:\n  push:\njobs:\n  a:\n    runs-on: {ok}\n");
        assert!(workflow_violations("ok.yml", &wf).is_empty(), "{ok}");
    }
    for bad in [
        "my-gpu-box",
        "[linux, x64]",
        "${{ matrix.runner }}",
        "\"custom\"",
    ] {
        let wf = format!("on:\n  push:\njobs:\n  a:\n    runs-on: {bad}\n");
        assert_eq!(workflow_violations("bad.yml", &wf).len(), 1, "{bad}");
    }
}

#[test]
fn actions_must_be_pinned_to_a_commit_sha() {
    let job = |uses: &str| {
        format!(
            "on:\n  push:\njobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: {uses}\n"
        )
    };
    for ok in [
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1",
        "github/codeql-action/upload-sarif@6f2ac81f5cd81fe351a197ba0b00d91d03cd2aee",
        "./.github/actions/local",
        "$/.github/workflows/reusable.yml",
        "docker://alpine@sha256:0000000000000000000000000000000000000000000000000000000000000000",
    ] {
        assert!(workflow_violations("ok.yml", &job(ok)).is_empty(), "{ok}");
    }
    for bad in [
        "actions/checkout@v4",
        "dtolnay/rust-toolchain@stable",
        "actions/checkout@3d3c42e",
        "docker://alpine:3.20",
    ] {
        let v = workflow_violations("bad.yml", &job(bad));
        assert_eq!(v.len(), 1, "{bad}: {v:?}");
        assert!(v[0].contains("not pinned"));
    }
}

#[test]
fn hosted_nightly_is_fine() {
    let wf = "on:\n  schedule:\n    - cron: '0 2 * * *'\n  workflow_dispatch:\njobs:\n  a:\n    runs-on: ubuntu-latest\n";
    assert!(workflow_violations("nightly.yml", wf).is_empty());
}

#[test]
fn spdx_header_must_be_near_the_top() {
    assert!(has_spdx_header(
        "// SPDX-License-Identifier: Apache-2.0\n//! doc\n"
    ));
    assert!(has_spdx_header(
        "# SPDX-License-Identifier: Apache-2.0\nimport os\n"
    ));
    // The tag inside a string literal or a non-comment line is not a header.
    assert!(!has_spdx_header(
        "//! doc\npub const SPDX_TAG: &str = \"SPDX-License-Identifier: Apache-2.0\";\n"
    ));
    assert!(!has_spdx_header(
        "let tag = \"# SPDX-License-Identifier: Apache-2.0\";\n"
    ));
    assert!(!has_spdx_header("//! doc\nfn main() {}\n"));
    assert!(!has_spdx_header(
        "a\nb\nc\nd\ne\n// SPDX-License-Identifier: Apache-2.0\n"
    ));
}

#[test]
fn parquet_object_adapters_are_found() {
    let src = "use parquet::arrow::async_reader::ParquetObjectReader;\nfn ok() {}\nlet w: ParquetObjectWriter;\n";
    assert_eq!(
        banned_idents_in(src),
        [(1, "ParquetObjectReader"), (3, "ParquetObjectWriter")]
    );
    assert!(banned_idents_in("fn fine() {}").is_empty());
}

#[test]
fn unsafe_keyword_lines_only() {
    let src = "#![forbid(unsafe_code)]\n#![deny(unsafe_op_in_unsafe_fn)]\n// unsafe in a comment\nlet s = \"unsafe\";\nunsafe { f() }\nfn g() {} // unsafe\npub unsafe fn h() {}\n";
    assert_eq!(unsafe_lines(src), vec![5, 7]);
    assert_eq!(
        unsafe_lines("let s = \"a\\\" unsafe\";"),
        Vec::<usize>::new()
    );
    // A '"' char literal does not hide a later unsafe.
    assert_eq!(unsafe_lines("let q = '\"'; unsafe { f() }"), vec![1]);
    assert_eq!(unsafe_lines("let q = b'\"'; unsafe { f() }"), vec![1]);
    assert_eq!(unsafe_lines("let q = '\\''; unsafe { f() }"), vec![1]);
    // A lifetime is not a char literal.
    assert_eq!(
        unsafe_lines("fn f<'a>(x: &'a str) { unsafe { g() } }"),
        vec![1]
    );
    assert_eq!(unsafe_lines("x('\"abc\" unsafe"), vec![1]);
    // An escaped backslash closes the string (review finding).
    assert_eq!(unsafe_lines("let p = \"\\\\\"; unsafe { f() }"), vec![1]);
    // Raw strings have no escapes; hashes delimit them.
    assert_eq!(unsafe_lines("let p = r\"C:\\\"; unsafe { f() }"), vec![1]);
    assert_eq!(
        unsafe_lines("let p = r#\"a \" unsafe\"#; f()"),
        Vec::<usize>::new()
    );
    assert_eq!(unsafe_lines("let p = br#\"unsafe\"#;\nunsafe {}"), vec![2]);
    // Strings and nested block comments span lines.
    assert_eq!(unsafe_lines("let s = \"a\nunsafe\";\nunsafe {}"), vec![3]);
    assert_eq!(unsafe_lines("/* a /* b */\nunsafe */ unsafe {}"), vec![2]);
    // C strings and raw C strings (review finding: exact example).
    assert_eq!(unsafe_lines("let p = cr\"C:\\\"; unsafe { f() }"), vec![1]);
    assert_eq!(
        unsafe_lines("let p = cr#\"a \" unsafe\"#; unsafe {}"),
        vec![1]
    );
    assert_eq!(
        unsafe_lines("let p = c\"unsafe\"; f()"),
        Vec::<usize>::new()
    );
    assert_eq!(
        unsafe_lines("let p = br##\"\"# unsafe\"##;\nunsafe {}"),
        vec![2]
    );
    // A line continuation inside a string still counts its newline.
    assert_eq!(
        unsafe_lines("let s = \"abc\\\n  def\";\nunsafe {}"),
        vec![3]
    );
    // An unterminated raw string is scanned as code (over-count, never hide).
    assert_eq!(unsafe_lines("r\"x unsafe"), vec![1]);
    assert_eq!(unsafe_lines("unsafe"), vec![1]);
}

#[test]
fn kernel_unsafe_only_in_unsafe_modules() {
    assert!(unsafe_allowed_in(
        "crates/krishiv-kernels/src/unsafe_dispatch.rs"
    ));
    assert!(!unsafe_allowed_in("crates/krishiv-kernels/src/chunk.rs"));
    assert!(!unsafe_allowed_in("crates/krishiv-kernels/src/unsafe/x.rs"));
    assert!(unsafe_allowed_in("crates/krishiv-memory/src/arena.rs"));
}

#[test]
fn unsafe_sites_are_counted_per_crate() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let src = root.join("crates/k/src/inner");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::create_dir_all(root.join("crates/clean/src")).unwrap();
    std::fs::create_dir_all(root.join("crates/no-src")).unwrap();
    std::fs::write(src.join("a.rs"), "unsafe { f() }\nunsafe fn g() {}\n").unwrap();
    std::fs::write(
        root.join("crates/k/src/b.rs"),
        "unsafe impl Send for X {}\n",
    )
    .unwrap();
    std::fs::write(root.join("crates/k/src/c.txt"), "unsafe\n").unwrap();
    std::fs::write(root.join("crates/clean/src/lib.rs"), "fn f() {}\n").unwrap();
    // Symlinks are not followed: no loop, no double count.
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&src, src.join("loop")).unwrap();
        std::os::unix::fs::symlink(src.join("a.rs"), root.join("crates/k/src/d.rs")).unwrap();
    }
    let sites = unsafe_sites_per_crate(&root).unwrap();
    assert_eq!(sites.into_iter().collect::<Vec<_>>(), [("k".to_owned(), 3)]);
    // Unreadable input is an error, not a lower count (fails closed).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let bad = root.join("crates/k/src/bad.rs");
        std::fs::write(&bad, "unsafe {}\n").unwrap();
        std::fs::set_permissions(&bad, std::fs::Permissions::from_mode(0o000)).unwrap();
        // Root can read anything; only check when the permission bites.
        if std::fs::read_to_string(&bad).is_err() {
            assert!(
                unsafe_sites_per_crate(&root)
                    .unwrap_err()
                    .contains("bad.rs")
            );
        }
        std::fs::set_permissions(&bad, std::fs::Permissions::from_mode(0o644)).unwrap();
    }
    assert!(unsafe_sites_per_crate(&root.join("missing")).is_err());
}

/// Fuzz-style: an `unsafe` placed after arbitrary literals and comments is
/// always found, on the right line.
#[test]
fn unsafe_after_random_literals_is_always_found() {
    const PIECES: &[&str] = &[
        "\"plain\"",
        "\"esc \\\" q\"",
        "\"bs \\\\\"",
        "\"cont \\\n x\"",
        "\"multi\nline\"",
        "b\"bytes\"",
        "c\"cstr\"",
        "r\"C:\\\"",
        "r#\"a \" b\"#",
        "r##\"x \"# y\"##",
        "br\"raw\\\"",
        "cr\"C:\\\"",
        "cr#\"q\"#",
        "'\"'",
        "'\\''",
        "'\\\\'",
        "b'\"'",
        "'\\u{1F600}'",
        "<'a>",
        "&'static str",
        "// line comment\n",
        "/* block */",
        "/* nested /* c */ \n */",
        "x",
        "1",
        ";",
        "(",
        ")",
        " ",
        "\n",
    ];
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        usize::try_from(seed % u64::try_from(n).unwrap()).unwrap()
    };
    for _ in 0..5000 {
        let mut src = String::new();
        for _ in 0..next(12) {
            src.push_str(PIECES[next(PIECES.len())]);
            src.push(' ');
        }
        let line = src.matches('\n').count() + 1;
        src.push_str("unsafe { f() }");
        assert!(unsafe_lines(&src).contains(&line), "{src:?} -> line {line}");
    }
}

#[test]
fn redacted_sql_is_built_only_by_the_token_redactor() {
    let call = "let r = RedactedSql::from_marks_unchecked(sql, marks);";
    assert!(confined_idents_in("crates/krishiv-sql/src/redact.rs", call).is_empty());
    assert!(confined_idents_in("crates\\krishiv-sql\\src\\redact.rs", call).is_empty());
    assert_eq!(
        confined_idents_in("crates/krishiv-plan/src/x.rs", call),
        vec![(1, "from_marks_unchecked")]
    );
}
