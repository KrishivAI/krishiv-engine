// SPDX-License-Identifier: Apache-2.0
//! Tests for the workflow, SPDX and banned-identifier lints.

use xtask::lint::{banned_idents_in, has_spdx_header, workflow_violations};

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
fn self_hosted_with_pull_request_trigger_fails() {
    let wf = "on:\n  pull_request:\njobs:\n  a:\n    runs-on: [self-hosted, krishiv-heavy]\n";
    assert_eq!(workflow_violations("bad.yml", wf).len(), 1);
    let inline = "on: [push, pull_request]\njobs:\n  a:\n    runs-on: [self-hosted]\n";
    let v = workflow_violations("bad2.yml", inline);
    // Inline `push` has no branch restriction, so it is refused as well.
    assert_eq!(v.len(), 2, "{v:?}");
    assert!(v.iter().any(|m| m.contains("`pull_request`")));
}

#[test]
fn krishiv_heavy_label_alone_counts_as_self_hosted() {
    let wf = "on:\n  pull_request:\njobs:\n  a:\n    runs-on: krishiv-heavy\n";
    assert_eq!(workflow_violations("bad.yml", wf).len(), 1);
}

#[test]
fn self_hosted_rejects_untrusted_triggers() {
    for trig in [
        "workflow_run",
        "issue_comment",
        "pull_request_review",
        "pull_request_review_comment",
        "pull_request",
    ] {
        let wf = format!("on:\n  {trig}:\njobs:\n  a:\n    runs-on: [self-hosted]\n");
        let v = workflow_violations("bad.yml", &wf);
        assert_eq!(v.len(), 1, "{trig}: {v:?}");
        assert!(v[0].contains(trig));
    }
}

#[test]
fn self_hosted_push_only_to_main() {
    let ok = "on:\n  push:\n    branches: [main]\njobs:\n  a:\n    runs-on: [self-hosted]\n";
    assert!(workflow_violations("ok.yml", ok).is_empty());
    let any = "on:\n  push:\njobs:\n  a:\n    runs-on: [self-hosted]\n";
    assert_eq!(workflow_violations("bad.yml", any).len(), 1);
    let other = "on:\n  push:\n    branches: [dev]\njobs:\n  a:\n    runs-on: [self-hosted]\n";
    assert_eq!(workflow_violations("bad.yml", other).len(), 1);
}

#[test]
fn hosted_workflows_may_use_pull_request() {
    let wf = "on:\n  pull_request:\n  workflow_run:\njobs:\n  a:\n    runs-on: ubuntu-latest\n";
    assert!(workflow_violations("ok.yml", wf).is_empty());
}

#[test]
fn self_hosted_nightly_is_fine() {
    let wf = "on:\n  schedule:\n    - cron: '0 2 * * *'\n  workflow_dispatch:\njobs:\n  a:\n    runs-on: [self-hosted, krishiv-heavy]\n";
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
