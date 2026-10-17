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
    let wf = "on:\n  pull_request:\n  workflow_run:\njobs:\n  a:\n    runs-on: ubuntu-latest\n";
    assert!(workflow_violations("ok.yml", wf).is_empty());
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
