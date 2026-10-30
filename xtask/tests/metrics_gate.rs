// SPDX-License-Identifier: Apache-2.0
//! Tests of `cargo xtask metrics-gate`: the default embedded build pulls
//! no telemetry SDK or encoder crate.

use std::path::Path;

use xtask::metrics_gate::{FORBIDDEN_FRAGMENTS, check_tree, run};
use xtask::precheck::{Action, Changes, Opts, Pkg, plan};

const CLEAN_TREE: &str = "\
krishiv v0.1.0-alpha (/checkout)
├── krishiv-common v0.1.0-alpha (/checkout/crates/krishiv-common)
│   └── serde v1.0.229
├── krishiv-session v0.1.0-alpha (/checkout/crates/krishiv-session)
│   └── krishiv-exec v0.1.0-alpha (/checkout/crates/krishiv-exec)
└── krishiv-types v0.1.0-alpha (/checkout/crates/krishiv-types)
";

const DIRTY_TREE: &str = "\
krishiv v0.1.0-alpha (/checkout)
├── krishiv-server v0.1.0-alpha (/checkout/crates/krishiv-server)
│   ├── opentelemetry v0.33.1
│   └── metrics-exporter-prometheus v0.18.3
└── krishiv-session v0.1.0-alpha (/checkout/crates/krishiv-session)
";

#[test]
fn clean_tree_passes() {
    assert_eq!(check_tree(CLEAN_TREE), Ok(()));
    assert_eq!(check_tree(""), Ok(()));
    assert!(FORBIDDEN_FRAGMENTS.contains(&"opentelemetry"));
    assert!(FORBIDDEN_FRAGMENTS.contains(&"prometheus"));
}

#[test]
fn a_checkout_path_naming_a_forbidden_fragment_is_not_a_dependency() {
    let tree = "krishiv v0.1.0-alpha (/home/me/prometheus-notes/krishiv)\n\
                └── serde v1.0.229\n";
    assert_eq!(check_tree(tree), Ok(()));
}

#[test]
fn forbidden_crates_fail_and_are_named() {
    let Err(message) = check_tree(DIRTY_TREE) else {
        panic!("dirty tree passed");
    };
    assert!(message.contains("opentelemetry v0.33.1"), "{message}");
    assert!(
        message.contains("metrics-exporter-prometheus v0.18.3"),
        "{message}"
    );
    assert!(!message.contains("krishiv-session"), "{message}");
}

#[test]
fn default_build_has_no_telemetry_deps() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    if let Err(message) = run(root) {
        panic!("{message}");
    }
}

#[test]
fn precheck_policy_runs_the_gate() {
    let changes = Changes::detect(
        &["xtask/src/metrics_gate.rs".to_owned()],
        &[Pkg {
            name: "xtask".to_owned(),
            dir: "xtask".to_owned(),
            deps: Vec::new(),
        }],
    );
    let opts = Opts::parse(&[]).unwrap();
    let steps = plan(&changes, &opts, &|_| true, "xtask");
    let policy = steps
        .iter()
        .find(|step| step.id == "policy")
        .expect("precheck has a policy step");
    let Action::Run(cmds) = &policy.action else {
        panic!("policy step should run");
    };
    assert!(
        cmds.iter().any(|cmd| cmd.args == ["metrics-gate"]),
        "policy runs: {:?}",
        cmds.iter().map(|cmd| &cmd.args).collect::<Vec<_>>()
    );
}
