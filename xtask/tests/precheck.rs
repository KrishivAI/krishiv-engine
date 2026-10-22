// SPDX-License-Identifier: Apache-2.0
//! Tests for `cargo xtask precheck`: change detection, scoping, the plan, and the
//! policy test that keeps it in step with the required CI checks.
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test helpers sit outside #[test] fns"
)]

use std::collections::BTreeSet;
use std::path::Path;

use xtask::precheck::{
    Action, Changes, EXTERNAL_CI_CHECKS, Mapping, Opts, Outcome, Pkg, REQUIRED_CI_CHECKS, STEP_IDS,
    Scope, Status, Step, cap_prefix, dependents_closure, failed, mapping, parse_packages, plan,
    render_table,
};

fn pkg(name: &str, deps: &[&str]) -> Pkg {
    Pkg {
        name: name.to_owned(),
        dir: if name == "xtask" {
            "xtask".to_owned()
        } else {
            format!("crates/{name}")
        },
        deps: deps.iter().map(|d| (*d).to_owned()).collect(),
    }
}

fn pkgs() -> Vec<Pkg> {
    vec![
        pkg("common", &[]),
        pkg("exec", &["common"]),
        pkg("cli", &["exec"]),
        pkg("other", &[]),
        pkg("xtask", &["common"]),
    ]
}

fn detect(paths: &[&str]) -> Changes {
    let p: Vec<String> = paths.iter().map(|s| (*s).to_owned()).collect();
    Changes::detect(&p, &pkgs())
}

fn opts(flags: &[&str]) -> Opts {
    Opts::parse(flags).unwrap()
}

fn make_plan(ch: &Changes, o: &Opts, installed: bool) -> Vec<Step> {
    plan(ch, o, &|_| installed, "/x/xtask")
}

fn step<'a>(steps: &'a [Step], id: &str) -> &'a Step {
    steps.iter().find(|s| s.id == id).unwrap()
}

fn skip_reason<'a>(steps: &'a [Step], id: &str) -> Option<&'a str> {
    match &step(steps, id).action {
        Action::Skip(why) => Some(why),
        _ => None,
    }
}

fn args_of(steps: &[Step], id: &str) -> Vec<String> {
    match &step(steps, id).action {
        Action::Run(cmds) => cmds[0].args.clone(),
        other => panic!("{id} is {other:?}"),
    }
}

#[test]
fn a_crate_change_touches_it_and_its_dependents() {
    let ch = detect(&["crates/common/src/lib.rs"]);
    assert_eq!(
        ch.scope,
        Scope::Packages {
            touched: vec!["common".to_owned()],
            with_dependents: ["cli", "common", "exec", "xtask"]
                .map(str::to_owned)
                .to_vec(),
        }
    );
}

#[test]
fn a_leaf_change_has_no_dependents() {
    let ch = detect(&["crates/cli/src/main.rs", "crates/other/Cargo.toml"]);
    let Scope::Packages {
        touched,
        with_dependents,
    } = ch.scope
    else {
        panic!()
    };
    assert_eq!(touched, ["cli", "other"]);
    assert_eq!(with_dependents, ["cli", "other"]);
    assert!(detect(&["crates/other/Cargo.toml"]).manifests);
}

#[test]
fn a_directory_prefix_does_not_match_a_sibling_crate() {
    let ch = detect(&["crates/exec-extra/src/lib.rs"]);
    assert_eq!(ch.scope, Scope::None);
}

#[test]
fn workspace_wide_inputs_scope_to_everything() {
    for p in [
        "Cargo.lock",
        "Cargo.toml",
        "rust-toolchain.toml",
        "clippy.toml",
        ".cargo/config.toml",
        "proto/a.proto",
    ] {
        assert_eq!(detect(&[p]).scope, Scope::Workspace, "{p}");
    }
}

#[test]
fn docs_only_changes_have_no_rust_scope() {
    let ch = detect(&["docs/a.md", "README.md", "  ", ""]);
    assert_eq!(ch.scope, Scope::None);
    assert_eq!(ch.paths, ["docs/a.md", "README.md"]);
    assert!(!ch.manifests && !ch.github && !ch.python && !ch.bench && !ch.scripts);
}

#[test]
fn path_classes_are_detected() {
    assert!(detect(&[".github/workflows/ci.yml"]).github);
    assert!(detect(&["python/krishiv/a.py"]).python);
    let b = detect(&["bench/README.md"]);
    assert!(b.bench && !b.bench_cov);
    for p in [
        "bench/src/krishiv_bench/a.py",
        "bench/tests/t.py",
        "bench/pyproject.toml",
        "bench/uv.lock",
    ] {
        assert!(detect(&[p]).bench_cov, "{p}");
    }
    assert!(detect(&["tests/scripts/gen.py"]).scripts);
}

#[test]
fn dependents_closure_is_transitive_and_stable() {
    let touched: BTreeSet<&str> = ["common"].into();
    assert_eq!(
        dependents_closure(&touched, &pkgs()),
        ["cli", "common", "exec", "xtask"]
    );
}

#[test]
fn packages_parse_from_cargo_metadata() {
    let json = r#"{"packages":[
        {"name":"a","manifest_path":"/w/crates/a/Cargo.toml","dependencies":[{"name":"serde"},{"name":"b"}]},
        {"name":"xtask","manifest_path":"/w/xtask/Cargo.toml","dependencies":[]}]}"#;
    let p = parse_packages(json, Path::new("/w")).unwrap();
    assert_eq!(
        p[0],
        Pkg {
            name: "a".into(),
            dir: "crates/a".into(),
            deps: vec!["serde".into(), "b".into()]
        }
    );
    assert_eq!(p[1].dir, "xtask");
    assert!(
        parse_packages("{", Path::new("/w"))
            .unwrap_err()
            .contains("bad cargo metadata")
    );
    assert!(
        parse_packages("{}", Path::new("/w"))
            .unwrap_err()
            .contains("no packages")
    );
    let bad = r#"{"packages":[{"name":"a"}]}"#;
    assert!(
        parse_packages(bad, Path::new("/w"))
            .unwrap_err()
            .contains("without name")
    );
}

#[test]
fn options_parse_and_validate() {
    let d = opts(&[]);
    assert_eq!(
        (d.base.as_str(), d.fast, d.full, d.memory_cap),
        ("origin/main", false, false, None)
    );
    let o = opts(&[
        "--base",
        "origin/dev",
        "--fast",
        "--full",
        "--memory-cap",
        "12G",
    ]);
    assert_eq!(
        (o.base.as_str(), o.fast, o.full, o.memory_cap.as_deref()),
        ("origin/dev", true, true, Some("12G"))
    );
    assert!(
        Opts::parse(&["--base"])
            .unwrap_err()
            .contains("needs a ref")
    );
    assert!(
        Opts::parse(&["--base", "--fast"])
            .unwrap_err()
            .contains("needs a ref")
    );
    assert!(
        Opts::parse(&["--bogus"])
            .unwrap_err()
            .contains("unknown precheck flag --bogus")
    );
    for bad in ["12", "G", "0G", "12GB", "-1G", "1.5G"] {
        assert!(
            Opts::parse(&["--memory-cap", bad])
                .unwrap_err()
                .contains("expects e.g. 12G"),
            "{bad}"
        );
    }
    assert!(
        Opts::parse(&["--memory-cap"])
            .unwrap_err()
            .contains("needs a size")
    );
}

#[test]
fn the_plan_has_one_row_per_step_id_in_order() {
    for ch in [
        detect(&[]),
        detect(&[
            "crates/exec/src/lib.rs",
            ".github/a.yml",
            "python/x.py",
            "bench/src/a.py",
        ]),
    ] {
        let ids: Vec<&str> = make_plan(&ch, &opts(&[]), true)
            .iter()
            .map(|s| s.id)
            .collect();
        assert_eq!(ids, STEP_IDS);
    }
}

#[test]
fn rust_steps_are_scoped_to_the_touched_crates() {
    let steps = make_plan(&detect(&["crates/common/src/lib.rs"]), &opts(&[]), true);
    let clippy = args_of(&steps, "clippy").join(" ");
    assert!(
        clippy.contains("-p common") && !clippy.contains("-p exec"),
        "{clippy}"
    );
    assert!(
        clippy.ends_with("-- -D warnings") && clippy.contains("--all-targets"),
        "{clippy}"
    );
    let test = args_of(&steps, "nextest").join(" ");
    assert!(
        test.contains("-p common -p exec -p cli -p xtask") || test.contains("-p cli"),
        "{test}"
    );
    let Action::Run(doc) = &step(&steps, "doc").action else {
        panic!()
    };
    assert_eq!(
        doc[0].env,
        [("RUSTDOCFLAGS".to_owned(), "-D warnings".to_owned())]
    );
    assert!(doc[0].args.join(" ").ends_with("-p common"));
}

#[test]
fn a_workspace_wide_change_runs_the_whole_workspace() {
    let steps = make_plan(&detect(&["Cargo.lock"]), &opts(&[]), true);
    assert!(args_of(&steps, "clippy").contains(&"--workspace".to_owned()));
    assert!(args_of(&steps, "nextest").contains(&"--workspace".to_owned()));
}

#[test]
fn no_rust_change_skips_the_rust_checks_with_a_reason() {
    let steps = make_plan(&detect(&["docs/a.md"]), &opts(&[]), true);
    for id in [
        "fmt",
        "clippy",
        "doc",
        "nextest",
        "layering",
        "diff-coverage",
    ] {
        assert_eq!(
            skip_reason(&steps, id),
            Some("no Rust crate changed"),
            "{id}"
        );
    }
    // The always-on policy checks still run.
    assert!(matches!(step(&steps, "pr-size").action, Action::Run(_)));
    assert!(matches!(step(&steps, "policy").action, Action::Run(_)));
}

#[test]
fn budgets_run_for_manifest_changes_or_full_and_never_with_fast() {
    let code = detect(&["crates/exec/src/lib.rs"]);
    assert!(
        skip_reason(&make_plan(&code, &opts(&[]), true), "budgets")
            .unwrap()
            .contains("--full")
    );
    assert!(matches!(
        step(&make_plan(&code, &opts(&["--full"]), true), "budgets").action,
        Action::Run(_)
    ));
    let manifest = detect(&["crates/exec/Cargo.toml"]);
    assert!(matches!(
        step(&make_plan(&manifest, &opts(&[]), true), "budgets").action,
        Action::Run(_)
    ));
    assert_eq!(
        skip_reason(
            &make_plan(&manifest, &opts(&["--fast", "--full"]), true),
            "budgets"
        ),
        Some("--fast")
    );
    // Only manifest changes run deny and machete.
    assert!(skip_reason(&make_plan(&code, &opts(&[]), true), "deny").is_some());
    assert!(matches!(
        step(&make_plan(&manifest, &opts(&[]), true), "machete").action,
        Action::Run(_)
    ));
}

#[test]
fn fast_skips_both_coverage_gates() {
    let ch = detect(&["crates/exec/src/lib.rs", "bench/src/krishiv_bench/a.py"]);
    let steps = make_plan(&ch, &opts(&["--fast"]), true);
    assert_eq!(skip_reason(&steps, "diff-coverage"), Some("--fast"));
    assert_eq!(skip_reason(&steps, "diff-coverage-python"), Some("--fast"));
    let slow = make_plan(&ch, &opts(&[]), true);
    assert!(matches!(
        step(&slow, "diff-coverage").action,
        Action::Run(_)
    ));
    assert!(matches!(
        step(&slow, "diff-coverage-python").action,
        Action::Run(_)
    ));
}

#[test]
fn github_changes_run_workflow_lint_and_zizmor_only_then() {
    let on = make_plan(&detect(&[".github/workflows/ci.yml"]), &opts(&[]), true);
    assert!(matches!(step(&on, "workflow-lint").action, Action::Run(_)));
    assert!(matches!(step(&on, "zizmor").action, Action::Run(_)));
    let off = make_plan(&detect(&["crates/exec/src/lib.rs"]), &opts(&[]), true);
    assert_eq!(
        skip_reason(&off, "workflow-lint"),
        Some("no .github change")
    );
    assert_eq!(skip_reason(&off, "zizmor"), Some("no .github change"));
}

#[test]
fn python_checks_follow_python_and_bench_changes() {
    let none = make_plan(&detect(&["crates/exec/src/lib.rs"]), &opts(&[]), true);
    assert!(skip_reason(&none, "python-quality").is_some());
    assert!(skip_reason(&none, "python-tests").is_some());
    assert_eq!(
        skip_reason(&none, "diff-coverage-python"),
        Some("no python/bench change")
    );

    let py = make_plan(&detect(&["python/krishiv/a.py"]), &opts(&[]), true);
    assert!(matches!(step(&py, "python-quality").action, Action::Run(_)));
    assert!(
        skip_reason(&py, "python-tests")
            .unwrap()
            .contains("maturin")
    );
    assert!(
        skip_reason(&py, "diff-coverage-python")
            .unwrap()
            .contains("maturin")
    );

    let bench = make_plan(&detect(&["bench/src/krishiv_bench/a.py"]), &opts(&[]), true);
    let Action::Run(q) = &step(&bench, "python-quality").action else {
        panic!()
    };
    assert!(q.iter().all(|c| c.dir == "bench" && c.program == "uv"));
    assert_eq!(q.len(), 4);
    assert!(matches!(
        step(&bench, "python-tests").action,
        Action::Run(_)
    ));

    let scripts = make_plan(&detect(&["tests/scripts/g.py"]), &opts(&[]), true);
    let Action::Run(q) = &step(&scripts, "python-quality").action else {
        panic!()
    };
    assert!(
        q.iter()
            .all(|c| c.args.contains(&"../../python".to_owned()))
    );
}

#[test]
fn python_and_bench_together_note_the_python_test_gap() {
    let s = make_plan(
        &detect(&["python/a.py", "bench/src/a.py"]),
        &opts(&[]),
        true,
    );
    assert!(step(&s, "python-tests").notes[0].contains("maturin"));
}

#[test]
fn missing_tools_are_reported_never_silent() {
    let ch = detect(&[
        "crates/exec/Cargo.toml",
        ".github/a.yml",
        "python/a.py",
        "bench/src/a.py",
    ]);
    let steps = make_plan(&ch, &opts(&[]), false);
    for (id, tool) in [
        ("zizmor", "zizmor"),
        ("deny", "cargo-deny"),
        ("machete", "cargo-machete"),
        ("nextest", "cargo-nextest"),
        ("python-quality", "uv"),
        ("python-tests", "uv"),
        ("diff-coverage", "cargo-llvm-cov"),
        ("diff-coverage-python", "uv"),
        ("text-lint", "typos"),
    ] {
        let why = skip_reason(&steps, id).unwrap_or_else(|| panic!("{id} ran"));
        assert!(
            why.starts_with("not installed:") && why.contains(tool),
            "{id}: {why}"
        );
    }
}

#[test]
fn text_lint_runs_what_is_installed_and_notes_the_rest() {
    let ch = detect(&[]);
    let only_typos = plan(&ch, &opts(&[]), &|b| b == "typos", "x");
    let s = step(&only_typos, "text-lint");
    let Action::Run(cmds) = &s.action else {
        panic!()
    };
    assert_eq!(cmds.len(), 1);
    assert_eq!(s.notes.len(), 2);
    assert!(
        s.notes.iter().any(|n| n.contains("taplo"))
            && s.notes.iter().any(|n| n.contains("markdownlint"))
    );
    let with_docker = plan(&ch, &opts(&[]), &|b| b == "docker", "x");
    let Action::Run(cmds) = &step(&with_docker, "text-lint").action else {
        panic!()
    };
    assert_eq!(cmds[0].program, "docker");
}

#[test]
fn xtask_subcommands_use_the_given_binary_and_base() {
    let steps = make_plan(&detect(&[]), &opts(&["--base", "origin/dev"]), true);
    assert_eq!(
        args_of(&steps, "pr-size"),
        ["pr-size", "--base", "origin/dev"]
    );
    let Action::Run(cmds) = &step(&steps, "pr-size").action else {
        panic!()
    };
    assert_eq!(cmds[0].program, "/x/xtask");
}

#[test]
fn heavy_commands_are_marked_for_the_memory_cap() {
    let steps = make_plan(
        &detect(&["crates/exec/Cargo.toml"]),
        &opts(&["--full"]),
        true,
    );
    for id in ["clippy", "doc", "nextest", "diff-coverage", "budgets"] {
        let Action::Run(cmds) = &step(&steps, id).action else {
            panic!("{id}")
        };
        assert!(cmds.iter().any(|c| c.heavy), "{id}");
    }
    let Action::Run(fmt) = &step(&steps, "fmt").action else {
        panic!()
    };
    assert!(!fmt[0].heavy);
    let prefix = cap_prefix("12G");
    assert_eq!(
        prefix.join(" "),
        "systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 --"
    );
}

fn outcome(id: &'static str, status: Status) -> Outcome {
    Outcome {
        id,
        status,
        detail: format!("{id} detail"),
        secs: 3,
    }
}

#[test]
fn only_a_failure_fails_the_run() {
    let ok = [
        outcome("fmt", Status::Pass),
        outcome("zizmor", Status::Skipped),
        outcome("text-lint", Status::Partial),
    ];
    assert!(!failed(&ok));
    let bad = [
        outcome("fmt", Status::Pass),
        outcome("clippy", Status::Fail),
    ];
    assert!(failed(&bad));
}

#[test]
fn the_table_lists_every_check_with_its_reason() {
    let t = render_table(&[
        outcome("fmt", Status::Pass),
        outcome("zizmor", Status::Skipped),
        outcome("clippy", Status::Fail),
    ]);
    assert!(t.lines().next().unwrap().starts_with("check"));
    assert!(t.contains("pass") && t.contains("skipped") && t.contains("FAIL"));
    assert!(t.contains("zizmor detail"));
    assert_eq!(t.lines().count(), 4);
}

// ---- policy: the required CI checks and precheck cannot drift apart ----

fn workflow_jobs() -> BTreeSet<String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut jobs = BTreeSet::new();
    for f in ["ci.yml", "coverage.yml", "pr-policy.yml", "security.yml"] {
        let text = std::fs::read_to_string(root.join(".github/workflows").join(f)).unwrap();
        let mut in_jobs = false;
        for line in text.lines() {
            if line.starts_with("jobs:") {
                in_jobs = true;
            } else if in_jobs
                && !line.starts_with(' ')
                && !line.is_empty()
                && !line.starts_with('#')
            {
                in_jobs = false;
            } else if in_jobs
                && let Some(id) = line.strip_prefix("  ").and_then(|l| l.strip_suffix(':'))
                && !id.starts_with(' ')
                && !id.starts_with('#')
            {
                jobs.insert(id.to_owned());
            }
        }
    }
    jobs
}

#[test]
fn every_required_ci_check_has_a_precheck_mapping() {
    let all = detect(&[
        "crates/exec/Cargo.toml",
        ".github/a.yml",
        "python/a.py",
        "bench/src/a.py",
    ]);
    let ids: BTreeSet<&str> = make_plan(&all, &opts(&["--full"]), true)
        .iter()
        .map(|s| s.id)
        .collect();
    let mut unmapped = Vec::new();
    for check in REQUIRED_CI_CHECKS {
        match mapping(check) {
            None => unmapped.push(*check),
            Some(Mapping::CiOnly(why)) => assert!(why.len() > 10, "{check} needs a real reason"),
            Some(Mapping::Local(steps)) => {
                assert!(!steps.is_empty(), "{check}");
                for s in steps {
                    assert!(ids.contains(s), "{check} maps to unknown precheck step {s}");
                }
            }
        }
    }
    assert!(
        unmapped.is_empty(),
        "required CI checks without a precheck mapping: {unmapped:?}"
    );
}

#[test]
fn every_step_backs_a_required_check() {
    // Every step must trace back to a required check, so a step cannot be
    // added or dropped without the mapping noticing.
    let claimed: BTreeSet<&str> = REQUIRED_CI_CHECKS
        .iter()
        .filter_map(|c| match mapping(c) {
            Some(Mapping::Local(s)) => Some(s.iter().copied()),
            _ => None,
        })
        .flatten()
        .collect();
    let all: BTreeSet<&str> = STEP_IDS.iter().copied().collect();
    let unclaimed: Vec<&&str> = all.difference(&claimed).collect();
    assert!(
        unclaimed.is_empty(),
        "steps not backing any required check: {unclaimed:?}"
    );
}

#[test]
fn required_checks_exist_as_workflow_jobs() {
    let jobs = workflow_jobs();
    for check in REQUIRED_CI_CHECKS {
        if EXTERNAL_CI_CHECKS.contains(check) {
            continue;
        }
        assert!(
            jobs.contains(*check),
            "required check {check} is not a job in the workflows (renamed?): {jobs:?}"
        );
    }
}

#[test]
fn external_checks_are_ci_only() {
    for c in EXTERNAL_CI_CHECKS {
        assert!(REQUIRED_CI_CHECKS.contains(c));
        assert!(matches!(mapping(c), Some(Mapping::CiOnly(_))), "{c}");
    }
}
