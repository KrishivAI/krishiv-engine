// SPDX-License-Identifier: Apache-2.0
//! Enumeration of vendored Spark `SQLQueryTestSuite` golden files (`tests/spark-golden/<tag>/`).
//!
//! `inputs/<name>.sql` pairs with `results/<name>.sql.out`. Variant outputs such as
//! `*.sql.out.java21` are not paired (they are JDK-specific alternates).
//!
//! The runner ([`run`]) parses the `.sql.out` files ([`parse`]), drives an [`run::Engine`] and reports
//! three separate numbers ([`report`]); [`manifest`] holds the exclusion manifest.

pub mod manifest;
pub mod parse;
pub mod report;
pub mod run;

use std::collections::BTreeSet;
use std::io;
use std::path::{Path, PathBuf};

/// Default tag vendored in the repo.
pub const DEFAULT_TAG: &str = "v4.2.0";

/// An `.sql` input with its `.sql.out` expected result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoldenPair {
    /// Name relative to `inputs/`, without the `.sql` suffix (e.g. `postgreSQL/int4`).
    pub name: String,
    /// Path of the `.sql` input.
    pub sql: PathBuf,
    /// Path of the `.sql.out` result.
    pub out: PathBuf,
}

/// All files of one golden directory.
#[derive(Debug, Clone, Default)]
pub struct GoldenSet {
    /// Paired inputs/results.
    pub pairs: Vec<GoldenPair>,
    /// Names with an `.sql` input but no `.sql.out`.
    pub unpaired_inputs: Vec<String>,
    /// Names with an `.sql.out` result but no `.sql` input.
    pub unpaired_outputs: Vec<String>,
}

impl GoldenSet {
    /// Directory of the vendored tag.
    #[must_use]
    pub fn tag_dir(tag: &str) -> PathBuf {
        crate::tests_dir().join("spark-golden").join(tag)
    }

    /// Scan `<dir>/inputs` and `<dir>/results`.
    pub fn scan(dir: &Path) -> io::Result<Self> {
        let inputs = collect(&dir.join("inputs"), ".sql")?;
        let outputs = collect(&dir.join("results"), ".sql.out")?;
        let mut set = Self::default();
        for name in &inputs {
            if outputs.contains(name) {
                set.pairs.push(GoldenPair {
                    name: name.clone(),
                    sql: dir.join("inputs").join(format!("{name}.sql")),
                    out: dir.join("results").join(format!("{name}.sql.out")),
                });
            } else {
                set.unpaired_inputs.push(name.clone());
            }
        }
        set.unpaired_outputs = outputs.difference(&inputs).cloned().collect();
        Ok(set)
    }
}

/// Names (relative path with `suffix` stripped, `/`-separated) of all files ending in `suffix`.
fn collect(root: &Path, suffix: &str) -> io::Result<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if let Some(rel) = path
                .strip_prefix(root)
                .ok()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .and_then(|p| p.strip_suffix(suffix).map(str::to_owned))
            {
                names.insert(rel);
            }
        }
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    type R = Result<(), Box<dyn std::error::Error>>;

    fn manifest_value(dir: &Path, key: &str) -> Result<usize, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(dir.join("MANIFEST"))?;
        let line = text
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{key}: ")))
            .ok_or("manifest key missing")?;
        Ok(line.trim().parse()?)
    }

    #[test]
    fn vendored_goldens_are_paired_and_match_manifest() -> R {
        let dir = GoldenSet::tag_dir(DEFAULT_TAG);
        let set = GoldenSet::scan(&dir)?;
        println!(
            "golden {DEFAULT_TAG}: {} pairs, {} inputs without result, {} results without input",
            set.pairs.len(),
            set.unpaired_inputs.len(),
            set.unpaired_outputs.len()
        );
        assert!(set.pairs.len() > 300, "too few pairs: {}", set.pairs.len());
        assert_eq!(
            set.pairs.len() + set.unpaired_inputs.len(),
            manifest_value(&dir, "sql_files")?
        );
        assert_eq!(
            set.pairs.len() + set.unpaired_outputs.len(),
            manifest_value(&dir, "out_files")?
        );
        assert!(
            set.unpaired_outputs.is_empty(),
            "orphan results: {:?}",
            set.unpaired_outputs
        );
        Ok(())
    }

    #[test]
    fn every_vendored_result_file_parses() -> R {
        let set = GoldenSet::scan(&GoldenSet::tag_dir(DEFAULT_TAG))?;
        let mut statements = 0;
        for pair in &set.pairs {
            statements += run::load_statements(pair)?.len();
        }
        assert!(statements > 10_000, "too few statements: {statements}");
        Ok(())
    }

    #[test]
    fn stub_engine_scores_zero_with_per_file_denominators() -> R {
        let set = GoldenSet::scan(&GoldenSet::tag_dir(DEFAULT_TAG))?;
        let manifest =
            manifest::ExclusionManifest::load(&manifest::ExclusionManifest::default_path())?;
        let run = run::run(&mut run::RejectAllEngine, &set, &manifest)?;
        let (parse, negative, result) = report::totals(&run);
        assert_eq!((parse.passed, negative.passed, result.passed), (0, 0, 0));
        assert!(parse.total > 0 && negative.total > 0 && result.total > 0);
        for f in &run.files {
            let denominators = f.parse.total + f.negative.total + f.result.total;
            assert!(
                denominators <= f.statements * 2,
                "{}: inconsistent denominators",
                f.name
            );
            assert_eq!(
                f.result.total + f.negative.total + f.unclassified_errors,
                f.statements,
                "{}",
                f.name
            );
        }
        assert_eq!(
            run.files.len() + run.excluded.len() + run.without_expected.len(),
            set.pairs.len() + set.unpaired_inputs.len()
        );
        Ok(())
    }
}
