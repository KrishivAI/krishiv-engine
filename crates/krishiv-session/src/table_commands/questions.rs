// SPDX-License-Identifier: Apache-2.0
//! The directory questions a run of DDL asks the policy, foreseen without running it: what the analyze API
//! lists for `CREATE TABLE`, CTAS, `ALTER TABLE ... RENAME TO` and `DROP NAMESPACE ... CASCADE`
//! (`analyze/dirs.rs`). Each follows the run's own steps up to its questions, through the same helpers, and
//! stops where the run would fail before it asks; the questions a run asks before a later failure stay.
//!
//! Nothing here changes anything: no catalog entry, no directory (the runtime's own warehouse directory is
//! looked up without being made, `Session::warehouse_path`: its name is chosen then, so the run makes the
//! directory listed; review D3). What needs more than the catalog and whether a
//! path exists is not foreseen, and the questions are listed as if it succeeds: the schema a `CREATE TABLE`
//! without columns reads from its files, and the run's later refusals (a write the format refuses).

use super::create::{NewTable, check_inferable, layout_of, new_table_location};
use super::files::LocationAsked;
use super::*;

/// One question: the securable (a path as the run's denial names it, or the cluster) and the privilege
/// asked.
pub(crate) type DirQuestion = (Securable, Privilege);

impl Session {
    /// The directory questions of a run of the `CREATE TABLE` `t` (not `TEMPORARY`), in its order; `query` is
    /// the analysis of a CTAS's query (`None`: the run fails before it asks). `original` is the text `t`'s
    /// spans index.
    pub(crate) fn create_table_questions(
        &self,
        t: &CreateTable,
        original: &str,
        query: Option<&krishiv_plan::analyzer::AnalyzedPlan>,
        env: &QueryEnv,
    ) -> Vec<DirQuestion> {
        let mut out = Vec::new();
        // An error is where the run stops: what it asked before stays.
        let _ = self.create_questions(t, original, query, env, &mut out);
        out
    }

    fn create_questions(
        &self,
        t: &CreateTable,
        original: &str,
        query: Option<&krishiv_plan::analyzer::AnalyzedPlan>,
        env: &QueryEnv,
        out: &mut Vec<DirQuestion>,
    ) -> Result<(), KrishivError> {
        let warehouse = self.warehouse_path();
        let Some(NewTable {
            name,
            partition_names,
            typed_partitions,
            provider,
            provider_written,
            ..
        }) = self.new_table(t, env)?
        else {
            return Ok(());
        };
        let mut options = options_of(refuse_secret_options(&t.options, original)?, original);
        // `authorize_store_options`: `MANAGE` on the cluster before the location (review D2).
        if self.store_options_asked(options.keys().map(String::as_str)) {
            out.push((Securable::System("cluster".to_owned()), Privilege::Manage));
        }
        let (external, location) = new_table_location(t, &mut options, &warehouse, &name)?;
        let layout = LayoutRequest {
            partition_by: &partition_names,
            bucket: t.bucket.as_ref(),
        };
        let case_sensitive = env.analysis.conf.case_sensitive;
        let Some(_) = &t.as_query else {
            let mut columns = self.declared_columns(t, original, &typed_partitions)?;
            if columns.is_empty() {
                // The schema (and partitioning) is read from the files: not foreseen.
                check_inferable(&layout, external, &provider_written)?;
            } else {
                layout_of(
                    &name,
                    &provider,
                    &mut columns,
                    layout.partition_by,
                    layout.bucket,
                    case_sensitive,
                )?;
            }
            // `add_table`: a managed table's directory must not exist yet.
            if !external && location.exists() {
                return Ok(());
            }
            return self.push_create(&location, !external, out);
        };
        // `create_table_from`.
        let Some(analyzed) = query else {
            return Ok(());
        };
        crate::char_write::refuse_indeterminate_schema(analyzed)?;
        let mut columns = crate::char_write::output_columns(analyzed, Some(true));
        self.refuse_collated_namespace(&name)?;
        crate::partitions::table_layout(
            &name,
            &mut columns,
            layout.partition_by,
            layout.bucket,
            case_sensitive,
        )?;
        if !external && location.exists() {
            return Ok(());
        }
        if !external {
            push_lexical(out, &location, &[Privilege::Modify, Privilege::Create]);
        }
        // The writer asks about the location as it locates it (`write_files`).
        let options: BTreeMap<String, OptionValue> = options
            .iter()
            .map(|(k, v)| (k.clone(), OptionValue::from_user(k, v)))
            .collect();
        let at = self.locate_write_text(&location.display().to_string(), &options, env)?;
        for privilege in [Privilege::Modify, Privilege::Create] {
            out.push((path_securable(&at.display), privilege));
        }
        // `add_table` of the written table.
        self.push_create(&location, !external, out)
    }

    /// `CREATE` on a new table's `location` as `add_table` asks it ([`Self::location_asked`]).
    fn push_create(
        &self,
        location: &Path,
        managed: bool,
        out: &mut Vec<DirQuestion>,
    ) -> Result<(), KrishivError> {
        match self.location_asked(location, managed)? {
            LocationAsked::Local(local) => push_lexical(out, &local, &[Privilege::Create]),
            LocationAsked::Store(qualified) => {
                out.push((path_securable(&qualified), Privilege::Create));
            }
        }
        Ok(())
    }

    /// The directory questions of a run of `ALTER TABLE a.table RENAME TO to`: a managed table's directory
    /// moves, so `MODIFY` on it (when it exists, as `deletable_dir` asks) and `CREATE` on the new one.
    pub(crate) fn rename_questions(
        &self,
        a: &AlterTable,
        to: &DdlName,
        env: &QueryEnv,
    ) -> Vec<DirQuestion> {
        let mut out = Vec::new();
        let _ = self.rename_questions_into(a, to, env, &mut out);
        out
    }

    fn rename_questions_into(
        &self,
        a: &AlterTable,
        to: &DdlName,
        env: &QueryEnv,
        out: &mut Vec<DirQuestion>,
    ) -> Result<(), KrishivError> {
        let (_, table) = self.altered_table(a, env)?;
        let to = self.table_name(to, env)?;
        let to = self.rename_target(&table, &to)?;
        if !table.managed {
            return Ok(());
        }
        let warehouse = self.warehouse_path();
        if let Some((source, _)) = self.deletable_location(&table)? {
            push_lexical(out, &source, &[Privilege::Modify]);
        }
        let target = managed_location(&warehouse, &to.namespace, &to.name);
        push_lexical(out, &target, &[Privilege::Create]);
        Ok(())
    }

    /// The directory questions of a run of `DROP NAMESPACE n CASCADE`: `MODIFY` on the directory of each
    /// managed table in it that exists, in the run's order, up to one the run cannot delete (it stops there).
    /// Asked only once the caller may list the namespace's tables (`analyze/dirs.rs`, review D1).
    pub(crate) fn drop_namespace_questions(
        &self,
        n: &DropNamespace,
        env: &QueryEnv,
    ) -> Vec<DirQuestion> {
        let mut out = Vec::new();
        if !matches!(n.behavior, Some(DropBehavior::Cascade)) {
            return out;
        }
        let Ok(name) = self.namespace_name(&n.name, env) else {
            return out;
        };
        for t in self.state.catalog.store.tables_in(&name) {
            match self.deletable_location(&t) {
                Ok(Some((location, _))) => push_lexical(&mut out, &location, &[Privilege::Modify]),
                Ok(None) => {}
                Err(_) => break,
            }
        }
        out
    }
}

/// Each of `privileges` on the local `path` as `authorize_path` names it in a denial (`.` and `..` folded in
/// the text, qualified); nothing for a path it refuses before it asks (relative, not UTF-8).
pub(crate) fn push_lexical(out: &mut Vec<DirQuestion>, path: &Path, privileges: &[Privilege]) {
    if let Ok(at) = crate::access::lexical_dir(path) {
        for privilege in privileges {
            out.push((path_securable(&at.display), *privilege));
        }
    }
}

/// The path securable of `display`, the text the run's denial names.
fn path_securable(display: &str) -> Securable {
    Securable::Path(OptionValue::from_user("path", display.to_owned()))
}
