// SPDX-License-Identifier: Apache-2.0
//! More session statements: the write and query entry points that need no state of their own.

use super::*;

impl Session {
    /// The directory a write of `text` goes to, as it physically is (links in
    /// its existing part resolved): the policy is asked about the directory
    /// that is written, never about a name that a link sends elsewhere.
    /// Cloud stores are built from the query snapshot with the write's
    /// per-relation `fs.*` options overriding it (Spark's per-relation Hadoop
    /// configuration).
    pub(crate) fn locate_write_path(
        &self,
        text: &str,
        options: &BTreeMap<String, OptionValue>,
        env: &QueryEnv,
    ) -> Result<krishiv_exec::files::LocatedPath, KrishivError> {
        let stores = self.scoped_stores(&env.config);
        let at = ObjectPaths::new(
            StoreId::new(LOCAL_STORE),
            self.state.cwd.clone(),
            Arc::clone(&self.state.io),
            stores,
        )
        .locate_with_options(text, options)?;
        crate::access::physical(at)
    }

    /// Before anything is written or replaced (an overwrite deletes first):
    /// `Modify` and `Create` on the path, both whether or not it exists, so a
    /// denial never tells whether it does.
    pub(crate) async fn authorize_write_path(
        &self,
        at: &krishiv_exec::files::LocatedPath,
        options: &BTreeMap<String, OptionValue>,
    ) -> Result<(), KrishivError> {
        // The encoder resolves secret options (`encryptionKey`) for this principal: USAGE first.
        let r = krishiv_plan::analyzer::authorize_secret_refs(
            &*self.state.policy,
            &self.state.principal,
            options.values(),
        )
        .await;
        crate::access::audit(self, StatementClass::Write, r)?;
        let securable = Securable::Path(OptionValue::from_user("path", at.display.clone()));
        for privilege in [Privilege::Modify, Privilege::Create] {
            let r = authorize(
                &*self.state.policy,
                &self.state.principal,
                &securable,
                privilege,
            )
            .await;
            crate::access::audit(self, StatementClass::Write, r)?;
        }
        Ok(())
    }

    /// The file format named `format`, which must be one the engine writes:
    /// the Spark file sources it does not write fail with
    /// `KRISHIV_UNSUPPORTED_FEATURE`.
    pub(crate) fn writable_format(
        &self,
        format: &str,
    ) -> Result<Arc<dyn krishiv_io::FileFormat>, KrishivError> {
        match self.state.runtime.formats.get(format) {
            Ok(found) => Ok(found),
            Err(e) => {
                if is_spark_file_source(format) {
                    return Err(KrishivUnsupportedFeature {
                        feature: format!("writing the {format} format"),
                    }
                    .into());
                }
                Err(e)
            }
        }
    }

    /// What a write of `schema` in `format` with `options` would refuse before it
    /// touches anything: a format the engine does not write, duplicate output
    /// columns, options the format rejects. `saveAsTable` in overwrite mode asks
    /// this before it drops the old table.
    pub(crate) fn check_write_ready(
        &self,
        format: &str,
        options: &BTreeMap<String, OptionValue>,
        schema: &SchemaRef,
        env: &QueryEnv,
    ) -> Result<(), KrishivError> {
        unique_columns(schema, env.analysis.conf.case_sensitive)?;
        let format = self.writable_format(format)?;
        let (parsed, _) = format.options().parse(options, &env.config)?;
        // The framework options the format schema skips (`maxRecordsPerFile`).
        max_records_per_file(options, format.name())?;
        // What the writer would refuse once it exists (a Parquet codec that is not one, a CSV
        // option combination, a schema the format cannot hold): the same check `write` makes.
        format
            .check_write(schema, &parsed)
            .map_err(|e| e.into_error("the table"))?;
        Ok(())
    }

    /// Writes the result of `physical` to the directory `path` in
    /// `format` with `mode` (`df.write`): the files under their final
    /// names, then the directory's commit (`_SUCCESS`, the removal of
    /// orphaned attempt files, and the removal of what an overwrite
    /// replaces). A failed or cancelled write deletes its files and leaves
    /// the directory as it was. The plan carries the write as data; the
    /// commit runs from the plan's result rows (one per file: `path`,
    /// `size`, `rows`). Returns the directories below `path` that hold the
    /// committed files (`col=value/...` of a partitioned write, in their
    /// escaped form; empty for an unpartitioned one or when nothing was
    /// written).
    pub(crate) async fn write_files(
        &self,
        physical: PhysicalPlan,
        env: &QueryEnv,
        target: WriteTarget,
        types: &[DataType],
        pre: &mut Option<Admitted>,
        dry: bool,
    ) -> Result<Vec<String>, KrishivError> {
        let WriteTarget {
            format,
            mode,
            path,
            options,
            partition_by,
        } = target;
        let mode = SaveMode::parse(&mode)?;
        unique_columns(physical.schema(), env.analysis.conf.case_sensitive)?;
        let format = self.writable_format(&format)?;
        let (parsed, _) = format.options().parse(&options, &env.config)?;
        let Some(text) = path.as_plain() else {
            return Err(KrishivUnsupportedFeature {
                feature: "write paths with credentials".to_owned(),
            }
            .into());
        };
        let schema = Arc::clone(physical.schema());
        // The analyzer's types where the plan's Arrow types don't say them.
        let column_types: BTreeMap<usize, DataType> = schema
            .fields()
            .iter()
            .zip(types)
            .enumerate()
            .filter(|(_, (f, t))| {
                t.to_arrow_field_with_storage(f.name(), f.data_type().clone(), f.is_nullable())
                    .is_ok_and(|tagged| !tagged.metadata().is_empty())
            })
            .map(|(i, (_, t))| (i, t.clone()))
            .collect();
        let partitions =
            resolve_partitions(&schema, &partition_by, env.analysis.conf.case_sensitive)?;
        if !partitions.is_empty() && partitions.len() == schema.fields().len() {
            return Err(AllPartitionColumnsNotAllowed {}.into());
        }
        let dynamic =
            !partitions.is_empty() && partition_overwrite_is_dynamic(&env.config, &options)?;
        let max_records = max_records_per_file(&options, format.name())?;
        let at = self.locate_write_path(text, &options, env)?;
        // The query's own context: its counters are the write's bytes.
        let io = env.io.clone();
        self.authorize_write_path(&at, &options).await?;
        // A link swapped into the checked path since fails the write
        // before anything is replaced, and again before the commit deletes
        // what an overwrite replaces.
        crate::access::still_physical(self, &at)?;
        // Checked only (`dry`): nothing is admitted or replaced.
        if dry {
            return Ok(Vec::new());
        }
        // The admission permit before anything is replaced: `DirectoryWrite::begin` deletes what
        // an overwrite replaces, which a refused or cancelled query must not have done.
        // Admitted by the caller already (it changed something before this write), else here.
        let mut admitted = Some(match pre.take() {
            Some(admitted) => admitted,
            None => self.admit_query(env, false).await?,
        });
        let written = async {
            let job = job_uuid();
            // Every name of the write is resolved relative to the directory
            // opened here, never through a link (P2-T199). The registration
            // lives until the write is over.
            let (store, _registration) = self.anchor_local_write(&at, &job)?;
            let Some(write) = DirectoryWrite::begin(
                &io,
                store.clone(),
                at.path.clone(),
                mode,
                job.clone(),
                dynamic,
            )
            .await?
            .map(|w| w.partitioned(!partitions.is_empty())) else {
                // `ignore` found the path: nothing is written.
                return Ok(Vec::new());
            };
            // Boxed: the statement futures are on the recursion of nested
            // scripts, so a larger write future would overflow their stack.
            let write = if Box::pin(uses_commit_log(
                &env.config,
                &self.state.commit_log_probes,
                &io,
                &store,
                &at,
                &job,
            ))
            .await?
            {
                Box::pin(write.logged(&io)).await?
            } else {
                write
            };
            let plan = physical.into_write(WritePlan {
                format: format.name().to_owned(),
                // The session's conf fallbacks travel with the plan: a
                // worker parses these options against its own config.
                options: format.options().with_conf_fallbacks(&options, &env.config),
                store: store.key().to_owned(),
                dir: at.path.as_ref().to_owned(),
                job_uuid: job,
                max_records_per_file: max_records,
                target_file_bytes: 0,
                partition_by: partitions.iter().map(|(i, _)| *i).collect(),
                column_types,
            })?;
            let worker = self.worker(env);
            let (mut rows, _, row_id) = self
                .run_on_query_logged(&plan, worker, env, false, admitted.take())
                .await?;
            let mut batches = Vec::new();
            while let Some(batch) = rows.next().await {
                batches.push(batch?);
            }
            drop(rows);
            let files = committed_files(&batches)?;
            // The query's stream has ended and its `system.queries` row says `finished`: a failure
            // of the steps that follow is the write's outcome, so the row is corrected.
            let after_run = async {
                if let Err(denied) = crate::access::still_physical(self, &at) {
                    // The files the query wrote are under the anchor: remove
                    // them (best effort) before refusing.
                    for f in &files {
                        drop(io.delete(&store, &f.path).await);
                    }
                    return Err(denied);
                }
                write
                    .commit(&io, format.as_ref(), schema, &parsed, files.clone())
                    .await
            }
            .await;
            let committed = match after_run {
                Ok(committed) => committed,
                Err(error) => {
                    self.state.queries.fail(&row_id, &error);
                    return Err(error);
                }
            };
            // The engine's own write may have replaced a cached path: entries
            // reading files are dropped, so the next action lists them again.
            // (An external change drops nothing and stays invisible.)
            self.refresh_files();
            // A write by path into a table's directory changes what the table reads.
            if let Some(local) = at.display.strip_prefix("file:") {
                self.state
                    .runtime
                    .results
                    .invalidate_path(std::path::Path::new(local));
            }
            let mut dirs = std::collections::BTreeSet::new();
            for f in &committed {
                let below: Vec<String> = f
                    .path
                    .prefix_match(&at.path)
                    .map(|parts| parts.map(|p| p.as_ref().to_owned()).collect())
                    .unwrap_or_default();
                if let Some((_, leaf)) = below.split_last()
                    && !leaf.is_empty()
                {
                    dirs.insert(leaf.join("/"));
                }
            }
            Ok(dirs.into_iter().collect())
        }
        .await;
        // A statement that ended before its query ran (an `ignore` that found its path, or an
        // error in the preparation) settles its row with that outcome.
        if let Some(unrun) = admitted.take() {
            match &written {
                Ok(_) => unrun.finish_ok(),
                Err(error) => unrun.fail(error),
            }
        }
        // A store error after the anchor is open may be a link swapped into
        // the path: the audited denial is what the caller sees then.
        if written.is_err()
            && let Err(denied) = crate::access::still_physical(self, &at)
            && denied.error_class().name() == "KRISHIV_INSUFFICIENT_PERMISSIONS"
        {
            return Err(denied);
        }
        written
    }

    /// The store a directory write at `at` goes through: for a local path
    /// on Unix a [`krishiv_io::BeneathStore`] anchored at the deepest
    /// existing directory above the target, registered for this write only
    /// (messages and `EXPLAIN` still name `file`); otherwise the plain
    /// store of `at`.
    pub(super) fn anchor_local_write(
        &self,
        at: &krishiv_exec::files::LocatedPath,
        job: &str,
    ) -> Result<(StoreId, Option<StoreRegistration>), KrishivError> {
        #[cfg(unix)]
        if let Some(local) = at.display.strip_prefix("file:") {
            match krishiv_io::BeneathStore::anchor(std::path::Path::new(local)) {
                Ok(anchored) => {
                    let id = at.store.private_to(job);
                    self.state
                        .io
                        .register_store(id.clone(), Arc::new(anchored) as Arc<dyn ObjectStore>);
                    let registration = StoreRegistration {
                        io: Arc::clone(&self.state.io),
                        id: id.clone(),
                    };
                    return Ok((id, Some(registration)));
                }
                Err(krishiv_io::AnchorError::Swapped(_)) => {
                    // A link or file where a directory was checked: the
                    // audited denial if the path now resolves elsewhere.
                    crate::access::still_physical(self, at)?;
                    return Err(anchor_failed(
                        at,
                        "a component of the path is not a directory",
                    ));
                }
                Err(krishiv_io::AnchorError::UnsupportedName(name)) => {
                    return Err(KrishivUnsupportedFeature {
                        feature: format!(
                            "writing below the directory {name:?} (a name with non-ASCII or \
                             reserved characters above the target)"
                        ),
                    }
                    .into());
                }
                Err(krishiv_io::AnchorError::Io(e)) => {
                    return Err(anchor_failed(at, &e.to_string()));
                }
            }
        }
        let _ = job;
        Ok((at.store.clone(), None))
    }

    /// `path` in the qualified form reads and writes use (`file:/a/b`),
    /// which the access policy is asked about.
    pub(crate) fn qualify_path(&self, path: &str) -> Result<String, KrishivError> {
        // No store is built here: cloud paths qualify through their canonical
        // display (as reads qualify them), local paths through `LocalPaths`.
        if let Ok(url) = parse_object_url(path) {
            return Ok(url.display);
        }
        Ok(
            LocalPaths::new(StoreId::new(LOCAL_STORE), self.state.cwd.clone())
                .locate(path)?
                .display,
        )
    }
}
