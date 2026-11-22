// SPDX-License-Identifier: Apache-2.0
//! Session methods of the write path: target paths, qualified paths and the helpers that read the session state.

use super::*;

impl Session {
    /// A query that reads something besides `system.*` is refused with `KRISHIV_SERVER_DRAINING`
    /// while the server drains (design P3-T066 §5.1); its row `log_id` closes as failed.
    pub(crate) fn refuse_query_while_draining(&self, log_id: &str) -> Result<(), KrishivError> {
        if self.drain_exempt || !self.state.runtime.is_draining() {
            return Ok(());
        }
        let error: KrishivError = krishiv_common::error::KrishivServerDraining {}.into();
        self.state.queries.record_failed_row(log_id, &error);
        Err(error)
    }

    /// While the server drains, a statement that writes, creates or drops is refused here, before
    /// its handler changes any catalog or file state (design P3-T066 §5.1): the command handlers
    /// do their work inside `sql`, so admission at collect would come too late. Queries are
    /// admitted when they run; `SET`, `USE`, `SHOW`, `DESCRIBE` and session variables change no
    /// shared state. The refusal leaves a failed `system.queries` row, as a refused query does.
    pub(crate) fn refuse_changes_while_draining(
        &self,
        class: krishiv_sql::StatementClass,
        redacted: &krishiv_common::redact::RedactedSql,
        env: &QueryEnv,
    ) -> Result<(), KrishivError> {
        use krishiv_sql::StatementClass::{Admin, Ddl, Write};
        if !self.state.runtime.is_draining() || !matches!(class, Write | Ddl | Admin) {
            return Ok(());
        }
        let error: KrishivError = krishiv_common::error::KrishivServerDraining {}.into();
        let id = self.record_refused(Some(redacted.as_str()), Some(class.name()), &error);
        *env.row
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(id);
        Err(error)
    }

    /// [`Self::sql`] and whether the statement was a command (not a query).
    pub(crate) async fn sql_kind(&self, query: &str) -> Result<(DataFrame, bool), KrishivError> {
        self.sql_ran(query, false)
            .await
            .and_then(super::Ran::into_frame)
    }

    /// [`Self::sql_kind`]; with `queries_only` a statement that is not a query is neither
    /// authorized nor run ([`super::Ran::NotAQuery`]) and records no `system.queries` row.
    pub(crate) async fn sql_ran(
        &self,
        query: &str,
        queries_only: bool,
    ) -> Result<super::Ran, KrishivError> {
        self.check_open()?;
        let (session, text) = (self.clone(), query.to_owned());
        let row: Arc<std::sync::Mutex<Option<String>>> = Arc::default();
        let marker = Arc::clone(&row);
        let (result, cancelled) = self
            .on_executor_reasoned(async move {
                let seq = session.state.statements.fetch_add(1, Ordering::Relaxed);
                use tracing::Instrument as _;
                let trace = crate::telemetry::QueryTrace::of(seq, &text);
                let span = trace.prepare_span(&crate::telemetry::Who::of(&session));
                session
                    .sql_here_marked(&text, marker, queries_only)
                    .instrument(span)
                    .await
                    .map(|ran| match ran {
                        super::Ran::Frame(df, command) => {
                            super::Ran::Frame(df.with_trace(trace), command)
                        }
                        refused @ super::Ran::NotAQuery(_) => refused,
                    })
            })
            .await;
        // A statement refused before it ran (parse, analysis, access) is a
        // row of `system.queries` too, with the text as the session may show it.
        // ...unless a query already ran inside `sql` (an `EXPLAIN ANALYZE`
        // that failed while it ran): that one has its row.
        if let Err(error) = &result
            && row.lock().unwrap_or_else(PoisonError::into_inner).is_none()
        {
            let id = self.state.queries.record_refused(
                crate::system::TextPolicy::from_config(&self.state.conf.snapshot()),
                Some(krishiv_sql::redact_sql(query).as_str()),
                None,
                error,
                self.operation_id(),
            );
            self.stamp_row(&id);
            // Ended by `cancel`/`interrupt_all` while it was analyzed (`action.rs`).
            if let Some(reason) = cancelled {
                self.state.queries.note_cancel_reason(&id, reason);
            }
        }
        result
    }

    /// This session's parser options, from its configuration. Public under `unstable-ir`, for a
    /// Connect server that parses `Expression.expression_string` while it translates a plan.
    #[cfg(feature = "unstable-ir")]
    pub fn parser_options(&self) -> ParserOptions {
        parser_options(&self.state.conf.snapshot(), self.state.profile)
    }

    /// This session's parser options, from its configuration.
    #[cfg(not(feature = "unstable-ir"))]
    pub(crate) fn parser_options(&self) -> ParserOptions {
        parser_options(&self.state.conf.snapshot(), self.state.profile)
    }

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
    ) -> Result<crate::access::Checked, KrishivError> {
        let at = self.locate_write_text(text, options, env)?;
        crate::access::physical(self, at, Privilege::Modify, StatementClass::Write)
    }

    /// The path a write of `text` goes to as the caller wrote it (qualified, `.` and `..` folded in the
    /// text, no link resolved): what [`Self::locate_write_path`] resolves, and the text its denial names.
    pub(crate) fn locate_write_text(
        &self,
        text: &str,
        options: &BTreeMap<String, OptionValue>,
        env: &QueryEnv,
    ) -> Result<krishiv_exec::files::LocatedPath, KrishivError> {
        let stores = self.scoped_stores(&env.config);
        ObjectPaths::new(
            StoreId::new(LOCAL_STORE),
            self.state.cwd.clone(),
            Arc::clone(&self.state.io),
            stores,
        )
        .locate_with_options(text, options)
    }

    /// A write's options, before its target is located: locating builds the
    /// store from them (a key file is read then), so a server-scoped option is
    /// refused before any store sees it. `USAGE` on the secrets the encoder
    /// resolves, then `Manage` for a server-scoped store key.
    pub(crate) async fn authorize_write_options(
        &self,
        options: &BTreeMap<String, OptionValue>,
        table_options: &BTreeMap<String, OptionValue>,
    ) -> Result<(), KrishivError> {
        // The encoder resolves secret options (`encryptionKey`) for this principal: USAGE first.
        let r = krishiv_plan::analyzer::authorize_secret_refs(
            &*self.state.policy,
            &self.state.principal,
            options.values(),
        )
        .await;
        crate::access::audit(self, StatementClass::Write, r)?;
        self.authorize_store_options(
            StatementClass::Write,
            &self.state.conf.snapshot(),
            // A table's own stored option was authorized when it was stored.
            options
                .iter()
                .filter(|(k, v)| table_options.get(*k) != Some(*v))
                .map(|(k, _)| k.as_str()),
        )
        .await
    }

    /// Before anything is written or replaced (an overwrite deletes first):
    /// `Modify` and `Create` on the path, both whether or not it exists, so a
    /// denial never tells whether it does. The options were checked before the
    /// path was located (`authorize_write_options`).
    pub(crate) async fn authorize_write_path(
        &self,
        checked: &crate::access::Checked,
    ) -> Result<(), KrishivError> {
        crate::access::authorize_checked(
            self,
            checked,
            &[Privilege::Modify, Privilege::Create],
            StatementClass::Write,
        )
        .await
    }

    /// `Manage` on the cluster when `options` hold a server-scoped store key
    /// (`krishiv_io::is_server_scoped_option`: endpoints, regions, accounts, key files; not inline
    /// credentials), or when the session's configuration `config` changed one from what the
    /// session was built with (P3-T067b). A store built from them could send the server's own
    /// credentials to another host. Admin only in server mode; embedded use (`AllowAll`) is
    /// unchanged.
    pub(crate) async fn authorize_store_options<'a>(
        &self,
        class: StatementClass,
        config: &krishiv_common::config::Config,
        options: impl Iterator<Item = &'a str>,
    ) -> Result<(), KrishivError> {
        let r = self
            .check_store_options(&*self.state.policy, config, options)
            .await;
        crate::access::audit(self, class, r)
    }

    /// [`Self::authorize_store_options`] without the audit record, asking `policy`, for a caller
    /// that logs it (an analysis, through [`crate::access::audit_read`], silent in a trial, whose
    /// policy is quiet too).
    pub(crate) async fn check_store_options<'a>(
        &self,
        policy: &dyn krishiv_plan::analyzer::AccessPolicy,
        config: &krishiv_common::config::Config,
        mut options: impl Iterator<Item = &'a str>,
    ) -> Result<(), KrishivError> {
        if !options.any(krishiv_io::is_server_scoped_option)
            && !self.state.conf.store_settings_changed(config)
        {
            return Ok(());
        }
        authorize(
            policy,
            &self.state.principal,
            &Securable::System("cluster".to_owned()),
            Privilege::Manage,
        )
        .await
    }

    /// Whether [`Self::authorize_store_options`] of `options` asks `Manage` on the cluster under the
    /// session's stored configuration (what a DDL run passes): the analyze API lists it then.
    pub(crate) fn store_options_asked<'a>(
        &self,
        mut options: impl Iterator<Item = &'a str>,
    ) -> bool {
        options.any(krishiv_io::is_server_scoped_option)
            || self
                .state
                .conf
                .store_settings_changed(&self.state.conf.snapshot())
    }

    /// `RESET key`: the key's `SET` check ([`Self::authorize_set_key`]), then the unset.
    pub(crate) async fn unset_key(&self, key: &str) -> Result<(), KrishivError> {
        self.authorize_set_key(key).await?;
        self.conf().unset(key)
    }

    /// `SET key = ..` (or `RESET key`) of a key only an administrator may change is `Manage`
    /// on the cluster (ADR 0046 §3): a registry key settable by `admin` (its Spark alias too),
    /// or a server-scoped store key (P3-T067b). Any other key passes. Asked before the value
    /// is stored, so a refused `SET` changes nothing. A `startup` key is refused by the
    /// configuration itself, for every principal (`CANNOT_MODIFY_STATIC_CONFIG`).
    ///
    /// Public for server adapters that set configuration outside SQL (Spark Connect's `Config`
    /// RPC), so they ask exactly what `SET` asks; not meant as user API.
    ///
    /// # Errors
    /// `KRISHIV_INSUFFICIENT_PERMISSIONS` (42501) when the principal lacks `Manage`; the
    /// denial is audited and counted like every other.
    pub async fn authorize_set_key(&self, key: &str) -> Result<(), KrishivError> {
        if krishiv_common::config::lookup(key)
            .or_else(|| {
                krishiv_common::config::lookup(krishiv_common::config::resolve_alternative(key))
            })
            .is_some_and(|e| e.settable_by == krishiv_common::config::SettableBy::Admin)
        {
            let r = authorize(
                &*self.state.policy,
                &self.state.principal,
                &Securable::System("cluster".to_owned()),
                Privilege::Manage,
            )
            .await;
            return crate::access::audit(self, StatementClass::Session, r);
        }
        if !krishiv_io::is_server_scoped_option(key) {
            return Ok(());
        }
        self.authorize_store_options(
            StatementClass::Session,
            &self.state.conf.snapshot(),
            std::iter::once(key),
        )
        .await
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
            bucket,
            format,
            mode,
            path,
            options,
            table_options,
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
        // The bucket id column `bucketBy` appended, and the columns each bucket is sorted by.
        let bucket_write = match &bucket {
            None => None,
            Some(b) => {
                // `with_bucket_id` appends the bucket id as the last column.
                let column = schema.fields().len().checked_sub(1).ok_or_else(|| {
                    internal("a bucketed write without its bucket id column".to_owned())
                })?;
                let sort_by = super::writes::resolve_columns(
                    &schema,
                    &b.sort_by,
                    env.analysis.conf.case_sensitive,
                )?;
                Some(krishiv_exec::physical::BucketWrite { column, sort_by })
            }
        };
        let dynamic =
            !partitions.is_empty() && partition_overwrite_is_dynamic(&env.config, &options)?;
        let max_records = max_records_per_file(&options, format.name())?;
        self.authorize_write_options(&options, &table_options)
            .await?;
        let checked = self.locate_write_path(text, &options, env)?;
        let at = checked.at.clone();
        // The query's own context: its counters are the write's bytes.
        let io = env.io.clone();
        self.authorize_write_path(&checked).await?;
        // A link swapped into the checked path since fails the write
        // before anything is replaced, and again before the commit deletes
        // what an overwrite replaces.
        crate::access::still_physical(self, &checked)?;
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
        // The query's profile waits here for the commit below.
        let hold: Arc<crate::profile_history::HistoryHold> = Arc::default();
        // Dropped with the future (a cancelled write), it records the held entry as cancelled.
        let hold_guard = crate::profile_history::HoldGuard::new(
            Arc::clone(&hold),
            self.state.runtime.profile_history.clone(),
        );
        let held_env = QueryEnv {
            history_hold: Some(Arc::clone(&hold)),
            ..env.clone()
        };
        let written = async {
            let job = job_uuid();
            // Every name of the write is resolved relative to the directory
            // opened here, never through a link (P2-T199). The registration
            // lives until the write is over.
            let (store, _registration) = self.anchor_local_write(&checked, &job)?;
            // An overwrite deletes what it replaces here: from now on a cancel
            // reaches the write through its query, not by dropping it (`action.rs`).
            crate::action::commit_point();
            let Some(write) = DirectoryWrite::begin(
                &io,
                store.clone(),
                at.path.clone(),
                mode,
                job.clone(),
                dynamic,
            )
            .await?
            .map(|w| w.partitioned(!partitions.is_empty() || bucket.is_some())) else {
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
                bucket: bucket_write,
            })?;
            let worker = self.worker(env);
            let (rows, _, row_id) = self
                .run_on_query_logged(&plan, worker, &held_env, false, admitted.take())
                .await?;
            let mut rows = crate::run_site::resolve_sites(rows, env);
            let mut batches = Vec::new();
            while let Some(batch) = rows.next().await {
                batches.push(batch?);
            }
            drop(rows);
            let files = committed_files(&batches)?;
            // The query's stream has ended and its `system.queries` row says `finished`: a failure
            // of the steps that follow is the write's outcome, so the row is corrected.
            let after_run = async {
                if let Err(denied) = crate::access::still_physical(self, &checked) {
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
        // the path: the audited denial is what the caller sees then. A denial
        // is already audited (and counted): it is not asked about again.
        let denied = written
            .as_ref()
            .is_err_and(|e| e.error_class().name() != "KRISHIV_INSUFFICIENT_PERMISSIONS")
            .then(|| crate::access::still_physical(self, &checked).err())
            .flatten()
            .filter(|d| d.error_class().name() == "KRISHIV_INSUFFICIENT_PERMISSIONS");
        hold_guard.settle(denied.as_ref().or(written.as_ref().err()));
        match denied {
            Some(denied) => Err(denied),
            None => written,
        }
    }

    /// The store a directory write at `at` goes through: for a local path
    /// on Unix a [`krishiv_io::BeneathStore`] anchored at the deepest
    /// existing directory above the target, registered for this write only
    /// (messages and `EXPLAIN` still name `file`); otherwise the plain
    /// store of `at`.
    pub(super) fn anchor_local_write(
        &self,
        checked: &crate::access::Checked,
        job: &str,
    ) -> Result<(StoreId, Option<StoreRegistration>), KrishivError> {
        let at = &checked.at;
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
                    crate::access::still_physical(self, checked)?;
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

/// Aborts a task when dropped.
pub(super) struct AbortOnDrop(pub(super) krishiv_exec::TaskHandle);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl Session {
    /// Asks the admission controller for a permit for the query whose
    /// `system.queries` row is `log_id`. A controller that answers at once
    /// leaves the row `running`; one that waits shows it `queued`. A refusal
    /// closes the row as failed with the controller's error class, and a
    /// caller that drops the query while it is queued closes it `cancelled`.
    pub(super) async fn admit(
        &self,
        log_id: &str,
        env: &QueryEnv,
        exempt: bool,
    ) -> Result<crate::admission::Permit, KrishivError> {
        // A query that reads only `system.*` is never queued behind the load it inspects.
        if exempt {
            return Ok(self.exempt_permit(log_id));
        }
        self.refuse_query_while_draining(log_id)?;
        let request = QueryRequest {
            principal: Arc::clone(&self.state.principal),
            tags: self.effective_tags(),
            class: WorkloadClass::Interactive,
            source: None,
            statement_class: None,
            estimate: env.catalog.estimate(),
        };
        let controller = Arc::clone(&self.state.admission);
        let mut admit = std::pin::pin!(controller.admit(&request));
        let mut queued = QueuedRow {
            state: Arc::clone(&self.state),
            id: log_id,
            open: true,
        };
        let admitted = match std::future::poll_fn(|cx| Poll::Ready(admit.as_mut().poll(cx))).await {
            Poll::Ready(done) => done,
            Poll::Pending => {
                self.state.queries.set_state(log_id, "queued");
                let done = admit.await;
                if done.is_ok() {
                    self.state.queries.set_state(log_id, "running");
                }
                done
            }
        };
        queued.open = false;
        match admitted {
            Ok(permit) => {
                let queued_ns = permit
                    .slot::<crate::resource_groups::GroupSlot>()
                    .map_or(0, crate::resource_groups::GroupSlot::queued_ns);
                self.state
                    .queries
                    .mark_started(log_id, permit.group(), queued_ns);
                Ok(permit)
            }
            Err(error) => {
                self.state.queries.record_failed_row(log_id, &error);
                Err(error)
            }
        }
    }

    /// Begins the `system.queries` row of a query and waits for its admission permit. A caller
    /// that is about to change something (a write replacing files) takes the permit first and
    /// passes it on, so a refused or cancelled query has changed nothing.
    pub(crate) async fn admit_query(
        &self,
        env: &QueryEnv,
        exempt: bool,
    ) -> Result<Admitted, KrishivError> {
        // The policy of the query's own configuration snapshot, passed to its row (no shared slot).
        let log_id = self.state.queries.begin_as(
            crate::system::TextPolicy::from_config(&env.config),
            env.text.as_deref(),
            env.statement_class,
            self.operation_id(),
        )?;
        crate::action::note_row(&log_id);
        // The request tags join the row's tags at once, so a queued query shows them too.
        self.stamp_row(&log_id);
        *env.row.lock().unwrap_or_else(PoisonError::into_inner) = Some(log_id.clone());
        let permit = self.admit(&log_id, env, exempt).await?;
        Ok(Admitted::new(Arc::clone(&self.state), log_id, permit))
    }
}
