// SPDX-License-Identifier: Apache-2.0
//! `CACHE [LAZY] TABLE`, `UNCACHE TABLE` and `CLEAR CACHE` (design P3-T083 3.5). A child of
//! `command` to keep `command.rs` under the file-size cap.

use super::*;

impl Session {
    /// Runs `CACHE [LAZY] TABLE name [AS query]`, `CLEAR CACHE` and `UNCACHE TABLE [IF EXISTS] name`,
    /// all returning an empty frame as in Spark (`CacheTableExec`). `CACHE TABLE` registers the
    /// relation at its `storageLevel` option (any letter case) or `spark.sql.defaultCacheStorageLevel`,
    /// logging other options as Spark does; without `LAZY` it then fills it with one count, and
    /// a failed count unregisters it (and drops the view `AS query` made) before it raises.
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_NOT_FOUND` for a missing table (`UNCACHE TABLE` without `IF EXISTS`);
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a dynamic name or an unknown storage level; the eager
    /// count's errors.
    pub(crate) async fn exec_cache(
        &self,
        c: &Cache,
        original: &str,
        ids: &[Span],
    ) -> Result<DataFrame, KrishivError> {
        match &c.kind {
            CacheKind::Clear => {
                self.clear_cache();
            }
            CacheKind::Uncache {
                if_exists, name, ..
            } => {
                let Some(parts) = ddl_table_parts(name) else {
                    return Err(KrishivUnsupportedFeature {
                        feature: "UNCACHE TABLE with a dynamic table name".to_owned(),
                    }
                    .into());
                };
                if let Some(key) = self.cache_key(&parts) {
                    self.uncache_table(&key);
                } else if !if_exists {
                    return Err(table_not_found(&parts));
                }
            }
            CacheKind::Cache {
                lazy,
                name,
                query,
                options,
                ..
            } => {
                // The options are never resolved: a `secret(..)` value would be kept as its text.
                crate::access::refuse_secret_options(options, original)?;
                let Some(parts) = ddl_table_parts(name) else {
                    return Err(KrishivUnsupportedFeature {
                        feature: "CACHE TABLE with a dynamic table name".to_owned(),
                    }
                    .into());
                };
                let level = self.cache_table_level(options, original)?;
                // `CACHE TABLE t AS query` first makes the temporary view `t`.
                let mut made_view = None;
                if let Some(q) = query {
                    let [view] = parts.as_slice() else {
                        return Err(table_not_found(&parts));
                    };
                    let body = original.get(q.span.start..q.span.end).unwrap_or_default();
                    let prefix = format!("CREATE TEMPORARY VIEW `{}` AS ", view.replace('`', "``"));
                    // The body's substituted `IDENTIFIER(..)` names, at their place in `create`.
                    let spans = crate::identifier_clause::spans_within(ids, q.span)
                        .into_iter()
                        .map(|s| Span::new(s.start + prefix.len(), s.end + prefix.len()))
                        .collect();
                    let create = format!("{prefix}{body}");
                    Box::pin(self.sql_here_spanned(&create, spans)).await?;
                    made_view = Some(view.clone());
                }
                let frame = self.table(
                    &parts
                        .iter()
                        .map(|p| format!("`{}`", p.replace('`', "``")))
                        .collect::<Vec<_>>()
                        .join("."),
                );
                let cached = async {
                    // `persist` calls made before this statement register first.
                    self.register_pending().await;
                    let key = self.register_frame(&frame, level).await?;
                    if !*lazy {
                        // Counting reads every batch into the cache without keeping the rows.
                        if let Err(error) = frame.count().await {
                            self.state.cache.unregister(&key);
                            return Err(error);
                        }
                    }
                    Ok(())
                }
                .await;
                if let Err(error) = cached {
                    if let Some(view) = made_view {
                        self.state.catalog.drop_table(&view);
                        self.state.cache.uncache_table(&view);
                    }
                    return Err(error);
                }
            }
            _ => {
                return Err(crate::session::internal("unknown CACHE form".to_owned()));
            }
        }
        self.empty_frame().await
    }

    /// The level `CACHE TABLE ... OPTIONS (...)` asks for: the `storageLevel` option (key and
    /// name in any letter case), else `spark.sql.defaultCacheStorageLevel`. Other options are
    /// logged and ignored, as Spark's `CacheTableExec` does.
    fn cache_table_level(
        &self,
        options: &[krishiv_sql::ast::TableOption],
        original: &str,
    ) -> Result<crate::StorageLevel, KrishivError> {
        let options = crate::table_commands::options_of(options, original);
        let mut level = None;
        let mut others = Vec::new();
        for (key, value) in &options {
            if key.eq_ignore_ascii_case("storagelevel") {
                level = Some(value);
            } else {
                others.push(format!("({key},{value})"));
            }
        }
        if !others.is_empty() {
            tracing::warn!(target: "krishiv::session", "Invalid options: {}", others.join(", "));
        }
        match level {
            None => Ok(crate::StorageLevel::default_of(&self.state.conf.snapshot())),
            Some(name) => crate::StorageLevel::from_name(&name.to_uppercase()).ok_or_else(|| {
                KrishivUnsupportedFeature {
                    feature: format!(
                        "the storage level `{name}` (Spark: Invalid StorageLevel: {name})"
                    ),
                }
                .into()
            }),
        }
    }
}
