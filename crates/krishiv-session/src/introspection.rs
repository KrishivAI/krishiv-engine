// SPDX-License-Identifier: Apache-2.0
//! `DESCRIBE` and `SHOW` on the catalog: tables, views, columns, namespaces, properties and `SHOW CREATE
//! TABLE`. Result schemas and row layouts are Spark 4.2's (the goldens' `-- !query schema` lines and a
//! throw-away container run); every result is a frame of strings (booleans for `isTemporary`), ordered as
//! the rows are produced.

use std::sync::Arc;

use krishiv_common::error::{
    ExpectPermanentViewNotTemp, KrishivError, KrishivUnsupportedFeature, SchemaNotFound,
    ShowColumnsWithConflictNamespace, UnresolvedColumnWithSuggestion,
};
use krishiv_sql::ast::{DdlName, Describe, DescribeKind, Show, ShowKind, ShowPattern};
use krishiv_types::DataType;

use crate::dataframe::DataFrame;
use crate::functions::{col, lit};
use crate::session::Session;
use crate::store::{CatalogPartition, CatalogStore, CatalogTable, NamespaceMeta};
use crate::table_commands::TableName;

type Cell = Option<String>;

fn s(v: impl Into<String>) -> Cell {
    Some(v.into())
}

fn unsupported(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}

/// Spark's `SHOW ... LIKE` pattern: `*` matches anything, `|` separates alternatives, the match ignores case.
fn matches(pattern: &str, name: &str) -> bool {
    // `reach[j]`: the pattern so far matches the first `j` characters of the name. O(pattern x name), so a
    // pattern like `*a*a*a*` cannot blow up.
    fn glob(p: &[char], n: &[char]) -> bool {
        let mut reach = vec![false; n.len() + 1];
        reach[0] = true;
        for &c in p {
            if c == '*' {
                for j in 1..=n.len() {
                    reach[j] = reach[j] || reach[j - 1];
                }
            } else {
                for j in (1..=n.len()).rev() {
                    reach[j] = reach[j - 1] && n[j - 1] == c;
                }
                reach[0] = false;
            }
        }
        reach[n.len()]
    }
    let name: Vec<char> = name.to_lowercase().chars().collect();
    pattern.split('|').any(|alt| {
        glob(
            &alt.trim().to_lowercase().chars().collect::<Vec<_>>(),
            &name,
        )
    })
}

fn keep(pattern: Option<&ShowPattern>, name: &str) -> bool {
    pattern.is_none_or(|p| matches(&p.pattern, name))
}

/// `Fri Oct 30 03:13:32 UTC 2026`, as Spark prints a creation time.
fn created_time(ms: i64) -> String {
    const DAYS: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let secs = ms.div_euclid(1000);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{} {} {day:02} {:02}:{:02}:{:02} UTC {year}",
        DAYS[usize::try_from(days.rem_euclid(7)).unwrap_or(0)],
        MONTHS[usize::try_from(month - 1).unwrap_or(0)],
        rem / 3600,
        rem % 3600 / 60,
        rem % 60,
    )
}

/// `Created Time` as the JSON command writes it: `yyyy-MM-dd'T'HH:mm:ss'Z'` in UTC.
fn created_time_iso(ms: i64) -> String {
    let secs = ms.div_euclid(1000);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60,
    )
}

/// Spark's `spark.redaction.regex` default, `(?i)secret|password|token|access[.]key`, applied to the key.
fn redact<'a>(key: &str, value: &'a str) -> &'a str {
    let k = key.to_lowercase();
    if ["secret", "password", "token", "access.key"]
        .iter()
        .any(|w| k.contains(w))
    {
        "*********(redacted)"
    } else {
        value
    }
}

/// A column type as `SHOW CREATE TABLE` prints it. Nested types keep their field names, which an upper-cased
/// `catalog_string` would change, so they are not printed yet.
fn type_sql(t: &DataType) -> Result<String, KrishivError> {
    let text = t.catalog_string();
    if text.contains('<') {
        return Err(unsupported("SHOW CREATE TABLE with a nested column type"));
    }
    Ok(text.to_uppercase())
}

/// A name as Spark's `quoteIfNeeded` writes it: bare when it is word characters
/// and not all digits, else in backquotes.
pub(crate) fn quote_if_needed(name: &str) -> String {
    let plain = !name.is_empty()
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.chars().all(|c| c.is_ascii_digit());
    if plain {
        name.to_owned()
    } else {
        format!("`{}`", name.replace('`', "``"))
    }
}

fn sql_literal(v: &str) -> String {
    format!("'{}'", v.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// Whether `ns` is the namespace of global temporary views.
fn is_global_namespace(ns: &str) -> bool {
    ns.eq_ignore_ascii_case(crate::catalog::GLOBAL_NAMESPACE)
}

impl Session {
    /// `SHOW TABLES | VIEWS IN global_temp`: the global temporary views.
    async fn global_views_frame(
        &self,
        columns: &[&str],
        pattern: Option<&ShowPattern>,
    ) -> Result<DataFrame, KrishivError> {
        let mut names = self.state.catalog.global_names();
        names.retain(|n| keep(pattern, n));
        names.sort();
        let mut rows: Vec<Vec<Cell>> = names
            .into_iter()
            .map(|n| vec![s(crate::catalog::GLOBAL_NAMESPACE), s(n), s("true")])
            .collect();
        // Spark lists the session's local temporary views after the global ones.
        rows.extend(
            self.temp_view_names(pattern)
                .into_iter()
                .map(|n| vec![s(""), s(n), s("true")]),
        );
        self.ordered_frame(columns, &["isTemporary"], rows).await
    }

    /// A frame of `columns` (string columns, except the names in `booleans`) with `rows`, in row order.
    pub(crate) async fn ordered_frame(
        &self,
        columns: &[&str],
        booleans: &[&str],
        rows: Vec<Vec<Cell>>,
    ) -> Result<DataFrame, KrishivError> {
        self.frame_with_order(columns, booleans, rows, true).await
    }

    /// [`Self::ordered_frame`], with the rows sorted into their written order
    /// only when `sorted`. Spark's `SHOW COLUMNS` plan is not a sort, so its
    /// golden results compare as a set; a sort node would compare them in order.
    async fn frame_with_order(
        &self,
        columns: &[&str],
        booleans: &[&str],
        rows: Vec<Vec<Cell>>,
        sorted: bool,
    ) -> Result<DataFrame, KrishivError> {
        let mut all = vec!["__krishiv_row_order"];
        all.extend_from_slice(columns);
        let keyed = rows
            .into_iter()
            .enumerate()
            .map(|(i, mut r)| {
                r.insert(0, Some(format!("{i:010}")));
                r
            })
            .collect();
        let df = self.string_frame(&all, keyed).await?;
        // Rows with a `key` column come out sorted by it; the others keep
        // their order through the hidden column.
        let df = if columns.contains(&"key") || !sorted {
            df
        } else {
            df.order_by([col("__krishiv_row_order")])
        };
        let df = df.select(columns.iter().map(|&c| {
            if booleans.contains(&c) {
                col(c).eq(lit("true")).alias(c)
            } else {
                col(c)
            }
        }));
        df.analyzed_in(self.query_env()).await?;
        Ok(df)
    }

    /// The text of `DESCRIBE EXTENDED` and `SHOW CREATE TABLE` is Spark 4.2's
    /// (`Created By`, `COLLATE UTF8_BINARY`); other profiles are refused rather
    /// than shown 4.2 output.
    fn require_current_profile(&self, what: &str) -> Result<(), KrishivError> {
        if self.state.profile == krishiv_plan::analyzer::SparkProfile::V4_2 {
            Ok(())
        } else {
            Err(unsupported(&format!("{what} under this Spark profile")))
        }
    }

    fn require_namespace(&self, namespace: &str) -> Result<(), KrishivError> {
        if self.state.catalog.store.has_namespace(namespace) {
            Ok(())
        } else {
            Err(SchemaNotFound {
                schema_name: format!("`{namespace}`"),
            }
            .into())
        }
    }

    /// The namespace a `SHOW ... IN name` names, else the current one.
    fn show_namespace(&self, name: Option<&DdlName>) -> Result<String, KrishivError> {
        let env = self.query_env();
        let ns = match name {
            Some(n) => self.namespace_name(n, &env)?,
            None => self.state.catalog.current_namespace(),
        };
        if !is_global_namespace(&ns) {
            self.require_namespace(&ns)?;
        }
        Ok(ns)
    }

    /// `SHOW ...` other than `SHOW CURRENT NAMESPACE`.
    pub(crate) async fn exec_show_catalog(
        &self,
        show: &Show,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let store = &self.state.catalog.store;
        match &show.kind {
            ShowKind::Tables {
                namespace, pattern, ..
            } => {
                let ns = self.show_namespace(namespace.as_ref())?;
                if is_global_namespace(&ns) {
                    return self
                        .global_views_frame(
                            &["namespace", "tableName", "isTemporary"],
                            pattern.as_ref(),
                        )
                        .await;
                }
                let mut rows: Vec<Vec<Cell>> = store
                    .tables_in(&ns)
                    .iter()
                    .filter(|t| keep(pattern.as_ref(), &t.name))
                    .map(|t| vec![s(&ns), s(&t.name), s("false")])
                    .collect();
                rows.extend(
                    self.temp_view_names(pattern.as_ref())
                        .into_iter()
                        .map(|n| vec![s(""), s(n), s("true")]),
                );
                self.ordered_frame(
                    &["namespace", "tableName", "isTemporary"],
                    &["isTemporary"],
                    rows,
                )
                .await
            }
            ShowKind::Views {
                namespace, pattern, ..
            } => {
                let ns = self.show_namespace(namespace.as_ref())?;
                if is_global_namespace(&ns) {
                    return self
                        .global_views_frame(
                            &["namespace", "viewName", "isTemporary"],
                            pattern.as_ref(),
                        )
                        .await;
                }
                let mut rows: Vec<Vec<Cell>> = store
                    .tables_in(&ns)
                    .iter()
                    .filter(|t| t.view.is_some() && keep(pattern.as_ref(), &t.name))
                    .map(|t| vec![s(&ns), s(&t.name), s("false")])
                    .collect();
                rows.extend(
                    self.temp_view_names(pattern.as_ref())
                        .into_iter()
                        .map(|n| vec![s(""), s(n), s("true")]),
                );
                self.ordered_frame(
                    &["namespace", "viewName", "isTemporary"],
                    &["isTemporary"],
                    rows,
                )
                .await
            }
            ShowKind::Columns {
                table, namespace, ..
            } => {
                let env = self.query_env();
                let mut name = self.table_name(table, &env)?;
                if let Some(ns) = namespace {
                    let from = self.namespace_name(ns, &env)?;
                    if !name.bare && !name.namespace.eq_ignore_ascii_case(&from) {
                        return Err(ShowColumnsWithConflictNamespace {
                            namespace_a: from,
                            namespace_b: name.namespace,
                        }
                        .into());
                    }
                    name.namespace = from;
                    name.bare = false;
                }
                let columns = self.columns_of_relation(&name).await?;
                let rows = columns.into_iter().map(|c| vec![s(c.0)]).collect();
                self.frame_with_order(&["col_name"], &[], rows, false).await
            }
            ShowKind::Namespaces { pattern, .. } => {
                let rows = store
                    .namespaces()
                    .into_iter()
                    .filter(|(n, _)| keep(pattern.as_ref(), n))
                    .map(|(n, _)| vec![s(n)])
                    .collect();
                self.ordered_frame(&["namespace"], &[], rows).await
            }
            ShowKind::TblProperties { name, key, .. } => {
                let env = self.query_env();
                let name = self.table_name(name, &env)?;
                if self.temp_entry_of(&name, &env) {
                    // A temporary view has no properties.
                    return self.ordered_frame(&["key", "value"], &[], Vec::new()).await;
                }
                let table = self.stored_relation(&name)?;
                if table.view.is_some() {
                    self.authorize_view_select(&name.namespace, &name.name, &env)
                        .await?;
                }
                let rows = match key {
                    Some(k) => vec![vec![
                        s(k),
                        s(table.properties.get(k).map_or_else(
                            || {
                                format!(
                                    "Table spark_catalog.{}.{} does not have property: {k}",
                                    table.namespace, table.name
                                )
                            },
                            |v| redact(k, v).to_owned(),
                        )),
                    ]],
                    None => table
                        .properties
                        .iter()
                        .map(|(k, v)| vec![s(k), s(redact(k, v))])
                        .collect(),
                };
                self.ordered_frame(&["key", "value"], &[], rows).await
            }
            ShowKind::CreateTable { name, serde, .. } => {
                if *serde {
                    return Err(unsupported("SHOW CREATE TABLE ... AS SERDE"));
                }
                let env = self.query_env();
                let name = self.table_name(name, &env)?;
                self.require_current_profile("SHOW CREATE TABLE")?;
                if self.temp_entry_of(&name, &env) {
                    return Err(ExpectPermanentViewNotTemp {
                        operation: "SHOW CREATE TABLE".to_owned(),
                        view_name: format!("`{}`", name.name),
                    }
                    .into());
                }
                let table = self.stored_relation(&name)?;
                if table.view.is_some() {
                    self.authorize_view_select(&name.namespace, &name.name, &env)
                        .await?;
                }
                self.ordered_frame(
                    &["createtab_stmt"],
                    &[],
                    vec![vec![s(create_statement(
                        &table,
                        crate::view_schema::binding_enabled(&self.state.conf.snapshot()),
                    )?)]],
                )
                .await
            }
            ShowKind::Partitions {
                name, partition, ..
            } => {
                let env = self.query_env();
                let name = self.table_name(name, &env)?;
                if self.temp_entry_of(&name, &env) {
                    return Err(unsupported("SHOW PARTITIONS on a temporary view"));
                }
                let table = self.stored_relation(&name)?;
                if table.view.is_some() {
                    return Err(unsupported("SHOW PARTITIONS on a view"));
                }
                self.show_partitions(&table, partition.as_ref(), original)
                    .await
            }
            _ => Err(unsupported("this SHOW command")),
        }
    }

    /// Temporary views matching `pattern`, by name.
    fn temp_view_names(&self, pattern: Option<&ShowPattern>) -> Vec<String> {
        let mut names: Vec<String> = self
            .state
            .catalog
            .names()
            .into_iter()
            .filter(|n| keep(pattern, n))
            .collect();
        names.sort();
        names
    }

    /// The stored table or view `name` names (temporary views have no stored form).
    fn stored_relation(&self, name: &TableName) -> Result<Arc<CatalogTable>, KrishivError> {
        self.state
            .catalog
            .store
            .table(&name.namespace, &name.name)
            .ok_or_else(|| name.not_found())
    }

    /// The (name, type, comment) of the columns of a table, view or temporary view.
    async fn columns_of_relation(
        &self,
        name: &TableName,
    ) -> Result<Vec<(String, DataType, Option<String>)>, KrishivError> {
        let env = self.query_env();
        if !name.bare
            && name
                .namespace
                .eq_ignore_ascii_case(crate::catalog::GLOBAL_NAMESPACE)
            && self.state.catalog.exists(
                &crate::catalog::global_key(&name.name),
                env.analysis.conf.case_sensitive,
            )
        {
            // The view's dependencies may have changed since it was created: ask the analyzer.
            let plan = Arc::new(krishiv_plan::LogicalPlan::table([
                crate::catalog::GLOBAL_NAMESPACE,
                name.name.as_str(),
            ]));
            let df = DataFrame::new(self.clone(), plan);
            let analysis = df.analyzed_in(env).await?;
            return Ok(analysis
                .analyzed
                .output
                .iter()
                .map(|c| (c.name.clone(), c.data_type.clone(), None))
                .collect());
        }
        if self.temp_entry_of(name, &env) {
            let df = self.table(&name.name);
            let analysis = df.analyzed_in(env).await?;
            return Ok(analysis
                .analyzed
                .output
                .iter()
                .map(|c| (c.name.clone(), c.data_type.clone(), None))
                .collect());
        }
        let table = self.stored_relation(name)?;
        if table.view.is_some() {
            self.authorize_view_select(&name.namespace, &name.name, &env)
                .await?;
        }
        Ok(table
            .columns
            .iter()
            .map(|c| (c.name.clone(), c.data_type.clone(), c.comment.clone()))
            .collect())
    }

    /// `DESCRIBE ...`.
    pub(crate) async fn exec_describe(
        &self,
        d: &Describe,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        match &d.kind {
            DescribeKind::Table {
                mode,
                table,
                partition,
                column,
                as_json,
                ..
            } => {
                let name = self.table_name(table, &env)?;
                let columns = self.columns_of_relation(&name).await?;
                let temp = self.temp_entry_of(&name, &env)
                    || (!name.bare
                        && name
                            .namespace
                            .eq_ignore_ascii_case(crate::catalog::GLOBAL_NAMESPACE)
                        && self.state.catalog.exists(
                            &crate::catalog::global_key(&name.name),
                            env.analysis.conf.case_sensitive,
                        ));
                let described = match partition {
                    Some(spec) => Some(
                        self.described_partition(&name, temp, spec, original)
                            .await?,
                    ),
                    None => None,
                };
                if *as_json {
                    if temp {
                        return Err(unsupported("DESCRIBE ... AS JSON of a temporary view"));
                    }
                    self.require_current_profile("DESCRIBE EXTENDED")?;
                    let t = self.stored_relation(&name)?;
                    let json = describe_json(&t, described.as_ref())?;
                    return self
                        .ordered_frame(&["json_metadata"], &[], vec![vec![s(json)]])
                        .await;
                }
                if !column.is_empty() {
                    if mode.is_some() {
                        // Spark adds statistics rows; the engine keeps none.
                        return Err(unsupported("DESCRIBE EXTENDED | FORMATTED ... column"));
                    }
                    return self.describe_column(&columns, column, false).await;
                }
                let mut rows: Vec<Vec<Cell>> = columns
                    .iter()
                    .map(|(n, t, c)| {
                        vec![
                            s(n),
                            s(t.catalog_string()),
                            s(c.clone().unwrap_or_default()),
                        ]
                    })
                    .collect();
                // A temporary view has no stored form: only its columns.
                let stored = if temp {
                    None
                } else {
                    Some(self.stored_relation(&name)?)
                };
                if let Some(t) = stored.as_ref().filter(|t| t.is_partitioned()) {
                    rows.push(vec![s("# Partition Information"), s(""), s("")]);
                    rows.push(vec![s("# col_name"), s("data_type"), s("comment")]);
                    rows.extend(t.partition_schema().into_iter().map(|c| {
                        vec![
                            s(&c.name),
                            s(c.data_type.catalog_string()),
                            s(c.comment.clone().unwrap_or_default()),
                        ]
                    }));
                }
                if let (Some(t), Some(p)) = (stored.as_ref(), described.as_ref()) {
                    if mode.is_some() {
                        self.require_current_profile("DESCRIBE EXTENDED")?;
                        rows.extend(
                            partition_details(t, p)
                                .into_iter()
                                .map(|(k, v)| vec![s(k), s(v), s("")]),
                        );
                    }
                } else if mode.is_some() && stored.is_some() {
                    rows.push(vec![s(""), s(""), s("")]);
                    rows.push(vec![s("# Detailed Table Information"), s(""), s("")]);
                    let info = self.detailed_information(&name)?;
                    rows.extend(info.into_iter().map(|(k, v)| vec![s(k), s(v), s("")]));
                }
                self.ordered_frame(&["col_name", "data_type", "comment"], &[], rows)
                    .await
            }
            DescribeKind::Query { query, .. } => {
                let plan = krishiv_sql::lower(query, self.next_source_id()).map_err(|e| e.error)?;
                let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
                let analysis = df.analyzed_in(env).await?;
                let rows = analysis
                    .analyzed
                    .output
                    .iter()
                    .map(|c| vec![s(&c.name), s(c.data_type.catalog_string()), None])
                    .collect();
                self.ordered_frame(&["col_name", "data_type", "comment"], &[], rows)
                    .await
            }
            DescribeKind::Namespace { extended, name, .. } => {
                let ns = self.namespace_name(name, &env)?;
                let meta = self
                    .state
                    .catalog
                    .store
                    .namespaces()
                    .into_iter()
                    .find(|(n, _)| *n == ns)
                    .map(|(_, m)| m)
                    .ok_or_else(|| {
                        KrishivError::from(SchemaNotFound {
                            schema_name: format!("`{ns}`"),
                        })
                    })?;
                let mut rows = self.namespace_rows(&ns, &meta)?;
                if *extended {
                    let props = meta
                        .properties
                        .iter()
                        .map(|(k, v)| format!("{k}={}", redact(k, v)))
                        .collect::<Vec<_>>()
                        .join(", ");
                    rows.push(vec![s("Properties"), s(format!("({props})"))]);
                }
                self.ordered_frame(&["info_name", "info_value"], &[], rows)
                    .await
            }
            _ => Err(unsupported("DESCRIBE FUNCTION")),
        }
    }

    fn namespace_rows(
        &self,
        ns: &str,
        meta: &NamespaceMeta,
    ) -> Result<Vec<Vec<Cell>>, KrishivError> {
        let location = match &meta.location {
            Some(l) => l.clone(),
            None => {
                let wh = self.warehouse_dir()?;
                let dir = if ns == crate::store::DEFAULT_NAMESPACE {
                    wh
                } else {
                    wh.join(format!("{ns}.db"))
                };
                format!("file:{}", dir.display())
            }
        };
        Ok(vec![
            vec![s("Catalog Name"), s(crate::catalog::CATALOG)],
            vec![s("Namespace Name"), s(ns)],
            vec![s("Comment"), s(meta.comment.clone().unwrap_or_default())],
            vec![s("Location"), s(location)],
            vec![s("Owner"), s("spark")],
        ])
    }

    /// One column's `info_name`/`info_value` rows.
    async fn describe_column(
        &self,
        columns: &[(String, DataType, Option<String>)],
        column: &[krishiv_sql::ast::ColumnName],
        _extended: bool,
    ) -> Result<DataFrame, KrishivError> {
        let wanted = match column {
            [krishiv_sql::ast::ColumnName::Ident(i)] => i.value.clone(),
            _ => return Err(unsupported("DESCRIBE of a nested column")),
        };
        let found = columns.iter().find(|c| c.0.eq_ignore_ascii_case(&wanted));
        let Some((name, data_type, comment)) = found else {
            return Err(UnresolvedColumnWithSuggestion {
                object_name: format!("`{wanted}`"),
                proposal: columns
                    .iter()
                    .map(|c| format!("`{}`", c.0))
                    .collect::<Vec<_>>()
                    .join(", "),
            }
            .into());
        };
        let rows = vec![
            vec![s("col_name"), s(name)],
            vec![s("data_type"), s(data_type.catalog_string())],
            vec![
                s("comment"),
                s(comment.clone().unwrap_or_else(|| "NULL".to_owned())),
            ],
        ];
        self.ordered_frame(&["info_name", "info_value"], &[], rows)
            .await
    }

    /// The `# Detailed Table Information` rows of `DESCRIBE EXTENDED|FORMATTED`.
    fn detailed_information(
        &self,
        name: &TableName,
    ) -> Result<Vec<(String, String)>, KrishivError> {
        self.require_current_profile("DESCRIBE EXTENDED")?;
        let t = self.stored_relation(name)?;
        let mut rows = vec![
            ("Catalog".to_owned(), crate::catalog::CATALOG.to_owned()),
            ("Database".to_owned(), t.namespace.clone()),
            ("Table".to_owned(), t.name.clone()),
            ("Created Time".to_owned(), created_time(t.created_ms)),
            ("Last Access".to_owned(), "UNKNOWN".to_owned()),
            ("Created By".to_owned(), "Spark 4.2.0".to_owned()),
        ];
        if let Some(view) = &t.view {
            rows.push(("Type".to_owned(), "VIEW".to_owned()));
            if let Some(c) = &t.comment {
                rows.push(("Comment".to_owned(), c.clone()));
            }
            rows.push(("View Text".to_owned(), view.sql.clone()));
            rows.push(("View Original Text".to_owned(), view.sql.clone()));
            let enabled = crate::view_schema::binding_enabled(&self.state.conf.snapshot());
            let mode = crate::view_schema::effective_mode(view.schema_mode, enabled);
            if let Some(mode) = mode {
                rows.push(("View Schema Mode".to_owned(), mode.as_str().to_owned()));
            }
            rows.push((
                "View Catalog and Namespace".to_owned(),
                format!("{}.{}", crate::catalog::CATALOG, view.namespace),
            ));
            // `EVOLUTION` keeps no query names: the view's own stand for them.
            let names: Vec<&str> = if mode == Some(crate::store::ViewSchemaMode::Evolution)
                || view.query_columns.len() != t.columns.len()
            {
                t.columns.iter().map(|c| c.name.as_str()).collect()
            } else {
                view.query_columns.iter().map(String::as_str).collect()
            };
            rows.push((
                "View Query Output Columns".to_owned(),
                format!(
                    "[{}]",
                    names
                        .iter()
                        .map(|n| format!("`{}`", n.replace('`', "``")))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        } else {
            rows.push((
                "Type".to_owned(),
                if t.managed { "MANAGED" } else { "EXTERNAL" }.to_owned(),
            ));
            rows.push(("Provider".to_owned(), t.provider.clone()));
            rows.extend(bucket_rows(&t));
            if let Some(c) = &t.comment {
                rows.push(("Comment".to_owned(), c.clone()));
            }
        }
        if !t.properties.is_empty() {
            rows.push((
                "Table Properties".to_owned(),
                format!(
                    "[{}]",
                    t.properties
                        .iter()
                        .map(|(k, v)| format!("{k}={}", redact(k, v)))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        }
        if t.view.is_none() {
            rows.extend(storage_rows(&t, &t.location));
            if t.is_partitioned() {
                rows.push(("Partition Provider".to_owned(), "Catalog".to_owned()));
            }
        }
        Ok(rows)
    }
}

/// A JSON string literal.
fn json_str(v: &str) -> String {
    serde_json::Value::String(v.to_owned()).to_string()
}

/// A JSON array of strings.
fn json_list<'a>(items: impl IntoIterator<Item = &'a str>) -> String {
    format!(
        "[{}]",
        items
            .into_iter()
            .map(json_str)
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// A JSON object of string values, keys in the order given, values redacted as `DESCRIBE` redacts them.
fn json_map<'a>(items: impl IntoIterator<Item = (&'a String, &'a String)>) -> String {
    format!(
        "{{{}}}",
        items
            .into_iter()
            .map(|(k, v)| format!("{}:{}", json_str(k), json_str(redact(k, v))))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// A column type as `DESCRIBE ... AS JSON` writes it (`DescribeTableJsonCommand`'s type objects).
fn json_type(t: &DataType) -> Result<String, KrishivError> {
    let name = match t {
        DataType::Boolean => "boolean",
        DataType::Byte => "byte",
        DataType::Short => "short",
        DataType::Integer => "int",
        DataType::Long => "long",
        DataType::Float => "float",
        DataType::Double => "double",
        DataType::Binary => "binary",
        DataType::Date => "date",
        DataType::Timestamp => "timestamp_ltz",
        DataType::TimestampNtz => "timestamp_ntz",
        DataType::String => return Ok(r#"{"name":"string","collation":"UTF8_BINARY"}"#.to_owned()),
        DataType::Decimal(d) => {
            return Ok(format!(
                r#"{{"name":"decimal","precision":{},"scale":{}}}"#,
                d.precision(),
                d.scale()
            ));
        }
        DataType::Array(_) | DataType::Map(_) | DataType::Struct(_) => {
            return Err(unsupported("DESCRIBE ... AS JSON of a nested column type"));
        }
        other => {
            return Ok(format!(
                r#"{{"name":{}}}"#,
                json_str(&other.catalog_string())
            ));
        }
    };
    Ok(format!(r#"{{"name":"{name}"}}"#))
}

/// `DESCRIBE EXTENDED [... PARTITION] ... AS JSON` of a stored table: one JSON object whose keys come in
/// the order Spark's `DescribeTableJsonCommand` writes them (a partition's differ: its values and location
/// come first, its statistics follow `last_access`, and the bucket keys sit after `provider`, before the comment).
fn describe_json(t: &CatalogTable, p: Option<&CatalogPartition>) -> Result<String, KrishivError> {
    if t.view.is_some() {
        return Err(unsupported("DESCRIBE ... AS JSON of a view"));
    }
    let columns = t
        .columns
        .iter()
        .map(|c| {
            let mut col = format!(
                r#"{{"name":{},"type":{},"nullable":{}"#,
                json_str(&c.name),
                json_type(&c.data_type)?,
                c.nullable
            );
            if let Some(comment) = &c.comment {
                col.push_str(&format!(r#","comment":{}"#, json_str(comment)));
            }
            if let Some(default) = &c.default {
                col.push_str(&format!(r#","default":{}"#, json_str(default)));
            }
            col.push('}');
            Ok(col)
        })
        .collect::<Result<Vec<_>, KrishivError>>()?;
    let mut f: Vec<String> = vec![
        format!(r#""table_name":{}"#, json_str(&t.name)),
        format!(r#""catalog_name":{}"#, json_str(crate::catalog::CATALOG)),
        format!(r#""namespace":{}"#, json_list([t.namespace.as_str()])),
        format!(r#""schema_name":{}"#, json_str(&t.namespace)),
        format!(r#""columns":[{}]"#, columns.join(",")),
    ];
    let buckets = t.layout.bucket.as_ref().map(|b| {
        [
            format!(r#""num_buckets":{}"#, b.num_buckets),
            format!(
                r#""bucket_columns":{}"#,
                json_list(b.columns.iter().map(String::as_str))
            ),
            format!(
                r#""sort_columns":{}"#,
                json_list(b.sort_columns.iter().map(String::as_str))
            ),
        ]
    });
    let location = match p {
        Some(p) => {
            let values = p
                .spec
                .iter()
                .map(|(k, v)| format!("{}:{}", json_str(k), json_str(v)))
                .collect::<Vec<_>>()
                .join(",");
            f.push(format!(r#""partition_values":{{{values}}}"#));
            &p.location
        }
        None => {
            f.extend(buckets.iter().flatten().cloned());
            &t.location
        }
    };
    f.push(format!(
        r#""location":{}"#,
        json_str(&format!("file:{}", location.display()))
    ));
    let options = &t.options;
    if !options.is_empty() {
        f.push(format!(
            r#""storage_properties":{}"#,
            json_map(options.iter())
        ));
    }
    if let Some(p) = p.filter(|p| !p.parameters.is_empty()) {
        f.push(format!(
            r#""partition_parameters":{}"#,
            json_map(p.parameters.iter())
        ));
    }
    let created = p.map_or(t.created_ms, |p| p.created_ms);
    f.push(format!(
        r#""created_time":{}"#,
        json_str(&created_time_iso(created))
    ));
    f.push(r#""last_access":"UNKNOWN""#.to_owned());
    if let Some(stats) = p.and_then(|p| p.stats) {
        f.push(format!(
            r#""partition_statistics":{}"#,
            json_str(&crate::partitions::stats_text(stats))
        ));
    }
    f.push(format!(r#""created_by":{}"#, json_str("Spark 4.2.0")));
    f.push(format!(
        r#""type":"{}""#,
        if t.managed { "MANAGED" } else { "EXTERNAL" }
    ));
    f.push(format!(r#""provider":{}"#, json_str(&t.provider)));
    if p.is_some() {
        f.extend(buckets.iter().flatten().cloned());
    }
    if let Some(c) = &t.comment {
        f.push(format!(r#""comment":{}"#, json_str(c)));
    }
    if !t.properties.is_empty() {
        f.push(format!(
            r#""table_properties":{}"#,
            json_map(t.properties.iter())
        ));
    }
    if t.is_partitioned() {
        f.push(r#""partition_provider":"Catalog""#.to_owned());
        f.push(format!(
            r#""partition_columns":{}"#,
            json_list(t.layout.partition_columns.iter().map(String::as_str))
        ));
    }
    f.push(r#""collation":"UTF8_BINARY""#.to_owned());
    Ok(format!("{{{}}}", f.join(",")))
}

/// `Num Buckets`, `Bucket Columns` and `Sort Columns` of a bucketed table (`BucketSpec.toLinkedHashMap`).
fn bucket_rows(t: &CatalogTable) -> Vec<(String, String)> {
    let Some(b) = &t.layout.bucket else {
        return Vec::new();
    };
    let list = |names: &[String]| {
        format!(
            "[{}]",
            names
                .iter()
                .map(|n| format!("`{}`", n.replace('`', "``")))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    vec![
        ("Num Buckets".to_owned(), b.num_buckets.to_string()),
        ("Bucket Columns".to_owned(), list(&b.columns)),
        ("Sort Columns".to_owned(), list(&b.sort_columns)),
    ]
}

/// `Location` and `Storage Properties` (`CatalogStorageFormat.toLinkedHashMap`) at `location`.
fn storage_rows(t: &CatalogTable, location: &std::path::Path) -> Vec<(String, String)> {
    let mut rows = vec![(
        "Location".to_owned(),
        format!("file:{}", location.display()),
    )];
    let options = &t.options;
    if !options.is_empty() {
        rows.push((
            "Storage Properties".to_owned(),
            format!(
                "[{}]",
                options
                    .iter()
                    .map(|(k, v)| format!("{k}={}", redact(k, v)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    }
    rows
}

/// The rows `DESCRIBE EXTENDED ... PARTITION` adds (`describeFormattedDetailedPartitionInfo`).
fn partition_details(t: &CatalogTable, p: &CatalogPartition) -> Vec<(String, String)> {
    let mut rows = vec![
        (String::new(), String::new()),
        ("# Detailed Partition Information".to_owned(), String::new()),
        ("Database".to_owned(), t.namespace.clone()),
        ("Table".to_owned(), t.name.clone()),
        (
            "Partition Values".to_owned(),
            format!(
                "[{}]",
                p.spec
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ),
    ];
    rows.extend(storage_rows(t, &p.location));
    if !p.parameters.is_empty() {
        rows.push((
            "Partition Parameters".to_owned(),
            format!(
                "[{}]",
                p.parameters
                    .iter()
                    .map(|(k, v)| format!("{k}={}", redact(k, v)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    }
    rows.push(("Created Time".to_owned(), created_time(p.created_ms)));
    rows.push(("Last Access".to_owned(), "UNKNOWN".to_owned()));
    if let Some(stats) = p.stats {
        rows.push((
            "Partition Statistics".to_owned(),
            crate::partitions::stats_text(stats),
        ));
    }
    rows.push((String::new(), String::new()));
    rows.push(("# Storage Information".to_owned(), String::new()));
    rows.extend(bucket_rows(t));
    rows.extend(storage_rows(t, &t.location));
    rows
}

/// `SHOW CREATE TABLE` text of a table or view, in Spark's layout; `binding_enabled` is
/// `spark.sql.legacy.viewSchemaBindingMode` (a view's `WITH SCHEMA` line).
fn create_statement(t: &CatalogTable, binding_enabled: bool) -> Result<String, KrishivError> {
    let column = |c: &crate::store::StoreColumn| -> Result<String, KrishivError> {
        let mut out = format!("  {} {}", quote_if_needed(&c.name), type_sql(&c.data_type)?);
        if !c.nullable {
            out.push_str(" NOT NULL");
        }
        if c.data_type == DataType::String {
            out.push_str(" COLLATE UTF8_BINARY");
        }
        if let Some(default) = &c.default {
            out.push_str(&format!(" DEFAULT {default}"));
        }
        if let Some(comment) = &c.comment {
            out.push_str(&format!(" COMMENT {}", sql_literal(comment)));
        }
        Ok(out)
    };
    let name = format!("spark_catalog.{}.{}", t.namespace, t.name);
    if let Some(view) = &t.view {
        let cols = t
            .columns
            .iter()
            .map(|c| match &c.comment {
                Some(m) => format!("  {} COMMENT {}", quote_if_needed(&c.name), sql_literal(m)),
                None => format!("  {}", quote_if_needed(&c.name)),
            })
            .collect::<Vec<_>>()
            .join(",\n");
        let mut out = format!("CREATE VIEW {}.{} (\n{cols})\n", t.namespace, t.name);
        if let Some(c) = &t.comment {
            out.push_str(&format!("COMMENT {}\n", sql_literal(c)));
        }
        if let Some(mode) = crate::view_schema::effective_mode(view.schema_mode, binding_enabled) {
            out.push_str(&format!("WITH SCHEMA {}\n", mode.as_str()));
        }
        out.push_str("AS ");
        out.push_str(&view.sql);
        out.push('\n');
        return Ok(out);
    }
    let cols = t
        .columns
        .iter()
        .map(column)
        .collect::<Result<Vec<_>, _>>()?
        .join(",\n");
    let mut out = format!("CREATE TABLE {name} (\n{cols})\nUSING {}\n", t.provider);
    let partitioning = |out: &mut String| {
        if t.is_partitioned() {
            out.push_str(&format!(
                "PARTITIONED BY ({})\n",
                t.layout
                    .partition_columns
                    .iter()
                    .map(|c| quote_if_needed(c))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if let Some(b) = &t.layout.bucket {
            out.push_str(&format!("CLUSTERED BY ({})\n", b.columns.join(", ")));
            if !b.sort_columns.is_empty() {
                out.push_str(&format!("SORTED BY ({})\n", b.sort_columns.join(", ")));
            }
            out.push_str(&format!("INTO {} BUCKETS\n", b.num_buckets));
        }
    };
    let pairs = |m: &std::collections::BTreeMap<String, String>| {
        m.iter()
            .map(|(k, v)| format!("  {} = {}", sql_literal(k), sql_literal(redact(k, v))))
            .collect::<Vec<_>>()
            .join(",\n")
    };
    let options = &t.options;
    if !options.is_empty() {
        out.push_str(&format!("OPTIONS (\n{})\n", pairs(options)));
    }
    partitioning(&mut out);
    if let Some(c) = &t.comment {
        out.push_str(&format!("COMMENT {}\n", sql_literal(c)));
    }
    if !t.managed {
        out.push_str(&format!(
            "LOCATION {}\n",
            sql_literal(&format!("file:{}", t.location.display()))
        ));
    }
    if !t.properties.is_empty() {
        out.push_str(&format!("TBLPROPERTIES (\n{})\n", pairs(&t.properties)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns_follow_spark() {
        assert!(matches("show_t*", "SHOW_T1"));
        assert!(matches("a|b*", "bcd"));
        assert!(!matches("a|b*", "cd"));
        assert!(matches("*", ""));
    }

    #[test]
    fn creation_times_print_like_spark() {
        assert_eq!(created_time(0), "Thu Jan 01 00:00:00 UTC 1970");
        assert_eq!(
            created_time(1_793_330_012_000),
            "Fri Oct 30 03:13:32 UTC 2026"
        );
    }
}
