// SPDX-License-Identifier: Apache-2.0
//! Statement classification for policy consumers: what a statement is
//! ([`StatementClass`], syntactic) and what it reads (its securables with
//! the columns used, from analysis).

use std::sync::Arc;

use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_plan::SourceId;
use krishiv_plan::analyzer::SecurableUse;
use krishiv_sql::{ExplainKind, StatementClass};

use crate::dataframe::DataFrame;
use crate::session::{Session, sql_context};

/// What a statement is and what it reads: the input of a read-only
/// surface's allow-list (over [`Self::class`]) and of the analyze API.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StatementInfo {
    /// The statement's class (syntactic; `EXPLAIN ANALYZE` takes the class
    /// of the statement it runs).
    pub class: StatementClass,
    /// How it explains, if it is an `EXPLAIN`.
    pub explain: Option<ExplainKind>,
    /// What its queries read, with the columns they reference, in
    /// first-use order (temporary views expanded: a view is read with the
    /// rights of the session that reads it).
    pub securables: Vec<SecurableUse>,
}

impl Session {
    /// Classifies the SQL statement `sql` and lists what it reads, without
    /// running it. Reads are found by analysis, which asks the session's
    /// access policy about each of them: a denied read fails here as it
    /// would when the statement runs.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_schema::{DataType, Field, Schema};
    /// use krishiv_session::{Session, StatementClass};
    ///
    /// let s = Session::new()?;
    /// # s.register_batches("t", Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)])), vec![])?;
    /// let info = futures::executor::block_on(s.statement_info("SELECT a FROM t"))?;
    /// assert_eq!(info.class, StatementClass::Read);
    /// assert_eq!(info.securables.len(), 1);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Parse and analysis errors (`KRISHIV_INSUFFICIENT_PERMISSIONS` for a
    /// denied read); `KRISHIV_UNSUPPORTED_FEATURE` for a statement whose
    /// reads are not whole queries (a multi-table `INSERT` with a shared
    /// `FROM`, reads through `UPDATE`/`DELETE` predicates or variable
    /// expressions).
    pub async fn statement_info(&self, sql: &str) -> Result<StatementInfo, KrishivError> {
        let (session, sql) = (self.clone(), sql.to_owned());
        self.on_executor(async move { session.statement_info_here(&sql).await })
            .await
    }

    async fn statement_info_here(&self, sql: &str) -> Result<StatementInfo, KrishivError> {
        let env = self.query_env();
        let redacted = krishiv_sql::redact_sql(sql);
        let statement = krishiv_sql::parse_statement(sql, &env.parser)
            .map_err(|e| sql_context(e.error, &redacted, e.span.start, e.span.end))?;
        let class = krishiv_sql::classify(&statement);
        let explain = krishiv_sql::explain_kind(&statement);
        let queries = krishiv_sql::statement_queries(&statement).ok_or_else(|| {
            KrishivError::from(KrishivUnsupportedFeature {
                feature: "listing the reads of this statement".to_owned(),
            })
        })?;
        let mut securables: Vec<SecurableUse> = Vec::new();
        for query in queries {
            let plan = krishiv_sql::lower(query, SourceId(0))
                .map_err(|e| sql_context(e.error, &redacted, e.span.start, e.span.end))?;
            let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
            let analysis = df
                .analyzed_in_with_text(self.query_env(), Some(&redacted))
                .await?;
            for used in &analysis.analyzed.securables {
                match securables
                    .iter_mut()
                    .find(|s| s.securable == used.securable && s.privilege == used.privilege)
                {
                    Some(seen) => seen.columns.extend(used.columns.iter().cloned()),
                    None => securables.push(used.clone()),
                }
            }
        }
        Ok(StatementInfo {
            class,
            explain,
            securables,
        })
    }
}
