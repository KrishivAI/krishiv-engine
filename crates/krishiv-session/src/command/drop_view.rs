// SPDX-License-Identifier: Apache-2.0
//! `DROP VIEW name` without a namespace: a temporary view of the session, or else the stored view of
//! the current namespace, dropped as `DROP VIEW <namespace>.<name>` with that statement's class,
//! access check and ownership change.

use krishiv_common::error::KrishivError;
use krishiv_sql::StatementClass;
use krishiv_sql::ast::{DdlName, DropPermanentView, DropView, ObjectName, Statement};

use crate::catalog::is_global_key;
use crate::dataframe::DataFrame;
use crate::session::Session;

impl Session {
    /// Runs `DROP VIEW [IF EXISTS] name` over the temporary views,
    /// returning an empty frame as in Spark.
    pub(crate) async fn exec_drop_view(
        &self,
        v: &DropView,
        env: &crate::session::QueryEnv,
    ) -> Result<DataFrame, KrishivError> {
        let Some(stored) = self.stored_drop_view(v, env) else {
            self.state.catalog.drop_table(&v.name.value);
            self.state.cache.forget_table(&v.name.value);
            return self.empty_frame().await;
        };
        // The stored view of the bare name: the checks and ownership change of
        // `DROP VIEW <current namespace>.<name>`, which `statement_targets` cannot give a
        // statement that may name a temporary view. The name is placed first, so a current
        // namespace that holds no stored views (`system`) keeps its `SCHEMA_NOT_FOUND`.
        self.stored_namespace_in(env)?;
        self.authorize_statement(&stored, StatementClass::Ddl, env)
            .await?;
        let drop = async { self.drop_bare_stored_view(&v.name.value, v.if_exists, env) };
        match self.owner_change(&stored, env) {
            Some(change) => self.with_owner_change(change, env, drop).await?,
            None => drop.await?,
        }
        self.empty_frame().await
    }

    /// The class `statement` runs under: `krishiv_sql::classify_with`, except that a bare
    /// `DROP VIEW name` of a stored view is `DDL`, the class of the qualified drop it runs as
    /// ([`Self::stored_drop_view`]), for its access check, the drain refusal, `system.queries`
    /// and the analyze API alike. Where the current namespace holds no stored views (`system`)
    /// the statement keeps its class and fails as before.
    pub(crate) fn statement_class(
        &self,
        statement: &Statement,
        env: &crate::session::QueryEnv,
    ) -> StatementClass {
        if let Statement::DropView(v) = statement
            && self.stored_drop_view(v, env).is_some()
            && self.stored_namespace_in(env).is_ok()
        {
            return StatementClass::Ddl;
        }
        krishiv_sql::classify_with(statement, &env.parser)
    }

    /// The `DROP VIEW <current namespace>.<name>` that `DROP VIEW name` stands for when `name`
    /// names no temporary view of the session (`None` when it does: that view is dropped).
    pub(crate) fn stored_drop_view(
        &self,
        v: &DropView,
        env: &crate::session::QueryEnv,
    ) -> Option<Statement> {
        if !is_global_key(&v.name.value)
            && self
                .state
                .catalog
                .exists(&v.name.value, env.analysis.conf.case_sensitive)
        {
            return None;
        }
        Some(Statement::DropPermanentView(DropPermanentView {
            if_exists: v.if_exists,
            name: DdlName::Parts(ObjectName {
                parts: vec![v.name.clone()],
                span: v.name.span,
            }),
            span: v.span,
        }))
    }
}
