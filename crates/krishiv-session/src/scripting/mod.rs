// SPDX-License-Identifier: Apache-2.0
//! The SQL script interpreter: `BEGIN ... END` with local variables, assignment, `IF`, `CASE`,
//! `WHILE`, `REPEAT`, `LOOP`, `LEAVE` and `ITERATE` (design note T196, unit 2b).
//!
//! A script runs eagerly, statement by statement, through the ordinary statement path: each leaf
//! is classified, authorized and run like the same statement typed alone, on a session *handle*
//! that carries the script's local frames ([`Session::with_locals`]). Locals reach the analyzer as
//! typed literals, so plans and the executor are unchanged. The script's result is the rows of the
//! last query statement that ran.
//!
//! `FOR` loops run their query once and give each row's columns to the body as locals. `EXIT`
//! handlers catch the errors raised while their compound runs (never `KRISHIV_*` errors or
//! cancellation), then end that compound.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use krishiv_common::error::{
    DuplicateAssignments, DuplicateVariableNameInsideDeclare, InvalidBooleanStatement,
    InvalidDefaultValueSubqueryExpression, InvalidVariableDeclarationQualifiedLocalVariable,
    InvalidVariableDeclarationReplaceLocalVariable, KrishivError, KrishivUnsupportedFeature,
    RowSubqueryTooManyRows, ScalarSubqueryTooManyRows,
    UnsupportedFeatureSqlScriptingDropTemporaryVariable, VariableAlreadyExists, VariableNotFound,
};
use krishiv_common::redact::RedactedSql;
use krishiv_plan::analyzer::{LocalFrame, SessionVariable};
use krishiv_plan::{Expr as PlanExpr, ExprNode, Literal, LogicalPlan, PlanNode};
use krishiv_sql::ast::{
    Compound, DataTypeSpec, DeclareVariable, Expr, ForStatement, Handler, HandlerBody,
    HandlerCondition, ObjectName, ScriptStatement, ScriptStatement as S, SetVariable,
    SetVariableKind, Statement,
};
use krishiv_types::DataType;

use crate::column::Column;
use crate::command::{
    arrow_value, contains_default, contains_subquery, is_bare_default, padded_select,
    unresolved_variable,
};
use crate::dataframe::DataFrame;
use crate::session::{Session, internal};
use crate::variables::{RESOLVE_SEARCH_PATH, StoredVariable, quote_normalized};

/// The most rows a script's result may have. The result is returned as literal rows, which keeps
/// it independent of the script's session; a larger one is refused rather than truncated.
const MAX_RESULT_ROWS: i32 = 256;

/// The most rows a `FOR` loop holds. Spark reads them lazily; this engine holds the query's rows for
/// the whole loop, so a larger one is refused rather than risking the process's memory.
const MAX_FOR_ROWS: usize = 100_000;

/// What a statement tells the statements around it to do next.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Flow {
    /// Continue with the next statement.
    Normal,
    /// `LEAVE label` is on its way to the labelled compound or loop.
    Leave(String),
    /// `ITERATE label` is on its way to the labelled loop.
    Iterate(String),
}

/// A local variable: its value and the query that re-evaluates its default.
struct Local {
    variable: SessionVariable,
    default_sql: Option<String>,
}

/// The variables of one `BEGIN ... END` body.
struct Scope {
    label: Option<String>,
    vars: BTreeMap<String, Local>,
    /// The `EXIT` handlers declared in the compound, in declaration order.
    handlers: Vec<Handler>,
}

/// Where an assignment goes.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Target {
    Local { scope: usize, key: String },
    Session { key: String },
}

/// One running script.
struct Run<'a> {
    session: &'a Session,
    /// The text of the whole script: statement spans are positions in it.
    text: &'a str,
    redacted: &'a RedactedSql,
    scopes: Vec<Scope>,
    /// The rows of the last query statement that ran.
    last: Option<(SchemaRef, Vec<RecordBatch>)>,
}

impl Session {
    /// Runs the script `c` (parsed from `text`) and returns the rows of its last query statement,
    /// or an empty result when no statement produced rows.
    pub(crate) async fn exec_script(
        &self,
        c: &Compound,
        text: &str,
        redacted: &RedactedSql,
    ) -> Result<DataFrame, KrishivError> {
        let mut run = Run {
            session: self,
            text,
            redacted,
            scopes: Vec::new(),
            last: None,
        };
        run.compound(c).await?;
        run.finish().await
    }
}

fn unsupported(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}

impl Run<'_> {
    /// A name as the frames key it: in lower case, whatever `spark.sql.caseSensitive` says, as
    /// `VariableSnapshot::get` looks it up (a divergence when that is on: `a` and `A` are one
    /// local).
    fn fold(&self, name: &str) -> String {
        name.to_lowercase()
    }

    /// The session handle whose queries see the locals as they are now.
    fn handle(&self) -> Session {
        self.session.with_locals(
            self.scopes
                .iter()
                .map(|scope| LocalFrame {
                    label: scope.label.clone(),
                    variables: scope
                        .vars
                        .iter()
                        .map(|(k, l)| (k.clone(), l.variable.clone()))
                        .collect(),
                })
                .collect(),
        )
    }

    /// The text of an expression of the script.
    fn slice(&self, e: &Expr) -> &str {
        self.text.get(e.span.start..e.span.end).unwrap_or_default()
    }

    // ---- statements ------------------------------------------------------------------------

    /// Runs a compound. After an `EXIT` handler the script goes on after the compound that declared it,
    /// as design note T196 section 2.4 says. A divergence, stated: Spark 4.2 ends the whole script
    /// silently when that compound is the last statement of an `IF` or `WHILE` body (the record's
    /// `exit_handler_in_loop_body_compound` cases); this engine continues after the compound there too.
    async fn compound(&mut self, c: &Compound) -> Result<Flow, KrishivError> {
        let label = c.label.as_ref().map(|l| self.fold(&l.value));
        let depth = self.scopes.len();
        // Handlers cover the whole compound, its declarations' defaults included.
        let handlers = c
            .body
            .iter()
            .filter_map(|item| match item {
                S::Handler(h) => Some((**h).clone()),
                _ => None,
            })
            .collect();
        self.scopes.push(Scope {
            label: label.clone(),
            vars: BTreeMap::new(),
            handlers,
        });
        let mut flow = Box::pin(self.items(&c.body)).await;
        if let Err(error) = &flow {
            // The scopes inside this one are gone when its handler runs.
            self.scopes.truncate(depth + 1);
            if let Some(handler) = self.find_handler(depth, error) {
                flow = Box::pin(self.exit_handler(&handler)).await;
            }
        }
        self.scopes.truncate(depth);
        Ok(match flow? {
            Flow::Leave(l) if label.as_deref() == Some(l.as_str()) => Flow::Normal,
            other => other,
        })
    }

    /// The handler of the compound at `depth` that takes `error`: a specific condition before
    /// `SQLEXCEPTION`, as in Spark. The errors of Krishiv's own classes, cancellation and the
    /// interpreter's bookkeeping are never taken: a statement Spark runs but Krishiv cannot must
    /// not read as an error the script handled.
    fn find_handler(&self, depth: usize, error: &KrishivError) -> Option<Handler> {
        let class = error.error_class();
        let name = class.name();
        if name.starts_with("KRISHIV_") || name == "OPERATION_CANCELED" || name == "INTERNAL_ERROR"
        {
            return None;
        }
        let state = error.sql_state().unwrap_or("");
        let prefix = name.split('.').next().unwrap_or(name);
        let rank = |condition: &HandlerCondition| -> Option<u8> {
            match condition {
                HandlerCondition::Name(n) if n.value.eq_ignore_ascii_case(name) => Some(0),
                HandlerCondition::Name(n) if n.value.eq_ignore_ascii_case(prefix) => Some(1),
                HandlerCondition::SqlState(s, _) if s.eq_ignore_ascii_case(state) => Some(2),
                HandlerCondition::NotFound(_) if state.starts_with("02") => Some(3),
                HandlerCondition::SqlException(_)
                    if !state.starts_with("XX") && !state.starts_with("02") =>
                {
                    Some(4)
                }
                _ => None,
            }
        };
        self.scopes
            .get(depth)?
            .handlers
            .iter()
            .filter_map(|h| h.conditions.iter().filter_map(rank).min().map(|r| (r, h)))
            .min_by_key(|(r, _)| *r)
            .map(|(_, h)| h.clone())
    }

    /// Runs the body of an `EXIT` handler; a `LEAVE` or `ITERATE` that escapes it ends the body only.
    async fn exit_handler(&mut self, handler: &Handler) -> Result<Flow, KrishivError> {
        match &handler.body {
            HandlerBody::Compound(c) => {
                Box::pin(self.compound(c)).await?;
            }
            HandlerBody::Sql(statement) => self.leaf(statement).await?,
            HandlerBody::Assign(set) => self.assign(set).await?,
            _ => return Err(unsupported("this handler body")),
        }
        Ok(Flow::Normal)
    }

    /// Runs `items` in order until one asks to leave or iterate.
    async fn items(&mut self, items: &[ScriptStatement]) -> Result<Flow, KrishivError> {
        for item in items {
            // Where a dropped `sql()` future lands: the script stops at the next statement.
            tokio::task::yield_now().await;
            let flow = Box::pin(self.item(item)).await?;
            if flow != Flow::Normal {
                return Ok(flow);
            }
        }
        Ok(Flow::Normal)
    }

    async fn item(&mut self, item: &ScriptStatement) -> Result<Flow, KrishivError> {
        match item {
            S::Declare(d) => Box::pin(self.declare(d)).await.map(|()| Flow::Normal),
            // Conditions match by name and handlers were registered when the compound started.
            S::Condition(_) | S::Handler(_) => Ok(Flow::Normal),
            S::Cursor(_) => Err(unsupported("SQL cursors")),
            S::For(f) => Box::pin(self.for_loop(f)).await,
            S::Assign(set) => Box::pin(self.assign(set)).await.map(|()| Flow::Normal),
            S::Sql(statement) => Box::pin(self.leaf(statement)).await.map(|()| Flow::Normal),
            S::Compound(c) => Box::pin(self.compound(c)).await,
            S::Leave(j) => Ok(Flow::Leave(self.fold(&j.label.value))),
            S::Iterate(j) => Ok(Flow::Iterate(self.fold(&j.label.value))),
            S::If(i) => {
                for branch in &i.branches {
                    if Box::pin(self.condition(&branch.condition)).await? {
                        return Box::pin(self.items(&branch.body)).await;
                    }
                }
                match &i.else_body {
                    Some(body) => Box::pin(self.items(body)).await,
                    None => Ok(Flow::Normal),
                }
            }
            S::Case(c) => {
                for branch in &c.branches {
                    let matched = match &c.subject {
                        Some(subject) => Box::pin(self.equal(subject, &branch.condition)).await?,
                        None => Box::pin(self.condition(&branch.condition)).await?,
                    };
                    if matched {
                        return Box::pin(self.items(&branch.body)).await;
                    }
                }
                match &c.else_body {
                    Some(body) => Box::pin(self.items(body)).await,
                    None => Ok(Flow::Normal),
                }
            }
            S::While(w) => {
                let label = w.label.as_ref().map(|l| self.fold(&l.value));
                while Box::pin(self.condition(&w.condition)).await? {
                    if let Some(flow) = Box::pin(self.round(&label, &w.body)).await? {
                        return Ok(flow);
                    }
                }
                Ok(Flow::Normal)
            }
            S::Repeat(r) => {
                let label = r.label.as_ref().map(|l| self.fold(&l.value));
                loop {
                    if let Some(flow) = Box::pin(self.round(&label, &r.body)).await? {
                        return Ok(flow);
                    }
                    if Box::pin(self.condition(&r.condition)).await? {
                        return Ok(Flow::Normal);
                    }
                }
            }
            S::Loop(l) => {
                let label = l.label.as_ref().map(|x| self.fold(&x.value));
                loop {
                    if let Some(flow) = Box::pin(self.round(&label, &l.body)).await? {
                        return Ok(flow);
                    }
                }
            }
            _ => Err(unsupported("this script statement")),
        }
    }

    /// `FOR [var AS] query DO ... END FOR`: the query runs once, then the body once per row with
    /// the row's columns as locals (`var.col` and `col`, assignable, gone after the loop).
    async fn for_loop(&mut self, f: &ForStatement) -> Result<Flow, KrishivError> {
        let label = f.label.as_ref().map(|l| self.fold(&l.value));
        let variable = f.variable.as_ref().map(|v| self.fold(&v.value));
        let inner = self
            .text
            .get(f.query.span.start..f.query.span.end)
            .unwrap_or_default();
        let sql = padded_select("", inner, f.query.span.start, "");
        let handle = self.handle();
        let analysis = handle.analyze_variable_query(&sql).await?;
        let names: Vec<String> = analysis
            .analyzed
            .output
            .iter()
            .map(|c| c.name.clone())
            .collect();
        let mut seen = HashSet::new();
        for name in &names {
            if !seen.insert(self.fold(name)) {
                return Err(VariableAlreadyExists {
                    variable_name: match &variable {
                        Some(v) => format!("`{v}`.`{name}`"),
                        None => format!("`{name}`"),
                    },
                }
                .into());
            }
        }
        let Some(batches) = handle
            .run_analyzed_variable_query_capped(&analysis, MAX_FOR_ROWS)
            .await?
        else {
            return Err(unsupported("a FOR loop over more than 100000 rows"));
        };
        for batch in &batches {
            for row in 0..batch.num_rows() {
                tokio::task::yield_now().await;
                let mut vars = BTreeMap::new();
                for (i, name) in names.iter().enumerate() {
                    let (data_type, value) = arrow_value(batch.column(i), row)?;
                    vars.insert(
                        self.fold(name),
                        Local {
                            variable: SessionVariable { data_type, value },
                            default_sql: None,
                        },
                    );
                }
                self.scopes.push(Scope {
                    label: variable.clone(),
                    vars,
                    handlers: Vec::new(),
                });
                let depth = self.scopes.len();
                let pass = Box::pin(self.round(&label, &f.body)).await;
                self.scopes.truncate(depth - 1);
                if let Some(flow) = pass? {
                    return Ok(flow);
                }
            }
        }
        Ok(Flow::Normal)
    }

    /// One pass over a loop body. `Some(flow)` ends the loop with that flow (`Normal` when the loop
    /// itself was left); `None` runs the next pass.
    async fn round(
        &mut self,
        label: &Option<String>,
        body: &[ScriptStatement],
    ) -> Result<Option<Flow>, KrishivError> {
        // An empty body must not spin past a dropped future.
        tokio::task::yield_now().await;
        Ok(match Box::pin(self.items(body)).await? {
            Flow::Normal => None,
            Flow::Leave(l) if label.as_deref() == Some(l.as_str()) => Some(Flow::Normal),
            Flow::Iterate(l) if label.as_deref() == Some(l.as_str()) => None,
            other => Some(other),
        })
    }

    /// A SQL statement of the script: classified and authorized like the statement typed alone,
    /// then run on the handle. A query's rows become the script's current result.
    async fn leaf(&mut self, statement: &Statement) -> Result<(), KrishivError> {
        if matches!(statement, Statement::DropVariable(_)) {
            return Err(UnsupportedFeatureSqlScriptingDropTemporaryVariable {}.into());
        }
        let handle = self.handle();
        let env = handle.query_env();
        let class = krishiv_sql::classify(statement);
        handle.authorize_statement(statement, class, &env).await?;
        // `EXECUTE IMMEDIATE ... INTO` assigns to locals too, which only this interpreter can do.
        if let Statement::ExecuteImmediate(e) = statement
            && !e.into.is_empty()
        {
            let frame = handle
                .immediate_frame(e, self.text, self.redacted, 0)
                .await?;
            return self.assign_row(e, &frame).await;
        }
        let frame = handle
            .run_statement(statement.clone(), env, self.text, self.redacted)
            .await?;
        // A query, or an `EXECUTE IMMEDIATE` of one, produces the script's current result; the
        // metadata commands (`SHOW`, `DESCRIBE`, `EXPLAIN`) and the rest do not.
        if matches!(
            statement,
            Statement::Query(_) | Statement::ExecuteImmediate(_)
        ) {
            let schema = frame.schema().await?;
            // A script's result is returned as literal rows: refuse a type they cannot hold now, not
            // after the rest of the script has run.
            for field in schema.fields() {
                crate::command::arrow_krishiv_type(field.data_type())?;
            }
            if !schema.fields().is_empty() {
                // One row more than the cap shows that it was exceeded; no more is read.
                let batches = frame.limit(MAX_RESULT_ROWS + 1).collect().await?;
                self.last = Some((schema, batches));
            }
        }
        Ok(())
    }

    /// `EXECUTE IMMEDIATE query INTO v, ...`: the one row of the query goes to the variables (locals
    /// first, then session variables), cast to their types; no row sets them to NULL.
    async fn assign_row(
        &mut self,
        e: &krishiv_sql::ast::ExecuteImmediate,
        frame: &DataFrame,
    ) -> Result<(), KrishivError> {
        let mut targets = Vec::with_capacity(e.into.len());
        for name in &e.into {
            let Some(target) = self.resolve(name) else {
                return Err(unresolved_variable(
                    name,
                    RESOLVE_SEARCH_PATH,
                    self.redacted,
                ));
            };
            targets.push(target);
        }
        let names: Vec<String> = e.into.iter().map(written).collect();
        distinct(&targets, &names)?;
        let mut types = Vec::with_capacity(targets.len());
        for (target, name) in targets.iter().zip(&names) {
            let Some((t, _)) = self.describe(target) else {
                return Err(VariableNotFound {
                    variable_name: name.clone(),
                }
                .into());
            };
            types.push(t);
        }
        let handle = self.handle();
        let row = handle.immediate_row(frame, targets.len()).await?;
        let mut values = Vec::with_capacity(targets.len());
        for (i, t) in types.iter().enumerate() {
            values.push(match row.as_ref().and_then(|r| r.get(i)) {
                Some(raw) => handle.cast_immediate(raw, t, self.redacted, e.span).await?,
                None => Literal::Null(Some(t.clone())),
            });
        }
        for (target, value) in targets.iter().zip(values) {
            self.store(target, value);
        }
        Ok(())
    }

    // ---- conditions ------------------------------------------------------------------------

    /// Evaluates `e` as a one-row query and returns its boolean; NULL is false.
    async fn condition(&self, e: &Expr) -> Result<bool, KrishivError> {
        let sql = padded_select("SELECT ", self.slice(e), e.span.start, "");
        self.boolean(&sql, e).await
    }

    /// `subject = value`, for a simple `CASE`.
    async fn equal(&self, subject: &Expr, value: &Expr) -> Result<bool, KrishivError> {
        let sql = format!("SELECT ({}) = ({})", self.slice(subject), self.slice(value));
        self.boolean(&sql, value).await
    }

    async fn boolean(&self, sql: &str, shown: &Expr) -> Result<bool, KrishivError> {
        let handle = self.handle();
        let analysis = handle.analyze_variable_query(sql).await?;
        let output = &analysis.analyzed.output;
        let not_boolean = || -> KrishivError {
            InvalidBooleanStatement {
                invalid_statement: self
                    .redacted
                    .fragment(shown.span.start..shown.span.end)
                    .to_owned(),
            }
            .into()
        };
        match output.as_slice() {
            [column] if column.data_type == DataType::Boolean => {}
            _ => return Err(not_boolean()),
        }
        let batches = handle.run_analyzed_variable_query(&analysis).await?;
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        if rows == 0 {
            return Err(internal("a script condition returned no row".to_owned()));
        }
        let Some(batch) = batches.iter().find(|b| b.num_rows() > 0) else {
            return Err(internal("a script condition returned no batch".to_owned()));
        };
        if rows > 1 {
            return Err(ScalarSubqueryTooManyRows {}.into());
        }
        Ok(matches!(
            arrow_value(batch.column(0), 0)?.1,
            Literal::Boolean(true)
        ))
    }

    // ---- variables -------------------------------------------------------------------------

    /// `DECLARE [VARIABLE] names [type] [DEFAULT expr]`: adds locals to the current compound.
    async fn declare(&mut self, d: &DeclareVariable) -> Result<(), KrishivError> {
        // Spark checks these when the statement runs, not before the script does.
        let first = d.names.first().map(|n| {
            n.parts
                .iter()
                .map(|p| format!("`{}`", p.value.replace('`', "``")))
                .collect::<Vec<_>>()
                .join(".")
        });
        if d.or_replace {
            return Err(InvalidVariableDeclarationReplaceLocalVariable {
                var_name: first.unwrap_or_default(),
            }
            .into());
        }
        if d.names.iter().any(|n| n.parts.len() > 1) {
            return Err(InvalidVariableDeclarationQualifiedLocalVariable {
                var_name: first.unwrap_or_default(),
            }
            .into());
        }
        let keys: Vec<String> = d
            .names
            .iter()
            .filter_map(|n| n.parts.first().map(|p| self.fold(&p.value)))
            .collect();
        let mut seen = HashSet::new();
        if let Some(dup) = keys.iter().find(|k| !seen.insert(k.as_str())) {
            return Err(DuplicateVariableNameInsideDeclare {
                variable_name: quote_normalized(dup),
            }
            .into());
        }
        if let Some(scope) = self.scopes.last()
            && let Some(existing) = keys.iter().find(|k| scope.vars.contains_key(*k))
        {
            return Err(VariableAlreadyExists {
                variable_name: quote_normalized(existing),
            }
            .into());
        }
        let declared = d
            .data_type
            .as_ref()
            .map(DataTypeSpec::to_data_type)
            .transpose()?;
        let (data_type, value, default_sql) = match (&declared, &d.default) {
            (_, None) => match declared {
                Some(t) => (t.clone(), Literal::Null(Some(t)), None),
                None => (DataType::Null, Literal::Null(None), None),
            },
            (declared, Some(default)) => {
                if contains_subquery(default) {
                    return Err(InvalidDefaultValueSubqueryExpression {
                        col_name: quote_normalized(keys.first().map_or("", String::as_str)),
                        default_value: self
                            .redacted
                            .fragment(default.span.start..default.span.end)
                            .to_owned(),
                        statement: "DECLARE VARIABLE".to_owned(),
                    }
                    .into());
                }
                let slice = self.slice(default);
                let sql = match declared {
                    Some(t) => padded_select(
                        "SELECT CAST(",
                        slice,
                        default.span.start,
                        &format!(" AS {t})"),
                    ),
                    None => padded_select("SELECT ", slice, default.span.start, ""),
                };
                let (inferred, value) = self
                    .handle()
                    .eval_variable_value(&sql, self.redacted, d.span, default.span)
                    .await?;
                (declared.clone().unwrap_or(inferred), value, Some(sql))
            }
        };
        if let Some(scope) = self.scopes.last_mut() {
            for key in keys {
                scope.vars.insert(
                    key,
                    Local {
                        variable: SessionVariable {
                            data_type: data_type.clone(),
                            value: value.clone(),
                        },
                        default_sql: default_sql.clone(),
                    },
                );
            }
        }
        Ok(())
    }

    /// Where `name` points: a local (by the rules of `VariableSnapshot::get`) or a session
    /// variable.
    fn resolve(&self, name: &ObjectName) -> Option<Target> {
        let parts: Vec<String> = name.parts.iter().map(|p| self.fold(&p.value)).collect();
        let session = |key: &str| {
            self.session
                .state
                .variables
                .get(key)
                .map(|_| Target::Session {
                    key: key.to_owned(),
                })
        };
        match parts.as_slice() {
            [v] => self
                .scopes
                .iter()
                .enumerate()
                .rev()
                .find(|(_, s)| s.vars.contains_key(v))
                .map(|(scope, _)| Target::Local {
                    scope,
                    key: v.clone(),
                })
                .or_else(|| session(v)),
            [q, v] => {
                let labelled = self
                    .scopes
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, s)| s.label.as_deref() == Some(q.as_str()));
                match labelled {
                    Some((scope, s)) => s.vars.contains_key(v).then(|| Target::Local {
                        scope,
                        key: v.clone(),
                    }),
                    None if q == "session" => session(v),
                    None => None,
                }
            }
            [system, session_part, v] if system == "system" && session_part == "session" => {
                session(v)
            }
            _ => None,
        }
    }

    /// The type and default query of the variable a target names.
    fn describe(&self, target: &Target) -> Option<(DataType, Option<String>)> {
        match target {
            Target::Local { scope, key } => {
                let local = self.scopes.get(*scope)?.vars.get(key)?;
                Some((local.variable.data_type.clone(), local.default_sql.clone()))
            }
            Target::Session { key } => {
                let stored = self.session.state.variables.get(key)?;
                Some((stored.data_type, stored.default_query))
            }
        }
    }

    fn store(&mut self, target: &Target, value: Literal) {
        match target {
            Target::Local { scope, key } => {
                if let Some(local) = self
                    .scopes
                    .get_mut(*scope)
                    .and_then(|s| s.vars.get_mut(key))
                {
                    local.variable.value = value;
                }
            }
            Target::Session { key } => {
                if let Some(stored) = self.session.state.variables.get(key) {
                    self.session
                        .state
                        .variables
                        .put(key.clone(), StoredVariable { value, ..stored });
                }
            }
        }
    }

    /// `SET name = expr [, ...]` and `SET (names) = (query)`: every right-hand side is evaluated
    /// before any target changes.
    async fn assign(&mut self, set: &SetVariable) -> Result<(), KrishivError> {
        match &set.kind {
            SetVariableKind::Single { assignments } => {
                let mut targets = Vec::with_capacity(assignments.len());
                for a in assignments {
                    let Some(target) = self.resolve(&a.name) else {
                        return Err(unresolved_variable(
                            &a.name,
                            RESOLVE_SEARCH_PATH,
                            self.redacted,
                        ));
                    };
                    targets.push(target);
                }
                let names: Vec<String> = assignments.iter().map(|a| written(&a.name)).collect();
                distinct(&targets, &names)?;
                let mut values = Vec::with_capacity(targets.len());
                for (a, target) in assignments.iter().zip(&targets) {
                    let Some((data_type, default_sql)) = self.describe(target) else {
                        return Err(VariableNotFound {
                            variable_name: names[values.len()].clone(),
                        }
                        .into());
                    };
                    let handle = self.handle();
                    let value = if is_bare_default(&a.value) {
                        match default_sql {
                            Some(sql) => {
                                handle
                                    .eval_variable_value(
                                        &sql,
                                        self.redacted,
                                        set.span,
                                        a.value.span,
                                    )
                                    .await?
                                    .1
                            }
                            None => Literal::Null(Some(data_type)),
                        }
                    } else {
                        if contains_default(&a.value) {
                            return Err(krishiv_common::error::DefaultPlacementInvalid {}.into());
                        }
                        let sql = padded_select(
                            "SELECT CAST(",
                            self.slice(&a.value),
                            a.value.span.start,
                            &format!(" AS {})", data_type.simple_string()),
                        );
                        handle
                            .eval_variable_value(&sql, self.redacted, set.span, a.value.span)
                            .await?
                            .1
                    };
                    values.push(value);
                }
                for (target, value) in targets.iter().zip(values) {
                    self.store(target, value);
                }
                Ok(())
            }
            SetVariableKind::Row { targets, query } => {
                let mut resolved = Vec::with_capacity(targets.len());
                for name in targets {
                    let Some(target) = self.resolve(name) else {
                        return Err(unresolved_variable(
                            name,
                            RESOLVE_SEARCH_PATH,
                            self.redacted,
                        ));
                    };
                    resolved.push(target);
                }
                let names: Vec<String> = targets.iter().map(written).collect();
                distinct(&resolved, &names)?;
                let mut types = Vec::with_capacity(resolved.len());
                for target in &resolved {
                    let Some((t, _)) = self.describe(target) else {
                        return Err(internal("a script variable vanished".to_owned()));
                    };
                    types.push(t);
                }
                let inner = self
                    .text
                    .get(query.span.start..query.span.end)
                    .unwrap_or_default();
                let inner_sql = padded_select("", inner, query.span.start, "");
                let handle = self.handle();
                let analysis = handle.analyze_variable_query(&inner_sql).await?;
                let width = analysis.analyzed.output.len();
                if width != resolved.len() {
                    return Err(krishiv_common::error::AssignmentArityMismatch {
                        num_target: resolved.len().to_string(),
                        num_expr: width.to_string(),
                    }
                    .into());
                }
                let casts = types
                    .iter()
                    .enumerate()
                    .map(|(i, t)| format!("CAST(q._c{i} AS {}) AS _c{i}", t.simple_string()))
                    .collect::<Vec<_>>()
                    .join(", ");
                let aliases = (0..types.len())
                    .map(|i| format!("_c{i}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let wrapper = format!("SELECT {casts} FROM ({inner_sql}) AS q({aliases})");
                let wrapped = handle.analyze_variable_query(&wrapper).await?;
                let batches = handle.run_analyzed_variable_query(&wrapped).await?;
                let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
                if rows > 1 {
                    return Err(RowSubqueryTooManyRows {}.into());
                }
                let values: Vec<Literal> = match batches.iter().find(|b| b.num_rows() > 0) {
                    Some(batch) => (0..types.len())
                        .map(|i| arrow_value(batch.column(i), 0).map(|(_, v)| v))
                        .collect::<Result<_, _>>()?,
                    None => types
                        .iter()
                        .map(|t| Literal::Null(Some(t.clone())))
                        .collect(),
                };
                for (target, value) in resolved.iter().zip(values) {
                    self.store(target, value);
                }
                Ok(())
            }
            _ => Err(unsupported("this SET form in a script")),
        }
    }

    // ---- the result ------------------------------------------------------------------------

    /// The script's result: the rows of the last query statement, as literal rows.
    async fn finish(self) -> Result<DataFrame, KrishivError> {
        let session = self.session;
        let Some((schema, batches)) = self.last else {
            return session.empty_frame().await;
        };
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        if i32::try_from(rows).map_or(true, |r| r > MAX_RESULT_ROWS) {
            return Err(unsupported("a script result of more than 256 rows"));
        }
        let one = || {
            DataFrame::new(
                session.clone(),
                Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
            )
        };
        let mut frames = Vec::with_capacity(rows);
        for batch in &batches {
            for row in 0..batch.num_rows() {
                let mut columns = Vec::with_capacity(batch.num_columns());
                for (i, field) in schema.fields().iter().enumerate() {
                    let (_, literal) = arrow_value(batch.column(i), row)?;
                    columns.push(
                        Column::new(PlanExpr::new(ExprNode::Literal(literal))).alias(field.name()),
                    );
                }
                frames.push(one().select(columns));
            }
        }
        let df = match frames.len() {
            0 => {
                let columns = schema
                    .fields()
                    .iter()
                    .map(|f| {
                        let t = DataType::from_arrow_field(f)?;
                        Ok(
                            Column::new(PlanExpr::new(ExprNode::Literal(Literal::Null(Some(t)))))
                                .alias(f.name()),
                        )
                    })
                    .collect::<Result<Vec<_>, KrishivError>>()?;
                one().select(columns).limit(0)
            }
            1 => frames.remove(0),
            _ => DataFrame::new(
                session.clone(),
                Arc::new(LogicalPlan::new(PlanNode::Union {
                    inputs: frames.iter().map(|f| Arc::clone(&f.plan)).collect(),
                    by_name: false,
                    allow_missing_columns: false,
                })),
            ),
        };
        df.analyzed_in(session.query_env()).await?;
        Ok(df)
    }
}

/// A variable name as the user wrote it, quoted part by part for an error.
fn written(name: &ObjectName) -> String {
    name.parts
        .iter()
        .map(|p| format!("`{}`", p.value.replace('`', "``")))
        .collect::<Vec<_>>()
        .join(".")
}

/// `DUPLICATE_ASSIGNMENTS` when two of `targets` are one variable; `names` are as written.
fn distinct(targets: &[Target], names: &[String]) -> Result<(), KrishivError> {
    for (i, target) in targets.iter().enumerate() {
        if targets[..i].contains(target) {
            return Err(DuplicateAssignments {
                name_list: names.get(i).cloned().unwrap_or_default(),
            }
            .into());
        }
    }
    Ok(())
}
