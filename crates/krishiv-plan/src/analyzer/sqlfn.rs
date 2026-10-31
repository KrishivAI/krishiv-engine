// SPDX-License-Identifier: Apache-2.0
//! SQL-bodied scalar functions (Spark's SQL UDFs), inlined at analysis.
//!
//! A call `f(a, b)` of a function the registry answers with
//! [`FunctionLookup::Sql`] is replaced by the function's body, resolved in
//! the function's own scope with each parameter bound to its (cast)
//! argument: the resolved plan holds no call of `f`, so nothing after
//! analysis knows functions of this kind exist. Each call resolves the body
//! afresh (fresh attribute ids), so two calls of one function share nothing.
//!
//! **Scope.** The body sees no column of the calling query and no `WITH`
//! name. Its parameters rank after everything else a name may mean (the
//! precedence Spark 4.2 gives them): the input of the operator the name is
//! in, a parameterless function written without parentheses
//! (`current_user`; first unless
//! `spark.sql.legacy.allowUdfParameterToShadowParameterlessFunction`), a
//! lateral column alias, the enclosing query's columns, then the
//! parameter. A parameter is visible at any depth of the body's subqueries;
//! an argument that reads columns of the calling query can be read from the
//! body itself and from a subquery directly in it (an outer reference), and
//! is `KRISHIV_UNSUPPORTED_FEATURE` from deeper (the analyzer resolves outer
//! references one query up only). Constant arguments have no such limit.
//! The arguments, defaults and the body of a function called from another
//! function's body see only their own function's parameters.
//!
//! **Arguments.** Positional, then named (`f(b => 1)`); defaults fill the
//! rest, resolved in an empty scope at each call. Each argument is cast to
//! its parameter's type as `CAST` would be, and the body to the declared
//! return type. A parameter is substituted by its argument expression, so
//! an argument that is not deterministic is only accepted for a parameter
//! the body reads once (Spark evaluates it once).
//!
//! **Recursion** (direct or through other functions) is
//! `CYCLIC_FUNCTION_REFERENCE`.
//!
//! **Not done here:** the body is resolved under the calling query's
//! configuration, not the one in force when the function was created, and
//! `NOT DETERMINISTIC` and the data-access characteristics are not
//! enforced: the body's own expressions say whether it is deterministic.

use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use krishiv_common::error::{
    CyclicFunctionReference, DuplicateRoutineParameterAssignmentBothPositionalAndNamed,
    DuplicateRoutineParameterAssignmentDoubleNamedArgumentReference, FailedToParseTooComplex,
    KrishivError, NotAScalarFunction, RequiredParameterNotFound, UnexpectedPositionalArgument,
    UnrecognizedParameterName, UserDefinedFunctionsCannotContainComplexFunctions,
    UserDefinedFunctionsReturnColumnCountMismatch, WrongNumArgsWithoutSuggestion,
};
use krishiv_common::stack::on_big_stack;
use krishiv_types::DataType;

use super::AnalysisError;
use super::bind::{same, unsupported};
use super::catalog::quoted;
use super::coerce;
use super::exprs::{Bound, Cx, SPARK_DOC_ROOT, WindowUse, cast_error, pretty};
use super::resolve::{Attr, Rel, Resolver};
use crate::walk::{Node, visit_distinct};
use crate::{
    AttrId, BoundColumn, CastMode, Expr, ExprNode, FunctionCall, LogicalPlan, Origin, PlanNode,
    ResolvedExpr, ResolvedFunction, SourceId,
};

/// A parameter of a [`SqlFunction`].
#[derive(Debug, Clone)]
pub struct SqlParam {
    /// The name, as written.
    pub name: String,
    /// The type arguments are cast to.
    pub data_type: DataType,
    /// The `DEFAULT` expression, unresolved; it resolves at each call, in a
    /// scope with no columns and no parameters.
    pub default: Option<Arc<Expr>>,
    /// The `COMMENT`.
    pub comment: Option<String>,
}

/// A SQL-bodied scalar function: what a registry answers for its name
/// ([`super::FunctionLookup::Sql`]).
#[derive(Debug, Clone)]
pub struct SqlFunction {
    /// The name as the registry keys it (lower case: function names are
    /// case-insensitive whatever `spark.sql.caseSensitive` says); it shows in
    /// errors.
    pub name: String,
    /// The catalog and namespace of a function stored in the session's
    /// catalog (`spark_catalog`, `default`); empty for a temporary one. Spark
    /// names a stored function by all three parts in the errors of a call.
    pub qualifier: Vec<String>,
    /// The parameters, in order.
    pub params: Vec<SqlParam>,
    /// The declared return type the body is cast to; `None` keeps the
    /// body's own type (a function created without `RETURNS`, which the
    /// session types by resolving the body once).
    pub returns: Option<DataType>,
    /// The columns of a table function (`RETURNS TABLE (...)`), whose body
    /// is a query (a scalar subquery expression holding the plan); `None`
    /// for a scalar function. Empty for `RETURNS TABLE` without a column
    /// list: the columns are the query's own.
    pub returns_table: Option<Vec<(String, DataType)>>,
    /// The body, unresolved.
    pub body: Arc<Expr>,
    /// The body as written, for `CANNOT_CONTAIN_COMPLEX_FUNCTIONS`.
    pub body_text: String,
    /// The text the body's spans index.
    pub source: SourceId,
    /// Whether the function was declared `DETERMINISTIC` (the default is
    /// that it is, as in Spark). Recorded for the front end; calls keep the
    /// determinism of the body's own expressions.
    pub deterministic: bool,
}

impl SqlFunction {
    /// The name as an error of a call quotes it: the qualifier, then the
    /// name.
    #[must_use]
    pub fn routine_name(&self) -> String {
        let mut parts = self.qualifier.clone();
        parts.push(self.name.clone());
        quoted(&parts)
    }

    /// The name with its qualifier, dotted and unquoted.
    #[must_use]
    pub fn qualified_name(&self) -> String {
        self.path_key()
    }

    /// What tells two functions apart in the path of nested calls.
    fn path_key(&self) -> String {
        let mut parts = self.qualifier.clone();
        parts.push(self.name.clone());
        parts.join(".")
    }

    /// The number of leading parameters a call must supply: those before
    /// the first one with a default.
    #[must_use]
    pub fn required(&self) -> usize {
        self.params
            .iter()
            .position(|p| p.default.is_some())
            .unwrap_or(self.params.len())
    }
}

/// The call being resolved: its function and the argument each parameter
/// stands for.
pub(super) struct Frame {
    /// The function's registry key.
    name: String,
    params: Vec<FrameParam>,
    /// The parameters as attributes, for `UNRESOLVED_COLUMN` proposals
    /// only (their ids are never used).
    attrs: Vec<Attr>,
}

struct FrameParam {
    name: String,
    arg: Bound,
    /// How often the body read it.
    reads: AtomicUsize,
}

/// What the references in an argument expression are.
#[derive(Default)]
struct Refs {
    columns: bool,
    outer: bool,
    subquery: bool,
}

fn refs(e: &Arc<Expr>) -> Refs {
    let mut r = Refs::default();
    let _ = visit_distinct(Node::Expr(e), |node| {
        if let Node::Expr(x) = node {
            match &x.node {
                ExprNode::Resolved(ResolvedExpr::Column(_)) => r.columns = true,
                ExprNode::Resolved(ResolvedExpr::OuterColumn(_)) => r.outer = true,
                ExprNode::Subquery { .. } => r.subquery = true,
                _ => {}
            }
        }
        ControlFlow::Continue(())
    });
    r
}

/// Whether `e` holds no call of a nondeterministic function.
fn is_deterministic(e: &Arc<Expr>) -> bool {
    visit_distinct(Node::Expr(e), |node| match node {
        Node::Expr(x)
            if matches!(
                &x.node,
                ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
                    deterministic: false,
                    ..
                }))
            ) =>
        {
            ControlFlow::Break(())
        }
        _ => ControlFlow::Continue(()),
    })
    .is_continue()
}

/// How many functions may be open at once.
const NESTING: usize = 128;

/// How many calls of SQL functions one analysis may inline. Every call
/// resolves the body afresh, so nested calls grow exponentially without a
/// budget; past it the statement is `FAILED_TO_PARSE_TOO_COMPLEX`.
const MAX_INLINED_CALLS: usize = 2_000;

/// Function levels resolved on the caller's stack before hopping to a
/// fresh one (a level holds several large frames).
const STEP: usize = 4;

impl Resolver<'_> {
    /// Replaces the call `call` (`e`, with its bound arguments `kids`) of
    /// `def` by the function's body.
    pub(super) fn sql_call(
        &mut self,
        def: &Arc<SqlFunction>,
        e: &Arc<Expr>,
        call: &FunctionCall,
        kids: &[Bound],
    ) -> Result<Bound, AnalysisError> {
        let n = call.args.len();
        if def.returns_table.is_some() {
            return Err(AnalysisError::new(
                NotAScalarFunction {
                    function_name: def.routine_name(),
                },
                e.origin,
            ));
        }
        if call.distinct || call.filter.is_some() {
            return Err(unsupported(
                "DISTINCT or FILTER on a SQL function",
                e.origin,
            ));
        }
        if self.fn_path.iter().any(|f| f == &def.path_key()) {
            let mut path = self.fn_path.clone();
            path.push(def.path_key());
            return Err(cyclic(&path, e.origin));
        }
        if self.fn_path.len() >= NESTING || self.sql_calls >= MAX_INLINED_CALLS {
            return Err(AnalysisError::new(FailedToParseTooComplex {}, e.origin));
        }
        self.sql_calls += 1;
        // Open while the defaults resolve too: a default may call the
        // function itself.
        self.fn_path.push(def.path_key());
        let hop = self.fn_path.len().is_multiple_of(STEP);
        let body = match self.arguments(def, e, call, &kids[..n]) {
            Ok(args) if hop => on_big_stack(|| self.inline(def, e, args))
                .unwrap_or_else(|| Err(AnalysisError::new(FailedToParseTooComplex {}, e.origin))),
            Ok(args) => self.inline(def, e, args),
            Err(error) => Err(error),
        };
        self.fn_path.pop();
        let body = body?;
        // The call as the user wrote it names the result; a stored function
        // is named by the catalog's identifier, as Spark names it.
        let (name, qualifier) = if def.qualifier.is_empty() {
            (call.name.clone(), Vec::new())
        } else {
            (def.name.clone(), def.qualifier.clone())
        };
        let shown = Arc::new(
            Expr::new(ExprNode::Function(FunctionCall {
                within_group: Vec::new(),
                name,
                qualifier,
                args: kids[..n].iter().map(Bound::shown).collect(),
                arg_names: call.arg_names.clone(),
                distinct: false,
                filter: None,
            }))
            .with_origin(e.origin),
        );
        let args_agg = kids.iter().find_map(|k| k.agg);
        let mut lateral = Vec::new();
        for id in kids.iter().flat_map(|k| k.lateral.iter()) {
            if !lateral.contains(id) {
                lateral.push(*id);
            }
        }
        Ok(Bound {
            agg: args_agg,
            win: kids.iter().find_map(|k| k.win),
            outer: kids.iter().filter_map(|k| k.outer).min(),
            lateral,
            shown: Some(shown),
            ..body
        })
    }

    /// The body of `def` resolved with `args` bound to its parameters, cast
    /// to the declared return type.
    fn inline(
        &mut self,
        def: &Arc<SqlFunction>,
        e: &Arc<Expr>,
        args: Vec<Bound>,
    ) -> Result<Bound, AnalysisError> {
        let frame = Self::frame_of(def, args);
        // The body sees no lambda variable of the calling query.
        let outer_lambdas = std::mem::take(&mut self.lambdas);
        let outer = self.frame.replace(Arc::clone(&frame));
        let resolved = self.body(def, e);
        self.frame = outer;
        self.lambdas = outer_lambdas;
        let body = resolved?;
        for (p, fp) in def.params.iter().zip(&frame.params) {
            if fp.reads.load(Ordering::Relaxed) > 1 && !is_deterministic(&fp.arg.expr) {
                return Err(unsupported(
                    format!(
                        "a nondeterministic argument for parameter {} of function {}, which the body reads more than once",
                        quoted(std::slice::from_ref(&p.name)),
                        def.routine_name(),
                    ),
                    e.origin,
                ));
            }
        }
        Ok(body)
    }

    /// The frame of a call of `def`: its parameters standing for `args`.
    fn frame_of(def: &SqlFunction, args: Vec<Bound>) -> Arc<Frame> {
        let params: Vec<FrameParam> = def
            .params
            .iter()
            .zip(args)
            .map(|(p, arg)| FrameParam {
                name: p.name.clone(),
                // What the argument computes and its type; what it
                // aggregates or windows belongs to the call, not the body.
                arg: Bound {
                    agg: None,
                    win: None,
                    outer: None,
                    lateral: Vec::new(),
                    ..arg
                },
                reads: AtomicUsize::new(0),
            })
            .collect();
        let attrs = def
            .params
            .iter()
            .zip(&params)
            .map(|(p, fp)| Attr {
                col: BoundColumn {
                    id: AttrId(u64::MAX),
                    name: p.name.clone(),
                    data_type: p.data_type.clone(),
                    nullable: fp.arg.ty.nullable,
                },
                qual: Arc::from([def.name.clone()]),
                source: None,
            })
            .collect();
        Arc::new(Frame {
            name: def.name.clone(),
            params,
            attrs,
        })
    }

    /// Replaces the table function call `FROM f(args)` of `def` by its body
    /// query, resolved with the arguments bound to the parameters and its
    /// columns renamed and cast to the declared ones.
    pub(super) fn sql_table_call(
        &mut self,
        p: &Arc<LogicalPlan>,
        def: &Arc<SqlFunction>,
        args: &[Arc<Expr>],
        env: &Arc<super::resolve::Env>,
    ) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        let key = def.path_key();
        if self.fn_path.iter().any(|f| f == &key) {
            let mut path = self.fn_path.clone();
            path.push(key);
            return Err(cyclic(&path, o));
        }
        if self.fn_path.len() >= NESTING || self.sql_calls >= MAX_INLINED_CALLS {
            return Err(AnalysisError::new(FailedToParseTooComplex {}, o));
        }
        self.sql_calls += 1;
        let mut kids = Vec::with_capacity(args.len());
        for a in args {
            let mut cx = Self::cx(&[], &[], env, "Project");
            cx.outer = false;
            kids.push(self.bind(a, &cx)?);
        }
        let call = FunctionCall {
            within_group: Vec::new(),
            name: def.name.clone(),
            qualifier: def.qualifier.clone(),
            args: args.to_vec(),
            arg_names: vec![None; args.len()],
            distinct: false,
            filter: None,
        };
        let e = Arc::new(Expr::new(ExprNode::Function(call.clone())).with_origin(o));
        let bound = self.arguments(def, &e, &call, &kids)?;
        let frame = Self::frame_of(def, bound);
        let Some(columns) = &def.returns_table else {
            return Err(unsupported("a scalar function as a table", o));
        };
        let ExprNode::Subquery { plan, .. } = &def.body.node else {
            return Err(unsupported("this kind of table function body", o));
        };
        self.fn_path.push(def.path_key());
        let outer_lambdas = std::mem::take(&mut self.lambdas);
        let outer = self.frame.replace(Arc::clone(&frame));
        let body_env = self.isolated_env();
        let hop = self.fn_path.len().is_multiple_of(STEP);
        let resolved = if hop {
            on_big_stack(|| self.rel(plan, &body_env))
                .unwrap_or_else(|| Err(AnalysisError::new(FailedToParseTooComplex {}, o)))
        } else {
            self.rel(plan, &body_env)
        };
        self.frame = outer;
        self.lambdas = outer_lambdas;
        self.fn_path.pop();
        let rel = resolved?;
        // Spark evaluates an argument once per call. The body inlines it, so
        // a nondeterministic one is accepted only where that is the same:
        // read at most once, by a body that scans no relation (which would
        // evaluate it once per row).
        let scans = visit_distinct(Node::Plan(plan), |node| match node {
            Node::Plan(x) if matches!(x.node, PlanNode::Read(_)) => ControlFlow::Break(()),
            _ => ControlFlow::Continue(()),
        })
        .is_break();
        for (param, fp) in def.params.iter().zip(&frame.params) {
            if !is_deterministic(&fp.arg.expr) && (scans || fp.reads.load(Ordering::Relaxed) > 1) {
                return Err(unsupported(
                    format!(
                        "a nondeterministic argument for parameter {} of table function {}, which the body reads more than once or over a relation",
                        quoted(std::slice::from_ref(&param.name)),
                        def.routine_name(),
                    ),
                    o,
                ));
            }
        }
        if columns.is_empty() {
            return Ok(rel);
        }
        if rel.out.len() != columns.len() {
            return Err(AnalysisError::new(
                UserDefinedFunctionsReturnColumnCountMismatch {
                    name: def.qualified_name(),
                    output_size: rel.out.len().to_string(),
                    return_param_size: columns.len().to_string(),
                },
                o,
            ));
        }
        let mut exprs = Vec::with_capacity(columns.len());
        for (a, (name, ty)) in rel.out.iter().zip(columns) {
            let cast = self.cast_arg(&super::ops::column_bound(a, o), ty, o)?;
            exprs.push(Arc::new(Expr::new(ExprNode::Alias {
                expr: cast.expr,
                name: name.clone(),
            })));
        }
        self.project_items(p, &rel, &exprs, env)
    }

    /// Resolves the body in its own scope and casts it to the return type.
    fn body(&mut self, def: &SqlFunction, e: &Arc<Expr>) -> Result<Bound, AnalysisError> {
        let env = self.isolated_env();
        let cx = Self::lone_cx(&env);
        let body = self.expr(&def.body, &cx)?;
        if body.agg.is_some() || body.win.is_some() {
            return Err(AnalysisError::new(
                UserDefinedFunctionsCannotContainComplexFunctions {
                    query_text: def.body_text.clone(),
                },
                e.origin,
            ));
        }
        match &def.returns {
            Some(to) => self.cast_arg(&body, to, e.origin),
            None => Ok(body),
        }
    }

    /// An environment with no enclosing query and no `WITH` names, of its
    /// own: nothing resolved in it is shared with another call.
    fn isolated_env(&mut self) -> Arc<super::resolve::Env> {
        let ctes_id = self.ctes_id();
        self.env(Vec::new(), ctes_id, Vec::new())
    }

    /// A scope with no columns: the body's, a default's.
    fn lone_cx(env: &Arc<super::resolve::Env>) -> Cx<'_> {
        Cx {
            scope: &[],
            rels: &[],
            lateral: None,
            aliases: &[],
            extra: None,
            agg_scope: None,
            grouping: false,
            lenient: false,
            outer: true,
            windows: WindowUse::Other,
            env,
            op: "Project",
        }
    }

    /// The argument of every parameter of `def` for the call `call` with
    /// bound arguments `kids`, each cast to its parameter's type.
    fn arguments(
        &mut self,
        def: &Arc<SqlFunction>,
        e: &Arc<Expr>,
        call: &FunctionCall,
        kids: &[Bound],
    ) -> Result<Vec<Bound>, AnalysisError> {
        let routine = def.routine_name();
        let total = def.params.len();
        let named: Vec<Option<&str>> = (0..call.args.len())
            .map(|i| call.arg_names.get(i).and_then(|n| n.as_deref()))
            .collect();
        let positional = named.iter().take_while(|n| n.is_none()).count();
        let mut slots: Vec<Option<Bound>> = vec![None; total];
        if named.iter().all(Option::is_none) {
            if kids.len() < def.required() || kids.len() > total {
                return Err(AnalysisError::new(
                    WrongNumArgsWithoutSuggestion {
                        function_name: routine,
                        expected_num: total.to_string(),
                        actual_num: kids.len().to_string(),
                        docroot: SPARK_DOC_ROOT.to_owned(),
                    },
                    e.origin,
                ));
            }
        } else if positional > total {
            return Err(AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: routine,
                    expected_num: total.to_string(),
                    actual_num: kids.len().to_string(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                e.origin,
            ));
        }
        for (slot, kid) in slots.iter_mut().zip(&kids[..positional]) {
            *slot = Some(kid.clone());
        }
        let cs = self.ctx.conf.case_sensitive;
        let mut last_named: Option<&str> = None;
        for (i, name) in named.iter().enumerate().skip(positional) {
            let Some(name) = name else {
                return Err(AnalysisError::new(
                    UnexpectedPositionalArgument {
                        routine_name: routine,
                        parameter_name: quoted(&[last_named.unwrap_or_default().to_owned()]),
                    },
                    call.args[i].origin,
                ));
            };
            last_named = Some(name);
            let origin = call.args[i].origin;
            let Some(p) = def.params.iter().position(|p| same(&p.name, name, cs)) else {
                let proposal: Vec<String> = def
                    .params
                    .iter()
                    .map(|p| quoted(std::slice::from_ref(&p.name)))
                    .collect();
                return Err(AnalysisError::new(
                    UnrecognizedParameterName {
                        routine_name: routine,
                        argument_name: quoted(&[(*name).to_owned()]),
                        proposal: proposal.join(", "),
                    },
                    origin,
                ));
            };
            if slots[p].is_some() {
                let parameter_name = quoted(std::slice::from_ref(&def.params[p].name));
                let error: KrishivError = if p < positional {
                    DuplicateRoutineParameterAssignmentBothPositionalAndNamed {
                        routine_name: routine,
                        parameter_name,
                    }
                    .into()
                } else {
                    DuplicateRoutineParameterAssignmentDoubleNamedArgumentReference {
                        routine_name: routine,
                        parameter_name,
                    }
                    .into()
                };
                return Err(AnalysisError::new(error, origin));
            }
            slots[p] = Some(kids[i].clone());
        }
        let mut out = Vec::with_capacity(total);
        for (i, (slot, p)) in slots.into_iter().zip(&def.params).enumerate() {
            let arg = match slot {
                Some(arg) => arg,
                None => match &p.default {
                    Some(default) => self.default_value(default)?,
                    None => {
                        return Err(AnalysisError::new(
                            RequiredParameterNotFound {
                                routine_name: routine,
                                parameter_name: quoted(std::slice::from_ref(&p.name)),
                                index: i.to_string(),
                            },
                            e.origin,
                        ));
                    }
                },
            };
            out.push(self.cast_arg(&arg, &p.data_type, e.origin)?);
        }
        Ok(out)
    }

    /// A parameter's default, resolved in a scope with no columns and no
    /// parameters (not the caller's, not the enclosing function's).
    fn default_value(&mut self, default: &Arc<Expr>) -> Result<Bound, AnalysisError> {
        let env = self.isolated_env();
        let cx = Self::lone_cx(&env);
        let outer = self.frame.take();
        let outer_lambdas = std::mem::take(&mut self.lambdas);
        let bound = self.expr(default, &cx);
        self.lambdas = outer_lambdas;
        self.frame = outer;
        bound
    }

    /// `b` cast to `to` as `CAST` would: nothing for a value of that type.
    fn cast_arg(&self, b: &Bound, to: &DataType, origin: Origin) -> Result<Bound, AnalysisError> {
        if b.ty.data_type.as_ref() == Some(to) {
            return Ok(b.clone());
        }
        let mode = if self.ctx.conf.ansi {
            CastMode::Ansi
        } else {
            CastMode::Legacy
        };
        if let Err(bad) = coerce::can_cast(b.ty.data_type.as_ref(), to, mode) {
            // The cast is Spark's implicit one: the error shows the value.
            let shown = format!("\"{}\"", pretty(&b.shown(), self.ctx.conf.session_zone));
            return Err(AnalysisError::new(cast_error(bad, shown, b, to), origin));
        }
        Ok(self.implicit_cast_as(b, Some(to.clone()), false))
    }

    /// The parameter `parts` names in the function being resolved, read
    /// from the body at `depth` levels of subquery.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for an argument that cannot be read at
    /// that depth: one that reads columns of the calling query, from a
    /// subquery of the body nested inside another subquery (the engine
    /// resolves outer references one query up only).
    pub(super) fn parameter(
        &self,
        parts: &[String],
        origin: Origin,
        depth: usize,
    ) -> Result<Option<Bound>, AnalysisError> {
        let Some(frame) = &self.frame else {
            return Ok(None);
        };
        let cs = self.ctx.conf.case_sensitive;
        let name = match parts {
            [name] => name,
            [function, name] if same(function, &frame.name, cs) => name,
            _ => return Ok(None),
        };
        let Some(p) = frame.params.iter().find(|p| same(&p.name, name, cs)) else {
            return Ok(None);
        };
        p.reads.fetch_add(1, Ordering::Relaxed);
        // A parameter substituted by its argument runs wherever it is read:
        // once per element of a lambda and once per row of a subquery. Spark
        // evaluates the argument once per call, so a nondeterministic one
        // cannot be read from there.
        if (depth > 0 || !self.lambdas.is_empty()) && !is_deterministic(&p.arg.expr) {
            return Err(unsupported(
                format!(
                    "a nondeterministic argument for parameter {} of function {}, read inside a lambda or a subquery of the body",
                    quoted(std::slice::from_ref(&p.name)),
                    quoted(std::slice::from_ref(&frame.name)),
                ),
                origin,
            ));
        }
        if depth == 0 {
            return Ok(Some(p.arg.clone()));
        }
        let r = refs(&p.arg.expr);
        if r.outer || r.subquery || (r.columns && depth > 1) {
            return Err(unsupported(
                format!(
                    "an argument that reads the calling query, for parameter {} of function {}, read from a subquery nested {} deep in its body",
                    quoted(std::slice::from_ref(&p.name)),
                    quoted(std::slice::from_ref(&frame.name)),
                    depth,
                ),
                origin,
            ));
        }
        if !r.columns {
            return Ok(Some(p.arg.clone()));
        }
        // Read from a subquery of the body, the argument's columns are the
        // calling query's: outer references.
        let lifted = Arc::clone(&p.arg.expr).transform_up(&mut |x: Arc<Expr>| {
            Ok::<_, std::convert::Infallible>(match &x.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) => Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::OuterColumn(c.clone())))
                        .with_origin(x.origin),
                ),
                _ => x,
            })
        });
        let Ok(expr) = lifted;
        Ok(Some(Bound {
            expr,
            outer: Some(0),
            ..p.arg.clone()
        }))
    }

    /// Whether the function being resolved has a parameter called `name`.
    pub(super) fn has_parameter(&self, name: &str) -> bool {
        let cs = self.ctx.conf.case_sensitive;
        self.frame
            .as_ref()
            .is_some_and(|f| f.params.iter().any(|p| same(&p.name, name, cs)))
    }

    /// The parameters of the function being resolved, as candidates for an
    /// `UNRESOLVED_COLUMN` proposal.
    pub(super) fn parameter_attrs(&self) -> &[Attr] {
        self.frame.as_ref().map_or(&[], |f| &f.attrs)
    }
}

/// `CYCLIC_FUNCTION_REFERENCE` for `path`.
fn cyclic(path: &[String], origin: Origin) -> AnalysisError {
    AnalysisError::new(
        CyclicFunctionReference {
            path: path.join(" -> "),
        },
        origin,
    )
}
