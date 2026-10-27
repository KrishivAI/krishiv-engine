// SPDX-License-Identifier: Apache-2.0
//! Session UDFs in the optimizer: deterministic calls of literal arguments
//! fold to their values (through the same binding execution uses), and
//! anything else stays in the plan.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::ArrayRef;
use krishiv_common::config::Config;
use krishiv_optimizer::{NoStatistics, optimize_with_udfs};
use krishiv_plan::{
    Expr, ExprNode, FunctionId, FunctionKind, Literal, LogicalPlan, Plan, PlanNode, ResolvedExpr,
    ResolvedFunction,
};
use krishiv_types::DataType;
use krishiv_udf_sdk::{
    ArgField, InvokeContext, ReturnField, ScalarUdf, Signature, UdfArg, UdfError, UdfInputType,
    UdfLibrary,
};

/// `my_udf(x) = x + 1` over `INT`, deterministic or not, counting its
/// invocations.
#[derive(Debug)]
struct PlusOne {
    deterministic: bool,
    calls: Arc<AtomicUsize>,
}

impl ScalarUdf for PlusOne {
    fn name(&self) -> &str {
        "my_udf"
    }

    fn version(&self) -> u32 {
        2
    }

    fn signature(&self) -> &Signature {
        use std::sync::OnceLock;
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(|| Signature::Exact(vec![UdfInputType::Exact(DataType::Integer)]))
    }

    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::Integer,
            nullable: true,
        })
    }

    fn properties(&self) -> krishiv_udf_sdk::UdfProperties {
        krishiv_udf_sdk::UdfProperties {
            deterministic: self.deterministic,
            null_handling: krishiv_udf_sdk::NullHandling::Strict,
        }
    }

    fn invoke(
        &self,
        args: &[UdfArg<'_>],
        rows: usize,
        _: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        use arrow_array::Int32Array;
        self.calls.fetch_add(1, Ordering::Relaxed);
        let input = args[0].array();
        let input = input
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| UdfError::new("my_udf needs INT"))?;
        assert_eq!(
            input.len(),
            if matches!(args[0], UdfArg::Scalar(_)) {
                1
            } else {
                rows
            }
        );
        Ok(Arc::new(
            input
                .iter()
                .map(|v| v.map(|v| v.wrapping_add(1)))
                .collect::<Int32Array>(),
        ))
    }
}

fn call(deterministic: bool) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: FunctionId::Udf {
                name: "my_udf".to_owned(),
                version: 2,
            },
            kind: FunctionKind::Scalar,
            args: vec![Arc::new(Expr::literal(Literal::Integer(41)))],
            distinct: false,
            filter: None,
            data_type: DataType::Integer,
            nullable: true,
            deterministic,
        },
    ))))
}

fn plan(call: Arc<Expr>) -> Plan {
    Plan::try_new(LogicalPlan::new(PlanNode::Project {
        input: Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
        exprs: vec![call],
    }))
    .unwrap()
}

fn config() -> Config {
    Config::from_pairs(Vec::<(String, String)>::new())
        .unwrap()
        .0
}

fn project_expr(plan: &Plan) -> Arc<Expr> {
    match &plan.root().node {
        PlanNode::Project { exprs, .. } => Arc::clone(&exprs[0]),
        other => panic!("a project, not {other:?}"),
    }
}

#[test]
fn a_deterministic_udf_of_literals_folds_to_its_value() {
    let lib = Arc::new(UdfLibrary::new());
    let calls = Arc::new(AtomicUsize::new(0));
    lib.register(Arc::new(PlusOne {
        deterministic: true,
        calls: Arc::clone(&calls),
    }))
    .unwrap();
    let optimized = optimize_with_udfs(&plan(call(true)), &NoStatistics, &lib, &config()).unwrap();
    // Folded (through one invocation), to the UDF's value.
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        project_expr(&optimized).node,
        Expr::new(ExprNode::Literal(Literal::Integer(42))).node
    );
}

#[test]
fn other_udf_calls_stay_in_the_plan() {
    let lib = Arc::new(UdfLibrary::new());
    let calls = Arc::new(AtomicUsize::new(0));
    lib.register(Arc::new(PlusOne {
        deterministic: true,
        calls: Arc::clone(&calls),
    }))
    .unwrap();
    let config = config();
    // Nondeterministic calls never fold.
    let optimized = optimize_with_udfs(&plan(call(false)), &NoStatistics, &lib, &config).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(matches!(
        &project_expr(&optimized).node,
        ExprNode::Resolved(ResolvedExpr::Function(_))
    ));
    // An unregistered version never folds either.
    let empty = UdfLibrary::new();
    let optimized = optimize_with_udfs(&plan(call(true)), &NoStatistics, &empty, &config).unwrap();
    assert!(matches!(
        &project_expr(&optimized).node,
        ExprNode::Resolved(ResolvedExpr::Function(_))
    ));
}

/// An identity UDF over every scalar literal type: folding covers each
/// `udf_array`/`udf_literal` arm, and nulls fold to null.
#[derive(Debug)]
struct TypedIdentity {
    data_type: DataType,
}

impl ScalarUdf for TypedIdentity {
    fn name(&self) -> &str {
        "typed_identity"
    }
    fn version(&self) -> u32 {
        1
    }
    fn signature(&self) -> &Signature {
        use std::sync::OnceLock;
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(|| Signature::Exact(vec![UdfInputType::Any]))
    }
    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: self.data_type.clone(),
            nullable: true,
        })
    }
    fn properties(&self) -> krishiv_udf_sdk::UdfProperties {
        krishiv_udf_sdk::UdfProperties {
            deterministic: true,
            null_handling: krishiv_udf_sdk::NullHandling::Strict,
        }
    }
    fn invoke(
        &self,
        args: &[UdfArg<'_>],
        _rows: usize,
        _: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        Ok(args[0].array())
    }
}

fn typed_call(data_type: DataType, literal: Literal) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: FunctionId::Udf {
                name: "typed_identity".to_owned(),
                version: 1,
            },
            kind: FunctionKind::Scalar,
            args: vec![Arc::new(Expr::literal(literal))],
            distinct: false,
            filter: None,
            data_type,
            nullable: true,
            deterministic: true,
        },
    ))))
}

fn fold_typed(
    data_type: DataType,
    literal: Literal,
    lib: &UdfLibrary,
    config: &Config,
) -> Arc<Expr> {
    let p = Plan::try_new(LogicalPlan::new(PlanNode::Project {
        input: Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
        exprs: vec![typed_call(data_type, literal)],
    }))
    .unwrap();
    project_expr(&optimize_with_udfs(&p, &NoStatistics, lib, config).unwrap())
}

#[test]
fn udf_folding_covers_every_scalar_type() {
    use krishiv_types::DecimalType;
    // Register one UDF per tested type (the library keys by name; reuse
    // the same name with the call's type, since folding reads the call).
    let config = config();
    let dec = |u: i128| Literal::Decimal {
        unscaled: u,
        data_type: DecimalType::try_new(10, 2).unwrap(),
    };
    let cases: Vec<(DataType, Literal)> = vec![
        (DataType::Boolean, Literal::Boolean(true)),
        (DataType::Byte, Literal::Byte(7)),
        (DataType::Short, Literal::Short(7)),
        (DataType::Integer, Literal::Integer(7)),
        (DataType::Long, Literal::Long(7)),
        (DataType::Float, Literal::Float(1.5)),
        (DataType::Double, Literal::Double(2.5)),
        (
            DataType::Decimal(DecimalType::try_new(10, 2).unwrap()),
            dec(150),
        ),
        (DataType::String, Literal::String("hi".to_owned())),
        (DataType::Binary, Literal::Binary(vec![1, 2])),
        (DataType::Date, Literal::Date(3)),
        (DataType::Timestamp, Literal::Timestamp(4)),
        (DataType::TimestampNtz, Literal::TimestampNtz(5)),
        (DataType::Integer, Literal::Null(Some(DataType::Integer))),
    ];
    for (data_type, literal) in cases {
        let lib = Arc::new(UdfLibrary::new());
        lib.register(Arc::new(TypedIdentity {
            data_type: data_type.clone(),
        }))
        .unwrap();
        let folded = fold_typed(data_type.clone(), literal.clone(), &lib, &config);
        assert!(
            matches!(folded.node, ExprNode::Literal(_)),
            "{data_type:?} folds to a literal, not {:?}",
            folded.node
        );
    }
}

#[test]
fn udf_folding_leaves_unfoldables_in_the_plan() {
    use krishiv_plan::Literal as L;
    let lib = Arc::new(UdfLibrary::new());
    lib.register(Arc::new(TypedIdentity {
        data_type: DataType::Integer,
    }))
    .unwrap();
    let config = config();
    // An interval literal has no UDF form.
    let interval = L::DayTimeInterval {
        micros: 1,
        start: krishiv_plan::DayTimeField::Day,
        end: krishiv_plan::DayTimeField::Day,
    };
    let folded = fold_typed(DataType::Integer, interval, &lib, &config);
    assert!(matches!(
        folded.node,
        ExprNode::Resolved(ResolvedExpr::Function(_))
    ));
    // DISTINCT, FILTER and aggregate calls never fold.
    let base = ResolvedFunction {
        id: FunctionId::Udf {
            name: "typed_identity".to_owned(),
            version: 1,
        },
        kind: FunctionKind::Scalar,
        args: vec![Arc::new(Expr::literal(L::Integer(1)))],
        distinct: false,
        filter: None,
        data_type: DataType::Integer,
        nullable: true,
        deterministic: true,
    };
    let variants = [
        ResolvedFunction {
            distinct: true,
            ..base.clone()
        },
        ResolvedFunction {
            filter: Some(Arc::new(Expr::literal(L::Boolean(true)))),
            ..base.clone()
        },
        ResolvedFunction {
            kind: FunctionKind::Aggregate,
            ..base
        },
    ];
    for mut g in variants {
        g.args = vec![Arc::new(Expr::literal(L::Integer(1)))];
        let p = Plan::try_new(LogicalPlan::new(PlanNode::Project {
            input: Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
            exprs: vec![Arc::new(Expr::new(ExprNode::Resolved(
                ResolvedExpr::Function(g),
            )))],
        }))
        .unwrap();
        let out = project_expr(&optimize_with_udfs(&p, &NoStatistics, &lib, &config).unwrap());
        assert!(matches!(
            out.node,
            ExprNode::Resolved(ResolvedExpr::Function(_))
        ));
    }
    // A failing call stays in the plan (errors surface at execution).
    #[derive(Debug)]
    struct Failing;
    impl ScalarUdf for Failing {
        fn name(&self) -> &str {
            "failing_udf"
        }
        fn version(&self) -> u32 {
            1
        }
        fn signature(&self) -> &Signature {
            use std::sync::OnceLock;
            static SIG: OnceLock<Signature> = OnceLock::new();
            SIG.get_or_init(|| Signature::Exact(vec![UdfInputType::Any]))
        }
        fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
            Ok(ReturnField {
                data_type: DataType::Integer,
                nullable: true,
            })
        }
        fn invoke(
            &self,
            _: &[UdfArg<'_>],
            _: usize,
            _: &InvokeContext,
        ) -> Result<ArrayRef, UdfError> {
            Err(UdfError::new("boom"))
        }
    }
    let lib2 = Arc::new(UdfLibrary::new());
    lib2.register(Arc::new(Failing)).unwrap();
    let p = Plan::try_new(LogicalPlan::new(PlanNode::Project {
        input: Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
        exprs: vec![Arc::new(Expr::new(ExprNode::Resolved(
            ResolvedExpr::Function(ResolvedFunction {
                id: FunctionId::Udf {
                    name: "failing_udf".to_owned(),
                    version: 1,
                },
                kind: FunctionKind::Scalar,
                args: vec![Arc::new(Expr::literal(L::Integer(1)))],
                distinct: false,
                filter: None,
                data_type: DataType::Integer,
                nullable: true,
                deterministic: true,
            }),
        )))],
    }))
    .unwrap();
    let out = project_expr(&optimize_with_udfs(&p, &NoStatistics, &lib2, &config).unwrap());
    assert!(matches!(
        out.node,
        ExprNode::Resolved(ResolvedExpr::Function(_))
    ));
    // A non-literal argument never folds.
    let resolved_col = || {
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
            krishiv_plan::BoundColumn {
                id: krishiv_plan::AttrId(1),
                name: "a".to_owned(),
                data_type: DataType::Integer,
                nullable: true,
            },
        ))))
    };
    let p = Plan::try_new(LogicalPlan::new(PlanNode::Project {
        input: Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
        exprs: vec![Arc::new(Expr::new(ExprNode::Resolved(
            ResolvedExpr::Function(ResolvedFunction {
                id: FunctionId::Udf {
                    name: "typed_identity".to_owned(),
                    version: 1,
                },
                kind: FunctionKind::Scalar,
                args: vec![resolved_col()],
                distinct: false,
                filter: None,
                data_type: DataType::Integer,
                nullable: true,
                deterministic: true,
            }),
        )))],
    }))
    .unwrap();
    let out = project_expr(&optimize_with_udfs(&p, &NoStatistics, &lib, &config).unwrap());
    assert!(matches!(
        out.node,
        ExprNode::Resolved(ResolvedExpr::Function(_))
    ));
    // An aggregate over expressions is left alone.
    let agg = Arc::new(LogicalPlan::new(PlanNode::Aggregate {
        input: Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
        grouping: krishiv_plan::Grouping::GroupBy(vec![typed_call(
            DataType::Integer,
            L::Integer(1),
        )]),
        aggregates: vec![],
    }));
    let p = Plan::try_new((*agg).clone()).unwrap();
    let out = optimize_with_udfs(&p, &NoStatistics, &lib, &config).unwrap();
    assert!(matches!(out.root().node, PlanNode::Aggregate { .. }));
}
