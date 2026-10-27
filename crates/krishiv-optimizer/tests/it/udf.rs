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
