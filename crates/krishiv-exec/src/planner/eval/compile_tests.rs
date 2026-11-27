// SPDX-License-Identifier: Apache-2.0
//! Logical predicates through [`super::compile`]: the cost class of each kind of call
//! ([`super::cost`]), an `AND` chain compiled to one conjunction with its movable conjuncts and
//! barriers, and `OR` with a costly right side (study 60 (d)).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_common::config::Config;
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, CastMode, Expr, ExprNode, FunctionId, FunctionKind, Literal,
    ResolvedExpr, ResolvedFunction,
};
use krishiv_types::DataType;

use super::cost::{self, Cost};
use super::{Column, EvalEnv, Input, PhysExpr, compile};

fn env(ansi: bool) -> EvalEnv {
    let mut config = Config::new();
    config
        .set(
            "spark.sql.ansi.enabled",
            if ansi { "true" } else { "false" },
        )
        .unwrap();
    EvalEnv::new(&config, Arc::from("test"))
}

fn col(id: u64, data_type: DataType) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
        BoundColumn {
            id: AttrId(id),
            name: format!("c{id}"),
            data_type,
            nullable: true,
        },
    ))))
}

/// `i`, a BIGINT column.
fn i() -> Arc<Expr> {
    col(1, DataType::Long)
}

/// `s`, a STRING column.
fn s() -> Arc<Expr> {
    col(2, DataType::String)
}

fn lit(l: Literal) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Literal(l)))
}

fn call_id(id: FunctionId, args: Vec<Arc<Expr>>, data_type: DataType) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id,
            kind: FunctionKind::Scalar,
            args,
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type,
            nullable: true,
            deterministic: true,
        },
    ))))
}

fn call(b: Builtin, args: Vec<Arc<Expr>>) -> Arc<Expr> {
    call_id(FunctionId::Builtin(b), args, DataType::Boolean)
}

fn registry(name: &str, args: Vec<Arc<Expr>>, data_type: DataType) -> Arc<Expr> {
    call_id(
        FunctionId::Registry {
            name: name.to_owned(),
            id: 0,
        },
        args,
        data_type,
    )
}

/// `s LIKE 'a%'`.
fn like_a() -> Arc<Expr> {
    call(
        Builtin::Like,
        vec![s(), lit(Literal::String("a%".to_owned()))],
    )
}

/// `s = 'a'`.
fn s_is_a() -> Arc<Expr> {
    call(Builtin::Eq, vec![s(), lit(Literal::String("a".to_owned()))])
}

/// `i >= 1`.
fn i_at_least_1() -> Arc<Expr> {
    call(Builtin::GtEq, vec![i(), lit(Literal::Long(1))])
}

fn cast(e: &Arc<Expr>, to: DataType) -> Arc<Expr> {
    Arc::new(Expr::clone(e).cast(to))
}

#[test]
fn each_kind_of_call_has_its_cost_class() {
    assert_eq!(cost::class(&like_a()), Cost::Costly);
    assert_eq!(cost::class(&s_is_a()), Cost::String);
    assert_eq!(cost::class(&i_at_least_1()), Cost::Cheap);
    // A registry function costs much only on a string argument.
    let lower = registry("lower", vec![s()], DataType::String);
    assert_eq!(cost::class(&lower), Cost::Costly);
    let abs = registry("abs", vec![i()], DataType::Long);
    assert_eq!(cost::class(&abs), Cost::Cheap);
    // So does IN.
    let in_strings = call(Builtin::In, vec![s(), lit(Literal::String("a".to_owned()))]);
    assert_eq!(cost::class(&in_strings), Cost::Costly);
    let in_longs = call(Builtin::In, vec![i(), lit(Literal::Long(1))]);
    assert_eq!(cost::class(&in_longs), Cost::Cheap);
    // A UDF is always costly.
    let udf = call_id(
        FunctionId::Udf {
            name: "f".to_owned(),
            version: 1,
        },
        vec![i()],
        DataType::Boolean,
    );
    assert_eq!(cost::class(&udf), Cost::Costly);
    // The class is the costliest part's, through casts and other calls.
    let not_like = call(Builtin::Not, vec![cast(&like_a(), DataType::Boolean)]);
    assert_eq!(cost::class(&not_like), Cost::Costly);
    let not_eq = call(Builtin::Not, vec![cast(&s_is_a(), DataType::Boolean)]);
    assert_eq!(cost::class(&not_eq), Cost::String);
    // Past the depth limit the walk stops and calls it cheap (the old path).
    let mut deep = like_a();
    let mut deep_eq = s_is_a();
    for _ in 0..70 {
        deep = cast(&deep, DataType::Boolean);
        deep_eq = cast(&deep_eq, DataType::Boolean);
    }
    assert_eq!(cost::class(&deep), Cost::Cheap);
    assert_eq!(cost::class(&deep_eq), Cost::Cheap);
}

fn batch() -> (Arc<Schema>, RecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("i", ArrowType::Int64, true),
        Field::new("s", ArrowType::Utf8, true),
    ]));
    let i: ArrayRef = Arc::new(Int64Array::from(vec![
        Some(1),
        Some(2),
        Some(3),
        None,
        Some(0),
    ]));
    let s: ArrayRef = Arc::new(StringArray::from(vec![
        Some("ab"),
        Some("a"),
        Some("a"),
        None,
        Some("a"),
    ]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![i, s]).unwrap();
    (schema, batch)
}

fn answers(e: &Arc<dyn PhysExpr>, batch: &RecordBatch) -> Vec<Option<bool>> {
    let out = e
        .eval(batch, 0)
        .unwrap()
        .into_array(batch.num_rows())
        .unwrap();
    out.as_boolean().iter().collect()
}

#[test]
fn an_and_chain_compiles_to_one_conjunction_of_its_conjuncts() {
    let (schema, batch) = batch();
    let layout = [AttrId(1), AttrId(2)];
    // (i >= 1 AND s LIKE 'a%') AND (CAST(s AS BIGINT) >= 1 OR s = 'a')
    for ansi in [false, true] {
        // CAST(s AS BIGINT), as the analyzer pins it under `ansi`.
        let s_as_long = Arc::new(Expr::new(ExprNode::Cast {
            expr: s(),
            to: DataType::Long,
            mode: if ansi {
                CastMode::Ansi
            } else {
                CastMode::Legacy
            },
        }));
        let raising = call(
            Builtin::Or,
            vec![
                call(Builtin::GtEq, vec![s_as_long, lit(Literal::Long(1))]),
                s_is_a(),
            ],
        );
        let chain = call(
            Builtin::And,
            vec![call(Builtin::And, vec![i_at_least_1(), like_a()]), raising],
        );
        let env = env(ansi);
        let input = Input {
            layout: &layout,
            schema: &schema,
            env: &env,
            subqueries: &[],
        };
        let e = compile(&chain, input).unwrap();
        let shown = format!("{e:?}");
        assert!(shown.starts_with("Conjunction"), "{shown}");
        assert_eq!(shown.matches("movable:").count(), 3, "{shown}");
        assert!(shown.contains("cost: Costly"), "{shown}");
        // Under ANSI the cast of a string raises on bad input: the OR is a barrier.
        assert_eq!(shown.contains("movable: false"), ansi, "{shown}");
        assert_eq!(e.data_type(), &ArrowType::Boolean);
        assert!(e.nullable());
        if ansi {
            // Row 0's `ab` does not cast, and Spark evaluates the OR on it (both earlier
            // conjuncts are true there).
            assert!(e.eval(&batch, 0).is_err());
        } else {
            assert_eq!(
                answers(&e, &batch),
                [None, Some(true), Some(true), None, Some(false)]
            );
        }
    }
}

#[test]
fn an_or_with_a_costly_right_side_answers_kleene() {
    let (schema, batch) = batch();
    let layout = [AttrId(1), AttrId(2)];
    let env = env(false);
    let input = Input {
        layout: &layout,
        schema: &schema,
        env: &env,
        subqueries: &[],
    };
    let or = call(
        Builtin::Or,
        vec![
            call(Builtin::Eq, vec![i(), lit(Literal::Long(2))]),
            like_a(),
        ],
    );
    let e = compile(&or, input).unwrap();
    assert_eq!(
        answers(&e, &batch),
        [Some(true), Some(true), Some(true), None, Some(true)]
    );
}

#[test]
fn the_runtime_filter_and_is_kleene() {
    let column = |index| -> Arc<dyn PhysExpr> {
        Arc::new(Column {
            index,
            data_type: ArrowType::Boolean,
            nullable: true,
        })
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("l", ArrowType::Boolean, true),
        Field::new("r", ArrowType::Boolean, true),
    ]));
    let l = BooleanArray::from(vec![Some(true), Some(true), Some(false), None, None]);
    let r = BooleanArray::from(vec![Some(true), None, None, Some(false), Some(true)]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(l), Arc::new(r)]).unwrap();
    let e = super::and(column(0), column(1));
    assert_eq!(
        answers(&e, &batch),
        [Some(true), None, Some(false), Some(false), None]
    );
}
