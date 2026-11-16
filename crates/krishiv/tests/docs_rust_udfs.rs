// SPDX-License-Identifier: Apache-2.0
//! The examples of `docs/reference/rust-udfs.md`, compiled and run so the page cannot rot (RS-20).
//! Keep them identical to the page, which uses only the `krishiv` crate.

use std::sync::{Arc, OnceLock};

use futures::executor::block_on;
use krishiv::arrow::array::{Array, ArrayRef, Int32Array, Int64Array, RecordBatch};
use krishiv::arrow::datatypes::{DataType as ArrowType, Field, Schema};
use krishiv::functions::{call_udf, col};
use krishiv::{
    AggregateUdf, ArgField, DataType, GroupsAccumulator, InvokeContext, ReturnField, ScalarUdf,
    Session, Signature, StateField, UdfArg, UdfError, UdfInputType,
};

#[derive(Debug)]
struct AddOne;

impl ScalarUdf for AddOne {
    fn name(&self) -> &str {
        "add_one"
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &Signature {
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(|| Signature::Exact(vec![UdfInputType::Exact(DataType::Integer)]))
    }

    fn return_field(&self, _args: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::Integer,
            nullable: true,
        })
    }

    fn invoke(
        &self,
        args: &[UdfArg<'_>],
        _rows: usize,
        _ctx: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        let input = args[0].array();
        let ints = input
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| UdfError::new("add_one needs an INT argument"))?;
        let mut out = Vec::with_capacity(ints.len());
        for v in ints.iter() {
            out.push(match v {
                None => None,
                Some(v) => Some(
                    v.checked_add(1)
                        .ok_or_else(|| UdfError::new("add_one overflowed"))?,
                ),
            });
        }
        Ok(Arc::new(Int32Array::from(out)))
    }
}

#[derive(Debug)]
struct Total;

impl AggregateUdf for Total {
    fn name(&self) -> &str {
        "total"
    }
    fn version(&self) -> u32 {
        1
    }
    fn signature(&self) -> &Signature {
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(|| Signature::Exact(vec![UdfInputType::Exact(DataType::Long)]))
    }
    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::Long,
            nullable: true,
        })
    }
    fn state_fields(&self, _: &[ArgField]) -> Result<Vec<StateField>, UdfError> {
        Ok(vec![StateField {
            name: "sum".into(),
            data_type: DataType::Long,
            nullable: true,
        }])
    }
    fn accumulator(
        &self,
        _: &[ArgField],
        _: &InvokeContext,
    ) -> Result<Box<dyn GroupsAccumulator>, UdfError> {
        Ok(Box::new(Sums(Vec::new())))
    }
}

/// One running sum per group; `None` is a group that has seen no rows.
#[derive(Debug)]
struct Sums(Vec<Option<i64>>);

impl Sums {
    fn add(&mut self, group: u32, v: i64) -> Result<(), UdfError> {
        let slot = &mut self.0[group as usize];
        let sum = slot
            .unwrap_or(0)
            .checked_add(v)
            .ok_or_else(|| UdfError::new("total overflowed"))?;
        *slot = Some(sum);
        Ok(())
    }

    fn column(&self, groups: &[u32]) -> ArrayRef {
        Arc::new(Int64Array::from_iter(
            groups.iter().map(|g| self.0[*g as usize]),
        ))
    }
}

impl GroupsAccumulator for Sums {
    fn resize(&mut self, groups: usize) {
        self.0.resize(groups, None);
    }

    fn update(&mut self, args: &[UdfArg<'_>], groups: &[u32]) -> Result<(), UdfError> {
        let values = args[0].array();
        let values = values
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| UdfError::new("total needs a BIGINT argument"))?;
        for (row, group) in groups.iter().enumerate() {
            self.add(*group, values.value(row))?;
        }
        Ok(())
    }

    fn merge(&mut self, states: &[ArrayRef], groups: &[u32]) -> Result<(), UdfError> {
        let partial = states[0]
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| UdfError::new("total state is not a BIGINT"))?;
        for (row, group) in groups.iter().enumerate() {
            if partial.is_valid(row) {
                self.add(*group, partial.value(row))?;
            }
        }
        Ok(())
    }

    fn state(&self, groups: &[u32]) -> Result<Vec<ArrayRef>, UdfError> {
        Ok(vec![self.column(groups)])
    }

    fn evaluate(&self, groups: &[u32]) -> Result<ArrayRef, UdfError> {
        Ok(self.column(groups))
    }

    fn size(&self) -> usize {
        self.0.capacity() * std::mem::size_of::<Option<i64>>()
    }
}

#[test]
fn the_scalar_example_runs() {
    let session = Session::new().unwrap();
    session.register_udf(AddOne).unwrap();
    let df =
        block_on(session.sql("SELECT add_one(id) AS v FROM VALUES (1), (NULL) AS t(id)")).unwrap();
    assert_eq!(
        block_on(df.collect())
            .unwrap()
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        2
    );
    let schema = Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int32, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1]))],
    )
    .unwrap();
    session
        .register_batches("people", schema, vec![batch])
        .unwrap();
    let df = session
        .table("people")
        .select([call_udf("add_one", [col("id")])]);
    assert_eq!(
        block_on(df.collect())
            .unwrap()
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        1
    );
}

#[test]
fn the_aggregate_example_runs() {
    let session = Session::new().unwrap();
    session.register_udaf(Total).unwrap();
    let df = block_on(session.sql(
        "SELECT k, total(v) FROM VALUES (1, 10L), (1, 5L), (2, 7L) AS t(k, v) GROUP BY k ORDER BY k",
    ))
    .unwrap();
    assert_eq!(
        block_on(df.collect())
            .unwrap()
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        2
    );
}
