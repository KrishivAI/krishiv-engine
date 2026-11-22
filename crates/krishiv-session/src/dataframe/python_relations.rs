// SPDX-License-Identifier: Apache-2.0
//! The embedded Python relation kinds on a frame (design P3-T085 §3.4): `mapInPandas` / `mapInArrow`,
//! `groupBy(..).applyInPandas` / `applyInArrow` and `cogroup(..).applyInPandas` / `applyInArrow`. Each builds
//! the plan node the Connect translator builds for the same call (`krishiv-connect`'s `relation_udf`), over a
//! UDF registered as an unregistered one (a top-bit version, [`crate::ANONYMOUS_VERSION_BIT`]).

use std::sync::Arc;

use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_plan::{AttrId, BoundColumn, PlanNode, PythonEvalType, PythonUdfRef};
use krishiv_types::DataType;

use super::{DataFrame, GroupedData, Sets};
use crate::column::Column;
use crate::python_udfs::PythonUdfDef;

/// The reference a plan names `def` (at `version`) by, and the result columns it declares.
fn relation_udf(
    def: &PythonUdfDef,
    version: u32,
    accept: &[PythonEvalType],
) -> Result<(PythonUdfRef, Vec<BoundColumn>), KrishivError> {
    if !accept.contains(&def.eval_type) {
        return Err(InternalError {
            message: format!(
                "a Python UDF of eval type {:?} cannot run as this relation",
                def.eval_type
            ),
        }
        .into());
    }
    let DataType::Struct(fields) = &def.return_type else {
        return Err(InternalError {
            message: format!(
                "the return type of a Python relation UDF must be a struct, got {}",
                def.return_type.sql()
            ),
        }
        .into());
    };
    let output = fields
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| BoundColumn {
            // Kept once per analysis, then made fresh.
            id: AttrId(i as u64),
            name: f.name().to_owned(),
            data_type: f.data_type().clone(),
            nullable: f.nullable(),
        })
        .collect();
    let udf = PythonUdfRef {
        name: def.name.to_ascii_lowercase(),
        version,
        eval_type: def.eval_type,
        thread_safe: def.thread_safe,
        deterministic: def.deterministic,
    };
    Ok((udf, output))
}

impl DataFrame {
    /// `mapInPandas` / `mapInArrow`: each partition goes to the Python UDF `def` as an iterator of batches
    /// and its rows are what it yields, typed as `def`'s struct return type.
    ///
    /// # Errors
    /// `KRISHIV_INTERNAL_ERROR` for a UDF that is not a map kind (205, 207) or whose return type is not a
    /// struct.
    pub fn map_in_python(
        &self,
        def: &PythonUdfDef,
        version: u32,
        barrier: bool,
    ) -> Result<Self, KrishivError> {
        let (udf, output) = relation_udf(
            def,
            version,
            &[PythonEvalType::MapPandasIter, PythonEvalType::MapArrowIter],
        )?;
        Ok(self.derive(PlanNode::MapPartitions {
            input: Arc::clone(&self.plan),
            udf,
            output,
            barrier,
        }))
    }

    /// `df.groupBy(left_keys).cogroup(other.groupBy(right_keys)).applyInPandas` / `applyInArrow`: per key,
    /// the UDF sees the left rows then the right rows (an empty side for a key on one side only).
    ///
    /// # Errors
    /// `KRISHIV_INTERNAL_ERROR` for a UDF that is not a cogrouped-map kind (206, 210) or whose return type is
    /// not a struct, or for different key counts.
    pub fn cogroup_apply_in_python(
        &self,
        other: &Self,
        left_keys: impl IntoIterator<Item = Column>,
        right_keys: impl IntoIterator<Item = Column>,
        def: &PythonUdfDef,
        version: u32,
    ) -> Result<Self, KrishivError> {
        let left_keys: Vec<_> = left_keys.into_iter().map(|c| Arc::new(c.expr)).collect();
        let right_keys: Vec<_> = right_keys.into_iter().map(|c| Arc::new(c.expr)).collect();
        if left_keys.len() != right_keys.len() {
            return Err(InternalError {
                message: "a cogroup's sides need the same number of grouping keys".to_owned(),
            }
            .into());
        }
        let (udf, output) = relation_udf(
            def,
            version,
            &[
                PythonEvalType::CogroupedMapPandas,
                PythonEvalType::CogroupedMapArrow,
            ],
        )?;
        let mut out = self.derive(PlanNode::CoGroupMap {
            left: Arc::clone(&self.plan),
            right: Arc::clone(&other.plan),
            left_keys,
            right_keys,
            udf,
            output,
        });
        // The right side's hidden tables and lazy checkpoints are owned by the result too, as a join's are.
        for k in &other.keep {
            if !out.keep.iter().any(|have| Arc::ptr_eq(have, k)) {
                out.keep.push(Arc::clone(k));
            }
        }
        Ok(out)
    }
}

impl GroupedData {
    /// `applyInPandas` / `applyInArrow`: the rows of each group go to the Python UDF `def`, which yields any
    /// rows, typed as its struct return type.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for `rollup` and `cube`; `KRISHIV_INTERNAL_ERROR` for a UDF that is not a
    /// grouped-map kind (201, 209, 215, 216) or whose return type is not a struct.
    pub fn apply_in_python(
        &self,
        def: &PythonUdfDef,
        version: u32,
    ) -> Result<DataFrame, KrishivError> {
        if !matches!(self.sets, Sets::GroupBy) {
            return Err(KrishivUnsupportedFeature {
                feature: "applyInPandas over rollup or cube".to_owned(),
            }
            .into());
        }
        let (udf, output) = relation_udf(
            def,
            version,
            &[
                PythonEvalType::GroupedMapPandas,
                PythonEvalType::GroupedMapArrow,
                PythonEvalType::GroupedMapPandasIter,
                PythonEvalType::GroupedMapArrowIter,
            ],
        )?;
        Ok(self.df.derive(PlanNode::GroupMap {
            input: Arc::clone(&self.df.plan),
            keys: self.keys.iter().cloned().map(Arc::new).collect(),
            udf,
            output,
        }))
    }
}
