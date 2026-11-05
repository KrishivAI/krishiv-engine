// SPDX-License-Identifier: Apache-2.0
//! The task descriptor a coordinator ships with a fragment's bytes
//! (design P2-T176 §2, Q2): a [`TaskSpec`] as an unstable protobuf message,
//! `krishiv.task.v1alpha`. No compatibility promise: P5-T004 freezes the
//! task envelope. It carries the submitted query's execution settings
//! ([`ExecConfig`]), so a remote task runs as a local one would. Scan splits
//! are refused until the coordinator deals them (P2-T178, design P2-T176
//! order wave C).

use std::collections::BTreeMap;
use std::time::Duration;

use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::ExecConfig;
use krishiv_exec::physical::{ExchangeId, ExchangeInput, TaskSpec};
use prost::Message as _;

/// `krishiv.task.v1alpha.Task`.
#[derive(Clone, PartialEq, prost::Message)]
struct Task {
    #[prost(uint32, tag = "1")]
    stage: u32,
    #[prost(uint32, tag = "2")]
    stage_attempt: u32,
    #[prost(uint32, tag = "3")]
    partition: u32,
    #[prost(uint32, tag = "4")]
    partitions: u32,
    #[prost(uint32, tag = "5")]
    attempt: u32,
    #[prost(message, repeated, tag = "6")]
    exchanges: Vec<Input>,
    #[prost(uint32, optional, tag = "7")]
    output: Option<u32>,
    #[prost(message, optional, tag = "8")]
    settings: Option<Settings>,
}

/// `krishiv.task.v1alpha.Settings`: the query's [`ExecConfig`].
#[derive(Clone, PartialEq, prost::Message)]
struct Settings {
    #[prost(uint64, tag = "1")]
    batch_size: u64,
    #[prost(uint64, tag = "2")]
    morsel_rows: u64,
    #[prost(uint64, tag = "3")]
    result_buffer_per_slot: u64,
    #[prost(uint64, tag = "4")]
    stall_threshold_nanos: u64,
    #[prost(bool, tag = "5")]
    fail_on_stall: bool,
    #[prost(uint64, optional, tag = "6")]
    seed: Option<u64>,
    #[prost(bool, tag = "7")]
    agg_row_layout: bool,
    #[prost(bool, tag = "8")]
    detailed_metrics: bool,
    #[prost(bool, tag = "9")]
    join_row_layout: bool,
    #[prost(uint32, tag = "10")]
    join_radix_bits: u32,
    #[prost(bool, tag = "11")]
    sort_hash_buckets: bool,
    #[prost(uint64, tag = "12")]
    join_merge_buffer_bytes: u64,
    #[prost(bool, tag = "13")]
    deterministic_order: bool,
}

fn wide(v: usize) -> u64 {
    u64::try_from(v).unwrap_or(u64::MAX)
}

fn narrow(v: u64, what: &str) -> Result<usize, KrishivError> {
    usize::try_from(v).map_err(|_| internal(format!("a task's {what} of {v}")))
}

impl Settings {
    fn of(c: &ExecConfig) -> Self {
        // Exhaustive: a new field fails to compile here until it travels.
        let ExecConfig {
            batch_size,
            morsel_rows,
            result_buffer_per_slot,
            stall_threshold,
            fail_on_stall,
            seed,
            agg_row_layout,
            detailed_metrics,
            join_row_layout,
            join_radix_bits,
            sort_hash_buckets,
            join_merge_buffer_bytes,
            deterministic_order,
        } = c.clone();
        Self {
            batch_size: wide(batch_size),
            morsel_rows: wide(morsel_rows),
            result_buffer_per_slot: wide(result_buffer_per_slot),
            stall_threshold_nanos: u64::try_from(stall_threshold.as_nanos()).unwrap_or(u64::MAX),
            fail_on_stall,
            seed,
            agg_row_layout,
            detailed_metrics,
            join_row_layout,
            join_radix_bits,
            sort_hash_buckets,
            join_merge_buffer_bytes: wide(join_merge_buffer_bytes),
            deterministic_order,
        }
    }

    fn config(&self) -> Result<ExecConfig, KrishivError> {
        Ok(ExecConfig {
            batch_size: narrow(self.batch_size, "batch size")?,
            morsel_rows: narrow(self.morsel_rows, "morsel rows")?,
            result_buffer_per_slot: narrow(self.result_buffer_per_slot, "result buffer")?,
            stall_threshold: Duration::from_nanos(self.stall_threshold_nanos),
            fail_on_stall: self.fail_on_stall,
            seed: self.seed,
            agg_row_layout: self.agg_row_layout,
            detailed_metrics: self.detailed_metrics,
            join_row_layout: self.join_row_layout,
            join_radix_bits: self.join_radix_bits,
            sort_hash_buckets: self.sort_hash_buckets,
            join_merge_buffer_bytes: narrow(self.join_merge_buffer_bytes, "merge buffer")?,
            deterministic_order: self.deterministic_order,
        })
    }
}

/// `krishiv.task.v1alpha.Input`: one entry of [`TaskSpec::exchanges`].
#[derive(Clone, PartialEq, prost::Message)]
struct Input {
    #[prost(uint64, tag = "1")]
    exchange: u64,
    #[prost(uint32, tag = "2")]
    stage_attempt: u32,
    #[prost(uint32, tag = "3")]
    maps: u32,
    #[prost(uint32, repeated, tag = "4")]
    reduce: Vec<u32>,
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// `task` of a query run with `settings`, as bytes.
///
/// # Errors
/// `INTERNAL_ERROR` for a task with scan splits (not shipped yet).
pub fn encode_task(task: &TaskSpec, settings: &ExecConfig) -> Result<Vec<u8>, KrishivError> {
    if !task.scans.is_empty() {
        return Err(internal(format!(
            "task {} of stage {}: scan splits are not shipped to workers yet",
            task.partition, task.stage
        )));
    }
    let message = Task {
        stage: task.stage,
        stage_attempt: task.stage_attempt,
        partition: task.partition,
        partitions: task.partitions,
        attempt: task.attempt,
        exchanges: task
            .exchanges
            .iter()
            .map(|(id, input)| Input {
                exchange: id.0,
                stage_attempt: input.stage_attempt,
                maps: input.maps,
                reduce: input.reduce.clone(),
            })
            .collect(),
        output: task.output,
        settings: Some(Settings::of(settings)),
    };
    Ok(message.encode_to_vec())
}

/// The task of `bytes` and its query's execution settings.
///
/// # Errors
/// `INTERNAL_ERROR` for bytes that are not a task, lack settings, or name
/// an exchange twice.
pub fn decode_task(bytes: &[u8]) -> Result<(TaskSpec, ExecConfig), KrishivError> {
    let message = Task::decode(bytes).map_err(|e| internal(format!("a task descriptor: {e}")))?;
    let settings = message
        .settings
        .as_ref()
        .ok_or_else(|| internal("a task descriptor without settings".to_owned()))?
        .config()?;
    let mut exchanges = BTreeMap::new();
    for input in message.exchanges {
        let entry = ExchangeInput {
            stage_attempt: input.stage_attempt,
            maps: input.maps,
            reduce: input.reduce,
        };
        if exchanges
            .insert(ExchangeId(input.exchange), entry)
            .is_some()
        {
            return Err(internal(format!(
                "a task descriptor reads exchange {} twice",
                input.exchange
            )));
        }
    }
    let task = TaskSpec {
        stage: message.stage,
        stage_attempt: message.stage_attempt,
        partition: message.partition,
        partitions: message.partitions,
        attempt: message.attempt,
        scans: BTreeMap::new(),
        exchanges,
        output: message.output,
    };
    Ok((task, settings))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_task_round_trips() {
        let task = TaskSpec {
            stage: 3,
            stage_attempt: 1,
            partition: 2,
            partitions: 4,
            attempt: 1,
            scans: BTreeMap::new(),
            exchanges: BTreeMap::from([(
                ExchangeId(7),
                ExchangeInput {
                    stage_attempt: 0,
                    maps: 5,
                    reduce: vec![2],
                },
            )]),
            output: Some(4),
        };
        let settings = ExecConfig {
            batch_size: 77,
            seed: Some(9),
            detailed_metrics: true,
            join_radix_bits: 3,
            stall_threshold: Duration::from_millis(1_234),
            ..ExecConfig::default()
        };
        let (back, got) = decode_task(&encode_task(&task, &settings).unwrap()).unwrap();
        assert_eq!(format!("{back:?}"), format!("{task:?}"));
        assert_eq!(format!("{got:?}"), format!("{settings:?}"));
        let root = TaskSpec {
            output: None,
            ..TaskSpec::whole()
        };
        let defaults = ExecConfig::default();
        let (root_back, got) = decode_task(&encode_task(&root, &defaults).unwrap()).unwrap();
        assert_eq!(root_back.output, None);
        assert_eq!(format!("{got:?}"), format!("{defaults:?}"));
    }

    #[test]
    fn garbage_is_refused() {
        assert!(decode_task(&[0xff, 0xff, 0xff]).is_err());
    }
}
