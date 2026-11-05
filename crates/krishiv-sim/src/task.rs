// SPDX-License-Identifier: Apache-2.0
//! The task descriptor a coordinator ships with a fragment's bytes
//! (design P2-T176 §2, Q2): a [`TaskSpec`] as an unstable protobuf message,
//! `krishiv.task.v1alpha`. No compatibility promise: P5-T004 freezes the
//! task envelope. Scan splits are refused until the coordinator deals them
//! (P2-T178, design P2-T176 order wave C).

use std::collections::BTreeMap;

use krishiv_common::error::{InternalError, KrishivError};
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

/// `task` as bytes.
///
/// # Errors
/// `INTERNAL_ERROR` for a task with scan splits (not shipped yet).
pub fn encode_task(task: &TaskSpec) -> Result<Vec<u8>, KrishivError> {
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
    };
    Ok(message.encode_to_vec())
}

/// The task of `bytes`.
///
/// # Errors
/// `INTERNAL_ERROR` for bytes that are not a task, or name an exchange
/// twice.
pub fn decode_task(bytes: &[u8]) -> Result<TaskSpec, KrishivError> {
    let message = Task::decode(bytes).map_err(|e| internal(format!("a task descriptor: {e}")))?;
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
    Ok(TaskSpec {
        stage: message.stage,
        stage_attempt: message.stage_attempt,
        partition: message.partition,
        partitions: message.partitions,
        attempt: message.attempt,
        scans: BTreeMap::new(),
        exchanges,
        output: message.output,
    })
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
        let back = decode_task(&encode_task(&task).unwrap()).unwrap();
        assert_eq!(format!("{back:?}"), format!("{task:?}"));
        let root = TaskSpec {
            output: None,
            ..TaskSpec::whole()
        };
        assert_eq!(
            decode_task(&encode_task(&root).unwrap()).unwrap().output,
            None
        );
    }

    #[test]
    fn garbage_is_refused() {
        assert!(decode_task(&[0xff, 0xff, 0xff]).is_err());
    }
}
