// SPDX-License-Identifier: Apache-2.0
//! The relation sink's aligned shape (a window aggregate, a scalar
//! iterator) over the fake runtime of `tests.rs`: rows joined with their
//! values in order, groups, a wrong value count, and memory.

use super::*;

/// The aligned kinds (a scalar iterator, a window aggregate): each row comes
/// back with its own value, whatever batches the task yields the values in
/// (Spark's JVM side zips its row queue with the results).
#[tokio::test]
async fn an_aligned_task_s_values_join_their_rows_in_order() {
    for yields in [Yield::Values, Yield::GroupValues] {
        let r = rig(
            RelationShape::Aligned { keys: None },
            yields,
            limits(2, 16),
            64 << 20,
        );
        let got = drive_rows(&r, vec![rows(&[1, 2, 3], None), rows(&[4, 5], None)])
            .await
            .unwrap();
        // Cut by `maxRecordsPerBatch` as a map is; no group marker.
        assert_eq!(r.shared.log(), ["B0:2", "B0:2", "B0:1", "E"]);
        assert_eq!(
            got,
            [(1, 0, 0), (2, 1, 1), (3, 2, 2), (4, 0, 0), (5, 1, 1)],
            "every row once, with its own value"
        );
        done(r);
    }
}

/// A window aggregate's partitions are groups, each ended by its marker; the
/// values of a partition may come in one batch at its end.
#[tokio::test]
async fn an_aligned_task_with_keys_sees_each_partition_as_a_group() {
    let r = rig(
        RelationShape::Aligned {
            keys: Some(vec![0]),
        },
        Yield::GroupValues,
        limits(0, 16),
        64 << 20,
    );
    let got = drive_rows(&r, vec![rows(&[5, 5, 7], None), rows(&[7, 9], None)])
        .await
        .unwrap();
    assert_eq!(
        r.shared.log(),
        ["B0:2", "G", "B0:1", "B0:1", "G", "B0:1", "G", "E"]
    );
    assert_eq!(got, [(5, 0, 0), (5, 1, 1), (7, 2, 2), (7, 0, 0), (9, 1, 1)]);
    done(r);
}

/// A task yielding another number of values than rows fails the phase with
/// the result class, naming rows (never a short or misaligned result).
#[tokio::test]
async fn an_aligned_task_yielding_another_number_of_values_than_rows_fails() {
    for more in [false, true] {
        let r = rig(
            RelationShape::Aligned { keys: None },
            Yield::ValuesOff(more),
            limits(0, 16),
            64 << 20,
        );
        let err = drive(&r, vec![rows(&[1, 2, 3], None)]).await.unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::KrishivPythonUdfResultMismatch,
            "{err}"
        );
        let want = if more {
            "4 values for 3 rows"
        } else {
            "2 values for 3 rows"
        };
        assert!(err.to_string().contains(want), "{err}");
        done(r);
    }
}

/// The kept rows share the input batch's one reservation with the frames
/// cut from them: nothing is left reserved after the phase, and a refused
/// reservation takes no input (the retry is exact).
#[tokio::test]
async fn an_aligned_sink_refusing_memory_takes_nothing() {
    let r = rig(
        RelationShape::Aligned { keys: None },
        Yield::Values,
        limits(0, 16),
        2 << 10,
    );
    let g = r.sink.global_state(&r.ctx).unwrap();
    let mut l = r.sink.local_state(&r.ctx, &g).unwrap();
    let big: Vec<i32> = (0..4096).collect();
    match r.sink.sink(rows(&big, None), &mut l, &g).unwrap() {
        SinkStatus::NeedMemory { .. } => {}
        _ => panic!("the batch is over the limit"),
    }
    assert!(r.shared.log().is_empty(), "nothing framed");
    drop((l, g));
    done(r);
}
