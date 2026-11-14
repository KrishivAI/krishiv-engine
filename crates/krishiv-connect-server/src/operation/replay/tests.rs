// SPDX-License-Identifier: Apache-2.0
use super::*;

fn held() -> Held {
    Held {
        ahead: None,
        reservation: None,
    }
}

/// Pushes responses of the given sizes; returns their ids.
fn fill(r: &Replay, sizes: &[usize]) -> Vec<String> {
    sizes
        .iter()
        .enumerate()
        .map(|(i, size)| {
            let bits = 0x1000 + i as u128;
            let id = uuid_text(bits);
            let response = pb::ExecutePlanResponse {
                response_id: id.clone(),
                ..Default::default()
            };
            assert!(r.push(response, bits, *size, held()));
            id
        })
        .collect()
}

fn take(r: &Replay, generation: u64, index: u64) -> String {
    match r.next(generation, index, true) {
        Next::Response(resp, _) => resp.response_id,
        other => panic!("{other:?}"),
    }
}

#[test]
fn uuid_text_round_trips_and_other_text_is_unknown() {
    let id = "00112233-4455-6677-8899-aabbccddeeff";
    let bits = uuid_bits(id).unwrap();
    assert_eq!(uuid_text(bits), id);
    assert_eq!(uuid_bits("00112233-4455-6677-8899-AABBCCDDEEFF"), None);
    assert_eq!(uuid_bits("0011223344556677-8899-aabbccddeeff-"), None);
    assert_eq!(uuid_bits("nope"), None);
}

#[test]
fn consumed_responses_stay_while_under_the_retry_buffer() {
    // Spark keeps responses back from index - 1 while the total before
    // each is under the buffer: with 25 bytes and 10-byte responses,
    // consuming 5 keeps 4, 3 and 2 (10 + 10 + 10 crosses 25 at 2).
    let r = Replay::new(25, true);
    let ids = fill(&r, &[10; 6]);
    let g = r.attach(None);
    for i in 1..=5 {
        assert_eq!(take(&r, g, i), ids[usize::try_from(i - 1).unwrap()]);
    }
    assert_eq!(
        r.bytes(),
        50,
        "2, 3 and 4 kept, 5 being sent, 6 not consumed"
    );
    assert!(matches!(r.next(g, 1, true), Next::Unavailable(_)));
    let Next::Unavailable(e) = r.next(g, 1, true) else {
        unreachable!()
    };
    assert_eq!(
        e.error_class().name(),
        "INVALID_CURSOR.POSITION_NOT_AVAILABLE"
    );
    // A reattach after 1 resumes at 2, which is still there.
    assert_eq!(take(&r, g, 2), ids[1]);
}

#[test]
fn a_zero_retry_buffer_keeps_only_the_response_being_sent() {
    let r = Replay::new(0, true);
    let ids = fill(&r, &[10; 3]);
    let g = r.attach(None);
    assert_eq!(take(&r, g, 1), ids[0]);
    assert_eq!(take(&r, g, 2), ids[1]);
    assert!(matches!(r.next(g, 1, true), Next::Unavailable(_)));
    assert_eq!(take(&r, g, 2), ids[1], "the last sent one stays");
}

#[test]
fn a_non_reattachable_buffer_hands_responses_out() {
    let r = Replay::new(0, false);
    fill(&r, &[10; 2]);
    let g = r.attach(None);
    take(&r, g, 1);
    assert_eq!(r.bytes(), 10);
    take(&r, g, 2);
    assert_eq!(r.bytes(), 0);
    assert_eq!(r.peak(), 20);
}

#[test]
fn release_until_drops_and_ids_are_never_forgotten() {
    let r = Replay::new(1 << 20, true);
    let ids = fill(&r, &[10; 4]);
    r.release_until(&ids[1]).unwrap();
    assert_eq!(r.bytes(), 20);
    // Released already: still known (Spark's `responseIdToIndex`).
    r.release_until(&ids[0]).unwrap();
    assert_eq!(r.index_of(&ids[1]).unwrap(), 2);
    let e = r.release_until("never").unwrap_err();
    assert_eq!(e.error_class().name(), "INVALID_CURSOR.POSITION_NOT_FOUND");
    let e = r.index_of(&uuid_text(7)).unwrap_err();
    assert_eq!(e.error_class().name(), "INVALID_CURSOR.POSITION_NOT_FOUND");
    let g = r.attach(None);
    assert_eq!(take(&r, g, 3), ids[2]);
}

#[test]
fn the_end_comes_after_every_response_and_a_new_sender_replaces_the_old() {
    let r = Replay::new(1 << 20, true);
    fill(&r, &[1; 2]);
    r.finish(Some(Status::internal("boom")));
    assert!(!r.push(pb::ExecutePlanResponse::default(), 9, 1, held()));
    let old = r.attach(None);
    take(&r, old, 1);
    let new = r.attach(None);
    assert!(matches!(r.next(old, 2, true), Next::Disconnected));
    take(&r, new, 2);
    assert!(matches!(r.next(new, 3, true), Next::Finished(Some(_))));
    // Caps reached: no response, but the end is still reported.
    assert!(matches!(r.next(new, 2, false), Next::Wait));
}

#[test]
fn detach_and_close() {
    let r = Replay::new(1 << 20, true);
    assert_eq!(r.detached_since(), None);
    let g = r.attach(None);
    let g2 = r.attach(None);
    r.detach(g);
    assert_eq!(
        r.detached_since(),
        None,
        "a replaced sender detaches nothing"
    );
    r.detach(g2);
    assert!(r.detached_since().is_some());
    r.attach(None);
    assert_eq!(r.detached_since(), None);
    fill(&r, &[5; 3]);
    r.close();
    assert!(r.is_closed());
    assert_eq!(r.bytes(), 0);
    assert!(matches!(r.next(g2 + 1, 1, true), Next::Disconnected));
    assert!(matches!(r.next(g, 1, true), Next::Disconnected));
    assert!(!r.push(pb::ExecutePlanResponse::default(), 9, 1, held()));
    assert_eq!(r.produced(), 3);
}

#[test]
fn a_finished_stream_with_nothing_buffered_ends_its_sender() {
    // Spark's `streamFinished` includes `isCleaned()` (finished, buffer
    // empty): a sender whose next responses were released ends.
    let r = Replay::new(100, true);
    let ids = fill(&r, &[10; 3]);
    let g = r.attach(None);
    assert_eq!(take(&r, g, 1), ids[0]);
    r.release_until(&ids[2]).unwrap();
    r.finish(None);
    assert!(matches!(r.next(g, 2, true), Next::Finished(None)));
}

#[test]
fn a_kept_response_shares_its_arrow_bytes_with_the_buffer() {
    use krishiv_connect::proto::v4_3::execute_plan_response::{ArrowBatch, ResponseType};
    let r = Replay::new(1 << 20, true);
    let bits = 0x2000_u128;
    let response = pb::ExecutePlanResponse {
        response_id: uuid_text(bits),
        response_type: Some(ResponseType::ArrowBatch(ArrowBatch {
            data: vec![7_u8; 4096].into(),
            ..Default::default()
        })),
        ..Default::default()
    };
    assert!(r.push(response, bits, 4096, held()));
    let data = |n: Next| match n {
        Next::Response(resp, _) => match resp.response_type {
            Some(ResponseType::ArrowBatch(b)) => b.data,
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    };
    let first = data(r.next(r.attach(None), 1, true));
    let again = data(r.next(r.attach(None), 1, true));
    assert_eq!(
        first.as_ptr(),
        again.as_ptr(),
        "the payload is shared, not copied"
    );
}

#[test]
fn a_sender_past_its_deadline_is_detached_since_its_deadline() {
    let r = Replay::new(1 << 20, true);
    let now = Instant::now();
    let deadline = now + std::time::Duration::from_millis(100);
    let g = r.attach(Some(deadline));
    assert!(!r.expire_sender(now), "not yet");
    assert_eq!(r.detached_since(), None);
    assert!(r.expire_sender(deadline));
    assert_eq!(r.detached_since(), Some(deadline));
    assert!(!r.expire_sender(deadline), "once");
    // Its stream, polled again, is not the attached sender any more: its
    // drop detaches nothing, and a new sender attaches as usual.
    r.detach(g);
    assert_eq!(r.detached_since(), Some(deadline));
    r.attach(None);
    assert_eq!(r.detached_since(), None);
    assert!(!r.expire_sender(deadline), "no deadline");
    r.close();
    assert!(!r.expire_sender(deadline));
}

#[test]
fn the_admission_is_held_until_close() {
    let budget = std::sync::Arc::new(tokio::sync::Semaphore::new(10));
    let r = Replay::new(0, true);
    let permit = std::sync::Arc::clone(&budget)
        .try_acquire_many_owned(4)
        .unwrap();
    assert!(r.admit(permit));
    assert_eq!(budget.available_permits(), 6);
    r.close();
    assert_eq!(budget.available_permits(), 10);
    let permit = std::sync::Arc::clone(&budget)
        .try_acquire_many_owned(4)
        .unwrap();
    assert!(!r.admit(permit), "closed: refused and returned");
    assert_eq!(budget.available_permits(), 10);
}
