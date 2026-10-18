// SPDX-License-Identifier: Apache-2.0
//! Loom models of the executor's handoffs: morsel claims, the finalize
//! countdown, cancellation wake-ups, the blocked result channel (woken by a
//! read or by dropping the stream), the driver's spill-versus-cancel wait
//! and the `blocking_done` handshake. Each protocol model was checked to
//! fail with a planted bug (Relaxed countdown; check-before-listen in the
//! wait loop and in `blocking_done`; `drop_receiver` not waking senders).
//! `RUSTFLAGS="--cfg loom" cargo test -p krishiv-exec --lib --release`.

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use krishiv_types::batch::Batch;
use loom::future::block_on;
use loom::sync::atomic::{AtomicBool, Ordering};
use loom::thread;

use crate::cancel::CancellationToken;
use crate::channel::Channel;
use crate::morsel::{Countdown, MorselCursor};

fn model(f: impl Fn() + Sync + Send + 'static) {
    let mut b = loom::model::Builder::new();
    b.preemption_bound = Some(3);
    b.check(f);
}

fn batch() -> Batch {
    let data = RecordBatch::try_new(
        crate::testing::schema(),
        vec![Arc::new(Int64Array::from(vec![1]))],
    );
    Batch::new(data.unwrap_or_else(|_| RecordBatch::new_empty(crate::testing::schema())))
}

/// Two drivers claim every morsel exactly once.
#[test]
fn morsels_are_claimed_once() {
    model(|| {
        let cursor = Arc::new(MorselCursor::new(3, None));
        let c = Arc::clone(&cursor);
        let other = thread::spawn(move || std::iter::from_fn(|| c.claim()).collect::<Vec<_>>());
        let mine: Vec<_> = std::iter::from_fn(|| cursor.claim()).collect();
        let mut all = other.join().unwrap_or_default();
        all.extend(mine);
        all.sort_unstable();
        assert_eq!(all, [0, 1, 2]);
    });
}

/// Exactly one driver finalizes, and it sees every driver's combine.
#[test]
fn the_last_driver_finalizes_after_every_combine() {
    model(|| {
        let countdown = Arc::new(Countdown::new(2));
        let combined = Arc::new([AtomicBool::new(false), AtomicBool::new(false)]);
        let driver = |i: usize| {
            let (countdown, combined) = (Arc::clone(&countdown), Arc::clone(&combined));
            move || {
                combined[i].store(true, Ordering::Relaxed);
                if countdown.arrive() {
                    assert!(combined.iter().all(|c| c.load(Ordering::Relaxed)));
                    return 1;
                }
                0
            }
        };
        let other = thread::spawn(driver(0));
        let mine = driver(1)();
        assert_eq!(mine + other.join().unwrap_or(0), 1, "one finalize");
    });
}

/// A cancel racing a waiter's registration always wakes it.
#[test]
fn cancellation_is_never_lost() {
    model(|| {
        let token = Arc::new(CancellationToken::new());
        let t = Arc::clone(&token);
        let canceller = thread::spawn(move || t.cancel());
        block_on(token.cancelled());
        assert!(token.is_cancelled());
        canceller.join().unwrap_or(());
    });
}

/// A driver blocked on a full result channel is woken by the reader.
#[test]
fn a_blocked_sender_is_woken_by_the_reader() {
    model(|| {
        let channel = Channel::new(1);
        let c = Arc::clone(&channel);
        let sender = thread::spawn(move || {
            assert_eq!(c.send_for_test(batch()), Some(true), "full");
            block_on(Arc::clone(&c).space());
            assert_eq!(c.send_for_test(batch()), Some(true));
            c.close(None);
        });
        let mut got = 0;
        block_on(std::future::poll_fn(|cx| {
            loop {
                match channel.poll_recv(cx) {
                    std::task::Poll::Ready(Some(_)) => got += 1,
                    std::task::Poll::Ready(None) => return std::task::Poll::Ready(()),
                    std::task::Poll::Pending => return std::task::Poll::Pending,
                }
            }
        }));
        assert_eq!(got, 2);
        sender.join().unwrap_or(());
    });
}

/// The driver's wait loop (`pause`): listen for spill requests, check
/// `pending`, then wait for a request or cancellation. A canceller that
/// cancels only after the request was serviced deadlocks the model if the
/// wait could miss the request.
#[test]
fn a_spill_request_racing_a_wait_is_serviced_before_cancel() {
    use crate::sync::Event;
    model(|| {
        let requested = Arc::new(Event::default());
        let pending = Arc::new(AtomicBool::new(false));
        let serviced = Arc::new(CancellationToken::new());
        let token = Arc::new(CancellationToken::new());
        let requester = {
            let (requested, pending) = (Arc::clone(&requested), Arc::clone(&pending));
            thread::spawn(move || {
                pending.store(true, Ordering::Release);
                requested.notify();
            })
        };
        let canceller = {
            let (serviced, token) = (Arc::clone(&serviced), Arc::clone(&token));
            thread::spawn(move || {
                block_on(serviced.cancelled());
                token.cancel();
            })
        };
        block_on(async {
            loop {
                let listener = requested.listen();
                if pending.load(Ordering::Acquire) {
                    pending.store(false, Ordering::Release);
                    serviced.cancel();
                    continue;
                }
                let cancelled = token.cancelled();
                let mut listener = std::pin::pin!(listener);
                let mut cancelled = std::pin::pin!(cancelled);
                let woke = std::future::poll_fn(|cx| {
                    if cancelled.as_mut().poll(cx).is_ready() {
                        return std::task::Poll::Ready(true);
                    }
                    listener.as_mut().poll(cx).map(|()| false)
                })
                .await;
                if woke {
                    break;
                }
            }
        });
        assert!(serviced.is_cancelled());
        requester.join().unwrap_or(());
        canceller.join().unwrap_or(());
    });
}

/// A driver blocked on a full result channel is woken when the stream is
/// dropped.
#[test]
fn a_dropped_stream_wakes_a_blocked_sender() {
    model(|| {
        let channel = Channel::new(1);
        let c = Arc::clone(&channel);
        let sender = thread::spawn(move || {
            // Full, unless the stream is already gone.
            if c.send_for_test(batch()) == Some(true) {
                block_on(Arc::clone(&c).space());
                assert_eq!(c.send_for_test(batch()), None, "receiver gone");
            }
        });
        channel.drop_receiver();
        sender.join().unwrap_or(());
    });
}

/// The coordinator's `blocking_done` returns only after the last blocking
/// closure ended, and is never left waiting by a racing end.
#[test]
fn blocking_done_waits_for_the_last_closure() {
    use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
    model(|| {
        let root = TreePool::new(&MemoryConfig {
            limit: Some(1 << 20),
            ..MemoryConfig::default()
        });
        let query = crate::QueryContext::new(
            crate::QueryId(1),
            crate::WorkloadClass::Batch,
            crate::ExecConfig::default(),
            root.child("q", None),
        );
        let ended = Arc::new(AtomicBool::new(false));
        let guard = query.blocking_started();
        let e = Arc::clone(&ended);
        let closure = thread::spawn(move || {
            e.store(true, Ordering::Release);
            drop(guard);
        });
        block_on(query.blocking_done());
        assert!(ended.load(Ordering::Acquire));
        closure.join().unwrap_or(());
    });
}
