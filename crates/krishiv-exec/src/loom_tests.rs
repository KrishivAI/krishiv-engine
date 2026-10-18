// SPDX-License-Identifier: Apache-2.0
//! Loom models of the executor's handoffs: morsel claims, the finalize
//! countdown, cancellation wake-ups and the blocked result channel.
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
