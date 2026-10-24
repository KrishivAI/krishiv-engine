// SPDX-License-Identifier: Apache-2.0
//! Spark Connect decoding and translation on arbitrary bytes. The first byte
//! picks the message (`Plan`, `Relation` or `Expression`), the rest is its
//! encoding. Decoding and translating must return a value or an error, never
//! panic, overflow the stack or take super-linear time or memory (run with
//! `-timeout` and `-rss_limit_mb`); a decoded message must re-encode to bytes
//! that decode to the same encoding and must print with its generated
//! (redacting) `Debug`; a translated plan must print, encode canonically and
//! equal itself structurally. Dropping a decoded message runs its generated
//! `Scrub`.
#![no_main]

use krishiv_connect::translate::{
    TranslateConf, plan_from_bytes, relation_from_bytes, translate_expression,
};
use krishiv_connect::wire::{decode_expression, decode_plan, decode_relation};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Mutex, OnceLock};

use libfuzzer_sys::fuzz_target;
use prost::Message;

fuzz_target!(|data: &[u8]| {
    let worker = worker();
    worker.cases.send(data.to_vec()).expect("the worker runs");
    let done = worker.done.lock().expect("one case at a time");
    done.recv().expect("the case does not panic");
});

/// One long-lived thread with a big stack runs every case: decoding,
/// printing, encoding and dropping a message nested a thousand levels deep
/// recurse that deep. A thread per case left about 200 bytes of RSS behind
/// per case under the fuzzer's AddressSanitizer build (measured: the same
/// input replayed 50,000 and 250,000 times peaked at 316 and 354 MB), which
/// reached the 2 GiB limit after about 8 million cases.
struct Worker {
    cases: SyncSender<Vec<u8>>,
    done: Mutex<Receiver<()>>,
}

fn worker() -> &'static Worker {
    static WORKER: OnceLock<Worker> = OnceLock::new();
    WORKER.get_or_init(|| {
        let (cases, inbox) = sync_channel::<Vec<u8>>(0);
        let (finished, done) = sync_channel(0);
        std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(move || {
                for data in inbox {
                    run(&data);
                    if finished.send(()).is_err() {
                        return;
                    }
                }
            })
            .expect("a thread");
        Worker {
            cases,
            done: Mutex::new(done),
        }
    })
}

fn run(data: &[u8]) {
    let Some((&which, bytes)) = data.split_first() else {
        return;
    };
    let conf = TranslateConf::default();
    match which % 3 {
        0 => {
            if let Ok(plan) = decode_plan(bytes) {
                let _ = format!("{:?}", *plan);
                reencodes(plan.encode_to_vec(), |b| {
                    decode_plan(b).map(|m| m.encode_to_vec())
                });
            }
            if let Ok(plan) = plan_from_bytes(bytes, &conf) {
                check_plan(&plan);
            }
        }
        1 => {
            if let Ok(relation) = decode_relation(bytes) {
                let _ = format!("{:?}", *relation);
                reencodes(relation.encode_to_vec(), |b| {
                    decode_relation(b).map(|m| m.encode_to_vec())
                });
            }
            if let Ok(plan) = relation_from_bytes(bytes, &conf) {
                check_plan(&plan);
            }
        }
        _ => {
            if let Ok(expression) = decode_expression(bytes) {
                let _ = format!("{:?}", *expression);
                reencodes(expression.encode_to_vec(), |b| {
                    decode_expression(b).map(|m| m.encode_to_vec())
                });
                if let Ok(e) = translate_expression(&expression, &conf) {
                    let _ = e.to_string();
                }
            }
        }
    }
}

/// `encoded`, decoded and encoded again, is `encoded`.
fn reencodes<E>(encoded: Vec<u8>, again: impl Fn(&[u8]) -> Result<Vec<u8>, E>) {
    assert!(
        matches!(again(&encoded), Ok(b) if b == encoded),
        "re-encoding changed the message"
    );
}

fn check_plan(plan: &krishiv_plan::Plan) {
    let _ = plan.to_string();
    assert!(plan.structurally_eq(plan));
    assert_eq!(plan.structural_hash(), plan.structural_hash());
}
