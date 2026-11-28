// SPDX-License-Identifier: Apache-2.0
//! `unsafe_utf8::string_array` (the one-pass check before
//! `new_unchecked`) against arrow's checked `try_new` over arbitrary bytes,
//! offsets and nulls. One-sided: whatever ours accepts, arrow accepts and
//! builds into an equal array. Ours may refuse what arrow-array 60.1 / 60.2
//! accepts only over bytes outside the offsets (ours checks the whole
//! buffer and every offset against it, as 60.0 did).
#![no_main]

use arrow_array::LargeStringArray;
use arrow_buffer::{Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use krishiv_kernels::unsafe_utf8::{StringArrayError, string_array};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Layout: [n][n offset steps][null mask byte][rest = values].
    let Some((&n, rest)) = data.split_first() else {
        return;
    };
    let n = usize::from(n % 32);
    if rest.len() < n + 1 {
        return;
    }
    let (steps, rest) = rest.split_at(n);
    let (mask, values) = rest.split_at(1);
    // Offsets from a start step, then non-negative steps (`OffsetBuffer`'s
    // invariant); they may run past the values.
    let mut offsets = vec![i64::from(steps.first().copied().unwrap_or(0) % 4)];
    for &s in steps.iter().skip(1) {
        let next = offsets.last().copied().unwrap_or(0) + i64::from(s % 8);
        offsets.push(next);
    }
    if offsets.len() < 2 {
        offsets.push(offsets[0]);
    }
    let count = offsets.len() - 1;
    // Bit 2 of the mask gives the null buffer one bit too many, so a wrong
    // length is fuzzed too.
    let null_len = count + usize::from(mask[0] & 4 == 4);
    let nulls = (mask[0] & 1 == 1).then(|| {
        NullBuffer::from(
            (0..null_len)
                .map(|i| (mask[0] >> (i % 8)) & 2 == 0)
                .collect::<Vec<_>>(),
        )
    });
    let (first, last) = (offsets[0], offsets[offsets.len() - 1]);
    let o = OffsetBuffer::new(ScalarBuffer::from(offsets));
    let v = Buffer::from_vec(values.to_vec());
    let ours = string_array(o.clone(), v.clone(), nulls.clone());
    let arrow = LargeStringArray::try_new(o, v, nulls);
    match (ours, arrow) {
        (Ok(a), Ok(b)) => assert_eq!(a, b),
        (Err(_), Err(_)) => {}
        // Stricter than arrow-array 60.1 / 60.2 only over the bytes outside
        // the offsets: the whole buffer is not UTF-8, or every value is empty
        // at an offset inside a character (60.2 checks the empty slice
        // `values[first..last]`; with `first < last` a split is refused by both).
        (Err(StringArrayError::NotUtf8), Ok(_)) if std::str::from_utf8(values).is_err() => {}
        (Err(StringArrayError::SplitCharacter), Ok(_)) if first == last => {}
        (a, b) => panic!("disagree: ours {a:?}, arrow {b:?}"),
    }
});
