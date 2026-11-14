// SPDX-License-Identifier: Apache-2.0
//! The Python UDF pool's reader of a worker's result stream (`krishiv_udf_pool::decode_result`, design
//! P3-T014 §3.3) on arbitrary bytes: a worker is untrusted, so any stream must give a column or an error,
//! never panic, abort on an allocation or take super-linear time or memory (run with `-timeout` and
//! `-rss_limit_mb`). The first byte picks the declared return type and the expected row count; the rest is
//! the stream.
#![no_main]

use arrow_schema::{DataType, Field};
use libfuzzer_sys::fuzz_target;
use std::sync::Arc;

fuzz_target!(|data: &[u8]| {
    let Some((&pick, stream)) = data.split_first() else {
        return;
    };
    let return_type = match pick % 4 {
        0 => DataType::Int64,
        1 => DataType::Utf8View,
        2 => DataType::Float64,
        _ => DataType::List(Arc::new(Field::new("element", DataType::Int64, true))),
    };
    let rows = usize::from(pick >> 2);
    if let Ok(column) = krishiv_udf_pool::decode_result(stream.to_vec(), &return_type, rows) {
        assert_eq!(column.data_type(), &return_type);
        assert_eq!(column.len(), rows);
    }
});
