// SPDX-License-Identifier: Apache-2.0
//! A fake Hive metastore for tests: one connection, real Thrift binary over TCP, and writers for the structs the
//! client reads (including a field it does not know and one with an unexpected wire type, which it must skip).

use std::net::TcpListener;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use thrift::protocol::{
    TBinaryInputProtocol, TBinaryOutputProtocol, TFieldIdentifier, TInputProtocol, TListIdentifier,
    TMapIdentifier, TMessageIdentifier, TMessageType, TOutputProtocol, TStructIdentifier, TType,
};
use thrift::transport::{TBufferedReadTransport, TBufferedWriteTransport, TIoChannel, TTcpChannel};

use super::client::HmsConfig;
use super::types::read_fields;

pub(crate) mod fake {
    use super::*;

    /// A one-connection fake metastore: reads each call's name and sequence number, skips the arguments, and
    /// lets `reply` write the answer. Real Thrift binary over TCP.
    pub(crate) fn serve(
        calls: usize,
        reply: impl FnMut(&str, i32, &mut dyn TOutputProtocol) -> thrift::Result<()> + Send + 'static,
    ) -> (String, JoinHandle<()>) {
        serve_connections(1, calls, reply)
    }

    /// A fake metastore that accepts `connections` connections one after the other and answers `calls` calls on
    /// each (a connection ends after its last call, so a client that reuses it finds it closed).
    pub(crate) fn serve_connections(
        connections: usize,
        calls: usize,
        mut reply: impl FnMut(&str, i32, &mut dyn TOutputProtocol) -> thrift::Result<()>
        + Send
        + 'static,
    ) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let handle = thread::spawn(move || {
            for _ in 0..connections {
                let (stream, _) = listener.accept().unwrap();
                let channel = TTcpChannel::with_stream(stream);
                let (read, write) = channel.split().unwrap();
                let mut input = TBinaryInputProtocol::new(TBufferedReadTransport::new(read), true);
                let mut output =
                    TBinaryOutputProtocol::new(TBufferedWriteTransport::new(write), true);
                for _ in 0..calls {
                    let Ok(head) = input.read_message_begin() else {
                        break;
                    };
                    read_fields(&mut input, |_, _, _| Ok(false)).unwrap();
                    input.read_message_end().unwrap();
                    reply(&head.name, head.sequence_number, &mut output).unwrap();
                    output.flush().unwrap();
                }
            }
        });
        (address, handle)
    }

    /// Writes `method`'s reply: `fields` writes the result struct's fields.
    pub(crate) fn reply_with(
        o: &mut dyn TOutputProtocol,
        method: &str,
        sequence: i32,
        fields: impl FnOnce(&mut dyn TOutputProtocol) -> thrift::Result<()>,
    ) -> thrift::Result<()> {
        o.write_message_begin(&TMessageIdentifier::new(
            method,
            TMessageType::Reply,
            sequence,
        ))?;
        o.write_struct_begin(&TStructIdentifier::new(format!("{method}_result")))?;
        fields(o)?;
        o.write_field_stop()?;
        o.write_struct_end()?;
        o.write_message_end()
    }

    pub(crate) fn len32(n: usize) -> i32 {
        i32::try_from(n).unwrap_or(i32::MAX)
    }

    pub(crate) fn field(o: &mut dyn TOutputProtocol, ty: TType, id: i16) -> thrift::Result<()> {
        o.write_field_begin(&TFieldIdentifier {
            name: None,
            field_type: ty,
            id: Some(id),
        })
    }

    pub(crate) fn string_field(
        o: &mut dyn TOutputProtocol,
        id: i16,
        value: &str,
    ) -> thrift::Result<()> {
        field(o, TType::String, id)?;
        o.write_string(value)?;
        o.write_field_end()
    }

    pub(crate) fn strings_field(
        o: &mut dyn TOutputProtocol,
        id: i16,
        values: &[&str],
    ) -> thrift::Result<()> {
        field(o, TType::List, id)?;
        o.write_list_begin(&TListIdentifier::new(TType::String, len32(values.len())))?;
        for v in values {
            o.write_string(v)?;
        }
        o.write_list_end()?;
        o.write_field_end()
    }

    pub(crate) fn map_field(
        o: &mut dyn TOutputProtocol,
        id: i16,
        entries: &[(&str, &str)],
    ) -> thrift::Result<()> {
        field(o, TType::Map, id)?;
        o.write_map_begin(&TMapIdentifier::new(
            TType::String,
            TType::String,
            len32(entries.len()),
        ))?;
        for (k, v) in entries {
            o.write_string(k)?;
            o.write_string(v)?;
        }
        o.write_map_end()?;
        o.write_field_end()
    }

    pub(crate) fn struct_field(
        o: &mut dyn TOutputProtocol,
        id: i16,
        body: impl FnOnce(&mut dyn TOutputProtocol) -> thrift::Result<()>,
    ) -> thrift::Result<()> {
        field(o, TType::Struct, id)?;
        o.write_struct_begin(&TStructIdentifier::new("s"))?;
        body(o)?;
        o.write_field_stop()?;
        o.write_struct_end()?;
        o.write_field_end()
    }

    /// A declared exception `id` (a struct whose field 1 is the message).
    pub(crate) fn exception(
        o: &mut dyn TOutputProtocol,
        id: i16,
        message: &str,
    ) -> thrift::Result<()> {
        struct_field(o, id, |o| string_field(o, 1, message))
    }

    pub(crate) fn cfg(address: &str) -> HmsConfig {
        HmsConfig {
            request_timeout: Duration::from_secs(5),
            ..HmsConfig::new(address)
        }
    }

    pub(crate) fn write_table(o: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        string_field(o, 1, "events")?;
        string_field(o, 2, "web")?;
        string_field(o, 3, "etl")?;
        field(o, TType::I32, 4)?;
        o.write_i32(1_700_000_000)?;
        o.write_field_end()?;
        struct_field(o, 7, |o| {
            field(o, TType::List, 1)?;
            o.write_list_begin(&TListIdentifier::new(TType::Struct, 2))?;
            for (name, ty, comment) in [
                ("id", "bigint", "the key"),
                ("payload", "map<string,string>", ""),
            ] {
                o.write_struct_begin(&TStructIdentifier::new("FieldSchema"))?;
                string_field(o, 1, name)?;
                string_field(o, 2, ty)?;
                string_field(o, 3, comment)?;
                o.write_field_stop()?;
                o.write_struct_end()?;
            }
            o.write_list_end()?;
            o.write_field_end()?;
            string_field(o, 2, "s3a://bucket/web/events")?;
            string_field(
                o,
                3,
                "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            )?;
            field(o, TType::Bool, 5)?;
            o.write_bool(true)?;
            o.write_field_end()?;
            field(o, TType::I32, 6)?;
            o.write_i32(-1)?;
            o.write_field_end()?;
            struct_field(o, 7, |o| {
                string_field(
                    o,
                    2,
                    "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
                )?;
                map_field(o, 3, &[("serialization.format", "1")])
            })?;
            // A field this client does not know, and a known id with another wire type: both skipped.
            string_field(o, 99, "ignored")?;
            field(o, TType::I32, 2)?;
            o.write_i32(7)?;
            o.write_field_end()
        })?;
        field(o, TType::List, 8)?;
        o.write_list_begin(&TListIdentifier::new(TType::Struct, 1))?;
        o.write_struct_begin(&TStructIdentifier::new("FieldSchema"))?;
        string_field(o, 1, "dt")?;
        string_field(o, 2, "string")?;
        o.write_field_stop()?;
        o.write_struct_end()?;
        o.write_list_end()?;
        o.write_field_end()?;
        map_field(
            o,
            9,
            &[
                ("EXTERNAL", "TRUE"),
                ("spark.sql.sources.provider", "parquet"),
            ],
        )?;
        string_field(o, 12, "EXTERNAL_TABLE")
    }
}
