// SPDX-License-Identifier: Apache-2.0
//! One Arrow IPC stream, handed to JavaScript a message at a time: the schema,
//! then one chunk per record batch, then the end-of-stream marker. Arrow JS's
//! `RecordBatchReader.from` reads the concatenation of the chunks.

use arrow_array::RecordBatch;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::Schema;
use krishiv_common::error::KrishivError;

use crate::error::internal;

/// The IPC stream of one result.
pub(crate) struct Encoder {
    writer: StreamWriter<Vec<u8>>,
}

impl std::fmt::Debug for Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encoder").finish_non_exhaustive()
    }
}

impl Encoder {
    /// A stream of `schema`, and its first chunk (the schema message).
    pub(crate) fn start(schema: &Schema) -> Result<(Self, Vec<u8>), KrishivError> {
        let mut writer =
            StreamWriter::try_new(Vec::new(), schema).map_err(|e| internal("IPC schema", e))?;
        let head = std::mem::take(writer.get_mut());
        Ok((Self { writer }, head))
    }

    /// The chunk of one record batch.
    pub(crate) fn batch(&mut self, batch: &RecordBatch) -> Result<Vec<u8>, KrishivError> {
        self.writer
            .write(batch)
            .map_err(|e| internal("IPC batch", e))?;
        Ok(std::mem::take(self.writer.get_mut()))
    }

    /// The last chunk: the end-of-stream marker.
    pub(crate) fn finish(&mut self) -> Result<Vec<u8>, KrishivError> {
        self.writer.finish().map_err(|e| internal("IPC end", e))?;
        Ok(std::mem::take(self.writer.get_mut()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int64Array, StringArray};
    use arrow_ipc::reader::StreamReader;
    use arrow_schema::{DataType, Field};

    use super::*;

    #[test]
    fn the_chunks_read_back_as_one_stream() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = |ids: Vec<i64>, names: Vec<Option<&str>>| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(ids)),
                    Arc::new(StringArray::from(names)),
                ],
            )
            .unwrap()
        };
        let (mut encoder, mut bytes) = Encoder::start(&schema).unwrap();
        let batches = [
            batch(vec![1, 2], vec![Some("a"), None]),
            batch(vec![3], vec![Some("c")]),
        ];
        for b in &batches {
            let chunk = encoder.batch(b).unwrap();
            assert!(!chunk.is_empty(), "every batch is its own chunk");
            bytes.extend(chunk);
        }
        bytes.extend(encoder.finish().unwrap());
        let read: Vec<RecordBatch> = StreamReader::try_new(bytes.as_slice(), None)
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(read, batches);
    }

    #[test]
    fn an_empty_result_is_a_schema_and_the_end() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int64, true)]);
        let (mut encoder, mut bytes) = Encoder::start(&schema).unwrap();
        bytes.extend(encoder.finish().unwrap());
        let reader = StreamReader::try_new(bytes.as_slice(), None).unwrap();
        assert_eq!(reader.schema().as_ref(), &schema);
        assert_eq!(reader.count(), 0);
        assert_eq!(format!("{encoder:?}"), "Encoder { .. }");
    }
}
