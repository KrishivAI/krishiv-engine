// SPDX-License-Identifier: Apache-2.0
//! Field-id mapping of files without ids: Iceberg's positional fallback
//! and the table's name mapping (split from `tests.rs` for its size).

use super::*;

/// Iceberg's fallback ids (FC S4): a file with no field id anywhere has
/// top-level ids 1..n by position; a file with any id is read by id only;
/// `Refuse` fails such a file instead; `Names` gives ids by a name mapping.
#[test]
fn field_ids_fall_back_to_positions_in_files_without_ids() {
    use crate::WithoutFieldIds;
    let id = |n: &str, i: &str| {
        f(n, DataType::Int32)
            .with_metadata(HashMap::from([(FIELD_ID_KEY.to_owned(), i.to_owned())]))
    };
    let plain = schema(vec![
        f("a", DataType::Int32),
        f("b", DataType::Int32),
        f("c", DataType::Int32),
    ]);
    let batch = |file: &SchemaRef| {
        RecordBatch::try_new(
            Arc::clone(file),
            (1..=3)
                .map(|v| Arc::new(Int32Array::from(vec![v])) as ArrayRef)
                .collect(),
        )
        .unwrap()
    };
    let read = |table: SchemaRef, file: &SchemaRef, mapping: FieldMapping| {
        let m = mapper(&table, file, mapping)?;
        let fr = m.file_request(&ScanRequest::all(table));
        Ok::<_, KrishivError>(m.map_batch(&project(&batch(file), &fr)).unwrap())
    };
    let one = |b: &RecordBatch, c: usize| -> Option<i32> {
        let a = b.column(c).as_primitive::<arrow_array::types::Int32Type>();
        a.is_valid(0).then(|| a.value(0))
    };
    // Renamed and reordered table columns: ids 3, 1 and 9 (past the end).
    let table = schema(vec![id("z", "3"), id("x", "1"), id("gone", "9")]);
    let by_position = FieldMapping::FieldIdOr(WithoutFieldIds::Position);
    let b = read(Arc::clone(&table), &plain, by_position.clone()).unwrap();
    assert_eq!(
        (one(&b, 0), one(&b, 1), one(&b, 2)),
        (Some(3), Some(1), None)
    );
    // Plain `FieldId` never reads by position.
    let b = read(Arc::clone(&table), &plain, FieldMapping::FieldId).unwrap();
    assert_eq!((one(&b, 0), one(&b, 1)), (None, None));
    // One id anywhere: by id only.
    let some = schema(vec![
        f("a", DataType::Int32),
        f("b", DataType::Int32),
        id("c", "1"),
    ]);
    let b = read(Arc::clone(&table), &some, by_position).unwrap();
    assert_eq!((one(&b, 0), one(&b, 1)), (None, Some(3)));
    let refuse = FieldMapping::FieldIdOr(WithoutFieldIds::Refuse);
    let e = read(Arc::clone(&table), &plain, refuse.clone()).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(read(Arc::clone(&table), &some, refuse).is_ok());
    // A name mapping: `c` is id 3, `a` (once called `x`) id 1; `b` is
    // unmapped, so id 9 reads NULL even though the file has a third column.
    let names = |id: i32, names: &[&str]| crate::MappedField {
        id: Some(id),
        names: names.iter().map(|n| (*n).to_owned()).collect(),
        fields: Vec::new(),
    };
    let mapping = FieldMapping::FieldIdOr(WithoutFieldIds::Names(Arc::new(
        crate::NameMapping::new(vec![names(3, &["c"]), names(1, &["x", "a"])]),
    )));
    let b = read(Arc::clone(&table), &plain, mapping.clone()).unwrap();
    assert_eq!(
        (one(&b, 0), one(&b, 1), one(&b, 2)),
        (Some(3), Some(1), None)
    );
    // A file with ids ignores the mapping.
    let b = read(table, &some, mapping).unwrap();
    assert_eq!((one(&b, 0), one(&b, 1)), (None, Some(3)));
}
