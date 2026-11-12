// SPDX-License-Identifier: Apache-2.0
//! `EngineData` over an Arrow 60 [`RecordBatch`].
//!
//! Ported from `delta_kernel` 0.30.1 `engine/arrow_data.rs` (Apache-2.0).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef, RecordBatch, RunArray, StructArray};
use arrow_schema::{DataType as ArrowType, FieldRef, Schema};
use arrow_select::filter::filter_record_batch;
use delta_kernel::engine_data::RowVisitor;
use delta_kernel::expressions::{ArrayData, ColumnName};
use delta_kernel::schema::{DataType, PrimitiveType, SchemaRef};
use delta_kernel::{EngineData, GetData, KernelError, KernelResult};

use super::get_data::{Getter, ListGetter, ListKind, MapGetter, StrAcc};
use super::schema::{array_data_to_arrow, arrow_schema};

/// Kernel `EngineData` backed by one Arrow record batch.
#[derive(Debug)]
pub struct KrishivEngineData {
    data: RecordBatch,
}

impl KrishivEngineData {
    /// Wraps a record batch.
    #[must_use]
    pub fn new(data: RecordBatch) -> Self {
        Self { data }
    }

    /// The wrapped record batch.
    #[must_use]
    pub fn record_batch(&self) -> &RecordBatch {
        &self.data
    }

    /// Unwraps the record batch.
    #[must_use]
    pub fn into_record_batch(self) -> RecordBatch {
        self.data
    }

    /// The record batch inside a kernel `EngineData`, or an error if it is another engine's data.
    pub fn extract(engine_data: &dyn EngineData) -> KernelResult<&RecordBatch> {
        engine_data
            .any_ref()
            .downcast_ref::<Self>()
            .map(|d| &d.data)
            .ok_or_else(|| KernelError::engine_data_type("KrishivEngineData"))
    }
}

impl From<RecordBatch> for KrishivEngineData {
    fn from(data: RecordBatch) -> Self {
        Self::new(data)
    }
}

impl From<StructArray> for KrishivEngineData {
    fn from(value: StructArray) -> Self {
        Self::new(RecordBatch::from(value))
    }
}

/// Columns and fields of a batch or a struct array, so the visitor drills into both the same way.
trait ColumnsAndFields {
    fn columns(&self) -> &[ArrayRef];
    fn fields(&self) -> &[FieldRef];
}

impl ColumnsAndFields for RecordBatch {
    fn columns(&self) -> &[ArrayRef] {
        self.columns()
    }
    fn fields(&self) -> &[FieldRef] {
        self.schema_ref().fields()
    }
}

impl ColumnsAndFields for StructArray {
    fn columns(&self) -> &[ArrayRef] {
        self.columns()
    }
    fn fields(&self) -> &[FieldRef] {
        self.fields()
    }
}

/// The state of a column path during extraction.
enum ColumnState<'a> {
    /// A parent of a requested leaf; the walk recurses into it.
    Parent,
    /// A requested leaf waiting for its getter.
    AwaitingGetter(&'a DataType),
    /// A requested leaf with its getter.
    HasGetter(Getter<'a>),
}

impl EngineData for KrishivEngineData {
    fn len(&self) -> usize {
        self.data.num_rows()
    }

    fn visit_rows(
        &self,
        leaf_columns: &[ColumnName],
        visitor: &mut dyn RowVisitor,
    ) -> KernelResult<()> {
        let leaf_types = visitor.selected_column_names_and_types().1;
        if leaf_types.len() != leaf_columns.len() {
            return Err(KernelError::missing_column(format!(
                "Visitor expected {} column names, but caller passed {}",
                leaf_types.len(),
                leaf_columns.len()
            )));
        }
        let mut column_map = HashMap::with_capacity(leaf_columns.len() * 2);
        for (column, data_type) in leaf_columns.iter().zip(leaf_types.iter()) {
            column_map.insert(column.clone(), ColumnState::AwaitingGetter(data_type));
            let mut cur = column.parent();
            while let Some(parent) = cur {
                column_map
                    .entry(parent.clone())
                    .or_insert(ColumnState::Parent);
                cur = parent.parent();
            }
        }
        extract_columns(&mut vec![], &mut column_map, &self.data)?;
        let mut owned = Vec::with_capacity(leaf_columns.len());
        for column in leaf_columns {
            match column_map.get(column) {
                Some(ColumnState::HasGetter(getter)) => owned.push(getter),
                _ => {
                    return Err(KernelError::missing_column(format!(
                        "Column {column} not found in the data"
                    )));
                }
            }
        }
        let getters: Vec<&dyn GetData<'_>> =
            owned.into_iter().map(|g| g as &dyn GetData<'_>).collect();
        visitor.visit(self.len(), &getters)
    }

    fn append_columns(
        &self,
        schema: SchemaRef,
        columns: Vec<ArrayData>,
    ) -> KernelResult<Box<dyn EngineData>> {
        let appended = arrow_schema(schema.as_ref())?;
        if appended.fields().len() != columns.len() {
            return Err(KernelError::generic(format!(
                "append_columns: {} schema fields but {} columns",
                appended.fields().len(),
                columns.len()
            )));
        }
        let mut fields = self.data.schema().fields().to_vec();
        fields.extend(appended.fields().iter().cloned());
        let mut arrays = self.data.columns().to_vec();
        for column in &columns {
            arrays.push(array_data_to_arrow(column)?);
        }
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
            .map_err(|e| KernelError::generic(format!("append_columns: {e}")))?;
        Ok(Box::new(Self::new(batch)))
    }

    fn apply_selection_vector(
        self: Box<Self>,
        mut selection_vector: Vec<bool>,
    ) -> KernelResult<Box<dyn EngineData>> {
        if selection_vector.len() > self.len() {
            return Err(KernelError::InvalidSelectionVector(format!(
                "Selection vector is larger than data length: {} > {}",
                selection_vector.len(),
                self.len()
            )));
        }
        // A short vector selects the rest.
        selection_vector.resize(self.len(), true);
        let filtered = filter_record_batch(&self.data, &selection_vector.into())
            .map_err(|e| KernelError::generic(format!("apply_selection_vector: {e}")))?;
        Ok(Box::new(Self::new(filtered)))
    }

    fn has_field(&self, name: &ColumnName) -> bool {
        let Some((first, rest)) = name.path().split_first() else {
            return false;
        };
        let Some((_, mut field)) = self.data.schema_ref().fields().find(first.as_str()) else {
            return false;
        };
        for component in rest {
            let ArrowType::Struct(nested) = field.data_type() else {
                return false;
            };
            let Some((_, next)) = nested.find(component.as_str()) else {
                return false;
            };
            field = next;
        }
        true
    }
}

fn extract_columns<'a>(
    path: &mut Vec<String>,
    column_map: &mut HashMap<ColumnName, ColumnState<'a>>,
    data: &'a dyn ColumnsAndFields,
) -> KernelResult<()> {
    for (column, field) in data.columns().iter().zip(data.fields()) {
        path.push(field.name().clone());
        if let Some(state) = column_map.get_mut(path.as_slice()) {
            match state {
                ColumnState::Parent => {
                    if let Some(struct_array) = column.as_struct_opt() {
                        extract_columns(path, column_map, struct_array)?;
                    }
                }
                ColumnState::AwaitingGetter(data_type) => {
                    let getter = if column.data_type() == &ArrowType::Null {
                        Getter::Null
                    } else {
                        extract_leaf_column(path, data_type, column)?
                    };
                    *state = ColumnState::HasGetter(getter);
                }
                ColumnState::HasGetter(_) => {
                    return Err(KernelError::internal_error(format!(
                        "Column {} already has a getter - duplicate column?",
                        ColumnName::new(path.iter())
                    )));
                }
            }
        }
        path.pop();
    }
    Ok(())
}

fn ree<'a>(col: &'a dyn Array) -> Option<Getter<'a>> {
    match col.data_type() {
        ArrowType::RunEndEncoded(_, _) => col
            .as_any()
            .downcast_ref::<RunArray<Int64Type>>()
            .map(Getter::Ree),
        _ => None,
    }
}

fn is_string(dt: &ArrowType) -> bool {
    matches!(
        dt,
        ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View
    )
}

fn is_struct(dt: &ArrowType) -> bool {
    matches!(dt, ArrowType::Struct(_))
}

fn list_like<'a>(col: &'a dyn Array, element_ok: fn(&ArrowType) -> bool) -> Option<Getter<'a>> {
    match col.data_type() {
        ArrowType::List(f)
        | ArrowType::LargeList(f)
        | ArrowType::ListView(f)
        | ArrowType::LargeListView(f)
            if element_ok(f.data_type()) => {}
        _ => return None,
    }
    ListKind::of(col).map(|k| Getter::List(ListGetter::new(k)))
}

fn extract_leaf_column<'a>(
    path: &[String],
    data_type: &DataType,
    col: &'a dyn Array,
) -> KernelResult<Getter<'a>> {
    macro_rules! prim {
        ($ty:ty, $variant:ident, $name:literal) => {
            col.as_primitive_opt::<$ty>()
                .map(Getter::$variant)
                .ok_or($name)
        };
    }
    let result: Result<Getter<'a>, &str> = match data_type {
        &DataType::BOOLEAN => col
            .as_boolean_opt()
            .map(Getter::Bool)
            .or_else(|| ree(col))
            .ok_or("bool"),
        &DataType::STRING => StrAcc::of(col)
            .map(Getter::Str)
            .or_else(|| ree(col))
            .ok_or("string"),
        &DataType::BINARY => col
            .as_binary_opt::<i32>()
            .map(Getter::Binary)
            .or_else(|| col.as_binary_opt::<i64>().map(Getter::LargeBinary))
            .or_else(|| col.as_binary_view_opt().map(Getter::BinaryView))
            .or_else(|| ree(col))
            .ok_or("binary"),
        &DataType::BYTE => prim!(Int8Type, I8, "byte"),
        &DataType::SHORT => prim!(Int16Type, I16, "short"),
        &DataType::INTEGER => col
            .as_primitive_opt::<Int32Type>()
            .map(Getter::I32)
            .or_else(|| ree(col))
            .ok_or("int"),
        &DataType::LONG => col
            .as_primitive_opt::<Int64Type>()
            .map(Getter::I64)
            .or_else(|| ree(col))
            .ok_or("long"),
        &DataType::FLOAT => prim!(Float32Type, F32, "float"),
        &DataType::DOUBLE => prim!(Float64Type, F64, "double"),
        &DataType::DATE => prim!(Date32Type, Date, "date"),
        &DataType::TIMESTAMP | &DataType::TIMESTAMP_NTZ => {
            prim!(TimestampMicrosecondType, Timestamp, "timestamp")
        }
        DataType::Primitive(PrimitiveType::Decimal(_)) => prim!(Decimal128Type, Decimal, "decimal"),
        DataType::Array(array) if matches!(array.element_type(), DataType::Struct(_)) => {
            // A nested visitor sees one row per element and cannot observe a skipped one.
            if array.contains_null() {
                return Err(KernelError::unexpected_column_type(format!(
                    "On {}: array<struct> columns with nullable elements are not visitable",
                    ColumnName::new(path)
                )));
            }
            list_like(col, is_struct).ok_or("array<struct>")
        }
        DataType::Array(_) => list_like(col, is_string).ok_or("array<string>"),
        DataType::Map(_) => col
            .as_map_opt()
            .filter(|a| is_string(a.key_type()) && is_string(a.value_type()))
            .and_then(MapGetter::new)
            .map(Getter::Map)
            .ok_or("map<string, string>"),
        other => {
            return Err(KernelError::unexpected_column_type(format!(
                "On {}: Unsupported type {other}",
                ColumnName::new(path)
            )));
        }
    };
    result.map_err(|expected| {
        KernelError::unexpected_column_type(format!(
            "Type mismatch on {}: expected {expected}, got {}",
            ColumnName::new(path),
            col.data_type()
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::builder::{ListBuilder, MapBuilder, StringBuilder};
    use arrow_array::{BooleanArray, Int32Array, Int64Array, StringArray};
    use arrow_schema::Field;
    use delta_kernel::expressions::{ArrayData, Scalar, column_name};
    use delta_kernel::schema::{ArrayType, DataType, StructField, StructType};

    use super::*;

    struct Collect {
        ints: Vec<Option<i32>>,
        strs: Vec<Option<String>>,
        lists: Vec<Option<Vec<String>>>,
        maps: Vec<Option<usize>>,
    }

    impl RowVisitor for Collect {
        fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
            use std::sync::LazyLock;
            static N: LazyLock<Vec<ColumnName>> = LazyLock::new(|| {
                vec![
                    column_name!("a"),
                    column_name!("s.b"),
                    column_name!("l"),
                    column_name!("m"),
                ]
            });
            static T: LazyLock<Vec<DataType>> = LazyLock::new(|| {
                vec![
                    DataType::INTEGER,
                    DataType::STRING,
                    DataType::from(ArrayType::new(DataType::STRING, true)),
                    DataType::from(delta_kernel::schema::MapType::new(
                        DataType::STRING,
                        DataType::STRING,
                        true,
                    )),
                ]
            });
            (&N, &T)
        }
        fn visit<'a>(
            &mut self,
            row_count: usize,
            getters: &[&'a dyn GetData<'a>],
        ) -> KernelResult<()> {
            for i in 0..row_count {
                self.ints.push(getters[0].get_int(i, "a")?);
                self.strs
                    .push(getters[1].get_str(i, "s.b")?.map(str::to_string));
                self.lists
                    .push(getters[2].get_list(i, "l")?.map(|l| l.materialize()));
                self.maps
                    .push(getters[3].get_map(i, "m")?.map(|m| m.materialize().len()));
            }
            // A wrong-typed request is an error, not a panic.
            assert!(getters[0].get_str(0, "a").is_err());
            Ok(())
        }
    }

    fn batch() -> RecordBatch {
        let a = Int32Array::from(vec![Some(1), None, Some(3)]);
        let b = StringArray::from(vec![Some("x"), Some("y"), None]);
        let s = StructArray::from(vec![(
            Arc::new(Field::new("b", ArrowType::Utf8, true)),
            Arc::new(b) as ArrayRef,
        )]);
        let mut lb = ListBuilder::new(StringBuilder::new());
        lb.values().append_value("p");
        lb.values().append_value("q");
        lb.append(true);
        lb.append(false);
        lb.append(true);
        let mut mb = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
        mb.keys().append_value("k");
        mb.values().append_value("v");
        mb.append(true).unwrap();
        mb.append(false).unwrap();
        mb.append(true).unwrap();
        let l = lb.finish();
        let m = mb.finish();
        RecordBatch::try_from_iter([
            ("a", Arc::new(a) as ArrayRef),
            ("s", Arc::new(s) as ArrayRef),
            ("l", Arc::new(l) as ArrayRef),
            ("m", Arc::new(m) as ArrayRef),
        ])
        .unwrap()
    }

    #[test]
    fn visit_rows_reads_nested_lists_and_maps() {
        let data = KrishivEngineData::new(batch());
        let mut v = Collect {
            ints: vec![],
            strs: vec![],
            lists: vec![],
            maps: vec![],
        };
        data.visit_rows(v.selected_column_names_and_types().0, &mut v)
            .unwrap();
        assert_eq!(v.ints, [Some(1), None, Some(3)]);
        assert_eq!(v.strs, [Some("x".into()), Some("y".into()), None]);
        assert_eq!(
            v.lists,
            [Some(vec!["p".into(), "q".into()]), None, Some(vec![])]
        );
        assert_eq!(v.maps, [Some(1), None, Some(0)]);
    }

    #[test]
    fn a_missing_column_is_an_error() {
        let data = KrishivEngineData::new(
            RecordBatch::try_from_iter([("a", Arc::new(Int32Array::from(vec![1])) as ArrayRef)])
                .unwrap(),
        );
        let mut v = Collect {
            ints: vec![],
            strs: vec![],
            lists: vec![],
            maps: vec![],
        };
        assert!(
            data.visit_rows(v.selected_column_names_and_types().0, &mut v)
                .is_err()
        );
    }

    #[test]
    fn a_short_selection_vector_keeps_the_rest_and_a_long_one_is_refused() {
        let mk = || {
            Box::new(KrishivEngineData::new(
                RecordBatch::try_from_iter([(
                    "a",
                    Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef,
                )])
                .unwrap(),
            ))
        };
        let kept = mk().apply_selection_vector(vec![false, true]).unwrap();
        assert_eq!(kept.len(), 3);
        assert!(mk().apply_selection_vector(vec![true; 5]).is_err());
    }

    #[test]
    fn has_field_follows_struct_paths() {
        let data = KrishivEngineData::new(batch());
        assert!(data.has_field(&column_name!("s.b")));
        assert!(!data.has_field(&column_name!("s.c")));
        assert!(!data.has_field(&column_name!("a.b")));
    }

    #[test]
    fn append_columns_adds_the_new_fields_after_the_old() {
        let data = KrishivEngineData::new(
            RecordBatch::try_from_iter([(
                "a",
                Arc::new(BooleanArray::from(vec![true, false])) as ArrayRef,
            )])
            .unwrap(),
        );
        let schema = Arc::new(StructType::new_unchecked([StructField::nullable(
            "n",
            DataType::LONG,
        )]));
        let col = ArrayData::try_new(
            ArrayType::new(DataType::LONG, true),
            [Scalar::Long(7), Scalar::Null(DataType::LONG)],
        )
        .unwrap();
        let out = data.append_columns(schema, vec![col]).unwrap();
        let rb = KrishivEngineData::extract(out.as_ref()).unwrap();
        assert_eq!(rb.schema().fields().len(), 2);
        let n = rb.column(1).as_primitive::<Int64Type>();
        assert_eq!((n.value(0), n.is_null(1)), (7, true));
    }
}
