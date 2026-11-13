// SPDX-License-Identifier: Apache-2.0
//! The kernel's `EngineData` and `GetData` over Arrow 60 record batches (design note P3-T021
//! section 3.1).
//!
//! Ported in shape from the kernel's default engine (`delta_kernel` 0.30.1,
//! `src/engine/arrow_data.rs` and `src/engine/arrow_get_data.rs`, Apache-2.0, copyright the
//! Delta Lake project authors), which is compiled only with the kernel's Arrow features. The
//! kernel implements `GetData` on Arrow's array types directly; here the orphan rule needs a
//! local `Getter` that borrows the array, resolved once per visit.
//!
//! Contract (`dk:src/engine_data.rs`):
//! - `visit_rows` resolves every requested leaf path (nested structs included), checks the
//!   column's Arrow type against the type the visitor declares, and calls the visitor once with
//!   one getter per leaf, in the requested order. A missing, duplicate or non-leaf path is an
//!   error; a column of Arrow type `Null` reads as all null.
//! - String lists, string maps and lists of structs have accessors; a list of structs whose
//!   declared element may be null is refused, since a nested visitor cannot see a skipped row.
//! - `apply_selection_vector`: rows past the end of a short vector are selected; a vector longer
//!   than the data is an error.

use std::any::Any;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
    GenericListArray, GenericListViewArray, Int8Array, Int16Array, Int32Array, Int64Array,
    LargeStringArray, MapArray, OffsetSizeTrait, RecordBatch, StringArray, StringViewArray,
    StructArray, TimestampMicrosecondArray,
};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType as ArrowType, FieldRef, Schema as ArrowSchema};
use arrow_select::filter::filter_record_batch;
use delta_kernel::engine_data::{
    GetData, ListItem, MapItem, RowVisitor, StringArrayAccessor, StructList, StructListAccessor,
};
use delta_kernel::expressions::{ArrayData, ColumnName};
use delta_kernel::schema::{DataType, PrimitiveType, SchemaRef};
use delta_kernel::{EngineData, KernelError, KernelResult};

use super::schema::{array_data_to_arrow, to_arrow_schema};

/// A record batch the kernel reads through `EngineData`.
#[derive(Debug, Clone)]
pub struct KrishivEngineData(RecordBatch);

impl KrishivEngineData {
    /// Wraps `batch`.
    #[must_use]
    pub fn new(batch: RecordBatch) -> Self {
        Self(batch)
    }

    /// The wrapped batch.
    #[must_use]
    pub fn batch(&self) -> &RecordBatch {
        &self.0
    }

    /// The wrapped batch, by value.
    #[must_use]
    pub fn into_batch(self) -> RecordBatch {
        self.0
    }

    /// `data` as Krishiv's engine data.
    ///
    /// # Errors
    /// `EngineDataType` when `data` came from another engine.
    pub fn from_engine_data(data: Box<dyn EngineData>) -> KernelResult<Box<Self>> {
        let any: Box<dyn Any + Send + Sync> = data.into_any();
        any.downcast::<Self>()
            .map_err(|_| KernelError::engine_data_type("KrishivEngineData"))
    }

    /// The batch inside `data`, borrowed.
    ///
    /// # Errors
    /// As [`Self::from_engine_data`].
    pub fn batch_of(data: &dyn EngineData) -> KernelResult<&RecordBatch> {
        data.any_ref()
            .downcast_ref::<Self>()
            .map(Self::batch)
            .ok_or_else(|| KernelError::engine_data_type("KrishivEngineData"))
    }
}

impl From<RecordBatch> for KrishivEngineData {
    fn from(batch: RecordBatch) -> Self {
        Self(batch)
    }
}

impl From<StructArray> for KrishivEngineData {
    fn from(array: StructArray) -> Self {
        Self(RecordBatch::from(array))
    }
}

/// A requested column's state while its getter is resolved.
enum Slot<'a> {
    /// A struct on the way to a requested leaf.
    Parent,
    /// A requested leaf of the declared type, not yet found.
    Awaiting(usize, &'a DataType),
}

impl EngineData for KrishivEngineData {
    fn visit_rows(
        &self,
        column_names: &[ColumnName],
        visitor: &mut dyn RowVisitor,
    ) -> KernelResult<()> {
        let types = visitor.selected_column_names_and_types().1;
        let getters = getters(
            self.0.columns(),
            self.0.schema_ref().fields(),
            column_names,
            types,
        )?;
        let refs: Vec<&dyn GetData<'_>> = getters.iter().map(|g| g as &dyn GetData<'_>).collect();
        visitor.visit(self.0.num_rows(), &refs)
    }

    fn len(&self) -> usize {
        self.0.num_rows()
    }

    fn append_columns(
        &self,
        schema: SchemaRef,
        columns: Vec<ArrayData>,
    ) -> KernelResult<Box<dyn EngineData>> {
        let added = to_arrow_schema(&schema)?;
        if added.fields().len() != columns.len() {
            return Err(KernelError::generic(format!(
                "append_columns: {} fields but {} columns",
                added.fields().len(),
                columns.len()
            )));
        }
        let mut fields = self.0.schema().fields().to_vec();
        fields.extend(added.fields().iter().cloned());
        let mut arrays = self.0.columns().to_vec();
        for column in &columns {
            arrays.push(array_data_to_arrow(column)?);
        }
        let batch = RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), arrays)
            .map_err(|e| KernelError::generic(e.to_string()))?;
        Ok(Box::new(Self(batch)))
    }

    fn apply_selection_vector(
        self: Box<Self>,
        mut selection_vector: Vec<bool>,
    ) -> KernelResult<Box<dyn EngineData>> {
        let rows = self.0.num_rows();
        if selection_vector.len() > rows {
            return Err(KernelError::InvalidSelectionVector(format!(
                "Selection vector is larger than data length: {} > {rows}",
                selection_vector.len()
            )));
        }
        if selection_vector.iter().all(|s| *s) {
            return Ok(self);
        }
        selection_vector.resize(rows, true);
        let filtered = filter_record_batch(&self.0, &BooleanArray::from(selection_vector))
            .map_err(|e| KernelError::generic(e.to_string()))?;
        Ok(Box::new(Self(filtered)))
    }

    fn has_field(&self, name: &ColumnName) -> bool {
        let Some((first, rest)) = name.path().split_first() else {
            return false;
        };
        let Some((_, mut field)) = self.0.schema_ref().fields().find(first) else {
            return false;
        };
        for part in rest {
            let ArrowType::Struct(children) = field.data_type() else {
                return false;
            };
            let Some((_, next)) = children.find(part) else {
                return false;
            };
            field = next;
        }
        true
    }
}

/// One getter per requested leaf of `columns`, in the order of `names`.
fn getters<'a>(
    columns: &'a [ArrayRef],
    fields: &'a [FieldRef],
    names: &[ColumnName],
    types: &'a [DataType],
) -> KernelResult<Vec<Getter<'a>>> {
    if names.len() != types.len() {
        return Err(KernelError::MissingColumn(format!(
            "Visitor expected {} column names, but caller passed {}",
            types.len(),
            names.len()
        )));
    }
    let mut slots: HashMap<&[String], Slot<'a>> = HashMap::with_capacity(names.len() * 2);
    for (i, (name, data_type)) in names.iter().zip(types).enumerate() {
        let path = name.path();
        if slots.insert(path, Slot::Awaiting(i, data_type)).is_some() {
            return Err(KernelError::MissingColumn(format!(
                "Column {name} is requested twice or is not a leaf"
            )));
        }
        for end in 1..path.len() {
            if let Some(Slot::Awaiting(..)) = slots.get(&path[..end]) {
                return Err(KernelError::MissingColumn(format!(
                    "Column {name} is under a requested leaf"
                )));
            }
            slots.insert(&path[..end], Slot::Parent);
        }
    }
    let mut found: Vec<Option<Getter<'a>>> = (0..names.len()).map(|_| None).collect();
    let mut path = Vec::new();
    resolve(columns, fields, None, &mut path, &slots, &mut found)?;
    found
        .into_iter()
        .zip(names)
        .map(|(g, name)| {
            g.ok_or_else(|| {
                KernelError::MissingColumn(format!("Column {name} not found in the data"))
            })
        })
        .collect()
}

/// Finds the requested leaves under `columns`. `parents` is the combined validity of the
/// structs above them: a null struct row hides its children's values, which Arrow allows to be
/// non-null.
fn resolve<'a>(
    columns: &'a [ArrayRef],
    fields: &'a [FieldRef],
    parents: Option<&NullBuffer>,
    path: &mut Vec<String>,
    slots: &HashMap<&[String], Slot<'a>>,
    found: &mut [Option<Getter<'a>>],
) -> KernelResult<()> {
    for (column, field) in columns.iter().zip(fields) {
        path.push(field.name().clone());
        match slots.get(path.as_slice()) {
            Some(Slot::Parent) => {
                if let Some(st) = column.as_struct_opt() {
                    let nulls = NullBuffer::union(parents, st.nulls());
                    resolve(
                        st.columns(),
                        st.fields(),
                        nulls.as_ref(),
                        path,
                        slots,
                        found,
                    )?;
                }
            }
            Some(Slot::Awaiting(i, data_type)) => {
                let mut getter = Getter::resolve(path, data_type, column.as_ref())?;
                getter.parents = parents.cloned();
                found[*i] = Some(getter);
            }
            None => {}
        }
        path.pop();
    }
    Ok(())
}

/// The strings of a list's values or a map's keys or values.
#[derive(Clone, Copy)]
enum Strings<'a> {
    Utf8(&'a StringArray),
    Large(&'a LargeStringArray),
    View(&'a StringViewArray),
}

impl<'a> Strings<'a> {
    fn of(array: &'a dyn Array) -> Option<Self> {
        if let Some(a) = array.as_string_opt::<i32>() {
            Some(Self::Utf8(a))
        } else if let Some(a) = array.as_string_opt::<i64>() {
            Some(Self::Large(a))
        } else {
            array.as_string_view_opt().map(Self::View)
        }
    }

    fn str(self, row: usize) -> Option<&'a str> {
        match self {
            Self::Utf8(a) => a.is_valid(row).then(|| a.value(row)),
            Self::Large(a) => a.is_valid(row).then(|| a.value(row)),
            Self::View(a) => a.is_valid(row).then(|| a.value(row)),
        }
    }
}

impl StringArrayAccessor for Strings<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Utf8(a) => a.len(),
            Self::Large(a) => a.len(),
            Self::View(a) => a.len(),
        }
    }

    fn value(&self, index: usize) -> &str {
        match self {
            Self::Utf8(a) => a.value(index),
            Self::Large(a) => a.value(index),
            Self::View(a) => a.value(index),
        }
    }

    fn is_valid(&self, index: usize) -> bool {
        match self {
            Self::Utf8(a) => a.is_valid(index),
            Self::Large(a) => a.is_valid(index),
            Self::View(a) => a.is_valid(index),
        }
    }
}

/// A list column of any of Arrow's four list layouts.
#[derive(Clone, Copy)]
enum Lists<'a> {
    List(&'a GenericListArray<i32>),
    Large(&'a GenericListArray<i64>),
    View(&'a GenericListViewArray<i32>),
    LargeView(&'a GenericListViewArray<i64>),
}

fn list_range<O: OffsetSizeTrait>(offsets: &[O], row: usize) -> Range<usize> {
    offsets[row].as_usize()..offsets[row + 1].as_usize()
}

fn view_range<O: OffsetSizeTrait>(offsets: &[O], sizes: &[O], row: usize) -> Range<usize> {
    let start = offsets[row].as_usize();
    start..start + sizes[row].as_usize()
}

impl<'a> Lists<'a> {
    fn of(array: &'a dyn Array) -> Option<Self> {
        array
            .as_list_opt::<i32>()
            .map(Self::List)
            .or_else(|| array.as_list_opt::<i64>().map(Self::Large))
            .or_else(|| array.as_list_view_opt::<i32>().map(Self::View))
            .or_else(|| array.as_list_view_opt::<i64>().map(Self::LargeView))
    }

    fn values(self) -> &'a ArrayRef {
        match self {
            Self::List(a) => a.values(),
            Self::Large(a) => a.values(),
            Self::View(a) => a.values(),
            Self::LargeView(a) => a.values(),
        }
    }

    fn is_valid(self, row: usize) -> bool {
        match self {
            Self::List(a) => a.is_valid(row),
            Self::Large(a) => a.is_valid(row),
            Self::View(a) => a.is_valid(row),
            Self::LargeView(a) => a.is_valid(row),
        }
    }

    fn range(self, row: usize) -> Range<usize> {
        match self {
            Self::List(a) => list_range(a.offsets(), row),
            Self::Large(a) => list_range(a.offsets(), row),
            Self::View(a) => view_range(a.offsets(), a.sizes(), row),
            Self::LargeView(a) => view_range(a.offsets(), a.sizes(), row),
        }
    }
}

/// The column behind one getter, its Arrow array resolved once.
enum Leaf<'a> {
    Null,
    Bool(&'a BooleanArray),
    Byte(&'a Int8Array),
    Short(&'a Int16Array),
    Int(&'a Int32Array),
    Long(&'a Int64Array),
    Float(&'a Float32Array),
    Double(&'a Float64Array),
    Date(&'a Date32Array),
    Timestamp(&'a TimestampMicrosecondArray),
    Decimal(&'a Decimal128Array),
    Str(Strings<'a>),
    Binary(&'a dyn Array),
    StringList(Lists<'a>, Strings<'a>),
    StructList(Lists<'a>, &'a StructArray),
    Map(&'a MapArray, Strings<'a>, Strings<'a>),
}

/// The `GetData` of one requested leaf.
struct Getter<'a> {
    leaf: Leaf<'a>,
    /// The combined validity of the structs above the leaf.
    parents: Option<NullBuffer>,
}

impl<'a> Getter<'a> {
    /// The getter of `column` at `path`, checked against the declared `data_type`.
    fn resolve(path: &[String], data_type: &DataType, column: &'a dyn Array) -> KernelResult<Self> {
        if column.data_type() == &ArrowType::Null {
            return Ok(Self {
                leaf: Leaf::Null,
                parents: None,
            });
        }
        let leaf = match data_type {
            &DataType::BOOLEAN => column.as_boolean_opt().map(Leaf::Bool).ok_or("bool"),
            &DataType::STRING => Strings::of(column).map(Leaf::Str).ok_or("string"),
            &DataType::BINARY => match column.data_type() {
                ArrowType::Binary | ArrowType::LargeBinary | ArrowType::BinaryView => {
                    Ok(Leaf::Binary(column))
                }
                _ => Err("binary"),
            },
            &DataType::BYTE => column
                .as_primitive_opt::<Int8Type>()
                .map(Leaf::Byte)
                .ok_or("byte"),
            &DataType::SHORT => column
                .as_primitive_opt::<Int16Type>()
                .map(Leaf::Short)
                .ok_or("short"),
            &DataType::INTEGER => column
                .as_primitive_opt::<Int32Type>()
                .map(Leaf::Int)
                .ok_or("int"),
            &DataType::LONG => column
                .as_primitive_opt::<Int64Type>()
                .map(Leaf::Long)
                .ok_or("long"),
            &DataType::FLOAT => column
                .as_primitive_opt::<Float32Type>()
                .map(Leaf::Float)
                .ok_or("float"),
            &DataType::DOUBLE => column
                .as_primitive_opt::<Float64Type>()
                .map(Leaf::Double)
                .ok_or("double"),
            &DataType::DATE => column
                .as_primitive_opt::<Date32Type>()
                .map(Leaf::Date)
                .ok_or("date"),
            &DataType::TIMESTAMP | &DataType::TIMESTAMP_NTZ => column
                .as_primitive_opt::<TimestampMicrosecondType>()
                .map(Leaf::Timestamp)
                .ok_or("timestamp"),
            DataType::Primitive(PrimitiveType::Decimal(_)) => column
                .as_primitive_opt::<Decimal128Type>()
                .map(Leaf::Decimal)
                .ok_or("decimal"),
            DataType::Array(a) if matches!(a.element_type(), DataType::Struct(_)) => {
                if a.contains_null() {
                    return Err(KernelError::unexpected_column_type(format!(
                        "On {}: array<struct> columns with nullable elements are not visitable",
                        ColumnName::new(path)
                    )));
                }
                Lists::of(column)
                    .and_then(|l| l.values().as_struct_opt().map(|s| Leaf::StructList(l, s)))
                    .ok_or("array<struct>")
            }
            DataType::Array(_) => Lists::of(column)
                .and_then(|l| Strings::of(l.values().as_ref()).map(|s| Leaf::StringList(l, s)))
                .ok_or("array<string>"),
            DataType::Map(_) => column
                .as_map_opt()
                .and_then(|m| {
                    Some(Leaf::Map(
                        m,
                        Strings::of(m.keys().as_ref())?,
                        Strings::of(m.values().as_ref())?,
                    ))
                })
                .ok_or("map<string, string>"),
            other => {
                return Err(KernelError::UnexpectedColumnType(format!(
                    "On {}: Unsupported type {other}",
                    ColumnName::new(path)
                )));
            }
        };
        leaf.map(|leaf| Self {
            leaf,
            parents: None,
        })
        .map_err(|expected| {
            KernelError::UnexpectedColumnType(format!(
                "Type mismatch on {}: expected {expected}, got {}",
                ColumnName::new(path),
                column.data_type()
            ))
        })
    }

    /// Whether no struct above the leaf is null at `row`.
    fn shown(&self, row: usize) -> bool {
        self.parents.as_ref().is_none_or(|n| n.is_valid(row))
    }

    fn mismatch<T>(&self, field: &str, asked: &str) -> KernelResult<Option<T>> {
        Err(KernelError::UnexpectedColumnType(format!(
            "{field} is not of type {asked}"
        )))
    }
}

macro_rules! primitive_get {
    ($name:ident, $variant:ident, $t:ty) => {
        fn $name(&'b self, row: usize, field: &str) -> KernelResult<Option<$t>> {
            match &self.leaf {
                Leaf::Null => Ok(None),
                Leaf::$variant(a) => Ok((self.shown(row) && a.is_valid(row)).then(|| a.value(row))),
                _ => self.mismatch(field, stringify!($t)),
            }
        }
    };
}

impl<'b, 'a: 'b> GetData<'b> for Getter<'a> {
    primitive_get!(get_bool, Bool, bool);
    primitive_get!(get_byte, Byte, i8);
    primitive_get!(get_short, Short, i16);
    primitive_get!(get_int, Int, i32);
    primitive_get!(get_long, Long, i64);
    primitive_get!(get_float, Float, f32);
    primitive_get!(get_double, Double, f64);
    primitive_get!(get_date, Date, i32);
    primitive_get!(get_timestamp, Timestamp, i64);
    primitive_get!(get_decimal, Decimal, i128);

    fn get_str(&'b self, row: usize, field: &str) -> KernelResult<Option<&'b str>> {
        match &self.leaf {
            Leaf::Null => Ok(None),
            Leaf::Str(s) => Ok(if self.shown(row) { s.str(row) } else { None }),
            _ => self.mismatch(field, "&str"),
        }
    }

    fn get_binary(&'b self, row: usize, field: &str) -> KernelResult<Option<&'b [u8]>> {
        match &self.leaf {
            Leaf::Null => Ok(None),
            Leaf::Binary(a) => {
                if !self.shown(row) || !a.is_valid(row) {
                    return Ok(None);
                }
                let a: &'b dyn Array = *a;
                Ok(a.as_binary_opt::<i32>()
                    .map(|b| b.value(row))
                    .or_else(|| a.as_binary_opt::<i64>().map(|b| b.value(row)))
                    .or_else(|| a.as_binary_view_opt().map(|b| b.value(row))))
            }
            _ => self.mismatch(field, "&[u8]"),
        }
    }

    fn get_list(&'b self, row: usize, field: &str) -> KernelResult<Option<ListItem<'b>>> {
        match &self.leaf {
            Leaf::Null => Ok(None),
            Leaf::StringList(list, values) => Ok((self.shown(row) && list.is_valid(row))
                .then(|| ListItem::new(values, list.range(row)))),
            _ => self.mismatch(field, "ListItem"),
        }
    }

    fn get_map(&'b self, row: usize, field: &str) -> KernelResult<Option<MapItem<'b>>> {
        match &self.leaf {
            Leaf::Null => Ok(None),
            Leaf::Map(map, keys, values) => Ok((self.shown(row) && map.is_valid(row))
                .then(|| MapItem::new(keys, values, list_range(map.value_offsets(), row)))),
            _ => self.mismatch(field, "MapItem"),
        }
    }

    fn get_struct_list(&'b self, row: usize, field: &str) -> KernelResult<Option<StructList<'b>>> {
        match &self.leaf {
            Leaf::Null => Ok(None),
            Leaf::StructList(list, _) => {
                Ok((self.shown(row) && list.is_valid(row)).then(|| StructList::new(self, row)))
            }
            _ => self.mismatch(field, "StructList"),
        }
    }
}

impl StructListAccessor for Getter<'_> {
    fn visit_elems_of_row(
        &self,
        row_index: usize,
        column_names: &[ColumnName],
        visitor: &mut dyn RowVisitor,
    ) -> KernelResult<()> {
        let Leaf::StructList(list, elements) = &self.leaf else {
            return Err(KernelError::unexpected_column_type("not a list of structs"));
        };
        let range = list.range(row_index);
        let sliced = StructArray::slice(elements, range.start, range.len());
        // A null element struct cannot become a row of the nested visit.
        if sliced.null_count() > 0 {
            return Err(KernelError::invalid_struct_data(
                "array<struct> has a null element; cannot visit",
            ));
        }
        let types = visitor.selected_column_names_and_types().1;
        let getters = getters(sliced.columns(), sliced.fields(), column_names, types)?;
        let refs: Vec<&dyn GetData<'_>> = getters.iter().map(|g| g as &dyn GetData<'_>).collect();
        visitor.visit(sliced.len(), &refs)
    }
}

#[cfg(test)]
#[path = "data_tests.rs"]
mod tests;
