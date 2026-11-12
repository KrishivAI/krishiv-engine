// SPDX-License-Identifier: Apache-2.0
//! `GetData` over Arrow 60 arrays.
//!
//! Follows `delta_kernel` 0.30.1 `engine/arrow_get_data.rs` (Apache-2.0). The kernel implements its traits
//! directly on Arrow array types; outside the kernel's crate that breaks the orphan rule, so one
//! [`Getter`] enum holds the borrowed array and implements [`GetData`] by matching on it. List element and
//! map key/value string accessors are resolved once, when the getter is built.

use std::ops::Range;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{
    Array, BinaryViewArray, BooleanArray, GenericBinaryArray, GenericListArray,
    GenericListViewArray, GenericStringArray, MapArray, OffsetSizeTrait, PrimitiveArray, RunArray,
    StringViewArray, StructArray,
};
use arrow_buffer::ArrowNativeType;
use delta_kernel::engine_data::{
    ListItem, MapItem, RowVisitor, StringArrayAccessor, StructList, StructListAccessor,
};
use delta_kernel::expressions::ColumnName;
use delta_kernel::{EngineData, GetData, KernelError, KernelResult};

use super::data::KrishivEngineData;

/// A string array of any of the three Arrow string layouts.
pub(crate) enum StrAcc<'a> {
    Utf8(&'a GenericStringArray<i32>),
    Large(&'a GenericStringArray<i64>),
    View(&'a StringViewArray),
}

impl<'a> StrAcc<'a> {
    pub(crate) fn of(array: &'a dyn Array) -> Option<Self> {
        array
            .as_string_opt::<i32>()
            .map(Self::Utf8)
            .or_else(|| array.as_string_opt::<i64>().map(Self::Large))
            .or_else(|| array.as_string_view_opt().map(Self::View))
    }
}

impl StringArrayAccessor for StrAcc<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Utf8(a) => Array::len(*a),
            Self::Large(a) => Array::len(*a),
            Self::View(a) => Array::len(*a),
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
            Self::Utf8(a) => Array::is_valid(*a, index),
            Self::Large(a) => Array::is_valid(*a, index),
            Self::View(a) => Array::is_valid(*a, index),
        }
    }
}

/// The four list-like layouts, which differ in how a row's offsets are computed.
pub(crate) enum ListKind<'a> {
    List(&'a GenericListArray<i32>),
    LargeList(&'a GenericListArray<i64>),
    View(&'a GenericListViewArray<i32>),
    LargeView(&'a GenericListViewArray<i64>),
}

fn list_offsets<O: OffsetSizeTrait>(a: &GenericListArray<O>, row: usize) -> Range<usize> {
    a.offsets()[row].as_usize()..a.offsets()[row + 1].as_usize()
}

fn view_offsets<O: OffsetSizeTrait>(a: &GenericListViewArray<O>, row: usize) -> Range<usize> {
    let start = a.offsets()[row].as_usize();
    start..start + a.sizes()[row].as_usize()
}

impl<'a> ListKind<'a> {
    pub(crate) fn of(array: &'a dyn Array) -> Option<Self> {
        array
            .as_list_opt::<i32>()
            .map(Self::List)
            .or_else(|| array.as_list_opt::<i64>().map(Self::LargeList))
            .or_else(|| array.as_list_view_opt::<i32>().map(Self::View))
            .or_else(|| array.as_list_view_opt::<i64>().map(Self::LargeView))
    }
    fn is_valid(&self, row: usize) -> bool {
        match self {
            Self::List(a) => a.is_valid(row),
            Self::LargeList(a) => a.is_valid(row),
            Self::View(a) => a.is_valid(row),
            Self::LargeView(a) => a.is_valid(row),
        }
    }
    fn values(&self) -> &'a dyn Array {
        match self {
            Self::List(a) => a.values().as_ref(),
            Self::LargeList(a) => a.values().as_ref(),
            Self::View(a) => a.values().as_ref(),
            Self::LargeView(a) => a.values().as_ref(),
        }
    }
    fn row_offsets(&self, row: usize) -> Range<usize> {
        match self {
            Self::List(a) => list_offsets(a, row),
            Self::LargeList(a) => list_offsets(a, row),
            Self::View(a) => view_offsets(a, row),
            Self::LargeView(a) => view_offsets(a, row),
        }
    }
}

/// A list column: its layout and, for a list of strings, the string accessor over its elements.
pub(crate) struct ListGetter<'a> {
    kind: ListKind<'a>,
    strings: Option<StrAcc<'a>>,
}

impl<'a> ListGetter<'a> {
    pub(crate) fn new(kind: ListKind<'a>) -> Self {
        let strings = StrAcc::of(kind.values());
        Self { kind, strings }
    }

    /// The element structs; a list of anything else is a type error for every row, even a null one.
    fn struct_elements(&self, field_name: &str) -> KernelResult<&StructArray> {
        self.kind.values().as_struct_opt().ok_or_else(|| {
            KernelError::unexpected_column_type(format!(
                "{field_name}: list values are not structs"
            ))
        })
    }
}

impl StructListAccessor for ListGetter<'_> {
    fn visit_elems_of_row(
        &self,
        row_index: usize,
        column_names: &[ColumnName],
        visitor: &mut dyn RowVisitor,
    ) -> KernelResult<()> {
        let offsets = self.kind.row_offsets(row_index);
        let sliced = self
            .struct_elements("struct-list")?
            .slice(offsets.start, offsets.len());
        // A null element struct cannot round-trip through a record batch, so reject it.
        if sliced.is_nullable() {
            return Err(KernelError::invalid_struct_data(
                "array<struct> elements are nullable; cannot visit",
            ));
        }
        KrishivEngineData::from(sliced).visit_rows(column_names, visitor)
    }
}

/// A map<string, string> column with its key and value accessors.
pub(crate) struct MapGetter<'a> {
    array: &'a MapArray,
    keys: StrAcc<'a>,
    values: StrAcc<'a>,
}

impl<'a> MapGetter<'a> {
    pub(crate) fn new(array: &'a MapArray) -> Option<Self> {
        Some(Self {
            keys: StrAcc::of(array.keys().as_ref())?,
            values: StrAcc::of(array.values().as_ref())?,
            array,
        })
    }
}

/// One requested leaf column, borrowed from the batch.
pub(crate) enum Getter<'a> {
    Null,
    Bool(&'a BooleanArray),
    I8(&'a PrimitiveArray<Int8Type>),
    I16(&'a PrimitiveArray<Int16Type>),
    I32(&'a PrimitiveArray<Int32Type>),
    I64(&'a PrimitiveArray<Int64Type>),
    F32(&'a PrimitiveArray<Float32Type>),
    F64(&'a PrimitiveArray<Float64Type>),
    Date(&'a PrimitiveArray<Date32Type>),
    Timestamp(&'a PrimitiveArray<TimestampMicrosecondType>),
    Decimal(&'a PrimitiveArray<Decimal128Type>),
    Str(StrAcc<'a>),
    Binary(&'a GenericBinaryArray<i32>),
    LargeBinary(&'a GenericBinaryArray<i64>),
    BinaryView(&'a BinaryViewArray),
    List(ListGetter<'a>),
    Map(MapGetter<'a>),
    /// A run-end-encoded column (string, binary, bool, int or long values).
    Ree(&'a RunArray<Int64Type>),
}

fn wrong<T>(field_name: &str, ty: &str) -> KernelResult<T> {
    Err(KernelError::unexpected_column_type(format!(
        "{field_name} is not of type {ty}"
    )))
}

/// The physical index of a row in a run-end-encoded array. A run array has no validity bitmap of its own;
/// nulls live in the values child, so each getter checks the child.
fn physical_index(
    run: &RunArray<Int64Type>,
    row_index: usize,
    field_name: &str,
) -> KernelResult<usize> {
    if row_index >= run.len() {
        return Err(KernelError::generic(format!(
            "Row index {row_index} out of bounds for field '{field_name}'"
        )));
    }
    Ok(run.run_ends().get_physical_index(row_index))
}

fn ree_values<'v, T>(
    got: Option<&'v T>,
    run: &RunArray<Int64Type>,
    what: &str,
) -> KernelResult<&'v T> {
    got.ok_or_else(|| {
        KernelError::generic(format!(
            "Expected {what} values in RunArray, got {:?}",
            run.values().data_type()
        ))
    })
}

macro_rules! prim_get {
    ($name:ident, $out:ty, $variant:ident, $ty:literal) => {
        fn $name(&'a self, row_index: usize, field_name: &str) -> KernelResult<Option<$out>> {
            match self {
                Getter::$variant(a) => Ok(a.is_valid(row_index).then(|| a.value(row_index))),
                Getter::Null => Ok(None),
                _ => wrong(field_name, $ty),
            }
        }
    };
}

impl<'a> GetData<'a> for Getter<'a> {
    prim_get!(get_byte, i8, I8, "i8");
    prim_get!(get_short, i16, I16, "i16");
    prim_get!(get_float, f32, F32, "f32");
    prim_get!(get_double, f64, F64, "f64");
    prim_get!(get_date, i32, Date, "i32");
    prim_get!(get_timestamp, i64, Timestamp, "i64");
    prim_get!(get_decimal, i128, Decimal, "i128");

    fn get_bool(&'a self, row_index: usize, field_name: &str) -> KernelResult<Option<bool>> {
        match self {
            Getter::Bool(a) => Ok(a.is_valid(row_index).then(|| a.value(row_index))),
            Getter::Ree(r) => {
                let i = physical_index(r, row_index, field_name)?;
                let v = ree_values(r.values().as_boolean_opt(), r, "BooleanArray")?;
                Ok((!v.is_null(i)).then(|| v.value(i)))
            }
            Getter::Null => Ok(None),
            _ => wrong(field_name, "bool"),
        }
    }

    fn get_int(&'a self, row_index: usize, field_name: &str) -> KernelResult<Option<i32>> {
        match self {
            Getter::I32(a) => Ok(a.is_valid(row_index).then(|| a.value(row_index))),
            Getter::Ree(r) => {
                let i = physical_index(r, row_index, field_name)?;
                let v = ree_values(r.values().as_primitive_opt::<Int32Type>(), r, "Int32Array")?;
                Ok((!v.is_null(i)).then(|| v.value(i)))
            }
            Getter::Null => Ok(None),
            _ => wrong(field_name, "i32"),
        }
    }

    fn get_long(&'a self, row_index: usize, field_name: &str) -> KernelResult<Option<i64>> {
        match self {
            Getter::I64(a) => Ok(a.is_valid(row_index).then(|| a.value(row_index))),
            Getter::Ree(r) => {
                let i = physical_index(r, row_index, field_name)?;
                let v = ree_values(r.values().as_primitive_opt::<Int64Type>(), r, "Int64Array")?;
                Ok((!v.is_null(i)).then(|| v.value(i)))
            }
            Getter::Null => Ok(None),
            _ => wrong(field_name, "i64"),
        }
    }

    fn get_str(&'a self, row_index: usize, field_name: &str) -> KernelResult<Option<&'a str>> {
        match self {
            Getter::Str(s) => {
                Ok(StringArrayAccessor::is_valid(s, row_index).then(|| s.value(row_index)))
            }
            Getter::Ree(r) => {
                let i = physical_index(r, row_index, field_name)?;
                let v = ree_values(r.values().as_string_opt::<i32>(), r, "StringArray")?;
                Ok((!v.is_null(i)).then(|| v.value(i)))
            }
            Getter::Null => Ok(None),
            _ => wrong(field_name, "&str"),
        }
    }

    fn get_binary(&'a self, row_index: usize, field_name: &str) -> KernelResult<Option<&'a [u8]>> {
        match self {
            Getter::Binary(a) => Ok(a.is_valid(row_index).then(|| a.value(row_index))),
            Getter::LargeBinary(a) => Ok(a.is_valid(row_index).then(|| a.value(row_index))),
            Getter::BinaryView(a) => Ok(a.is_valid(row_index).then(|| a.value(row_index))),
            Getter::Ree(r) => {
                let i = physical_index(r, row_index, field_name)?;
                let v = ree_values(r.values().as_binary_opt::<i32>(), r, "BinaryArray")?;
                Ok((!v.is_null(i)).then(|| v.value(i)))
            }
            Getter::Null => Ok(None),
            _ => wrong(field_name, "&[u8]"),
        }
    }

    fn get_list(
        &'a self,
        row_index: usize,
        field_name: &str,
    ) -> KernelResult<Option<ListItem<'a>>> {
        match self {
            Getter::List(l) => {
                if !l.kind.is_valid(row_index) {
                    return Ok(None);
                }
                let strings = l.strings.as_ref().ok_or_else(|| {
                    KernelError::unexpected_column_type(format!(
                        "{field_name}: list values are not a supported string type"
                    ))
                })?;
                Ok(Some(ListItem::new(strings, l.kind.row_offsets(row_index))))
            }
            Getter::Null => Ok(None),
            _ => wrong(field_name, "ListItem"),
        }
    }

    fn get_map(&'a self, row_index: usize, field_name: &str) -> KernelResult<Option<MapItem<'a>>> {
        match self {
            Getter::Map(m) => {
                if !m.array.is_valid(row_index) {
                    return Ok(None);
                }
                let start = m.array.offsets()[row_index].as_usize();
                let end = m.array.offsets()[row_index + 1].as_usize();
                Ok(Some(MapItem::new(&m.keys, &m.values, start..end)))
            }
            Getter::Null => Ok(None),
            _ => wrong(field_name, "MapItem"),
        }
    }

    fn get_struct_list(
        &'a self,
        row_index: usize,
        field_name: &str,
    ) -> KernelResult<Option<StructList<'a>>> {
        match self {
            Getter::List(l) => {
                l.struct_elements(field_name)?;
                if !l.kind.is_valid(row_index) {
                    return Ok(None);
                }
                Ok(Some(StructList::new(l, row_index)))
            }
            Getter::Null => Ok(None),
            _ => wrong(field_name, "StructList"),
        }
    }
}
