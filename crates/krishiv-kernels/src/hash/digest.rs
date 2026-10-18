// SPDX-License-Identifier: Apache-2.0
//! `md5`, `sha1`, `sha2` and `crc32` over the bytes of a string or binary
//! column (a `STRING` argument is its UTF-8 bytes).

use arrow_array::builder::GenericStringBuilder;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, Int64Array, StringArray};
use arrow_schema::DataType;
use md5::Md5;
use sha1::Sha1;
use sha2::{Digest, Sha224, Sha256, Sha384, Sha512};

use crate::error::KernelError;
use crate::ops::Operand;

/// The bytes of row `i` of a string/binary column; `None` for NULL.
fn bytes_at(a: &dyn Array, i: usize) -> Option<&[u8]> {
    if a.is_null(i) {
        return None;
    }
    Some(match a.data_type() {
        DataType::Utf8 => a.as_string::<i32>().value(i).as_bytes(),
        DataType::LargeUtf8 => a.as_string::<i64>().value(i).as_bytes(),
        DataType::Binary => a.as_binary::<i32>().value(i),
        DataType::LargeBinary => a.as_binary::<i64>().value(i),
        _ => return None,
    })
}

fn check_input(a: &dyn Array) -> Result<(), KernelError> {
    match a.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => Ok(()),
        other => Err(KernelError::internal(format!(
            "a digest takes a string or binary column, not {other}"
        ))),
    }
}

const HEX: &[u8; 16] = b"0123456789abcdef";

fn hex(bytes: &[u8], out: &mut String) {
    for b in bytes {
        out.push(char::from(HEX[usize::from(b >> 4)]));
        out.push(char::from(HEX[usize::from(b & 15)]));
    }
}

fn hex_digests(
    input: &dyn Array,
    mut digest: impl FnMut(usize, &[u8], &mut String) -> bool,
) -> Result<StringArray, KernelError> {
    check_input(input)?;
    let mut out = GenericStringBuilder::<i32>::with_capacity(input.len(), 0);
    let mut scratch = String::new();
    for i in 0..input.len() {
        scratch.clear();
        match bytes_at(input, i) {
            Some(b) if digest(i, b, &mut scratch) => out.append_value(&scratch),
            _ => out.append_null(),
        }
    }
    Ok(out.finish())
}

/// `md5(input)`: 32 lowercase hex digits.
///
/// # Errors
/// `INTERNAL_ERROR` if `input` is not a string or binary column.
pub fn md5(input: &dyn Array) -> Result<StringArray, KernelError> {
    hex_digests(input, |_, b, out| {
        hex(&Md5::digest(b), out);
        true
    })
}

/// `sha1(input)` (also `sha`): 40 lowercase hex digits.
///
/// # Errors
/// As [`md5()`].
pub fn sha1(input: &dyn Array) -> Result<StringArray, KernelError> {
    hex_digests(input, |_, b, out| {
        hex(&Sha1::digest(b), out);
        true
    })
}

/// `sha2(input, bits)`: SHA-224/256/384/512 for `bits` 224, 256, 384, 512
/// (0 is 256); any other length, and a NULL length, gives NULL.
///
/// # Errors
/// As [`md5()`]; `INTERNAL_ERROR` if a `bits` column differs in length.
pub fn sha2(input: &dyn Array, bits: Operand<'_, Int32Type>) -> Result<StringArray, KernelError> {
    if let Operand::Array(b) = bits
        && b.len() != input.len()
    {
        return Err(KernelError::internal(format!(
            "sha2 operands differ in length: {} vs {}",
            input.len(),
            b.len()
        )));
    }
    let bits_at = |i: usize| match bits {
        Operand::Array(a) => a.is_valid(i).then(|| a.value(i)),
        Operand::Scalar(v) => v,
    };
    hex_digests(input, |i, b, out| match bits_at(i) {
        Some(0 | 256) => {
            hex(&Sha256::digest(b), out);
            true
        }
        Some(224) => {
            hex(&Sha224::digest(b), out);
            true
        }
        Some(384) => {
            hex(&Sha384::digest(b), out);
            true
        }
        Some(512) => {
            hex(&Sha512::digest(b), out);
            true
        }
        _ => false,
    })
}

/// The CRC-32 (IEEE 802.3, reflected) table.
const fn crc_table() -> [u32; 256] {
    let mut t = [0_u32; 256];
    let mut n = 0_u32;
    while n < 256 {
        let mut c = n;
        let mut k = 0_u32;
        while k < 8 {
            c = if c & 1 == 1 {
                0xedb8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k = k.wrapping_add(1);
        }
        t[n as usize] = c;
        n = n.wrapping_add(1);
    }
    t
}

static CRC_TABLE: [u32; 256] = crc_table();

fn crc32_of(bytes: &[u8]) -> u32 {
    !bytes.iter().fold(!0_u32, |c, &b| {
        CRC_TABLE[usize::from(u8::try_from((c ^ u32::from(b)) & 0xff).unwrap_or_default())]
            ^ (c >> 8)
    })
}

/// `crc32(input)`: the CRC-32 checksum as a `bigint`.
///
/// # Errors
/// As [`md5()`].
pub fn crc32(input: &dyn Array) -> Result<Int64Array, KernelError> {
    check_input(input)?;
    Ok((0..input.len())
        .map(|i| bytes_at(input, i).map(|b| i64::from(crc32_of(b))))
        .collect())
}
