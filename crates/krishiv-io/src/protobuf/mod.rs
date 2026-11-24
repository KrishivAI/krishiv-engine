// SPDX-License-Identifier: Apache-2.0
//! Protobuf messages outside a file: Spark's `from_protobuf` and `to_protobuf` (the `spark-protobuf` module, SQL functions since 4.0).
//!
//! A [`DescriptorSet`] reads a `FileDescriptorSet` (`protoc --include_imports --descriptor_set_out`) with our own reader (`desc`: a
//! protobuf library may not sit under `krishiv-plan`, `xtask/layers.toml`); the wire
//! codec here is our own because Spark's rules differ from a generic one (last scalar wins, merged message fields, map entries kept
//! in wire order with duplicate keys, unknown-field checks, presence rules for proto3 scalars). [`Decoder`] turns a column of
//! messages into one Arrow array of Spark's type for the message (`schema`); [`Encoder`] writes a struct column as messages.
//!
//! Not supported (refused, not guessed): the Java-class form (no descriptor), proto2 extensions
//! (`spark.sql.function.protobufExtensions.enabled`), and `Struct`, `Value`, `ListValue` and `FieldMask` inside an `Any` that is
//! printed as JSON.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the wire format is bit-level: Java's `(int) varint`, `Integer.toUnsignedLong` and zigzag are these casts, on purpose"
)]
#![expect(
    clippy::match_same_arms,
    reason = "arms stay separate per protobuf kind so each reads as the rule of that kind"
)]

mod decode;
mod desc;
mod encode;
mod json;
mod options;
mod schema;
#[cfg(test)]
mod tests;
mod wire;

use std::sync::Arc;

use krishiv_common::error::{
    CannotLoadProtobufClass, CannotParseProtobufDescriptor, KrishivError,
    ProtobufDependencyNotFound, ProtobufDescriptorFileNotFound, ProtobufMessageNotFound,
};
use krishiv_types::DataType;

pub use self::decode::Decoder;
use self::desc::{MessageDescriptor, Pool, Refused};
pub use self::encode::{Encoder, spark_type_of};
pub use self::options::{ParseMode, ProtoOptions};

use crate::error::unsupported;

/// Where the descriptor set of a call comes from: its third argument.
#[derive(Debug, Clone)]
pub enum DescriptorArg {
    /// None (a NULL, or a call with two arguments): the message name is a Java class name.
    None,
    /// A path to a descriptor file, not read: the analyzer reads a path under the session's access policy and passes
    /// [`Self::File`]. An empty path is no descriptor for `from_protobuf` and unreadable for `to_protobuf`; any other path is refused
    /// here, so no descriptor file is ever read outside that check.
    Path(String),
    /// The bytes of a descriptor set (a binary argument).
    Bytes(Vec<u8>),
    /// The bytes of the descriptor file a path named, read by the analyzer after the access check
    /// ([`DescriptorSet::read_file`]). Unlike [`Self::Bytes`], empty contents are still a descriptor set.
    File(Vec<u8>),
}

/// The constant arguments of a `from_protobuf` or `to_protobuf` call, with the descriptor set read.
#[derive(Debug, Clone)]
pub struct Call {
    message: String,
    descriptor: Option<Vec<u8>>,
    options: Vec<(String, String)>,
}

impl Call {
    /// The call of `descriptor`. `from_protobuf` takes an empty path or empty bytes as no descriptor; `to_protobuf` (`to`) fails on
    /// the empty path and takes the empty bytes as an empty set.
    ///
    /// # Errors
    /// `CANNOT_PARSE_PROTOBUF_DESCRIPTOR` for `to_protobuf`'s empty path, and `KRISHIV_UNSUPPORTED_FEATURE` for a path the analyzer
    /// did not read (a descriptor path that is not a string literal).
    pub fn new(
        message: &str,
        descriptor: DescriptorArg,
        options: Vec<(String, String)>,
        to: bool,
    ) -> Result<Self, KrishivError> {
        let descriptor = match descriptor {
            DescriptorArg::None => None,
            DescriptorArg::Path(p) if p.is_empty() && !to => None,
            // Spark's Hadoop `Path` refuses the empty string.
            DescriptorArg::Path(p) if p.is_empty() => {
                return Err(CannotParseProtobufDescriptor {}.into());
            }
            DescriptorArg::Path(_) => {
                return Err(unsupported(
                    "a protobuf descriptor file path that is not a string literal",
                ));
            }
            DescriptorArg::Bytes(b) if b.is_empty() && !to => None,
            DescriptorArg::Bytes(b) | DescriptorArg::File(b) => Some(b),
        };
        Ok(Self {
            message: message.to_owned(),
            descriptor,
            options,
        })
    }

    /// The message name.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    fn set(&self) -> Result<DescriptorSet, KrishivError> {
        let Some(bytes) = &self.descriptor else {
            // Spark would load a generated Java class of that name.
            return Err(CannotLoadProtobufClass {
                protobuf_class_name: self.message.clone(),
                explanation: if self.message.contains('.') {
                    "Ensure the class include in the jar".to_owned()
                } else {
                    "Ensure the class name includes package prefix".to_owned()
                },
            }
            .into());
        };
        DescriptorSet::parse(bytes)
    }

    /// The reader of the call.
    ///
    /// # Errors
    /// The descriptor, message, option and schema errors of Spark, in Spark's order.
    pub fn decoder(&self) -> Result<Decoder, KrishivError> {
        let set = self.set()?;
        Decoder::new(
            &set,
            &self.message,
            self.options.iter().map(|(k, v)| (k.as_str(), v.as_str())),
        )
    }

    /// The writer of the call for values of Spark type `input`.
    ///
    /// # Errors
    /// The descriptor and conversion errors of Spark.
    pub fn encoder(&self, input: &DataType) -> Result<Encoder, KrishivError> {
        Encoder::new(&self.set()?, &self.message, input)
    }
}

/// A parsed `FileDescriptorSet`.
#[derive(Debug, Clone)]
pub struct DescriptorSet {
    pool: Arc<Pool>,
}

impl DescriptorSet {
    /// Parses the bytes of a `FileDescriptorSet`.
    ///
    /// # Errors
    /// `CANNOT_PARSE_PROTOBUF_DESCRIPTOR` when the bytes are not a descriptor set, `PROTOBUF_DEPENDENCY_NOT_FOUND` naming the
    /// first import the set does not hold (Spark's order).
    pub fn parse(bytes: &[u8]) -> Result<Self, KrishivError> {
        match desc::parse(bytes) {
            Ok(Ok(pool)) => Ok(Self { pool }),
            Ok(Err(Refused::Unsupported(feature))) => Err(unsupported(feature)),
            Ok(Err(Refused::MissingDependency(name))) => Err(ProtobufDependencyNotFound {
                dependency_name: name,
            }
            .into()),
            Err(_) => Err(CannotParseProtobufDescriptor {}.into()),
        }
    }

    /// Reads the descriptor file `path` names, at `local` (a local path; a `file:` URI is accepted). Only the analyzer calls it,
    /// after the access policy allowed the path; `path` is the path as the call wrote it, for the error.
    ///
    /// # Errors
    /// `PROTOBUF_DESCRIPTOR_FILE_NOT_FOUND` when the file does not exist, `CANNOT_PARSE_PROTOBUF_DESCRIPTOR` when it cannot be
    /// read (a directory, an empty path, a file past 64 MiB).
    pub fn read_file(path: &str, local: &str) -> Result<Vec<u8>, KrishivError> {
        let local = local
            .strip_prefix("file://")
            .or_else(|| local.strip_prefix("file:"))
            .unwrap_or(local);
        // `file:/C:/x` is `C:/x` on Windows: `/C:/x` does not open there.
        let local = if cfg!(windows) {
            without_slashes_before_drive(local)
        } else {
            local
        };
        // A descriptor set is kilobytes; a file past the cap is not one (and is never read whole).
        const MAX_BYTES: u64 = 64 << 20;
        let fail = |e: &std::io::Error| -> KrishivError {
            // An empty path or a directory is a parse failure in Spark (a Hadoop `Path` error); a missing file is not found.
            if local.is_empty() || e.kind() != std::io::ErrorKind::NotFound {
                CannotParseProtobufDescriptor {}.into()
            } else {
                ProtobufDescriptorFileNotFound {
                    file_path: path.to_owned(),
                }
                .into()
            }
        };
        // Spark follows links (a mounted ConfigMap is one); the file read is the file opened, whatever the path now names.
        let before = std::fs::metadata(local).map_err(|e| fail(&e))?;
        let file = std::fs::File::open(local).map_err(|e| fail(&e))?;
        let opened = file.metadata().map_err(|e| fail(&e))?;
        // Unix: the same device and inode. Windows' file identity (volume serial and file index) is not in
        // stable std: there the handle's creation time (100 ns ticks), last write time and size must be
        // the path's, so a file swapped in between the check and the open is refused unless it matches all
        // three (Windows port, 0.3.0).
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if opened.dev() != before.dev() || opened.ino() != before.ino() {
                return Err(CannotParseProtobufDescriptor {}.into());
            }
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            let stamp =
                |m: &std::fs::Metadata| (m.creation_time(), m.last_write_time(), m.file_size());
            if stamp(&opened) != stamp(&before) {
                return Err(CannotParseProtobufDescriptor {}.into());
            }
        }
        #[cfg(not(any(unix, windows)))]
        let _ = (&before, &opened);
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut std::io::Read::take(file, MAX_BYTES + 1), &mut bytes)
            .map_err(|e| fail(&e))?;
        if bytes.len() as u64 > MAX_BYTES {
            return Err(CannotParseProtobufDescriptor {}.into());
        }
        Ok(bytes)
    }

    /// The first top-level message whose short or full name is `name`, files in set order (Spark's lookup).
    ///
    /// # Errors
    /// `PROTOBUF_MESSAGE_NOT_FOUND`.
    pub(crate) fn message(&self, name: &str) -> Result<MessageDescriptor, KrishivError> {
        self.pool
            .top_level()
            .into_iter()
            .find(|m| m.name() == name || m.full_name() == name)
            .ok_or_else(|| {
                ProtobufMessageNotFound {
                    message_name: name.to_owned(),
                }
                .into()
            })
    }

    /// The message of a full name (`pkg.Outer.Inner`), nested ones included.
    pub(crate) fn message_by_full_name(&self, name: &str) -> Option<MessageDescriptor> {
        self.pool.message_by_full_name(name)
    }
}

/// `text` without the slashes before a leading drive (`/C:/x` is `C:/x`); other text as it is.
fn without_slashes_before_drive(text: &str) -> &str {
    let rest = text.trim_start_matches('/');
    let b = rest.as_bytes();
    let drive = b.len() >= 2
        && b[0].is_ascii_alphabetic()
        && b[1] == b':'
        && (b.len() == 2 || matches!(b[2], b'/' | b'\\'));
    if drive { rest } else { text }
}
