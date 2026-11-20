// SPDX-License-Identifier: Apache-2.0
//! The descriptor files of `from_protobuf` / `to_protobuf` (P3-T087): a string literal third argument names a file, which is a read
//! by path like any other. Bind qualifies it, asks the policy for `Select` on it before anything touches the file system (a denied
//! principal learns nothing about the file, not even whether it exists), then reads it once; Resolve hands the bytes to the
//! function's result rule and writes them into the plan as a binary literal, so the call is bound from those bytes and no file is
//! read again. A failed read is kept and raised by the result rule, after its argument checks, where Spark reads the file.

use krishiv_common::error::{
    KrishivError, KrishivUnsupportedFeature, ProtobufDescriptorFileNotFound,
};
use krishiv_common::secret::OptionValue;
use krishiv_io::protobuf::DescriptorSet;

use super::{Bindings, authorize_as, is_builtin_qualifier};
use crate::analyzer::{AnalysisContext, AnalysisError, Privilege, Securable};
use crate::{Expr, ExprNode, Literal, Origin};

/// The descriptor path of a `from_protobuf` / `to_protobuf` call: its third argument when that is a non-empty string literal
/// (Spark's three-argument constructor takes a string literal there as the path, the four-argument one any foldable string; a
/// path that is not a literal is refused by the call rather than read). An empty path is no file (`from_protobuf`) or a parse
/// error (`to_protobuf`) without a read.
pub(super) fn descriptor_path(e: &Expr) -> Option<&str> {
    let ExprNode::Function(f) = &e.node else {
        return None;
    };
    let named =
        f.name.eq_ignore_ascii_case("from_protobuf") || f.name.eq_ignore_ascii_case("to_protobuf");
    if !named
        || !(f.qualifier.is_empty() || is_builtin_qualifier(&f.qualifier))
        || !matches!(f.args.len(), 3 | 4)
    {
        return None;
    }
    match &f.args[2].node {
        ExprNode::Literal(Literal::String(s)) if !s.is_empty() => Some(s),
        _ => None,
    }
}

/// Reads the descriptor file `path` names into [`Bindings::descriptors`], once per query, after the policy allowed `Select` on
/// the qualified path (a denial fails the analysis here, as for any read by path).
pub(super) async fn bind_descriptor(
    ctx: &AnalysisContext,
    b: &mut Bindings,
    checked: &mut Vec<(Securable, usize)>,
    path: &str,
    origin: Origin,
) -> Result<(), AnalysisError> {
    if b.descriptors.contains_key(path) {
        return Ok(());
    }
    // The plan holds the file's contents: it depends on the file, like a read by path.
    b.reads_files = true;
    let read = match &ctx.files {
        None => Err(KrishivUnsupportedFeature {
            feature: "reading a protobuf descriptor file by path".to_owned(),
        }
        .into()),
        Some(files) => match files.qualify(&OptionValue::from_user("path", path)) {
            Err(e) => Err(e),
            Ok(qualified) => {
                let securable = Securable::Path(qualified.clone());
                authorize_as(ctx, b, checked, securable, Privilege::Select, origin).await?;
                read(path, &qualified)
            }
        },
    };
    b.descriptors.insert(path.to_owned(), read);
    Ok(())
}

/// The bytes of the local file `qualified` names (Spark reads a descriptor with `java.io.File`: anything but a local path is not
/// found). `path` is the path as written, for the error.
fn read(path: &str, qualified: &OptionValue) -> Result<Vec<u8>, KrishivError> {
    match qualified.as_plain() {
        Some(text) if text.starts_with("file:") || !text.contains("://") => {
            DescriptorSet::read_file(path, text)
        }
        _ => Err(ProtobufDescriptorFileNotFound {
            file_path: path.to_owned(),
        }
        .into()),
    }
}
