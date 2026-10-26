// SPDX-License-Identifier: Apache-2.0
//! The options of the `text` format (Spark's names and defaults) and the
//! parsed [`TextOptions`] the decoder and encoder use.

use std::sync::LazyLock;

use crate::error::Result;
use crate::options::{FormatOptions, OptionDefault, OptionKind, OptionSchema, OptionSpec};
use crate::text::{Charset, LineSep, check_write_codec};

fn string(name: &'static str, doc: &'static str) -> OptionSpec {
    OptionSpec {
        name,
        kind: OptionKind::String,
        default: OptionDefault::None,
        conf_fallback: None,
        metadata_key: false,
        doc,
    }
}

fn flag(name: &'static str, default: bool, doc: &'static str) -> OptionSpec {
    OptionSpec {
        name,
        kind: OptionKind::Bool,
        default: OptionDefault::Bool(default),
        conf_fallback: None,
        metadata_key: false,
        doc,
    }
}

/// The options as one list (a brace-delimited macro: `rustfmt` leaves the
/// table one option per line).
macro_rules! options {
    ($($spec:expr;)*) => { vec![$($spec),*] };
}

static SPECS: LazyLock<Vec<OptionSpec>> = LazyLock::new(|| {
    options! {
        flag("wholetext", false, "Each file is a single row holding its whole content.");
        string("lineSep", "The line separator (read: \\r, \\r\\n, \\n; write: \\n).");
        string("encoding", "The charset: UTF-8, ISO-8859-1 or US-ASCII.");
        string("charset", "Alias of `encoding`.");
        string("compression", "Write codec: only none and uncompressed so far.");
        string("codec", "Alias of `compression`.");
    }
});

pub(super) static OPTIONS: LazyLock<OptionSchema> = LazyLock::new(|| OptionSchema {
    format: "text",
    options: SPECS.as_slice(),
});

/// The parsed options of a read or write.
#[derive(Debug, Clone)]
pub(crate) struct TextOptions {
    pub wholetext: bool,
    pub charset: Charset,
    pub line_sep: LineSep,
}

impl TextOptions {
    /// The options of a read (`writing` false) or a write.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_FORMAT_OPTION_VALUE` / `KRISHIV_UNSUPPORTED_FEATURE`
    /// for a value that cannot be honoured.
    pub(crate) fn new(o: &FormatOptions, writing: bool) -> Result<Self> {
        if writing {
            check_write_codec(o)?;
        }
        let charset = Charset::parse(o.str("encoding").or_else(|| o.str("charset")))?;
        let line_sep = LineSep::parse(o.str("lineSep"), charset, "text")?;
        Ok(Self {
            wholetext: o.bool("wholetext").unwrap_or(false),
            charset,
            line_sep,
        })
    }
}
