// SPDX-License-Identifier: Apache-2.0
//! The read options of a Spark release before 4.3, which has no nanosecond timestamp types
//! (`FileReads::with_nanos_types_known`).

use std::borrow::Cow;
use std::collections::BTreeMap;

use krishiv_common::secret::OptionValue;
use krishiv_io::OptionSchema;

use super::FileReads;

/// The two Parquet options the pin sets.
const PINNED: [&str; 2] = ["timestampNanosTypes", "nanosTypesKnown"];

impl FileReads {
    /// Reads as a Spark release before 4.3 when `known` is false: the nanosecond timestamp types
    /// are never inferred and a Parquet footer schema naming one is not used, whatever
    /// `spark.sql.timestampNanosTypes.enabled` says (that release has no such setting).
    #[must_use]
    pub fn with_nanos_types_known(mut self, known: bool) -> Self {
        self.nanos_types_known = known;
        self
    }

    /// `options` with the pin applied: only a format that declares the options (Parquet) takes it,
    /// since any other would warn about, or under strict options refuse, names the user never set.
    pub(super) fn nanos_pin<'a>(
        &self,
        schema: &OptionSchema,
        options: &'a BTreeMap<String, OptionValue>,
    ) -> Cow<'a, BTreeMap<String, OptionValue>> {
        if self.nanos_types_known || schema.spec(PINNED[1]).is_none() {
            return Cow::Borrowed(options);
        }
        let mut o = options.clone();
        o.retain(|k, _| !PINNED.iter().any(|p| k.eq_ignore_ascii_case(p)));
        for name in PINNED {
            o.insert(
                name.to_owned(),
                OptionValue::from_user(name, "false".to_owned()),
            );
        }
        Cow::Owned(o)
    }
}
