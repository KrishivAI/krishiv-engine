// SPDX-License-Identifier: Apache-2.0
//! The sandbox's one error and its error class.

use krishiv_common::error::{
    ErrorClass, InternalError, KrishivError, KrishivUdfSandboxUnavailable,
};

/// Why a UDF worker could not be sandboxed.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    /// The kernel lacks a layer the sandbox needs (Landlock, seccomp, or the Landlock ABI a network allowlist
    /// needs), or applying a layer failed (`KRISHIV_UDF_SANDBOX_UNAVAILABLE`, SQLSTATE 0A000). The server
    /// refuses server-mode UDFs with this error unless an administrator sets `krishiv.udf.sandbox=off`.
    #[error("{}", KrishivError::from(self))]
    Unavailable {
        /// What is missing or what failed, in words.
        reason: String,
    },
    /// The trampoline was started with arguments that do not decode to a [`crate::SandboxSpec`]: a caller bug,
    /// since only the server builds them (`INTERNAL_ERROR`, SQLSTATE XX000).
    #[error("{}", KrishivError::from(self))]
    BadSpec(String),
}

impl SandboxError {
    /// The error class this error is reported with.
    pub fn error_class(&self) -> ErrorClass {
        match self {
            SandboxError::Unavailable { .. } => ErrorClass::KrishivUdfSandboxUnavailable,
            SandboxError::BadSpec(_) => ErrorClass::InternalError,
        }
    }

    pub(crate) fn unavailable(reason: impl Into<String>) -> Self {
        SandboxError::Unavailable {
            reason: reason.into(),
        }
    }
}

impl From<&SandboxError> for KrishivError {
    fn from(e: &SandboxError) -> Self {
        match e {
            SandboxError::Unavailable { reason } => KrishivUdfSandboxUnavailable {
                reason: reason.clone(),
            }
            .into(),
            SandboxError::BadSpec(message) => InternalError {
                message: format!("UDF sandbox trampoline: {message}"),
            }
            .into(),
        }
    }
}

impl From<SandboxError> for KrishivError {
    fn from(e: SandboxError) -> Self {
        KrishivError::from(&e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_carry_their_class_by_value_and_by_reference() {
        let unavailable = SandboxError::unavailable("no Landlock");
        assert_eq!(
            unavailable.error_class(),
            ErrorClass::KrishivUdfSandboxUnavailable
        );
        let by_ref = KrishivError::from(&unavailable);
        let by_value = KrishivError::from(unavailable);
        assert_eq!(by_ref, by_value);
        assert!(by_value.to_string().contains("no Landlock"));
        let bad = KrishivError::from(SandboxError::BadSpec("unknown flag".to_owned()));
        assert!(
            bad.to_string()
                .contains("UDF sandbox trampoline: unknown flag"),
            "{bad}"
        );
    }
}
