// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostics emitted when translation preserves, drops, or degrades data.

use serde::{Deserialize, Serialize};

use crate::format::FormatId;

/// Severity level for translation diagnostics.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

/// Structured diagnostic attached to translation output.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TranslationDiagnostic {
    pub severity: DiagnosticSeverity,
    pub code: String,
    pub message: String,
    pub source: Option<FormatId>,
    pub target: Option<FormatId>,
    pub path: Option<String>,
}

impl TranslationDiagnostic {
    /// Creates a warning diagnostic.
    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: DiagnosticSeverity::Warning,
            code: code.into(),
            message: message.into(),
            source: None,
            target: None,
            path: None,
        }
    }

    /// Adds a JSON-path-like location to a diagnostic.
    pub fn at_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Adds source and target format metadata to a diagnostic.
    pub fn with_formats(
        mut self,
        source: impl Into<FormatId>,
        target: impl Into<FormatId>,
    ) -> Self {
        self.source = Some(source.into());
        self.target = Some(target.into());
        self
    }
}
