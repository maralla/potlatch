//! Generic ACP capability model.
//!
//! Vendors (e.g. Cursor CLI) provide protocol extensions beyond the standard
//! ACP spec. Agents implement [`CapabilityProvider`] to provide the
//! capabilities they support; vendor extensions convert between the neutral
//! types defined here and their vendor-specific wire format. Agents never
//! reference vendor-specific names or JSON shapes.

/// One choice in an ask question.
#[derive(Debug, Clone)]
pub struct AskChoice {
    /// Vendor option id (e.g. "guide"); empty when the vendor supplies no id.
    pub id: String,
    /// Human-readable label for the choice.
    pub label: String,
}

/// A vendor-neutral ask question, parsed from the vendor request by the
/// extension. Agents read this; they never see the raw vendor JSON.
#[derive(Debug, Clone, Default)]
pub struct AskQuestion {
    /// Question prompt text.
    pub text: String,
    /// Ordered choices (may be empty).
    pub choices: Vec<AskChoice>,
}

/// A vendor-neutral answer. The vendor extension converts this to its wire
/// format.
#[derive(Debug, Clone)]
pub enum AskAnswer {
    /// Select the choice with this id.
    Choice(String),
    /// Free-text answer the vendor maps onto its protocol.
    FreeText(String),
    /// Let the vendor pick its default (e.g. the first option).
    Auto,
}

/// Single capability trait. Agents implement this to provide the capabilities
/// they support; vendor extensions read from it during session creation.
/// Default implementations return the no-op answer for capabilities the agent
/// doesn't implement.
pub trait CapabilityProvider: Send + Sync {
    /// Answer an ask question. Default: [`AskAnswer::Auto`] (the vendor picks
    /// its default, e.g. the first option).
    fn ask(&self, _question: &AskQuestion) -> AskAnswer {
        AskAnswer::Auto
    }
}
