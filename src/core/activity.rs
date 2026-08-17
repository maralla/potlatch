use std::sync::Arc;

/// A drop guard returned by an [`ActivityReporter`].
pub trait ActivityToken: Send {}

impl<T: Send> ActivityToken for T {}

/// Reports long-running work without coupling agents or model code to a UI.
pub trait ActivityReporter: Send + Sync {
    fn start(&self, label: String) -> Box<dyn ActivityToken>;
}

/// Activity reporter used when no UI integration is installed.
#[derive(Default)]
pub struct NoopActivityReporter;

impl ActivityReporter for NoopActivityReporter {
    fn start(&self, _label: String) -> Box<dyn ActivityToken> {
        Box::new(())
    }
}

/// Shared activity reporter injected into workflow and agent runtimes.
pub type SharedActivityReporter = Arc<dyn ActivityReporter>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_reporter_returns_a_drop_token() {
        let reporter = NoopActivityReporter;
        let token = reporter.start("test".into());
        drop(token);
    }
}
