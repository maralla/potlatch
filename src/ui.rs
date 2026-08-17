//! Terminal-friendly logging: compact, colored lines instead of server-style traces.

use std::cell::RefCell;
use std::fmt;
use std::io::{self, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use tracing::Level;
use tracing::field::{Field, Visit};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::{FormatEvent, Writer};
use tracing_subscriber::fmt::{FmtContext, FormatFields};
use tracing_subscriber::registry::LookupSpan;

use crate::core::banner::Banner;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";

const BADGE_WIDTH: usize = 11;
const SYSTEM_BADGE: &str = "potlatch";
const SPINNER_CLEAR: &str = "\r\x1b[2K\r";
const LOG_PREFIX_WIDTH: usize = 18;
const SPINNER_PREFIX_WIDTH: usize = 4;

static OUTPUT_LOCK: Mutex<()> = Mutex::new(());
static SPINNER: OnceLock<Arc<SpinnerState>> = OnceLock::new();

thread_local! {
    static AGENT_BADGE: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Binds the current thread's log badge to a specific agent instance (e.g. `worker-0`).
pub struct AgentBadgeGuard {
    previous: Option<String>,
}

impl AgentBadgeGuard {
    pub fn new(agent_id: &str) -> Self {
        let previous = AGENT_BADGE.with(|badge| badge.replace(Some(agent_id.to_string())));
        Self { previous }
    }
}

impl Drop for AgentBadgeGuard {
    fn drop(&mut self) {
        AGENT_BADGE.with(|badge| {
            *badge.borrow_mut() = self.previous.take();
        });
    }
}

fn agent_badge_from_context() -> Option<String> {
    AGENT_BADGE.with(|badge| badge.borrow().clone())
}

struct SpinnerState {
    active: AtomicUsize,
    running: AtomicBool,
    visible: AtomicBool,
    enabled: bool,
    /// Active labels as `(slot_id, label)` pairs. Each `activity()` call
    /// registers an entry; the `ActivityGuard` removes it on drop. When only
    /// one entry remains, the spinner shows its label directly.
    labels: Mutex<Vec<(usize, String)>>,
    next_slot: AtomicUsize,
}

pub struct ActivityGuard {
    state: Option<Arc<SpinnerState>>,
    slot: usize,
}

impl Drop for ActivityGuard {
    fn drop(&mut self) {
        if let Some(state) = &self.state {
            state.active.fetch_sub(1, Ordering::SeqCst);
            if let Ok(mut labels) = state.labels.lock() {
                labels.retain(|(id, _)| *id != self.slot);
            }
        }
    }
}

/// Show the activity spinner until the returned guard is dropped.
pub fn activity(label: impl Into<String>) -> ActivityGuard {
    let Some(state) = SPINNER.get().cloned() else {
        return ActivityGuard {
            state: None,
            slot: 0,
        };
    };
    if !state.enabled {
        return ActivityGuard {
            state: None,
            slot: 0,
        };
    }

    let slot = state.next_slot.fetch_add(1, Ordering::SeqCst);
    if let Ok(mut labels) = state.labels.lock() {
        labels.push((slot, label.into()));
    }
    state.active.fetch_add(1, Ordering::SeqCst);
    ActivityGuard {
        state: Some(state),
        slot,
    }
}

/// Terminal-backed activity reporter used by the CLI workflow.
#[derive(Default)]
pub struct UiActivityReporter;

impl crate::core::activity::ActivityReporter for UiActivityReporter {
    fn start(&self, label: String) -> Box<dyn crate::core::activity::ActivityToken> {
        Box::new(activity(label))
    }
}

/// Install the Potlatch terminal log formatter.
pub fn init() {
    let use_color = io::stdout().is_terminal();
    init_spinner(use_color);

    let filter = EnvFilter::builder()
        .with_default_directive(Level::INFO.into())
        .from_env_lossy()
        .add_directive("potlatch::acp_fs=warn".parse().expect("valid directive"))
        .add_directive("potlatch::acp_modes=warn".parse().expect("valid directive"))
        .add_directive("potlatch::acp_slash=warn".parse().expect("valid directive"))
        .add_directive(
            "potlatch::agent_stderr=warn"
                .parse()
                .expect("valid directive"),
        )
        .add_directive("hyper_util=warn".parse().expect("valid directive"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time()
        .with_level(false)
        .with_ansi(use_color)
        .with_writer(io::stdout)
        .event_format(TuiFormatter { use_color })
        .init();
}

fn init_spinner(enabled: bool) {
    let state = SPINNER
        .get_or_init(|| {
            Arc::new(SpinnerState {
                active: AtomicUsize::new(0),
                running: AtomicBool::new(false),
                visible: AtomicBool::new(false),
                enabled,
                labels: Mutex::new(Vec::new()),
                next_slot: AtomicUsize::new(0),
            })
        })
        .clone();

    if !enabled || state.running.swap(true, Ordering::SeqCst) {
        return;
    }

    thread::spawn(move || {
        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let mut idx = 0usize;
        let mut visible = false;
        loop {
            let active = state.active.load(Ordering::SeqCst);
            let _terminal = OUTPUT_LOCK.lock().ok();
            let mut out = io::stdout().lock();
            if active > 0 {
                let label = state
                    .labels
                    .lock()
                    .ok()
                    .and_then(|labels| {
                        if labels.len() == 1 {
                            Some(labels[0].1.clone())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| "agents".to_string());
                let text = activity_text(active, &label);
                let text = truncate_to_terminal_width(&text, SPINNER_PREFIX_WIDTH);
                let _ = write!(out, "{SPINNER_CLEAR}  {} {}{}", frames[idx], DIM, text);
                let _ = write!(out, "{RESET}");
                let _ = out.flush();
                visible = true;
                state.visible.store(true, Ordering::SeqCst);
                idx = (idx + 1) % frames.len();
            } else if visible {
                let _ = write!(out, "{SPINNER_CLEAR}");
                let _ = out.flush();
                visible = false;
                state.visible.store(false, Ordering::SeqCst);
            }
            drop(out);
            drop(_terminal);
            thread::sleep(Duration::from_millis(120));
        }
    });
}

/// Startup banner shown once before agents begin polling.
pub fn print_banner(banner: &Banner) {
    let use_color = io::stdout().is_terminal();
    let _terminal = OUTPUT_LOCK.lock().ok();
    let mut out = io::stdout().lock();

    let _ = writeln!(out);
    if use_color {
        let _ = writeln!(
            out,
            "{BOLD}{CYAN}  Potlatch{RESET} {DIM}Fully automatic agentic platform{RESET}"
        );
    } else {
        let _ = writeln!(out, "  Potlatch  Fully automatic agentic platform");
    }

    for field in banner.fields() {
        let _ = write!(out, "  ");
        let _ = label_value(&mut out, use_color, &field.key, &field.value);
    }
    let _ = writeln!(out);
}

fn label_value(out: &mut impl Write, color: bool, key: &str, value: &str) -> io::Result<()> {
    if color {
        writeln!(
            out,
            "{DIM}{key:<7}{RESET} {value}",
            DIM = DIM,
            RESET = RESET,
            key = key,
            value = value
        )
    } else {
        writeln!(out, "{key:<7} {value}", key = key, value = value)
    }
}

struct TuiFormatter {
    use_color: bool,
}

struct EventMessage(String);

impl Visit for EventMessage {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            self.0 = strip_debug_quotes(&format!("{value:?}"));
        }
    }
}

impl<S, N> FormatEvent<S, N> for TuiFormatter
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        _ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> fmt::Result {
        let level = *event.metadata().level();
        let target = event.metadata().target();

        let mut msg = EventMessage(String::new());
        event.record(&mut msg);
        let message = msg.0;
        if message.is_empty() {
            return Ok(());
        }

        if should_suppress(target, level) {
            return Ok(());
        }

        let _terminal = OUTPUT_LOCK.lock().ok();
        if let Some(state) = SPINNER.get()
            && state.enabled
            && state.visible.swap(false, Ordering::SeqCst)
        {
            write!(writer, "{SPINNER_CLEAR}")?;
        }

        let (prefix_badge, text) = split_badge_prefix(&message);
        let context_agent = agent_badge_from_context();
        let badge_source = prefix_badge
            .or_else(|| extract_embedded_badge(&message))
            .or(context_agent.as_deref());
        let badge = badge_for_source(badge_source);
        let icon = level_icon(level);
        let badge_color = badge_color(badge_source, self.use_color);
        let text = truncate_to_terminal_width(text, LOG_PREFIX_WIDTH);

        write!(writer, "  {icon} ")?;
        if self.use_color && !badge_color.is_empty() {
            write!(writer, "{badge_color}")?;
        }
        write!(writer, "{badge:<BADGE_WIDTH$}")?;
        if self.use_color {
            write!(writer, "{RESET}")?;
        }
        write!(writer, "  ")?;

        if level == Level::ERROR && self.use_color {
            write!(writer, "{RED}{text}{RESET}", RED = RED, RESET = RESET)?;
        } else if level == Level::WARN && self.use_color {
            write!(
                writer,
                "{YELLOW}{text}{RESET}",
                YELLOW = YELLOW,
                RESET = RESET
            )?;
        } else {
            write!(writer, "{text}")?;
        }

        writeln!(writer)
    }
}

fn extract_embedded_badge(message: &str) -> Option<&str> {
    message.split_whitespace().find(|word| is_badge_id(word))
}

fn level_icon(level: Level) -> &'static str {
    match level {
        Level::ERROR => "✗",
        Level::WARN => "⚠",
        Level::INFO => "›",
        Level::DEBUG => "·",
        Level::TRACE => "·",
    }
}

fn strip_debug_quotes(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        s[1..s.len() - 1].replace("\\n", "\n").replace("\\\"", "\"")
    } else {
        s.to_string()
    }
}

fn is_agent_id(part: &str) -> bool {
    let Some((role, _)) = split_badge_id(part) else {
        return false;
    };
    !is_non_agent_badge_role(role)
}

fn is_badge_id(part: &str) -> bool {
    split_badge_id(part).is_some()
}

fn split_badge_id(part: &str) -> Option<(&str, u32)> {
    let (role, n) = part.split_once('-')?;
    if role.is_empty()
        || !role
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    {
        return None;
    }
    let n = n.parse::<u32>().ok()?;
    Some((role, n))
}

fn is_non_agent_badge_role(role: &str) -> bool {
    role.eq_ignore_ascii_case("issue")
}

fn split_badge_prefix(message: &str) -> (Option<&str>, &str) {
    if let Some((head, tail)) = message.split_once(": ")
        && is_badge_id(head)
    {
        return (Some(head), tail);
    }
    (None, message)
}

fn badge_for_source(source: Option<&str>) -> String {
    source.unwrap_or(SYSTEM_BADGE).to_string()
}

fn badge_color(source: Option<&str>, use_color: bool) -> &'static str {
    if !use_color {
        return "";
    }
    if source.is_some_and(is_agent_id) {
        CYAN
    } else {
        DIM
    }
}

fn terminal_width() -> Option<usize> {
    terminal_size::terminal_size()
        .map(|(terminal_size::Width(width), _)| width as usize)
        .or_else(|| {
            std::env::var("COLUMNS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
        })
        .filter(|w| *w >= 40)
}

fn truncate_to_terminal_width(text: &str, prefix_width: usize) -> String {
    let text = compact_terminal_text(text);
    let Some(width) = terminal_width() else {
        return text;
    };
    let available = width.saturating_sub(prefix_width);
    truncate_to_width(&text, available)
}

fn activity_text(active: usize, label: &str) -> String {
    if active == 1 {
        label.to_string()
    } else {
        format!("{active} agents working")
    }
}

fn truncate_to_width(text: &str, width: usize) -> String {
    let text = compact_terminal_text(text);
    if display_width(&text) <= width {
        return text;
    }

    if width <= 1 {
        return "…".to_string();
    }

    let content_width = width - 1;
    let head_width = content_width / 2;
    let tail_width = content_width - head_width;

    let head = take_display_prefix(&text, head_width);
    let tail = take_display_suffix(&text, tail_width);
    format!("{head}…{tail}")
}

fn display_width(text: &str) -> usize {
    text.chars()
        .map(|ch| if ch.is_ascii() { 1 } else { 2 })
        .sum()
}

fn take_display_prefix(text: &str, width: usize) -> String {
    let mut out = String::new();
    let mut used = 0usize;
    for ch in text.chars() {
        let w = if ch.is_ascii() { 1 } else { 2 };
        if used + w > width {
            break;
        }
        out.push(ch);
        used += w;
    }
    out
}

fn take_display_suffix(text: &str, width: usize) -> String {
    let mut chars = Vec::new();
    let mut used = 0usize;
    for ch in text.chars().rev() {
        let w = if ch.is_ascii() { 1 } else { 2 };
        if used + w > width {
            break;
        }
        chars.push(ch);
        used += w;
    }
    chars.into_iter().rev().collect()
}

fn compact_terminal_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn should_suppress(target: &str, level: Level) -> bool {
    if level <= Level::INFO
        && (target.contains("::acp_fs")
            || target.contains("::acp_modes")
            || target.contains("::acp_slash")
            || target.contains("::agent_stderr"))
    {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_badge_prefix_extracts_agent_id() {
        let (badge, text) = split_badge_prefix("worker-0: Polling for new issues...");
        assert_eq!(badge, Some("worker-0"));
        assert_eq!(text, "Polling for new issues...");
    }

    #[test]
    fn split_badge_prefix_extracts_non_agent_id() {
        let (badge, text) = split_badge_prefix("issue-79: MR !101 diff");
        assert_eq!(badge, Some("issue-79"));
        assert_eq!(text, "MR !101 diff");
    }

    #[test]
    fn split_badge_prefix_leaves_unprefixed_messages() {
        let (badge, text) = split_badge_prefix("GitLab client ready");
        assert_eq!(badge, None);
        assert_eq!(text, "GitLab client ready");
    }

    #[test]
    fn is_agent_id_recognizes_roles() {
        assert!(is_agent_id("worker-0"));
        assert!(is_agent_id("reviewer-2"));
        assert!(is_agent_id("pmo-0"));
        assert!(is_agent_id("custom_agent-12"));
        assert!(!is_agent_id("issue-79"));
        assert!(!is_agent_id("worker"));
        assert!(!is_agent_id("worker-next"));
    }

    #[test]
    fn extract_embedded_badge_finds_non_agent_badge() {
        assert_eq!(
            extract_embedded_badge("MR !101 diff: issue-79 (1a8fd2e) -> main"),
            Some("issue-79")
        );
    }

    #[test]
    fn agent_badge_guard_sets_thread_context() {
        assert!(agent_badge_from_context().is_none());
        let _guard = AgentBadgeGuard::new("worker-0");
        assert_eq!(agent_badge_from_context().as_deref(), Some("worker-0"));
        drop(_guard);
        assert!(agent_badge_from_context().is_none());
    }

    #[test]
    fn badge_for_source_uses_source_or_system_badge() {
        assert_eq!(badge_for_source(Some("worker-0")), "worker-0");
        assert_eq!(badge_for_source(None), SYSTEM_BADGE);
    }

    #[test]
    fn badge_color_uses_system_color_for_non_agent_badges() {
        assert_eq!(badge_color(Some("worker-0"), true), CYAN);
        assert_eq!(badge_color(Some("issue-79"), true), DIM);
        assert_eq!(badge_color(None, true), DIM);
        assert_eq!(badge_color(Some("issue-79"), false), "");
    }

    #[test]
    fn truncate_to_width_prevents_wrapping() {
        assert_eq!(truncate_to_width("abcdef", 4), "a…ef");
        assert_eq!(truncate_to_width("abc", 4), "abc");
        assert_eq!(truncate_to_width("a\nb", 10), "a b");
        assert_eq!(
            truncate_to_width("/very/long/path/to/repository", 14),
            "/very/…ository"
        );
    }

    #[test]
    fn compact_terminal_text_collapses_cli_error_spacing() {
        assert_eq!(
            compact_terminal_text(
                "glab api issue list failed:               ERROR                Get \"https://example/api\": net/http: TLS      handshake timeout.\n"
            ),
            "glab api issue list failed: ERROR Get \"https://example/api\": net/http: TLS handshake timeout."
        );
        assert_eq!(
            compact_terminal_text(
                "Git fetch failed: Connection closed\nfatal: Could not read from remote repository.\n\nPlease make sure you have the correct access rights."
            ),
            "Git fetch failed: Connection closed fatal: Could not read from remote repository. Please make sure you have the correct access rights."
        );
    }

    #[test]
    fn activity_text_uses_single_agent_label_as_is() {
        assert_eq!(
            activity_text(1, "worker-0 addressing MR !83 feedback"),
            "worker-0 addressing MR !83 feedback"
        );
        assert_eq!(
            activity_text(2, "worker-0 implementing issue #1"),
            "2 agents working"
        );
    }

    #[test]
    fn activity_guard_shows_remaining_label_after_others_drop() {
        // Simulate: ops-0 starts, then worker-0 and worker-1 start, then both
        // workers finish. The spinner should show ops-0's label, not a stale
        // worker label.
        let state = Arc::new(SpinnerState {
            active: AtomicUsize::new(0),
            running: AtomicBool::new(false),
            visible: AtomicBool::new(false),
            enabled: true,
            labels: Mutex::new(Vec::new()),
            next_slot: AtomicUsize::new(0),
        });

        // Ops starts
        let _ops_guard = ActivityGuard {
            state: Some(Arc::clone(&state)),
            slot: 0,
        };
        state.active.fetch_add(1, Ordering::SeqCst);
        state
            .labels
            .lock()
            .unwrap()
            .push((0, "ops-0 analyzing logs".into()));

        // Worker-0 starts
        let w0_guard = ActivityGuard {
            state: Some(Arc::clone(&state)),
            slot: 1,
        };
        state.active.fetch_add(1, Ordering::SeqCst);
        state
            .labels
            .lock()
            .unwrap()
            .push((1, "worker-0 implementing issue #570".into()));

        // Worker-1 starts
        let w1_guard = ActivityGuard {
            state: Some(Arc::clone(&state)),
            slot: 2,
        };
        state.active.fetch_add(1, Ordering::SeqCst);
        state
            .labels
            .lock()
            .unwrap()
            .push((2, "worker-1 implementing issue #572".into()));

        // Three active — generic label
        assert_eq!(state.active.load(Ordering::SeqCst), 3);
        let labels = state.labels.lock().unwrap();
        assert_eq!(labels.len(), 3);
        drop(labels);

        // Workers finish (drop their guards)
        drop(w0_guard);
        drop(w1_guard);

        // Only ops remains — its label should be the one shown
        assert_eq!(state.active.load(Ordering::SeqCst), 1);
        let labels = state.labels.lock().unwrap();
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].1, "ops-0 analyzing logs");
    }
}
