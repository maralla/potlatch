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

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const MAGENTA: &str = "\x1b[35m";
const BLUE: &str = "\x1b[34m";

const BADGE_WIDTH: usize = 11;
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
    label: Mutex<Option<String>>,
}

pub struct ActivityGuard {
    state: Option<Arc<SpinnerState>>,
}

impl Drop for ActivityGuard {
    fn drop(&mut self) {
        if let Some(state) = &self.state
            && state.active.fetch_sub(1, Ordering::SeqCst) == 1
            && let Ok(mut label) = state.label.lock()
        {
            *label = None;
        }
    }
}

/// Show the activity spinner until the returned guard is dropped.
pub fn activity(label: impl Into<String>) -> ActivityGuard {
    let Some(state) = SPINNER.get().cloned() else {
        return ActivityGuard { state: None };
    };
    if !state.enabled {
        return ActivityGuard { state: None };
    }

    if let Ok(mut current) = state.label.lock() {
        *current = Some(label.into());
    }
    state.active.fetch_add(1, Ordering::SeqCst);
    ActivityGuard { state: Some(state) }
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
        );

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
                label: Mutex::new(None),
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
                    .label
                    .lock()
                    .ok()
                    .and_then(|label| label.clone())
                    .unwrap_or_else(|| "agents".to_string());
                let text = if active == 1 {
                    format!("{label} working")
                } else {
                    format!("{active} agents working")
                };
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
pub fn print_banner(config_path: &str, gitlab_repo: Option<&str>, agents: &[String]) {
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

    let _ = write!(out, "  ");
    let _ = label_value(&mut out, use_color, "config", config_path);
    if let Some(repo) = gitlab_repo {
        let _ = write!(out, "  ");
        let _ = label_value(&mut out, use_color, "repo", repo);
    }
    let agents_line = if agents.is_empty() {
        "(none configured)".to_string()
    } else {
        agents.join(", ")
    };
    let _ = write!(out, "  ");
    let _ = label_value(&mut out, use_color, "agents", &agents_line);
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

        if should_suppress(target, level, &message) {
            return Ok(());
        }

        let _terminal = OUTPUT_LOCK.lock().ok();
        if let Some(state) = SPINNER.get()
            && state.enabled
            && state.visible.swap(false, Ordering::SeqCst)
        {
            write!(writer, "{SPINNER_CLEAR}")?;
        }

        let (prefix_agent, text) = split_agent_prefix(&message);
        let context_agent = agent_badge_from_context();
        let agent = prefix_agent
            .or_else(|| extract_embedded_agent(&message))
            .or(context_agent.as_deref());
        let badge = agent
            .map(agent_badge_label)
            .unwrap_or_else(|| badge_from_target(target));
        let icon = level_icon(level);
        let badge_color = agent_badge_color(agent, target, self.use_color);
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

fn extract_embedded_agent(message: &str) -> Option<&str> {
    message.split_whitespace().find(|word| is_agent_id(word))
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
    let Some((role, n)) = part.split_once('-') else {
        return false;
    };
    matches!(role, "worker" | "reviewer" | "pmo" | "ops") && n.parse::<u32>().is_ok()
}

fn split_agent_prefix(message: &str) -> (Option<&str>, &str) {
    if let Some((head, tail)) = message.split_once(": ")
        && is_agent_id(head)
    {
        return (Some(head), tail);
    }
    (None, message)
}

fn agent_badge_label(agent: &str) -> String {
    agent.to_string()
}

fn badge_from_target(target: &str) -> String {
    if target.contains("::agents::gitlab") {
        "gitlab".to_string()
    } else if target.contains("::agents::") {
        target.rsplit("::").next().unwrap_or("agent").to_string()
    } else if target.contains("::acp") {
        "acp".to_string()
    } else if target.contains("::config") {
        "config".to_string()
    } else {
        "potlatch".to_string()
    }
}

fn agent_badge_color(agent: Option<&str>, target: &str, use_color: bool) -> &'static str {
    if !use_color {
        return "";
    }
    let role = agent
        .and_then(|a| a.split_once('-').map(|(r, _)| r))
        .or_else(|| {
            if target.contains("::worker") {
                Some("worker")
            } else if target.contains("::reviewer") {
                Some("reviewer")
            } else if target.contains("::pmo") {
                Some("pmo")
            } else {
                None
            }
        });
    match role {
        Some("worker") => CYAN,
        Some("reviewer") => MAGENTA,
        Some("pmo") => YELLOW,
        Some("ops") => BLUE,
        _ if target.contains("::gitlab") => GREEN,
        _ if target.contains("::acp") => DIM,
        _ => BOLD,
    }
}

fn is_idle_status(text: &str) -> bool {
    text.contains("0 MRs merged")
        || text.contains("No action-required issues found")
        || text.contains("Idle, no issues to work on")
        || text.contains("Checking for issues requiring action")
        || text.contains("Polling for new issues")
        || text.contains("Poll interval:")
        || text.contains("Watching MR !")
        || text.contains("GitLab client for")
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
    let text = text.replace(['\r', '\n'], " ");
    let Some(width) = terminal_width() else {
        return text;
    };
    let available = width.saturating_sub(prefix_width);
    truncate_to_width(&text, available)
}

fn truncate_to_width(text: &str, width: usize) -> String {
    let text = text.replace(['\r', '\n'], " ");
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

fn should_suppress(target: &str, level: Level, message: &str) -> bool {
    if level == Level::INFO && message.starts_with("Loading config from:") {
        return true;
    }
    if level == Level::INFO && message.starts_with("Starting configured agents:") {
        return true;
    }
    let (_, text) = split_agent_prefix(message);
    if level == Level::INFO && is_idle_status(text) {
        return true;
    }
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
    fn split_agent_prefix_extracts_agent_id() {
        let (agent, text) = split_agent_prefix("worker-0: Polling for new issues...");
        assert_eq!(agent, Some("worker-0"));
        assert_eq!(text, "Polling for new issues...");
    }

    #[test]
    fn split_agent_prefix_leaves_unprefixed_messages() {
        let (agent, text) = split_agent_prefix("GitLab client ready");
        assert_eq!(agent, None);
        assert_eq!(text, "GitLab client ready");
    }

    #[test]
    fn is_agent_id_recognizes_roles() {
        assert!(is_agent_id("worker-0"));
        assert!(is_agent_id("reviewer-2"));
        assert!(!is_agent_id("worker"));
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
    fn idle_status_matches_heartbeat_messages() {
        assert!(is_idle_status("0 MRs merged"));
        assert!(!is_idle_status("Created MR !12 for issue #3"));
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
}
