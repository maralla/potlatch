//! Terminal-friendly logging: compact, colored lines instead of server-style traces.

use std::fmt;
use std::io::{self, IsTerminal, Write};

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

/// Install the Potlatch terminal log formatter.
pub fn init() {
    let use_color = io::stdout().is_terminal();

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
        .event_format(TuiFormatter { use_color })
        .init();
}

/// Startup summary shown once before agents begin polling.
pub fn print_banner(config_path: &str, gitlab_repo: Option<&str>, agents: &[String]) {
    let use_color = io::stdout().is_terminal();
    let mut out = io::stdout().lock();

    let _ = writeln!(out);
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
    if use_color {
        let _ = write!(
            out,
            "  {DIM}tip{RRESET}    RUST_LOG=debug for full diagnostics\n\n",
            RRESET = RESET
        );
    } else {
        let _ = writeln!(out, "  tip     RUST_LOG=debug for full diagnostics\n");
    }
}

fn label_value(out: &mut impl Write, color: bool, key: &str, value: &str) -> io::Result<()> {
    if color {
        write!(
            out,
            "{DIM}{key:<7}{RESET} {value}\n",
            DIM = DIM,
            RESET = RESET,
            key = key,
            value = value
        )
    } else {
        write!(out, "{key:<7} {value}\n", key = key, value = value)
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

        let (agent, text) = split_agent_prefix(&message);
        let agent = agent.or_else(|| extract_embedded_agent(&message));
        let badge = agent
            .map(agent_badge_label)
            .unwrap_or_else(|| badge_from_target(target));
        let icon = level_icon(level);
        let badge_color = agent_badge_color(agent, target, self.use_color);
        let idle = is_idle_status(text);

        write!(writer, "  {icon} ")?;
        if self.use_color && !badge_color.is_empty() {
            write!(writer, "{badge_color}")?;
        }
        write!(writer, "{badge:<BADGE_WIDTH$}")?;
        if self.use_color {
            write!(writer, "{RESET}")?;
        }
        write!(writer, "  ")?;

        if idle && self.use_color {
            write!(writer, "{DIM}{text}{RESET}", DIM = DIM, RESET = RESET)?;
        } else if level == Level::ERROR && self.use_color {
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
    if let Some((head, tail)) = message.split_once(": ") {
        if is_agent_id(head) {
            return (Some(head), tail);
        }
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
}

fn should_suppress(target: &str, level: Level, message: &str) -> bool {
    if level == Level::INFO && message.starts_with("Loading config from:") {
        return true;
    }
    if level == Level::INFO && message.starts_with("Starting configured agents:") {
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
    fn idle_status_matches_heartbeat_messages() {
        assert!(is_idle_status("0 MRs merged"));
        assert!(!is_idle_status("Created MR !12 for issue #3"));
    }
}
