//! Model-free web agent backed by one persistent browser instance.

mod browser;

use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;
use tracing::info;

use crate::core::agent::CoreAgent;
use crate::core::bus::{AgentInbox, AgentRequest, AgentToolDefinition};
use crate::core::config::{AgentSection, Config};
use crate::core::periodic::{JitterPolicy, PeriodicTaskSpec};
use crate::core::runtime::AgentRuntime;
use crate::core::workflow::AgentBuildContext;
use browser::{ChromeBrowser, GoogleVerificationRequired};

const REQUEST_TASK: &str = "requests";
const INBOX_WAIT: Duration = Duration::from_millis(200);
const DEFAULT_MAX_RESULTS: usize = 8;
const MAX_CONFIGURED_RESULTS: usize = 10;
const MAX_QUERY_CHARS: usize = 500;
const MAX_URL_CHARS: usize = 2_048;
const MAX_RENDERED_MARKDOWN_BYTES: usize = 200_000;
const WEB_SEARCH_OPERATION: &str = "google_search";
const WEB_FETCH_OPERATION: &str = "web_fetch";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebAgentSettings {
    #[serde(default = "default_max_results")]
    max_results: usize,
}

fn default_max_results() -> usize {
    DEFAULT_MAX_RESULTS
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WebSearchRequest {
    query: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WebFetchRequest {
    url: String,
}

trait WebBackend: Send {
    fn search(&mut self, query: &str, max_results: usize) -> Result<String>;
    fn fetch_rendered_markdown(&mut self, url: &str) -> Result<String>;
}

struct ChromeWebBackend {
    browser: Option<ChromeBrowser>,
}

pub struct WebAgent {
    runtime: AgentRuntime,
    inbox: AgentInbox,
    backend: Box<dyn WebBackend>,
    max_results: usize,
}

impl ChromeWebBackend {
    fn new() -> Self {
        Self { browser: None }
    }
}

impl ChromeWebBackend {
    /// Take the browser out for an enveloped operation, launching a fresh
    /// one when a previous timeout abandoned it.
    fn take_or_launch_browser(&mut self) -> Result<ChromeBrowser> {
        match self.browser.take() {
            Some(browser) => Ok(browser),
            None => ChromeBrowser::launch(),
        }
    }
}

impl WebBackend for ChromeWebBackend {
    fn search(&mut self, query: &str, max_results: usize) -> Result<String> {
        let browser = self.take_or_launch_browser()?;
        let query = query.to_string();
        let (browser, result) = run_browser_enveloped(browser, move |mut browser| {
            let result = search_google(&mut browser, &query, max_results);
            (browser, result)
        });
        if result.as_ref().is_err_and(should_relaunch_browser) {
            // A failed CDP operation can leave the tab or process unusable.
            // Relaunch lazily for the next independent request.
            self.browser = None;
        } else {
            self.browser = browser;
        }
        result
    }

    fn fetch_rendered_markdown(&mut self, url: &str) -> Result<String> {
        // Non-HTML content never needs the browser: raw file endpoints, JSON
        // APIs, and plain-text error pages (raw.githubusercontent returns
        // text/plain even for 404s) are fetched over plain HTTP. Rendering
        // them through Chrome was observed to wedge the browser and, with
        // the agent single-instance, time out every later web call.
        if let Some(body) = fetch_non_html_body(url)? {
            return Ok(body);
        }

        let browser = self.take_or_launch_browser()?;
        let url = url.to_string();
        let (browser, result) = run_browser_enveloped(browser, move |browser| {
            let result = browser.fetch_rendered_markdown(&url);
            (browser, result)
        });
        self.browser = browser;
        result
    }
}

/// Run one browser operation under a hard wall-clock envelope slightly under
/// the agent-bus timeout (45s). The browser is returned on completion so it
/// is reused across requests; on timeout it stays with the wedged thread and
/// the next operation relaunches a fresh one — the single web agent must
/// never stay wedged behind an operation the caller already gave up on.
fn run_browser_enveloped<T: Send + 'static>(
    browser: ChromeBrowser,
    operation: impl FnOnce(ChromeBrowser) -> (ChromeBrowser, Result<T>) + Send + 'static,
) -> (Option<ChromeBrowser>, Result<T>) {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = sender.send(operation(browser));
    });
    match receiver.recv_timeout(BROWSER_ENVELOPE) {
        Ok((browser, result)) => (Some(browser), result),
        Err(_) => (
            None,
            Err(anyhow::anyhow!(
                "browser operation exceeded {BROWSER_ENVELOPE:?}; the browser was abandoned and will relaunch"
            )),
        ),
    }
}

/// The hard wall-clock for one browser-backed web operation. Slightly under
/// the agent-bus envelope (45s) so a timeout surfaces as this error instead
/// of a bus timeout that leaves the single web agent wedged.
const BROWSER_ENVELOPE: Duration = Duration::from_secs(40);

/// Fetch `url` over plain HTTP and return the body when it is NOT an HTML
/// document (those go through the browser for rendering). `Ok(None)` means
/// the browser path should handle it.
fn fetch_non_html_body(url: &str) -> Result<Option<String>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(20))
        .user_agent(concat!(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 ",
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        ))
        .build()
        .context("build plain HTTP client for web fetch")?;
    let response = client
        .get(url)
        .send()
        .context("fetch {url} over plain HTTP")?;
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();
    if content_type.contains("text/html") || content_type.contains("application/xhtml") {
        return Ok(None);
    }
    let status = response.status();
    let body = response.text().context("read non-HTML response body")?;
    if !status.is_success() {
        anyhow::bail!("HTTP {status} fetching {url}: {}", first_lines(&body));
    }
    Ok(Some(body))
}

/// The first non-empty lines of a body, for an error message.
fn first_lines(body: &str) -> String {
    body.lines()
        .filter(|line| !line.trim().is_empty())
        .take(3)
        .collect::<Vec<_>>()
        .join(" | ")
        .chars()
        .take(300)
        .collect()
}

fn should_relaunch_browser(error: &anyhow::Error) -> bool {
    error.downcast_ref::<GoogleVerificationRequired>().is_none()
}

impl WebAgent {
    fn handle_request(&mut self, request: AgentRequest) {
        let result = handle_agent_request(
            self.backend.as_mut(),
            self.max_results,
            &request.operation,
            request.payload.clone(),
        );
        request.respond(result);
    }
}

impl CoreAgent for WebAgent {
    type Settings = WebAgentSettings;
    const FIXED_INSTANCES: Option<usize> = Some(1);

    fn name() -> &'static str {
        "web"
    }

    fn validate_settings(
        _config: &Config,
        _section: &AgentSection,
        settings: &Self::Settings,
    ) -> Result<()> {
        ensure!(
            (1..=MAX_CONFIGURED_RESULTS).contains(&settings.max_results),
            "max_results must be between 1 and {MAX_CONFIGURED_RESULTS}"
        );
        Ok(())
    }

    fn runtime(&self) -> &AgentRuntime {
        &self.runtime
    }

    fn periodic_tasks(&self) -> Vec<PeriodicTaskSpec> {
        vec![PeriodicTaskSpec {
            id: REQUEST_TASK,
            interval: Duration::ZERO,
            jitter: JitterPolicy::BeforeEachCycle,
            jitter_max_ms: 0,
            autostart: true,
        }]
    }

    fn run_periodic_task(&mut self, task_id: &str) -> Result<()> {
        ensure!(task_id == REQUEST_TASK, "unknown web task {task_id:?}");
        if let Some(request) = self.inbox.recv_timeout(INBOX_WAIT)? {
            self.handle_request(request);
        }
        Ok(())
    }

    fn build(ctx: AgentBuildContext<Self::Settings>) -> Result<Self> {
        let bus = ctx
            .workflow
            .bus
            .as_ref()
            .context("web agent requires the cross-agent bus")?;
        let inbox = bus.register(
            Self::name(),
            vec![web_search_tool_definition(), web_fetch_tool_definition()],
        )?;
        Ok(Self {
            runtime: ctx.runtime.clone(),
            inbox,
            backend: Box::new(ChromeWebBackend::new()),
            max_results: ctx.settings.max_results,
        })
    }

    fn on_start(&mut self) -> Result<()> {
        info!(
            "{}: Web agent ready; registered tools `web_search` and `web_fetch`",
            self.agent_id()
        );
        Ok(())
    }

    fn on_shutdown(&mut self) {}
}

fn handle_agent_request(
    backend: &mut dyn WebBackend,
    max_results: usize,
    operation: &str,
    payload: Value,
) -> Result<Value> {
    match operation {
        WEB_SEARCH_OPERATION => handle_web_search_request(backend, max_results, payload),
        WEB_FETCH_OPERATION => handle_web_fetch_request(backend, payload),
        _ => bail!("unsupported web operation {operation:?}"),
    }
}

fn handle_web_search_request(
    backend: &mut dyn WebBackend,
    max_results: usize,
    payload: Value,
) -> Result<Value> {
    let request: WebSearchRequest =
        serde_json::from_value(payload).context("invalid Google Search request")?;
    let query = request.query.trim();
    ensure!(!query.is_empty(), "search query must not be empty");
    ensure!(
        query.chars().count() <= MAX_QUERY_CHARS,
        "search query exceeds {MAX_QUERY_CHARS} characters"
    );
    Ok(Value::String(backend.search(query, max_results)?))
}

fn handle_web_fetch_request(backend: &mut dyn WebBackend, payload: Value) -> Result<Value> {
    let request: WebFetchRequest =
        serde_json::from_value(payload).context("invalid rendered web fetch request")?;
    let url = validate_web_url(&request.url)?;
    let markdown = backend.fetch_rendered_markdown(&url)?;
    Ok(Value::String(truncate_rendered_markdown(markdown)))
}

fn validate_web_url(raw: &str) -> Result<String> {
    let raw = raw.trim();
    ensure!(!raw.is_empty(), "web fetch URL must not be empty");
    ensure!(
        raw.chars().count() <= MAX_URL_CHARS,
        "web fetch URL exceeds {MAX_URL_CHARS} characters"
    );
    let url = reqwest::Url::parse(raw).context("invalid web fetch URL")?;
    ensure!(
        matches!(url.scheme(), "http" | "https"),
        "web fetch URL must use http or https"
    );
    ensure!(url.host().is_some(), "web fetch URL must include a host");
    ensure!(
        url.username().is_empty() && url.password().is_none(),
        "web fetch URL must not contain credentials"
    );
    Ok(url.to_string())
}

fn truncate_rendered_markdown(mut markdown: String) -> String {
    if markdown.len() <= MAX_RENDERED_MARKDOWN_BYTES {
        return markdown;
    }
    let mut end = MAX_RENDERED_MARKDOWN_BYTES;
    while !markdown.is_char_boundary(end) {
        end -= 1;
    }
    markdown.truncate(end);
    markdown
}

fn web_search_tool_definition() -> AgentToolDefinition {
    AgentToolDefinition {
        name: "web_search".to_string(),
        description: "Search Google through the dedicated browser web agent and return Defuddle Markdown extracted directly from the rendered search results page. Use it when current public web information is needed.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The Google Search query.",
                    "minLength": 1,
                    "maxLength": MAX_QUERY_CHARS
                }
            },
            "required": ["query"],
            "additionalProperties": false
        }),
        operation: WEB_SEARCH_OPERATION.to_string(),
    }
}

fn web_fetch_tool_definition() -> AgentToolDefinition {
    AgentToolDefinition {
        name: "web_fetch".to_string(),
        description: "Preferred tool for opening web pages and URLs returned by `web_search` whenever rendered or JavaScript-generated content may be needed. It opens the HTTP(S) address in the persistent system browser and returns only the extracted Markdown content, without a JSON wrapper or metadata.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The absolute HTTP(S) URL to render.",
                    "minLength": 1,
                    "maxLength": MAX_URL_CHARS
                }
            },
            "required": ["url"],
            "additionalProperties": false
        }),
        operation: WEB_FETCH_OPERATION.to_string(),
    }
}

fn search_google(browser: &mut ChromeBrowser, query: &str, max_results: usize) -> Result<String> {
    browser.submit_google_query(query, max_results)?;

    if browser
        .google_verification_required()
        .context("check Google verification after submitting query")?
    {
        return Err(GoogleVerificationRequired.into());
    }
    browser.wait_for_google_results()?;
    let markdown = browser
        .extract_rendered_markdown()
        .context("extract Google search results page with Defuddle")?;
    ensure!(
        !markdown.trim().is_empty(),
        "Defuddle returned no Google search result content"
    );
    Ok(markdown)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    struct RecordingBackend {
        queries: Vec<(String, usize)>,
        search_markdown: Option<String>,
        fetched_urls: Vec<String>,
        rendered_markdown: Option<String>,
    }

    impl WebBackend for RecordingBackend {
        fn search(&mut self, query: &str, max_results: usize) -> Result<String> {
            self.queries.push((query.to_string(), max_results));
            self.search_markdown
                .take()
                .context("test backend has no search Markdown")
        }

        fn fetch_rendered_markdown(&mut self, url: &str) -> Result<String> {
            self.fetched_urls.push(url.to_string());
            self.rendered_markdown
                .take()
                .context("test backend has no rendered Markdown")
        }
    }

    #[test]
    fn request_is_validated_and_forwarded_without_a_model() {
        let mut backend = RecordingBackend {
            queries: Vec::new(),
            search_markdown: Some(
                "Rust\n====\n\n[Rust Programming Language](https://www.rust-lang.org/)".to_string(),
            ),
            fetched_urls: Vec::new(),
            rendered_markdown: None,
        };
        let response = handle_agent_request(
            &mut backend,
            7,
            WEB_SEARCH_OPERATION,
            json!({"query": "  rust language  "}),
        )
        .unwrap();
        assert_eq!(backend.queries, vec![("rust language".to_string(), 7)]);
        assert_eq!(
            response.as_str().unwrap(),
            "Rust\n====\n\n[Rust Programming Language](https://www.rust-lang.org/)"
        );
    }

    #[test]
    fn rejects_empty_long_and_unknown_requests() {
        let mut backend = RecordingBackend {
            queries: Vec::new(),
            search_markdown: None,
            fetched_urls: Vec::new(),
            rendered_markdown: None,
        };
        assert!(handle_agent_request(&mut backend, 8, "other", json!({"query": "rust"})).is_err());
        assert!(
            handle_agent_request(
                &mut backend,
                8,
                WEB_SEARCH_OPERATION,
                json!({"query": "  "})
            )
            .is_err()
        );
        assert!(
            handle_agent_request(
                &mut backend,
                8,
                WEB_SEARCH_OPERATION,
                json!({"query": "x".repeat(MAX_QUERY_CHARS + 1)})
            )
            .is_err()
        );
        assert!(backend.queries.is_empty());
    }

    #[test]
    fn web_fetch_validates_and_returns_rendered_markdown() {
        let mut backend = RecordingBackend {
            queries: Vec::new(),
            search_markdown: None,
            fetched_urls: Vec::new(),
            rendered_markdown: Some("# Rendered\n\nHello **world**.".to_string()),
        };
        let response = handle_agent_request(
            &mut backend,
            8,
            WEB_FETCH_OPERATION,
            json!({"url": " https://example.com/start "}),
        )
        .unwrap();
        assert_eq!(
            backend.fetched_urls,
            vec!["https://example.com/start".to_string()]
        );
        assert_eq!(response.as_str().unwrap(), "# Rendered\n\nHello **world**.");

        for invalid in [
            "",
            "not a URL",
            "file:///etc/passwd",
            "https://user:secret@example.com/",
        ] {
            assert!(
                handle_agent_request(
                    &mut backend,
                    8,
                    WEB_FETCH_OPERATION,
                    json!({"url": invalid})
                )
                .is_err()
            );
        }
    }

    #[test]
    fn rendered_markdown_truncation_preserves_utf8_boundaries() {
        let markdown = format!("{}é", "x".repeat(MAX_RENDERED_MARKDOWN_BYTES - 1));
        let markdown = truncate_rendered_markdown(markdown);
        assert!(markdown.is_char_boundary(markdown.len()));
        assert_eq!(markdown.len(), MAX_RENDERED_MARKDOWN_BYTES - 1);
    }

    #[test]
    fn web_settings_have_a_bounded_result_count() {
        let valid = Config::from_toml_str("[agent.web]\nmax_results = 10").unwrap();
        let section = valid.agent("web").unwrap();
        let settings = WebAgent::parse_settings(&valid, section).unwrap();
        WebAgent::validate_settings(&valid, section, &settings).unwrap();

        let invalid = Config::from_toml_str("[agent.web]\nmax_results = 0").unwrap();
        let section = invalid.agent("web").unwrap();
        let settings = WebAgent::parse_settings(&invalid, section).unwrap();
        assert!(WebAgent::validate_settings(&invalid, section, &settings).is_err());
    }

    #[test]
    fn web_agent_owns_its_remote_tool_definition() {
        let search = web_search_tool_definition();
        assert_eq!(search.name, "web_search");
        assert_eq!(search.operation, WEB_SEARCH_OPERATION);
        assert_eq!(search.parameters["required"], serde_json::json!(["query"]));

        let fetch = web_fetch_tool_definition();
        assert_eq!(fetch.name, "web_fetch");
        assert_eq!(fetch.operation, WEB_FETCH_OPERATION);
        assert_eq!(fetch.parameters["required"], serde_json::json!(["url"]));
        assert!(!fetch.description.is_empty());
    }

    #[test]
    fn verification_keeps_the_visible_browser_open_for_the_user() {
        let verification = anyhow::Error::new(GoogleVerificationRequired);
        assert!(!should_relaunch_browser(&verification));
        assert!(should_relaunch_browser(&anyhow::anyhow!(
            "CDP disconnected"
        )));
    }

    #[test]
    fn optional_smoke_test_searches_google_with_the_persistent_browser() {
        if std::env::var("POTLATCH_GOOGLE_SEARCH_SMOKE").as_deref() != Ok("1") {
            return;
        }
        let mut backend = ChromeWebBackend::new();
        let markdown = backend
            .search("Rust programming language", 3)
            .expect("Google Search through Chrome");
        assert!(!markdown.trim().is_empty());
        assert!(markdown.contains("http"));
    }

    #[test]
    fn optional_smoke_test_fetches_rendered_markdown_with_the_persistent_browser() {
        if std::env::var("POTLATCH_WEB_FETCH_SMOKE").as_deref() != Ok("1") {
            return;
        }
        let target = std::env::var("POTLATCH_WEB_FETCH_SMOKE_URL")
            .unwrap_or_else(|_| "https://example.com/".to_string());
        let mut backend = ChromeWebBackend::new();
        let response = handle_agent_request(
            &mut backend,
            DEFAULT_MAX_RESULTS,
            WEB_FETCH_OPERATION,
            json!({"url": target}),
        )
        .expect("render web page through Chrome and Defuddle");
        let markdown = response.as_str().unwrap();
        assert!(!markdown.trim().is_empty());
        assert!(!markdown.contains("<html"));
        assert!(!markdown.contains(".turbo-progress-bar"));
        assert!(!markdown.contains("\"featureFlags\""));
    }
}
