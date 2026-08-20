//! Model-free Google Search service backed by one persistent browser instance.

mod browser;

use std::time::Duration;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(test)]
use serde_json::json;
use tracing::info;

use crate::core::agent::CoreAgent;
use crate::core::bus::{AgentInbox, AgentRequest, AgentToolDefinition};
use crate::core::periodic::{JitterPolicy, PeriodicTaskSpec};
use crate::core::runtime::AgentRuntime;
use crate::core::workflow::AgentBuildContext;
use browser::{ChromeBrowser, GoogleVerificationRequired};

const REQUEST_TASK: &str = "requests";
const INBOX_WAIT: Duration = Duration::from_millis(200);
const DEFAULT_MAX_RESULTS: usize = 8;
const MAX_CONFIGURED_RESULTS: usize = 10;
const MAX_QUERY_CHARS: usize = 500;
const SEARCH_OPERATION: &str = "google_search";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchAgentSettings {
    #[serde(default = "default_max_results")]
    max_results: usize,
}

fn default_max_results() -> usize {
    DEFAULT_MAX_RESULTS
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchRequest {
    query: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct SearchResponse {
    query: String,
    results: Vec<SearchResult>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct SearchResult {
    title: String,
    url: String,
    snippet: String,
}

trait SearchBackend: Send {
    fn search(&mut self, query: &str, max_results: usize) -> Result<Vec<SearchResult>>;
}

struct ChromeSearchBackend {
    browser: Option<ChromeBrowser>,
}

pub struct SearchAgent {
    runtime: AgentRuntime,
    inbox: AgentInbox,
    backend: Box<dyn SearchBackend>,
    max_results: usize,
}

impl ChromeSearchBackend {
    fn new() -> Self {
        Self { browser: None }
    }

    fn browser(&mut self) -> Result<&mut ChromeBrowser> {
        if self.browser.is_none() {
            self.browser = Some(ChromeBrowser::launch()?);
        }
        self.browser
            .as_mut()
            .context("search browser failed to initialize")
    }
}

impl SearchBackend for ChromeSearchBackend {
    fn search(&mut self, query: &str, max_results: usize) -> Result<Vec<SearchResult>> {
        let result = search_google(self.browser()?, query, max_results);
        if result.as_ref().is_err_and(should_relaunch_browser) {
            // A failed CDP operation can leave the tab or process unusable.
            // Relaunch lazily for the next independent request.
            self.browser.take();
        }
        result
    }
}

fn should_relaunch_browser(error: &anyhow::Error) -> bool {
    error.downcast_ref::<GoogleVerificationRequired>().is_none()
}

impl SearchAgent {
    fn handle_request(&mut self, request: AgentRequest) {
        let result = handle_search_request(
            self.backend.as_mut(),
            self.max_results,
            &request.operation,
            request.payload.clone(),
        );
        request.respond(result);
    }
}

impl CoreAgent for SearchAgent {
    type Settings = SearchAgentSettings;
    const FIXED_INSTANCES: Option<usize> = Some(1);

    fn name() -> &'static str {
        "search"
    }

    fn validate_settings(
        _config: &crate::core::config::Config,
        _section: &crate::core::config::AgentSection,
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
        ensure!(task_id == REQUEST_TASK, "unknown search task {task_id:?}");
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
            .context("search agent requires the cross-agent bus")?;
        let inbox = bus.register(Self::name(), vec![search_tool_definition()])?;
        Ok(Self {
            runtime: ctx.runtime.clone(),
            inbox,
            backend: Box::new(ChromeSearchBackend::new()),
            max_results: ctx.settings.max_results,
        })
    }

    fn on_start(&mut self) -> Result<()> {
        info!(
            "{}: Search agent ready; registered tool `search`",
            self.agent_id()
        );
        Ok(())
    }

    fn on_shutdown(&mut self) {}
}

fn handle_search_request(
    backend: &mut dyn SearchBackend,
    max_results: usize,
    operation: &str,
    payload: Value,
) -> Result<Value> {
    ensure!(
        operation == SEARCH_OPERATION,
        "unsupported search operation {operation:?}"
    );
    let request: SearchRequest =
        serde_json::from_value(payload).context("invalid Google Search request")?;
    let query = request.query.trim();
    ensure!(!query.is_empty(), "search query must not be empty");
    ensure!(
        query.chars().count() <= MAX_QUERY_CHARS,
        "search query exceeds {MAX_QUERY_CHARS} characters"
    );
    let results = backend.search(query, max_results)?;
    serde_json::to_value(SearchResponse {
        query: query.to_string(),
        results,
    })
    .context("encode Google Search response")
}

fn search_tool_definition() -> AgentToolDefinition {
    AgentToolDefinition {
        name: "search".to_string(),
        description: "Search Google through the dedicated browser search agent and return rendered result titles, URLs, and snippets. Use it when current public web information is needed.".to_string(),
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
        operation: SEARCH_OPERATION.to_string(),
    }
}

fn search_google(
    browser: &mut ChromeBrowser,
    query: &str,
    max_results: usize,
) -> Result<Vec<SearchResult>> {
    browser.submit_google_query(query)?;

    if browser
        .google_verification_required()
        .context("check Google verification after submitting query")?
    {
        return Err(GoogleVerificationRequired.into());
    }
    browser.wait_for_google_results()?;

    let expression = format!(
        r#"(() => {{
            const output = [];
            const seen = new Set();
            for (const heading of document.querySelectorAll("a h3")) {{
                const anchor = heading.closest("a");
                if (!anchor || !anchor.href || seen.has(anchor.href)) continue;
                if (!anchor.href.startsWith("http://") && !anchor.href.startsWith("https://")) continue;
                seen.add(anchor.href);
                const container = anchor.closest("div.MjjYud, div.N54PNb, div.g") || anchor.parentElement;
                const text = (container?.innerText || "").split("\n")
                    .map(line => line.trim()).filter(Boolean);
                const title = (heading.innerText || heading.textContent || "").trim();
                const snippet = text.filter(line => line !== title).slice(0, 4).join(" ");
                output.push({{ title, url: anchor.href, snippet }});
                if (output.length >= {max_results}) break;
            }}
            return output;
        }})()"#,
    );
    let value = browser
        .evaluate_json(&expression)
        .context("extract rendered Google search results")?;
    let results: Vec<SearchResult> =
        serde_json::from_value(value).context("decode rendered Google Search results")?;
    ensure!(
        !results.is_empty(),
        "Google returned no extractable search results"
    );
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RecordingBackend {
        queries: Vec<(String, usize)>,
        results: Vec<SearchResult>,
    }

    impl SearchBackend for RecordingBackend {
        fn search(&mut self, query: &str, max_results: usize) -> Result<Vec<SearchResult>> {
            self.queries.push((query.to_string(), max_results));
            Ok(std::mem::take(&mut self.results))
        }
    }

    fn result() -> SearchResult {
        SearchResult {
            title: "Rust".to_string(),
            url: "https://www.rust-lang.org/".to_string(),
            snippet: "A language empowering everyone.".to_string(),
        }
    }

    #[test]
    fn request_is_validated_and_forwarded_without_a_model() {
        let mut backend = RecordingBackend {
            queries: Vec::new(),
            results: vec![result()],
        };
        let response = handle_search_request(
            &mut backend,
            7,
            "google_search",
            json!({"query": "  rust language  "}),
        )
        .unwrap();
        assert_eq!(backend.queries, vec![("rust language".to_string(), 7)]);
        assert_eq!(response["results"][0]["title"], "Rust");
    }

    #[test]
    fn rejects_empty_long_and_unknown_requests() {
        let mut backend = RecordingBackend {
            queries: Vec::new(),
            results: Vec::new(),
        };
        assert!(handle_search_request(&mut backend, 8, "other", json!({"query": "rust"})).is_err());
        assert!(
            handle_search_request(&mut backend, 8, "google_search", json!({"query": "  "}))
                .is_err()
        );
        assert!(
            handle_search_request(
                &mut backend,
                8,
                "google_search",
                json!({"query": "x".repeat(MAX_QUERY_CHARS + 1)})
            )
            .is_err()
        );
        assert!(backend.queries.is_empty());
    }

    #[test]
    fn search_settings_have_a_bounded_result_count() {
        let valid =
            crate::core::config::Config::from_toml_str("[agent.search]\nmax_results = 10").unwrap();
        let section = valid.agent("search").unwrap();
        let settings = SearchAgent::parse_settings(&valid, section).unwrap();
        SearchAgent::validate_settings(&valid, section, &settings).unwrap();

        let invalid =
            crate::core::config::Config::from_toml_str("[agent.search]\nmax_results = 0").unwrap();
        let section = invalid.agent("search").unwrap();
        let settings = SearchAgent::parse_settings(&invalid, section).unwrap();
        assert!(SearchAgent::validate_settings(&invalid, section, &settings).is_err());
    }

    #[test]
    fn search_agent_owns_its_remote_tool_definition() {
        let tool = search_tool_definition();
        assert_eq!(tool.name, "search");
        assert_eq!(tool.operation, SEARCH_OPERATION);
        assert_eq!(tool.parameters["required"], serde_json::json!(["query"]));
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
        if std::env::var("BREEZE_GOOGLE_SEARCH_SMOKE").as_deref() != Ok("1") {
            return;
        }
        let mut backend = ChromeSearchBackend::new();
        let results = backend
            .search("Rust programming language", 3)
            .expect("Google Search through Chrome");
        assert!(!results.is_empty());
        assert!(results.iter().all(|result| !result.url.is_empty()));
    }
}
