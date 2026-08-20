//! Stateful browser tool backed by an existing Chrome or Chromium installation.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use headless_chrome::browser::LaunchOptions;
use headless_chrome::protocol::cdp::Page::{
    SetDownloadBehavior, SetDownloadBehaviorBehaviorOption,
};
use headless_chrome::{Browser, Tab};
use reqwest::Url;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{SessionState, SessionStates, Tool};

const MAX_HTML_BYTES: usize = 50_000;
const MAX_SELECTOR_BYTES: usize = 2_000;
const MAX_TEXT_BYTES: usize = 100_000;
const MAX_KEY_BYTES: usize = 64;
const BROWSER_TIMEOUT: Duration = Duration::from_secs(30);

pub struct BrowserTool {
    state: Arc<BrowserState>,
}

struct BrowserState {
    driver: Mutex<Option<Box<dyn BrowserDriver>>>,
}

trait BrowserDriver: Send {
    fn navigate(&mut self, url: &str) -> Result<String>;
    fn snapshot(&mut self) -> Result<PageSnapshot>;
    fn click(&mut self, selector: &str) -> Result<()>;
    fn type_text(&mut self, selector: &str, text: &str, clear: bool) -> Result<()>;
    fn press_key(&mut self, key: &str) -> Result<()>;
}

struct ChromeDriver {
    _browser: Browser,
    tab: Arc<Tab>,
}

#[derive(Debug, PartialEq, Eq)]
struct PageSnapshot {
    url: String,
    title: String,
    html: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
enum BrowserAction {
    Navigate {
        url: String,
    },
    Snapshot,
    Click {
        selector: String,
    },
    Type {
        selector: String,
        text: String,
        #[serde(default)]
        clear: bool,
    },
    PressKey {
        key: String,
    },
}

impl BrowserState {
    fn new() -> Self {
        Self {
            driver: Mutex::new(None),
        }
    }

    fn with_driver<T>(
        &self,
        operation: impl FnOnce(&mut dyn BrowserDriver) -> Result<T>,
    ) -> Result<T> {
        let mut driver = self
            .driver
            .lock()
            .map_err(|_| anyhow::anyhow!("browser state lock poisoned"))?;
        if driver.is_none() {
            *driver = Some(Box::new(ChromeDriver::launch()?));
        }
        operation(
            driver
                .as_deref_mut()
                .context("browser failed to initialize")?,
        )
    }
}

impl SessionState for BrowserState {
    fn shutdown(&self) {
        if let Ok(mut driver) = self.driver.lock() {
            driver.take();
        }
    }
}

impl ChromeDriver {
    fn launch() -> Result<Self> {
        let executable = detect_browser_executable()?;
        let options = LaunchOptions::default_builder()
            .path(Some(executable))
            .headless(true)
            .sandbox(true)
            .ignore_certificate_errors(false)
            .idle_browser_timeout(Duration::from_secs(3_600))
            .build()
            .context("build Chrome launch options")?;
        let browser = Browser::new(options).context("launch headless Chrome or Chromium")?;
        browser.set_default_timeout(BROWSER_TIMEOUT);
        let tab = browser.new_tab().context("open browser tab")?;
        tab.set_default_timeout(BROWSER_TIMEOUT);
        tab.call_method(SetDownloadBehavior {
            behavior: SetDownloadBehaviorBehaviorOption::Deny,
            download_path: None,
        })
        .context("disable browser downloads")?;
        Ok(Self {
            _browser: browser,
            tab,
        })
    }
}

impl BrowserDriver for ChromeDriver {
    fn navigate(&mut self, url: &str) -> Result<String> {
        self.tab
            .navigate_to(url)
            .with_context(|| format!("navigate browser to {url}"))?
            .wait_until_navigated()
            .context("wait for browser navigation")?;
        Ok(self.tab.get_url())
    }

    fn snapshot(&mut self) -> Result<PageSnapshot> {
        Ok(PageSnapshot {
            url: self.tab.get_url(),
            title: self.tab.get_title().context("read page title")?,
            html: self.tab.get_content().context("serialize rendered DOM")?,
        })
    }

    fn click(&mut self, selector: &str) -> Result<()> {
        self.tab
            .wait_for_element(selector)
            .with_context(|| format!("wait for CSS selector {selector:?}"))?
            .click()
            .with_context(|| format!("click CSS selector {selector:?}"))?;
        Ok(())
    }

    fn type_text(&mut self, selector: &str, text: &str, clear: bool) -> Result<()> {
        let element = self
            .tab
            .wait_for_element(selector)
            .with_context(|| format!("wait for CSS selector {selector:?}"))?;
        if clear {
            element
                .call_js_fn(
                    r#"function () {
                        if ("value" in this) {
                            this.value = "";
                            this.dispatchEvent(new Event("input", { bubbles: true }));
                            this.dispatchEvent(new Event("change", { bubbles: true }));
                        } else {
                            this.textContent = "";
                        }
                    }"#,
                    Vec::new(),
                    false,
                )
                .with_context(|| format!("clear element {selector:?}"))?;
        }
        element
            .type_into(text)
            .with_context(|| format!("type into CSS selector {selector:?}"))?;
        Ok(())
    }

    fn press_key(&mut self, key: &str) -> Result<()> {
        self.tab
            .press_key(key)
            .with_context(|| format!("press browser key {key:?}"))?;
        Ok(())
    }
}

impl BrowserTool {
    pub fn new(states: &mut SessionStates) -> Self {
        if states.get::<BrowserState>().is_none() {
            states.insert(Arc::new(BrowserState::new()));
        }
        Self {
            state: states
                .get::<BrowserState>()
                .expect("browser state was just inserted"),
        }
    }
}

impl Tool for BrowserTool {
    fn name(&self) -> &str {
        "browser"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Control an existing local Chrome or Chromium in an isolated headless session. Use this tool for Google Search when current web information is needed: navigate to https://www.google.com/search?q=<URL-encoded query>, then call snapshot to inspect the rendered results. It can also navigate other pages and interact through CSS selectors. The browser is never downloaded automatically.",
            "parameters": {
                "type": "object",
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "action": { "const": "navigate" },
                            "url": {
                                "type": "string",
                                "description": "HTTP or HTTPS URL to load. For Google Search, use https://www.google.com/search?q=<URL-encoded query>."
                            }
                        },
                        "required": ["action", "url"],
                        "additionalProperties": false
                    },
                    {
                        "type": "object",
                        "properties": {
                            "action": { "const": "snapshot" }
                        },
                        "required": ["action"],
                        "additionalProperties": false
                    },
                    {
                        "type": "object",
                        "properties": {
                            "action": { "const": "click" },
                            "selector": {
                                "type": "string",
                                "description": "CSS selector of the element to click"
                            }
                        },
                        "required": ["action", "selector"],
                        "additionalProperties": false
                    },
                    {
                        "type": "object",
                        "properties": {
                            "action": { "const": "type" },
                            "selector": {
                                "type": "string",
                                "description": "CSS selector of the input element"
                            },
                            "text": {
                                "type": "string",
                                "description": "Text to enter"
                            },
                            "clear": {
                                "type": "boolean",
                                "description": "Clear the element before typing",
                                "default": false
                            }
                        },
                        "required": ["action", "selector", "text"],
                        "additionalProperties": false
                    },
                    {
                        "type": "object",
                        "properties": {
                            "action": { "const": "press_key" },
                            "key": {
                                "type": "string",
                                "description": "Chrome key name such as Enter, Escape, or Tab"
                            }
                        },
                        "required": ["action", "key"],
                        "additionalProperties": false
                    }
                ]
            }
        })
    }

    fn execute(&self, args: &Value, _cwd: &str) -> Result<String> {
        let action: BrowserAction =
            serde_json::from_value(args.clone()).context("invalid browser arguments")?;
        validate_action(&action)?;
        self.state
            .with_driver(|driver| execute_action(driver, action))
    }
}

fn execute_action(driver: &mut dyn BrowserDriver, action: BrowserAction) -> Result<String> {
    validate_action(&action)?;
    match action {
        BrowserAction::Navigate { url } => {
            let url = validate_navigation_url(&url)?;
            let final_url = driver.navigate(url.as_str())?;
            Ok(format!("Navigated to {final_url}"))
        }
        BrowserAction::Snapshot => Ok(format_snapshot(driver.snapshot()?)),
        BrowserAction::Click { selector } => {
            driver.click(&selector)?;
            Ok(format!("Clicked {selector:?}"))
        }
        BrowserAction::Type {
            selector,
            text,
            clear,
        } => {
            driver.type_text(&selector, &text, clear)?;
            Ok(format!("Typed into {selector:?}"))
        }
        BrowserAction::PressKey { key } => {
            driver.press_key(&key)?;
            Ok(format!("Pressed {key:?}"))
        }
    }
}

fn validate_action(action: &BrowserAction) -> Result<()> {
    match action {
        BrowserAction::Navigate { url } => {
            validate_navigation_url(url)?;
        }
        BrowserAction::Snapshot => {}
        BrowserAction::Click { selector } => validate_selector(selector)?,
        BrowserAction::Type { selector, text, .. } => {
            validate_selector(selector)?;
            ensure!(
                text.len() <= MAX_TEXT_BYTES,
                "browser text exceeds {MAX_TEXT_BYTES} bytes"
            );
        }
        BrowserAction::PressKey { key } => validate_key(key)?,
    }
    Ok(())
}

fn validate_navigation_url(raw: &str) -> Result<Url> {
    let url = Url::parse(raw).context("browser URL must be absolute")?;
    ensure!(
        matches!(url.scheme(), "http" | "https"),
        "browser navigation only supports HTTP and HTTPS"
    );
    ensure!(
        url.username().is_empty() && url.password().is_none(),
        "browser URL must not embed credentials"
    );
    Ok(url)
}

fn validate_selector(selector: &str) -> Result<()> {
    ensure!(
        !selector.trim().is_empty(),
        "CSS selector must not be empty"
    );
    ensure!(
        selector.len() <= MAX_SELECTOR_BYTES,
        "CSS selector exceeds {MAX_SELECTOR_BYTES} bytes"
    );
    Ok(())
}

fn validate_key(key: &str) -> Result<()> {
    ensure!(!key.trim().is_empty(), "browser key must not be empty");
    ensure!(
        key.len() <= MAX_KEY_BYTES,
        "browser key exceeds {MAX_KEY_BYTES} bytes"
    );
    ensure!(
        !key.chars().any(char::is_control),
        "browser key must not contain control characters"
    );
    Ok(())
}

fn format_snapshot(snapshot: PageSnapshot) -> String {
    let header = format!("URL: {}\nTitle: {}\nHTML:\n", snapshot.url, snapshot.title);
    let available = MAX_HTML_BYTES.saturating_sub(header.len());
    let html = truncate_at_char_boundary(&snapshot.html, available);
    if html.len() < snapshot.html.len() {
        format!("{header}{html}\n[...rendered HTML truncated at {MAX_HTML_BYTES} bytes...]")
    } else {
        format!("{header}{html}")
    }
}

fn truncate_at_char_boundary(value: &str, max_bytes: usize) -> &str {
    if value.len() <= max_bytes {
        return value;
    }
    let mut end = max_bytes;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn detect_browser_executable() -> Result<PathBuf> {
    if let Some(explicit) = std::env::var_os("CHROME") {
        let path = PathBuf::from(explicit);
        ensure!(
            path.is_file(),
            "CHROME does not point to an existing executable: {}",
            path.display()
        );
        ensure!(
            is_chrome_or_chromium(&path),
            "CHROME must point to Chrome or Chromium"
        );
        return Ok(path);
    }

    let path_dirs: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).collect())
        .unwrap_or_default();
    detect_browser_in(
        &path_dirs,
        platform_browser_names(),
        &standard_browser_paths(),
    )
    .context(
        "no Chrome or Chromium executable found; install one or set CHROME to its executable path",
    )
}

fn detect_browser_in(
    path_dirs: &[PathBuf],
    names: &[&str],
    standard_paths: &[PathBuf],
) -> Option<PathBuf> {
    for directory in path_dirs {
        for name in names {
            let candidate = directory.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    standard_paths.iter().find(|path| path.is_file()).cloned()
}

fn is_chrome_or_chromium(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_ascii_lowercase)
        .is_some_and(|name| {
            (name.contains("chrome") || name.contains("chromium"))
                && !name.contains("edge")
                && !name.contains("msedge")
        })
}

fn platform_browser_names() -> &'static [&'static str] {
    #[cfg(windows)]
    {
        &["chrome.exe", "chromium.exe"]
    }
    #[cfg(not(windows))]
    {
        &[
            "google-chrome-stable",
            "google-chrome",
            "chromium",
            "chromium-browser",
            "chrome",
        ]
    }
}

fn standard_browser_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    #[cfg(target_os = "linux")]
    {
        paths.extend(
            [
                "/opt/google/chrome/chrome",
                "/usr/bin/google-chrome-stable",
                "/usr/bin/google-chrome",
                "/usr/bin/chromium",
                "/usr/bin/chromium-browser",
                "/snap/bin/chromium",
            ]
            .into_iter()
            .map(PathBuf::from),
        );
    }
    #[cfg(target_os = "macos")]
    {
        paths.extend(
            [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium",
            ]
            .into_iter()
            .map(PathBuf::from),
        );
    }
    #[cfg(windows)]
    {
        for base in [
            std::env::var_os("PROGRAMFILES"),
            std::env::var_os("PROGRAMFILES(X86)"),
            std::env::var_os("LOCALAPPDATA"),
        ]
        .into_iter()
        .flatten()
        {
            let base = PathBuf::from(base);
            paths.push(base.join("Google/Chrome/Application/chrome.exe"));
            paths.push(base.join("Chromium/Application/chrome.exe"));
        }
    }
    paths
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[derive(Default)]
    struct RecordingDriver {
        calls: Vec<String>,
        snapshot: Option<PageSnapshot>,
    }

    impl BrowserDriver for RecordingDriver {
        fn navigate(&mut self, url: &str) -> Result<String> {
            self.calls.push(format!("navigate:{url}"));
            Ok(url.to_string())
        }

        fn snapshot(&mut self) -> Result<PageSnapshot> {
            self.calls.push("snapshot".to_string());
            self.snapshot
                .take()
                .context("recording snapshot not configured")
        }

        fn click(&mut self, selector: &str) -> Result<()> {
            self.calls.push(format!("click:{selector}"));
            Ok(())
        }

        fn type_text(&mut self, selector: &str, text: &str, clear: bool) -> Result<()> {
            self.calls
                .push(format!("type:{selector}:{text}:clear={clear}"));
            Ok(())
        }

        fn press_key(&mut self, key: &str) -> Result<()> {
            self.calls.push(format!("key:{key}"));
            Ok(())
        }
    }

    #[test]
    fn dispatches_stateful_browser_actions_to_the_same_driver() {
        let mut driver = RecordingDriver::default();
        execute_action(
            &mut driver,
            BrowserAction::Navigate {
                url: "https://example.com/search".to_string(),
            },
        )
        .unwrap();
        execute_action(
            &mut driver,
            BrowserAction::Type {
                selector: "input[name=q]".to_string(),
                text: "rust browser".to_string(),
                clear: true,
            },
        )
        .unwrap();
        execute_action(
            &mut driver,
            BrowserAction::PressKey {
                key: "Enter".to_string(),
            },
        )
        .unwrap();
        assert_eq!(
            driver.calls,
            [
                "navigate:https://example.com/search",
                "type:input[name=q]:rust browser:clear=true",
                "key:Enter",
            ]
        );
    }

    #[test]
    fn rejects_non_web_navigation_before_calling_the_driver() {
        let mut driver = RecordingDriver::default();
        let error = execute_action(
            &mut driver,
            BrowserAction::Navigate {
                url: "file:///etc/passwd".to_string(),
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("HTTP and HTTPS"));
        assert!(driver.calls.is_empty());
    }

    #[test]
    fn rejects_empty_selectors_and_keys() {
        let mut driver = RecordingDriver::default();
        assert!(
            execute_action(
                &mut driver,
                BrowserAction::Click {
                    selector: " ".to_string(),
                },
            )
            .is_err()
        );
        assert!(
            execute_action(&mut driver, BrowserAction::PressKey { key: String::new() },).is_err()
        );
        assert!(driver.calls.is_empty());
    }

    #[test]
    fn snapshot_returns_rendered_html_and_truncates_on_a_character_boundary() {
        let mut driver = RecordingDriver {
            snapshot: Some(PageSnapshot {
                url: "https://example.com".to_string(),
                title: "Example".to_string(),
                html: "é".repeat(MAX_HTML_BYTES),
            }),
            ..RecordingDriver::default()
        };
        let output = execute_action(&mut driver, BrowserAction::Snapshot).unwrap();
        assert!(output.starts_with("URL: https://example.com\nTitle: Example\nHTML:\n"));
        assert!(output.contains("rendered HTML truncated"));
        assert!(output.is_char_boundary(output.len()));
    }

    #[test]
    fn detects_only_chrome_and_chromium_candidates() {
        let dir = super::super::test_util::unique_test_dir();
        let chrome = dir.path().join("chromium");
        let edge = dir.path().join("microsoft-edge");
        fs::write(&chrome, "").unwrap();
        fs::write(&edge, "").unwrap();
        assert_eq!(
            detect_browser_in(&[dir.path().to_path_buf()], &["chromium"], &[]),
            Some(chrome)
        );
        assert!(is_chrome_or_chromium(Path::new("google-chrome")));
        assert!(is_chrome_or_chromium(Path::new("chromium")));
        assert!(!is_chrome_or_chromium(Path::new("microsoft-edge")));
        assert!(!is_chrome_or_chromium(Path::new("msedge.exe")));
    }

    #[test]
    fn browser_schema_lists_every_supported_action() {
        let mut states = SessionStates::new();
        let schema = BrowserTool::new(&mut states).schema();
        let serialized = schema.to_string();
        for action in ["navigate", "snapshot", "click", "type", "press_key"] {
            assert!(serialized.contains(&format!("\"const\":\"{action}\"")));
        }
    }

    #[test]
    fn browser_schema_explains_how_to_use_google_search() {
        let mut states = SessionStates::new();
        let schema = BrowserTool::new(&mut states).schema();
        let description = schema["description"].as_str().unwrap();
        assert!(description.contains("Google Search"));
        assert!(description.contains("google.com/search?q="));
        assert!(description.contains("snapshot"));
    }

    #[test]
    fn browser_tools_reuse_one_lazy_session_state() {
        let mut states = SessionStates::new();
        let first = BrowserTool::new(&mut states);
        let second = BrowserTool::new(&mut states);
        assert!(Arc::ptr_eq(&first.state, &second.state));
        assert!(first.state.driver.lock().unwrap().is_none());
    }

    #[test]
    fn shutdown_without_a_started_browser_is_safe() {
        let state = BrowserState::new();
        state.shutdown();
        assert!(state.driver.lock().unwrap().is_none());
    }

    #[test]
    fn shutdown_drops_a_started_browser_driver() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct DropDriver(Arc<AtomicBool>);

        impl Drop for DropDriver {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        impl BrowserDriver for DropDriver {
            fn navigate(&mut self, _url: &str) -> Result<String> {
                unreachable!()
            }
            fn snapshot(&mut self) -> Result<PageSnapshot> {
                unreachable!()
            }
            fn click(&mut self, _selector: &str) -> Result<()> {
                unreachable!()
            }
            fn type_text(&mut self, _selector: &str, _text: &str, _clear: bool) -> Result<()> {
                unreachable!()
            }
            fn press_key(&mut self, _key: &str) -> Result<()> {
                unreachable!()
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        let state = BrowserState {
            driver: Mutex::new(Some(Box::new(DropDriver(Arc::clone(&dropped))))),
        };
        state.shutdown();
        assert!(dropped.load(Ordering::SeqCst));
        assert!(state.driver.lock().unwrap().is_none());
    }

    #[test]
    fn optional_smoke_test_uses_an_installed_browser() {
        if std::env::var_os("BREEZE_BROWSER_SMOKE_TEST").is_none() {
            return;
        }
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 1_024];
            let _ = stream.read(&mut request);
            let body = "<!doctype html><html><title>Smoke</title><p>ok</p></html>";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        let mut driver = ChromeDriver::launch().unwrap();
        driver.navigate(&format!("http://{address}/")).unwrap();
        let snapshot = driver.snapshot().unwrap();
        assert_eq!(snapshot.title, "Smoke");
        assert!(snapshot.html.contains("<p>ok</p>"));
        server.join().unwrap();
    }
}
