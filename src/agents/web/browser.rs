//! Private Chrome/Chromium controller owned by the web agent.

use std::ffi::OsStr;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use headless_chrome::browser::LaunchOptions;
use headless_chrome::protocol::cdp::Page::{
    AddScriptToEvaluateOnNewDocument, SetDownloadBehavior, SetDownloadBehaviorBehaviorOption,
};
use headless_chrome::{Browser, Tab};
use serde_json::Value;
use tracing::info;

const BROWSER_TIMEOUT: Duration = Duration::from_secs(30);
// Defuddle 0.19.2 full browser bundle (MIT); see defuddle.LICENSE.txt.
const DEFUDDLE_SCRIPT: &str = include_str!("defuddle.full.js");
const SEARCH_PROFILE_ENV: &str = "BREEZE_WEB_PROFILE";

#[derive(Debug)]
pub(super) struct GoogleVerificationRequired;

impl fmt::Display for GoogleVerificationRequired {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(
            "Google requires verification in the visible search browser; complete it and retry",
        )
    }
}

impl std::error::Error for GoogleVerificationRequired {}

pub(super) struct ChromeBrowser {
    _browser: Browser,
    tab: Arc<Tab>,
}

impl ChromeBrowser {
    pub(super) fn launch() -> Result<Self> {
        let executable = detect_browser_executable()?;
        let profile = resolve_search_profile_dir(
            std::env::var_os(SEARCH_PROFILE_ENV).map(PathBuf::from),
            std::env::var_os("HOME").map(PathBuf::from),
        )?;
        fs::create_dir_all(&profile)
            .with_context(|| format!("create search browser profile {}", profile.display()))?;
        let options = LaunchOptions::default_builder()
            .path(Some(executable))
            .user_data_dir(Some(profile))
            // Keep Chrome visible so a user can complete consent or verification
            // once; the persistent profile retains that state across restarts.
            .headless(false)
            .sandbox(true)
            .window_size(Some((1365, 900)))
            .ignore_default_args(vec![OsStr::new("--enable-automation")])
            .args(vec![
                OsStr::new("--disable-blink-features=AutomationControlled"),
                OsStr::new("--lang=en-US"),
            ])
            .ignore_certificate_errors(false)
            .idle_browser_timeout(Duration::from_secs(3_600))
            .build()
            .context("build Chrome launch options")?;
        let browser = Browser::new(options).context("launch Chrome or Chromium")?;
        browser.set_default_timeout(BROWSER_TIMEOUT);
        let tab = browser.new_tab().context("open search browser tab")?;
        tab.set_default_timeout(BROWSER_TIMEOUT);
        tab.call_method(SetDownloadBehavior {
            behavior: SetDownloadBehaviorBehaviorOption::Deny,
            download_path: None,
        })
        .context("disable browser downloads")?;
        tab.call_method(AddScriptToEvaluateOnNewDocument {
            source: r#"
                Object.defineProperty(navigator, "webdriver", { get: () => undefined });
                Object.defineProperty(navigator, "languages", { get: () => ["en-US", "en"] });
            "#
            .to_string(),
            world_name: None,
            include_command_line_api: None,
            run_immediately: None,
        })
        .context("install search browser compatibility script")?;
        tab.call_method(AddScriptToEvaluateOnNewDocument {
            source: DEFUDDLE_SCRIPT.to_string(),
            world_name: None,
            include_command_line_api: None,
            run_immediately: None,
        })
        .context("install Defuddle page extraction script")?;
        Ok(Self {
            _browser: browser,
            tab,
        })
    }

    pub(super) fn navigate_to(&self, url: &str) -> Result<String> {
        info!("opening {url}");
        self.tab
            .navigate_to(url)
            .with_context(|| format!("navigate search browser to {url}"))?
            .wait_until_navigated()
            .context("wait for search browser navigation")?;
        Ok(self.tab.get_url())
    }

    pub(super) fn submit_google_query(&self, query: &str, max_results: usize) -> Result<()> {
        // Land on Google first so the region-dependent consent interstitial is
        // handled before we open the results page directly. Opening the results
        // URL directly avoids the slow character-by-character typing into the
        // search box.
        self.navigate_to("https://www.google.com/")?;
        self.dismiss_google_consent()?;
        if self
            .google_verification_required()
            .context("check Google verification before submitting query")?
        {
            return Err(GoogleVerificationRequired.into());
        }
        self.navigate_to(&google_search_url(query, max_results))?;
        Ok(())
    }

    fn dismiss_google_consent(&self) -> Result<()> {
        // Consent is region-dependent and absent in many environments.
        self.tab
            .evaluate(
                r##"(() => {
                    const consent = document.querySelector("#L2AGLb")
                        || [...document.querySelectorAll("button")]
                            .find(button => /accept all/i.test(button.innerText || ""));
                    if (consent) consent.click();
                })()"##,
                false,
            )
            .context("handle Google consent page")?;
        Ok(())
    }

    pub(super) fn google_verification_required(&self) -> Result<bool> {
        Ok(self
            .evaluate_json(
                r#"(() => {
                    const text = (document.body?.innerText || "").toLowerCase();
                    return text.includes("our systems have detected unusual traffic")
                        || text.includes("verify you are not a robot");
                })()"#,
            )?
            .as_bool()
            .unwrap_or(false))
    }

    pub(super) fn wait_for_google_results(&self) -> Result<()> {
        self.wait_for_render()
            .context("wait for Google results page to finish rendering")?;
        let state = self
            .evaluate_json(GOOGLE_RESULTS_STATE_SCRIPT)
            .context("classify Google results page")?;
        match state.as_str() {
            Some("results") | Some("no_results") => Ok(()),
            Some("verification") => Err(GoogleVerificationRequired.into()),
            Some(other) => anyhow::bail!("unexpected Google results state {other:?}"),
            None => anyhow::bail!("Google results state was not a string: {state}"),
        }
    }

    /// Wait until the page's client-side rendering has settled. Page-agnostic
    /// and selector-free: a `MutationObserver` resolves once the DOM has been
    /// quiet for a settling window, with a bounded timeout. Use after
    /// [`navigate_to`] (which waits for `networkAlmostIdle`) on any page whose
    /// content is produced by post-load JS.
    pub(super) fn wait_for_render(&self) -> Result<()> {
        self.evaluate_await_void(RENDER_SETTLED_SCRIPT)
            .context("wait for page render to settle")
    }

    pub(super) fn fetch_rendered_markdown(&self, url: &str) -> Result<String> {
        self.navigate_to(url)?;
        self.wait_for_render()?;
        self.extract_rendered_markdown()
    }

    pub(super) fn extract_rendered_markdown(&self) -> Result<String> {
        Ok(self
            .evaluate_json(
                r#"(() => {
                    if (typeof globalThis.Defuddle !== "function") {
                        throw new Error("Defuddle browser bundle is unavailable");
                    }
                    const result = new globalThis.Defuddle(document, {
                        markdown: true,
                        useAsync: false
                    }).parse();
                    if (!result || typeof result.content !== "string") {
                        throw new Error("Defuddle returned no Markdown content");
                    }
                    return result.content;
                })()"#,
            )
            .context("extract rendered page with Defuddle")?
            .as_str()
            .context("Defuddle Markdown result was not a string")?
            .to_string())
    }

    pub(super) fn evaluate_json(&self, expression: &str) -> Result<Value> {
        let serialized = self
            .tab
            .evaluate(&format!("JSON.stringify({expression})"), true)
            .context("evaluate search browser expression")?
            .value
            .context("search browser expression returned no value")?;
        decode_evaluated_json(serialized)
    }

    /// Evaluate an expression expected to return a `Promise` and block until it
    /// resolves, discarding the resolved value. Used by [`wait_for_render`] so
    /// the DOM — via a `MutationObserver` injected inside the expression —
    /// drives completion instead of Rust-side polling.
    fn evaluate_await_void(&self, expression: &str) -> Result<()> {
        self.tab
            .evaluate(expression, true)
            .context("evaluate await expression")?;
        Ok(())
    }
}

/// Resolve once the page's client-side rendering has settled: a
/// [`MutationObserver`] watches `document.body` and the Promise resolves after
/// the DOM has been **quiet** (no mutations) for `SETTLE_MS`. This is
/// selector-free and page-agnostic — it works for any site that renders with
/// JS after `networkAlmostIdle` (what `wait_until_navigated` waits for), which
/// a fixed sleep cannot reliably cover.
///
/// A `MAX_WAIT_MS` `setTimeout` rejects so the Promise never hangs forever on a
/// page that keeps churning. The settle window restarts on every mutation, so
/// only a *sustained* pause counts as "done".
const RENDER_SETTLED_SCRIPT: &str = r##"new Promise((resolve, reject) => {
    const SETTLE_MS = 500;
    const MAX_WAIT_MS = 15000;
    let settleTimer;
    const done = () => { observer.disconnect(); clearTimeout(maxTimer); resolve(); };
    const observer = new MutationObserver(() => {
        clearTimeout(settleTimer);
        settleTimer = setTimeout(done, SETTLE_MS);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    settleTimer = setTimeout(done, SETTLE_MS);
    const maxTimer = setTimeout(() => {
        observer.disconnect();
        clearTimeout(settleTimer);
        reject(new Error("timed out waiting for the page to finish rendering"));
    }, MAX_WAIT_MS);
})"##;

/// One-shot, selector-free structural classification of a Google results page
/// **after** rendering has settled (see [`RENDER_SETTLED_SCRIPT`]). Returns
/// `"results"`, `"no_results"`, or `"verification"` — never `"loading"`, since
/// by this point the DOM is stable. Purely structural (no text matching) so it
/// is robust to Google copy/locale changes. Verification is checked first so a
/// stale `a h3` from a previous page cannot mask a wall.
const GOOGLE_RESULTS_STATE_SCRIPT: &str = r##"(() => {
    if (document.querySelector("#captcha-form, form[action*='sorry'], #recaptcha")) {
        return "verification";
    }
    const search = document.querySelector("#search");
    if (search) {
        return search.querySelector("a h3") ? "results" : "no_results";
    }
    return "no_results";
})()"##;

fn decode_evaluated_json(serialized: Value) -> Result<Value> {
    let serialized = serialized
        .as_str()
        .context("search browser expression did not return serialized JSON")?;
    serde_json::from_str(serialized).context("decode search browser expression JSON")
}

/// Build a Google Search results URL. Navigating directly to the results page
/// is much faster than typing the query into the search box and submitting the
/// form, and `num` requests the desired result count server-side.
fn google_search_url(query: &str, max_results: usize) -> String {
    let mut url = reqwest::Url::parse("https://www.google.com/search").expect("Google base URL");
    url.query_pairs_mut()
        .append_pair("q", query)
        .append_pair("num", &max_results.to_string());
    url.to_string()
}

fn resolve_search_profile_dir(explicit: Option<PathBuf>, home: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(explicit) = explicit {
        ensure!(
            !explicit.as_os_str().is_empty(),
            "{SEARCH_PROFILE_ENV} must not be empty"
        );
        return Ok(explicit);
    }
    let home = home.context("HOME is not set; set BREEZE_WEB_PROFILE explicitly")?;
    Ok(home.join(".potlatch").join("web-chrome-profile"))
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

    #[test]
    fn detection_accepts_only_chrome_and_chromium() {
        assert!(is_chrome_or_chromium(Path::new("/usr/bin/google-chrome")));
        assert!(is_chrome_or_chromium(Path::new("/usr/bin/chromium")));
        assert!(!is_chrome_or_chromium(Path::new("/usr/bin/msedge")));
        assert!(!is_chrome_or_chromium(Path::new("/usr/bin/firefox")));
    }

    #[test]
    fn path_detection_uses_supported_names_only() {
        let dir = crate::harness::tools::test_util::unique_test_dir();
        fs::write(dir.path().join("chromium"), "").unwrap();
        fs::write(dir.path().join("firefox"), "").unwrap();
        assert_eq!(
            detect_browser_in(&[dir.path().to_path_buf()], &["chromium"], &[]),
            Some(dir.path().join("chromium"))
        );
        assert_eq!(
            detect_browser_in(&[dir.path().to_path_buf()], &["chrome"], &[]),
            None
        );
    }

    #[test]
    fn search_profile_is_persistent_and_overridable() {
        let home = PathBuf::from("/home/tester");
        assert_eq!(
            resolve_search_profile_dir(None, Some(home.clone())).unwrap(),
            home.join(".potlatch/web-chrome-profile")
        );
        assert_eq!(
            resolve_search_profile_dir(
                Some(PathBuf::from("/var/lib/potlatch/search-profile")),
                Some(home)
            )
            .unwrap(),
            PathBuf::from("/var/lib/potlatch/search-profile")
        );
        assert!(resolve_search_profile_dir(None, None).is_err());
    }

    #[test]
    fn evaluated_json_decodes_objects_and_primitives() {
        assert_eq!(
            decode_evaluated_json(Value::String(r#"[{"title":"Rust"}]"#.to_string())).unwrap(),
            serde_json::json!([{"title": "Rust"}])
        );
        assert_eq!(
            decode_evaluated_json(Value::String("false".to_string())).unwrap(),
            Value::Bool(false)
        );
        assert!(decode_evaluated_json(Value::Null).is_err());
    }

    #[test]
    fn google_search_url_encodes_query_and_result_count() {
        let url = google_search_url("rust & crates", 10);
        let parsed = reqwest::Url::parse(&url).unwrap();
        assert_eq!(parsed.scheme(), "https");
        assert_eq!(parsed.host_str(), Some("www.google.com"));
        assert_eq!(parsed.path(), "/search");
        let pairs: std::collections::HashMap<_, _> = parsed.query_pairs().collect();
        assert_eq!(pairs.get("q").map(|v| v.as_ref()), Some("rust & crates"));
        assert_eq!(pairs.get("num").map(|v| v.as_ref()), Some("10"));
    }
}
