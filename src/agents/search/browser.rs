//! Private Chrome/Chromium controller owned by the search agent.

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

const BROWSER_TIMEOUT: Duration = Duration::from_secs(30);
const SEARCH_PROFILE_ENV: &str = "BREEZE_SEARCH_PROFILE";

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
        Ok(Self {
            _browser: browser,
            tab,
        })
    }

    pub(super) fn navigate_to(&self, url: &str) -> Result<String> {
        self.tab
            .navigate_to(url)
            .with_context(|| format!("navigate search browser to {url}"))?
            .wait_until_navigated()
            .context("wait for search browser navigation")?;
        Ok(self.tab.get_url())
    }

    pub(super) fn submit_google_query(&self, query: &str) -> Result<()> {
        self.navigate_to("https://www.google.com/")?;
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
        if self
            .google_verification_required()
            .context("check Google verification before submitting query")?
        {
            return Err(GoogleVerificationRequired.into());
        }
        let input = self
            .tab
            .wait_for_element("textarea[name='q'], input[name='q']")
            .context("wait for Google search input")?;
        input.click().context("focus Google search input")?;
        input.type_into(query).context("type Google search query")?;
        self.tab
            .press_key("Enter")
            .context("submit Google search query")?;
        self.tab
            .wait_until_navigated()
            .context("wait for Google search results navigation")?;
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
        self.tab
            .wait_for_element("a h3")
            .context("wait for rendered Google search results")?;
        Ok(())
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
}

fn decode_evaluated_json(serialized: Value) -> Result<Value> {
    let serialized = serialized
        .as_str()
        .context("search browser expression did not return serialized JSON")?;
    serde_json::from_str(serialized).context("decode search browser expression JSON")
}

fn resolve_search_profile_dir(explicit: Option<PathBuf>, home: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(explicit) = explicit {
        ensure!(
            !explicit.as_os_str().is_empty(),
            "{SEARCH_PROFILE_ENV} must not be empty"
        );
        return Ok(explicit);
    }
    let home = home.context("HOME is not set; set BREEZE_SEARCH_PROFILE explicitly")?;
    Ok(home.join(".potlatch").join("search-chrome-profile"))
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
            home.join(".potlatch/search-chrome-profile")
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
}
