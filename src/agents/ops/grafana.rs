use anyhow::{Context, Result, bail, ensure};
use chrono::{DateTime, Utc};
use reqwest::Url;
use serde::Deserialize;
use serde_json::{Value, json};
use std::fmt;
use std::time::Duration;

const DEFAULT_ORG_ID: u64 = 1;
const TIME_FIELD: &str = "@timestamp";
const FETCH_BATCH_SIZE: usize = 1_000;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GrafanaLogSourceSettings {
    pub(super) url: String,
    pub(super) datasource_uid: String,
    pub(super) index: String,
    #[serde(default = "default_org_id")]
    pub(super) org_id: u64,
    pub(super) username: String,
    pub(super) password: String,
    #[serde(default)]
    pub(super) filter: String,
}

impl fmt::Debug for GrafanaLogSourceSettings {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GrafanaLogSourceSettings")
            .field("url", &self.url)
            .field("datasource_uid", &self.datasource_uid)
            .field("index", &self.index)
            .field("org_id", &self.org_id)
            .field("username", &self.username)
            .field("password", &"[REDACTED]")
            .field("filter", &self.filter)
            .finish()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct GrafanaLogSource {
    base_url: Url,
    datasource_uid: String,
    index: String,
    org_id: u64,
    username: String,
    password: String,
    filter: String,
}

impl fmt::Debug for GrafanaLogSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GrafanaLogSource")
            .field("base_url", &self.base_url)
            .field("datasource_uid", &self.datasource_uid)
            .field("index", &self.index)
            .field("org_id", &self.org_id)
            .field("username", &self.username)
            .field("password", &"[REDACTED]")
            .field("filter", &self.filter)
            .finish()
    }
}

fn default_org_id() -> u64 {
    DEFAULT_ORG_ID
}

fn validate_nonempty(value: &str, field: &str, idx: usize) -> Result<()> {
    ensure!(
        !value.trim().is_empty(),
        "{field} is required for Grafana [agent.ops].logs[{idx}]"
    );
    Ok(())
}

impl GrafanaLogSourceSettings {
    pub(super) fn validate(&self, idx: usize) -> Result<()> {
        validate_nonempty(&self.url, "url", idx)?;
        let url = Url::parse(self.url.trim())
            .with_context(|| format!("invalid url for Grafana [agent.ops].logs[{idx}]"))?;
        ensure!(
            matches!(url.scheme(), "http" | "https"),
            "url must use http or https for Grafana [agent.ops].logs[{idx}]"
        );
        ensure!(
            url.username().is_empty() && url.password().is_none(),
            "url must not embed credentials for Grafana [agent.ops].logs[{idx}]"
        );
        validate_nonempty(&self.datasource_uid, "datasource_uid", idx)?;
        ensure!(
            self.datasource_uid
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_')),
            "datasource_uid contains unsupported characters for Grafana [agent.ops].logs[{idx}]"
        );
        validate_nonempty(&self.index, "index", idx)?;
        validate_nonempty(&self.username, "username", idx)?;
        validate_nonempty(&self.password, "password", idx)?;
        Ok(())
    }

    pub(super) fn into_source(self) -> Result<GrafanaLogSource> {
        let mut base_url = Url::parse(self.url.trim()).context("invalid Grafana url")?;
        let base_path = base_url.path().trim_end_matches('/').to_string();
        base_url.set_path(&base_path);
        base_url.set_query(None);
        base_url.set_fragment(None);
        Ok(GrafanaLogSource {
            base_url,
            datasource_uid: self.datasource_uid.trim().to_string(),
            index: self.index.trim().to_string(),
            org_id: self.org_id,
            username: self.username,
            password: self.password,
            filter: self.filter.trim().to_string(),
        })
    }
}

impl GrafanaLogSource {
    pub(super) fn target(&self) -> String {
        format!(
            "{} datasource={} index={}",
            self.base_url, self.datasource_uid, self.index
        )
    }

    fn endpoint(&self) -> Result<Url> {
        let path = format!(
            "{}/api/datasources/proxy/uid/{}/_msearch",
            self.base_url.path().trim_end_matches('/'),
            self.datasource_uid
        );
        let mut endpoint = self.base_url.clone();
        endpoint.set_path(&path);
        Ok(endpoint)
    }
}

pub(super) fn fetch_logs(
    source: &GrafanaLogSource,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> Result<String> {
    let endpoint = source.endpoint()?;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .context("build Grafana HTTP client")?;

    let mut lines = Vec::new();
    let mut windows = vec![TimeWindow {
        from,
        to,
        include_to: true,
    }];
    while let Some(window) = windows.pop() {
        let body = build_msearch_body(source, window)?;
        let response = client
            .post(endpoint.clone())
            .basic_auth(&source.username, Some(&source.password))
            .header("X-Grafana-Org-Id", source.org_id)
            .header("Content-Type", "application/x-ndjson")
            .body(body)
            .send()
            .with_context(|| format!("query Grafana Elasticsearch source at {endpoint}"))?;
        let status = response.status();
        let response_body = response
            .text()
            .context("read Grafana Elasticsearch response")?;
        if !status.is_success() {
            let summary: String = response_body.chars().take(1_000).collect();
            bail!("Grafana Elasticsearch request failed with HTTP {status}: {summary}");
        }

        let page = parse_msearch_response(&response_body, FETCH_BATCH_SIZE)?;
        if page.total_hits > FETCH_BATCH_SIZE {
            let (older, newer) = window.split()?;
            windows.push(older);
            windows.push(newer);
            continue;
        }
        lines.extend(page.lines);
    }
    Ok(lines.join("\n"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TimeWindow {
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    include_to: bool,
}

impl TimeWindow {
    fn split(self) -> Result<(Self, Self)> {
        let from_millis = self.from.timestamp_millis();
        let to_millis = self.to.timestamp_millis();
        ensure!(
            to_millis - from_millis > 1,
            "more than {FETCH_BATCH_SIZE} Elasticsearch records share one millisecond"
        );
        let midpoint_millis = from_millis + (to_millis - from_millis) / 2;
        let midpoint = DateTime::from_timestamp_millis(midpoint_millis)
            .context("split Grafana log time window")?;
        Ok((
            Self {
                from: self.from,
                to: midpoint,
                include_to: false,
            },
            Self {
                from: midpoint,
                to: self.to,
                include_to: self.include_to,
            },
        ))
    }
}

fn build_msearch_body(source: &GrafanaLogSource, window: TimeWindow) -> Result<String> {
    let mut time_bounds = serde_json::Map::new();
    time_bounds.insert("gte".to_string(), json!(window.from.timestamp_millis()));
    time_bounds.insert(
        if window.include_to { "lte" } else { "lt" }.to_string(),
        json!(window.to.timestamp_millis()),
    );
    let mut filters = vec![json!({ "range": { TIME_FIELD: time_bounds } })];
    if !source.filter.is_empty() {
        filters.push(raw_filter(&source.filter));
    }

    let header = json!({ "index": source.index });
    let query = json!({
        "size": FETCH_BATCH_SIZE,
        "track_total_hits": true,
        "sort": [{ TIME_FIELD: "desc" }],
        "query": {
            "bool": {
                "filter": filters
            }
        }
    });
    Ok(format!(
        "{}\n{}\n",
        serde_json::to_string(&header)?,
        serde_json::to_string(&query)?
    ))
}

fn raw_filter(filter: &str) -> Value {
    match serde_json::from_str::<Value>(filter) {
        Ok(value) if value.is_object() => value,
        _ => json!({
            "query_string": {
                "query": filter,
                "analyze_wildcard": true
            }
        }),
    }
}

#[derive(Debug, PartialEq)]
struct MsearchPage {
    total_hits: usize,
    lines: Vec<String>,
}

fn parse_msearch_response(body: &str, page_size: usize) -> Result<MsearchPage> {
    let root: Value =
        serde_json::from_str(body).context("decode Grafana Elasticsearch response")?;
    let responses = root
        .get("responses")
        .and_then(Value::as_array)
        .context("Grafana Elasticsearch response has no responses array")?;
    let mut lines = Vec::new();
    let mut total_hits = 0;
    for response in responses {
        if let Some(error) = response.get("error") {
            bail!("Elasticsearch query failed: {}", compact_json(error));
        }
        let hits = response
            .pointer("/hits/hits")
            .and_then(Value::as_array)
            .context("Elasticsearch response has no hits.hits array")?;
        let response_total = response
            .pointer("/hits/total/value")
            .or_else(|| response.pointer("/hits/total"))
            .and_then(Value::as_u64)
            .context("Elasticsearch response has no exact hits.total")?;
        total_hits +=
            usize::try_from(response_total).context("Elasticsearch hit count overflow")?;
        for hit in hits {
            let source = hit
                .get("_source")
                .context("Elasticsearch hit has no _source")?;
            lines.push(compact_json(source));
        }
    }
    ensure!(lines.len() <= page_size, "Elasticsearch exceeded page size");
    Ok(MsearchPage { total_hits, lines })
}

fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn settings() -> GrafanaLogSourceSettings {
        GrafanaLogSourceSettings {
            url: "https://grafana.example.com/".to_string(),
            datasource_uid: "elastic-1".to_string(),
            index: "app-logs".to_string(),
            org_id: 2,
            username: "ops".to_string(),
            password: "super-secret".to_string(),
            filter: r#"{"term":{"__tag__:__user_defined_id__":"10.0.0.3"}}"#.to_string(),
        }
    }

    #[test]
    fn settings_debug_redacts_password() {
        let debug = format!("{:?}", settings());
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("super-secret"));

        let source = settings().into_source().unwrap();
        let debug = format!("{source:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("super-secret"));
    }

    #[test]
    fn settings_reject_embedded_credentials() {
        let mut invalid = settings();
        invalid.url = "https://user:secret@grafana.example.com".to_string();
        assert!(
            invalid
                .validate(0)
                .unwrap_err()
                .to_string()
                .contains("embed")
        );
    }

    #[test]
    fn msearch_body_contains_time_window_raw_filter_and_internal_page_size() {
        let source = settings().into_source().unwrap();
        let from = Utc.with_ymd_and_hms(2026, 8, 19, 7, 0, 0).unwrap();
        let to = Utc.with_ymd_and_hms(2026, 8, 19, 9, 0, 0).unwrap();
        let body = build_msearch_body(
            &source,
            TimeWindow {
                from,
                to,
                include_to: true,
            },
        )
        .unwrap();
        assert!(body.ends_with('\n'));
        let lines: Vec<Value> = body
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["index"], "app-logs");
        assert_eq!(lines[1]["size"], FETCH_BATCH_SIZE);
        assert_eq!(lines[1]["track_total_hits"], true);
        assert_eq!(lines[1]["sort"][0]["@timestamp"], "desc");
        assert_eq!(lines[1]["sort"].as_array().unwrap().len(), 1);
        assert!(lines[1].get("_source").is_none());
        assert_eq!(
            lines[1]["query"]["bool"]["filter"][1]["term"]["__tag__:__user_defined_id__"],
            "10.0.0.3"
        );
        assert_eq!(
            lines[1]["query"]["bool"]["filter"][0]["range"]["@timestamp"]["gte"],
            from.timestamp_millis()
        );
        assert_eq!(
            lines[1]["query"]["bool"]["filter"][0]["range"]["@timestamp"]["lte"],
            to.timestamp_millis()
        );
    }

    #[test]
    fn non_json_filter_uses_elasticsearch_query_string() {
        assert_eq!(
            raw_filter("level:ERROR")["query_string"]["query"],
            "level:ERROR"
        );
    }

    #[test]
    fn split_time_windows_are_disjoint_and_keep_the_outer_bounds() {
        let window = TimeWindow {
            from: Utc.with_ymd_and_hms(2026, 8, 19, 7, 0, 0).unwrap(),
            to: Utc.with_ymd_and_hms(2026, 8, 19, 9, 0, 0).unwrap(),
            include_to: true,
        };
        let (older, newer) = window.split().unwrap();
        assert_eq!(older.from, window.from);
        assert_eq!(older.to, newer.from);
        assert!(!older.include_to);
        assert_eq!(newer.to, window.to);
        assert!(newer.include_to);
    }

    #[test]
    fn msearch_response_becomes_json_lines() {
        let body = r#"{
            "responses": [{
                "hits": {
                    "total": {"value": 2, "relation": "eq"},
                    "hits": [
                        {"_source": {"@timestamp": "2026-08-19T09:00:00Z", "message": "first"}},
                        {"_source": {"@timestamp": "2026-08-19T08:59:00Z", "message": "second"}}
                    ]
                }
            }]
        }"#;
        let parsed = parse_msearch_response(body, 2).unwrap();
        assert_eq!(parsed.total_hits, 2);
        assert_eq!(parsed.lines.len(), 2);
        assert!(parsed.lines[0].contains("\"message\":\"first\""));
        assert!(parsed.lines[1].contains("\"message\":\"second\""));
    }

    #[test]
    fn msearch_response_surfaces_elasticsearch_errors() {
        let error = parse_msearch_response(
            r#"{"responses":[{"error":{"type":"query_shard_exception","reason":"bad field"}}]}"#,
            FETCH_BATCH_SIZE,
        )
        .unwrap_err();
        assert!(error.to_string().contains("query_shard_exception"));
    }
}
