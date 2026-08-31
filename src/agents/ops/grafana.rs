use anyhow::{Context, Result, bail, ensure};
use chrono::{DateTime, Utc};
use reqwest::Url;
use serde::Deserialize;
use serde_json::{Value, json};
use std::fmt;
use std::time::Duration;

const DEFAULT_ORG_ID: u64 = 1;
const TIME_FIELD: &str = "@timestamp";
const PAGE_SIZE: usize = 1_000;

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
    let mut current_to = to;
    loop {
        let body = build_msearch_body(source, from, current_to)?;
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

        let page = parse_msearch_response(&response_body)?;
        let page_len = page.lines.len();
        lines.extend(page.lines);

        // Fewer than a full page means this was the last batch within the
        // [from, to] window; no more data to fetch.
        if page_len < PAGE_SIZE {
            break;
        }

        let Some(oldest) = page.oldest_timestamp else {
            break;
        };

        // The oldest timestamp on a full page may have more records than fit on
        // this page. Before narrowing the cursor strictly below `oldest`, query
        // the true count at `oldest`; if it exceeds what we already collected
        // (`oldest_count_on_page`), drain the remaining records at `oldest`
        // with `from`/`size` offset pagination. Without this, any records at
        // `oldest` beyond the page would be skipped forever, because the next
        // page excludes `oldest` via the strict `lt` upper bound.
        // The Grafana/OpenSearch proxy does not support Scroll or `search_after`
        // cursors, so offset pagination is the only option.
        let true_at_oldest = count_at_timestamp(&client, &endpoint, source, oldest)?;
        if true_at_oldest > page.oldest_count_on_page {
            drain_timestamp_overflow(
                &client,
                &endpoint,
                source,
                oldest,
                page.oldest_count_on_page,
                &mut lines,
            )?;
        }

        // Narrow the upper bound to the oldest timestamp on this page and
        // re-query for strictly-older records. The upper bound is `lt`
        // (strict), so records at this exact timestamp are excluded from the
        // next page — no overlap, and none are skipped because any overflow at
        // `oldest` was drained above.
        current_to = DateTime::from_timestamp_millis(oldest)
            .context("decode oldest timestamp for next page cursor")?;
    }
    Ok(lines.join("\n"))
}

/// Drain the remaining records at a single timestamp using `from`/`size`
/// offset pagination. Called after a full page whose oldest timestamp has more
/// records than fit on the page: the first `start_from` records at that
/// timestamp were already collected, so fetching continues from
/// `from = start_from` in PAGE_SIZE increments until a short page is returned.
/// The Grafana/OpenSearch proxy does not support Scroll or `search_after`
/// cursors, so offset pagination is the only option.
fn drain_timestamp_overflow(
    client: &reqwest::blocking::Client,
    endpoint: &reqwest::Url,
    source: &GrafanaLogSource,
    timestamp_millis: i64,
    start_from: usize,
    lines: &mut Vec<String>,
) -> Result<()> {
    let mut from = start_from;
    loop {
        let body = build_timestamp_msearch_body(source, timestamp_millis, from)?;
        let response = client
            .post(endpoint.clone())
            .basic_auth(&source.username, Some(&source.password))
            .header("X-Grafana-Org-Id", source.org_id)
            .header("Content-Type", "application/x-ndjson")
            .body(body)
            .send()
            .with_context(|| {
                format!("drain overflow at timestamp {timestamp_millis} from {from} at {endpoint}")
            })?;
        let status = response.status();
        let response_body = response
            .text()
            .context("read timestamp overflow response")?;
        if !status.is_success() {
            let summary: String = response_body.chars().take(1_000).collect();
            bail!("Grafana timestamp overflow request failed with HTTP {status}: {summary}");
        }

        let page = parse_msearch_response(&response_body)?;
        let page_len = page.lines.len();
        lines.extend(page.lines);
        if page_len < PAGE_SIZE {
            break;
        }
        from += PAGE_SIZE;
    }
    Ok(())
}

/// Query the exact total number of records at a single timestamp. Used to
/// decide whether the oldest timestamp on a full page has more records than
/// were collected on that page, which would require draining the overflow.
fn count_at_timestamp(
    client: &reqwest::blocking::Client,
    endpoint: &reqwest::Url,
    source: &GrafanaLogSource,
    timestamp_millis: i64,
) -> Result<usize> {
    let body = build_count_msearch_body(source, timestamp_millis)?;
    let response = client
        .post(endpoint.clone())
        .basic_auth(&source.username, Some(&source.password))
        .header("X-Grafana-Org-Id", source.org_id)
        .header("Content-Type", "application/x-ndjson")
        .body(body)
        .send()
        .with_context(|| format!("count records at timestamp {timestamp_millis} at {endpoint}"))?;
    let status = response.status();
    let response_body = response.text().context("read count response")?;
    if !status.is_success() {
        let summary: String = response_body.chars().take(1_000).collect();
        bail!("Grafana count request failed with HTTP {status}: {summary}");
    }
    parse_count_response(&response_body)
}

fn build_msearch_body(
    source: &GrafanaLogSource,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> Result<String> {
    let mut time_bounds = serde_json::Map::new();
    time_bounds.insert("gte".to_string(), json!(from.timestamp_millis()));
    // `lt` (not `lte`) on the upper bound: the next page's range starts
    // strictly below the previous page's oldest timestamp, so records at that
    // exact millisecond are fetched once on the page that includes them as the
    // newest of their sub-range, with no overlap across pages.
    time_bounds.insert("lt".to_string(), json!(to.timestamp_millis()));
    let mut filters = vec![json!({ "range": { TIME_FIELD: time_bounds } })];
    if !source.filter.is_empty() {
        filters.push(raw_filter(&source.filter));
    }

    let header = json!({ "index": source.index });
    let query = json!({
        "size": PAGE_SIZE,
        "track_total_hits": false,
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

/// Build an `_msearch` body that fetches records at a single timestamp using
/// `from`/`size` offset pagination. Used to drain all records at a timestamp
/// when more than PAGE_SIZE share one millisecond (or the oldest timestamp on a
/// full page has more records than fit on that page).
fn build_timestamp_msearch_body(
    source: &GrafanaLogSource,
    timestamp_millis: i64,
    from: usize,
) -> Result<String> {
    let mut time_bounds = serde_json::Map::new();
    time_bounds.insert("gte".to_string(), json!(timestamp_millis));
    time_bounds.insert("lte".to_string(), json!(timestamp_millis));
    let mut filters = vec![json!({ "range": { TIME_FIELD: time_bounds } })];
    if !source.filter.is_empty() {
        filters.push(raw_filter(&source.filter));
    }

    let header = json!({ "index": source.index });
    let query = json!({
        "size": PAGE_SIZE,
        "from": from,
        "track_total_hits": false,
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

/// Build an `_msearch` body that counts records at a single timestamp. Uses
/// `size: 0` and `track_total_hits: true` so Elasticsearch returns only the
/// hit count without transferring any documents.
fn build_count_msearch_body(source: &GrafanaLogSource, timestamp_millis: i64) -> Result<String> {
    let mut time_bounds = serde_json::Map::new();
    time_bounds.insert("gte".to_string(), json!(timestamp_millis));
    time_bounds.insert("lte".to_string(), json!(timestamp_millis));
    let mut filters = vec![json!({ "range": { TIME_FIELD: time_bounds } })];
    if !source.filter.is_empty() {
        filters.push(raw_filter(&source.filter));
    }

    let header = json!({ "index": source.index });
    let query = json!({
        "size": 0,
        "track_total_hits": true,
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

/// Parse the total hit count from a count `_msearch` response.
fn parse_count_response(body: &str) -> Result<usize> {
    let root: Value =
        serde_json::from_str(body).context("decode Grafana Elasticsearch count response")?;
    let responses = root
        .get("responses")
        .and_then(Value::as_array)
        .context("Grafana Elasticsearch count response has no responses array")?;
    let mut total = 0usize;
    for response in responses {
        if let Some(error) = response.get("error") {
            bail!("Elasticsearch count query failed: {}", compact_json(error));
        }
        let count = response
            .pointer("/hits/total/value")
            .or_else(|| response.pointer("/hits/total"))
            .and_then(Value::as_u64)
            .context("Elasticsearch count response has no hits.total.value")?;
        total = total
            .checked_add(usize::try_from(count).context("Elasticsearch count overflow")?)
            .context("Elasticsearch total count overflow")?;
    }
    Ok(total)
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
    lines: Vec<String>,
    /// Oldest `@timestamp` (millis) on the page — the last record (desc sort).
    /// `None` only when the page is empty.
    oldest_timestamp: Option<i64>,
    /// Number of records on this page whose `@timestamp` equals
    /// `oldest_timestamp`. Because records are sorted `@timestamp desc`, these
    /// are the trailing records of the page. Used to decide whether more
    /// records at the oldest timestamp need draining before narrowing the
    /// cursor below it.
    oldest_count_on_page: usize,
}

fn parse_msearch_response(body: &str) -> Result<MsearchPage> {
    let root: Value =
        serde_json::from_str(body).context("decode Grafana Elasticsearch response")?;
    let responses = root
        .get("responses")
        .and_then(Value::as_array)
        .context("Grafana Elasticsearch response has no responses array")?;
    let mut lines = Vec::new();
    let mut oldest_timestamp = None;
    let mut oldest_count_on_page = 0usize;
    for response in responses {
        if let Some(error) = response.get("error") {
            bail!("Elasticsearch query failed: {}", compact_json(error));
        }
        let hits = response
            .pointer("/hits/hits")
            .and_then(Value::as_array)
            .context("Elasticsearch response has no hits.hits array")?;
        for hit in hits {
            let source = hit
                .get("_source")
                .context("Elasticsearch hit has no _source")?;
            lines.push(compact_json(source));
            if let Some(ts) = source.get(TIME_FIELD).and_then(Value::as_i64) {
                match oldest_timestamp {
                    None => {
                        oldest_timestamp = Some(ts);
                        oldest_count_on_page = 1;
                    }
                    Some(o) if ts < o => {
                        // Descending sort: a strictly smaller timestamp is a new
                        // oldest; reset the count for this new minimum.
                        oldest_timestamp = Some(ts);
                        oldest_count_on_page = 1;
                    }
                    Some(o) if ts == o => {
                        oldest_count_on_page += 1;
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(MsearchPage {
        lines,
        oldest_timestamp,
        oldest_count_on_page,
    })
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
    fn msearch_body_contains_time_window_raw_filter_and_page_size() {
        let source = settings().into_source().unwrap();
        let from = Utc.with_ymd_and_hms(2026, 8, 19, 7, 0, 0).unwrap();
        let to = Utc.with_ymd_and_hms(2026, 8, 19, 9, 0, 0).unwrap();
        let body = build_msearch_body(&source, from, to).unwrap();
        assert!(body.ends_with('\n'));
        let lines: Vec<Value> = body
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["index"], "app-logs");
        assert_eq!(lines[1]["size"], PAGE_SIZE);
        assert_eq!(lines[1]["track_total_hits"], false);
        assert_eq!(lines[1]["sort"][0]["@timestamp"], "desc");
        assert_eq!(lines[1]["sort"].as_array().unwrap().len(), 1);
        assert!(lines[1].get("search_after").is_none());
        assert_eq!(
            lines[1]["query"]["bool"]["filter"][1]["term"]["__tag__:__user_defined_id__"],
            "10.0.0.3"
        );
        assert_eq!(
            lines[1]["query"]["bool"]["filter"][0]["range"]["@timestamp"]["gte"],
            from.timestamp_millis()
        );
        // Upper bound is `lt` (strict) so the next page excludes the previous
        // page's oldest timestamp, giving no overlap across pages.
        assert_eq!(
            lines[1]["query"]["bool"]["filter"][0]["range"]["@timestamp"]["lt"],
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
    fn timestamp_body_drains_one_timestamp_with_from_offset() {
        let source = settings().into_source().unwrap();
        let ts: i64 = 1_724_054_400_000;
        let body = build_timestamp_msearch_body(&source, ts, PAGE_SIZE).unwrap();
        let lines: Vec<Value> = body
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1]["size"], PAGE_SIZE);
        assert_eq!(lines[1]["from"], PAGE_SIZE);
        // gte == lte == ts: fetch only records at this one timestamp.
        assert_eq!(
            lines[1]["query"]["bool"]["filter"][0]["range"]["@timestamp"]["gte"],
            ts
        );
        assert_eq!(
            lines[1]["query"]["bool"]["filter"][0]["range"]["@timestamp"]["lte"],
            ts
        );
        assert_eq!(lines[1]["sort"][0]["@timestamp"], "desc");
        // Same raw filter is applied.
        assert_eq!(
            lines[1]["query"]["bool"]["filter"][1]["term"]["__tag__:__user_defined_id__"],
            "10.0.0.3"
        );
    }

    #[test]
    fn count_body_counts_one_timestamp_with_size_zero() {
        let source = settings().into_source().unwrap();
        let ts: i64 = 1_724_054_400_000;
        let body = build_count_msearch_body(&source, ts).unwrap();
        let lines: Vec<Value> = body
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1]["size"], 0);
        assert_eq!(lines[1]["track_total_hits"], true);
        // No `from` or `sort` for a pure count query.
        assert!(lines[1].get("from").is_none());
        assert!(lines[1].get("sort").is_none());
        // gte == lte == ts: count only records at this one timestamp.
        assert_eq!(
            lines[1]["query"]["bool"]["filter"][0]["range"]["@timestamp"]["gte"],
            ts
        );
        assert_eq!(
            lines[1]["query"]["bool"]["filter"][0]["range"]["@timestamp"]["lte"],
            ts
        );
        // Same raw filter is applied.
        assert_eq!(
            lines[1]["query"]["bool"]["filter"][1]["term"]["__tag__:__user_defined_id__"],
            "10.0.0.3"
        );
    }

    #[test]
    fn parse_count_response_extracts_total_value() {
        let body = r#"{
            "responses": [{
                "hits": {
                    "total": {"value": 2500, "relation": "eq"},
                    "hits": []
                }
            }]
        }"#;
        assert_eq!(parse_count_response(body).unwrap(), 2500);
    }

    #[test]
    fn parse_count_response_sums_across_shards() {
        let body = r#"{
            "responses": [
                {"hits": {"total": {"value": 1000, "relation": "eq"}, "hits": []}},
                {"hits": {"total": {"value": 1500, "relation": "eq"}, "hits": []}}
            ]
        }"#;
        assert_eq!(parse_count_response(body).unwrap(), 2500);
    }

    #[test]
    fn parse_count_response_surfaces_elasticsearch_errors() {
        let error = parse_count_response(
            r#"{"responses":[{"error":{"type":"query_shard_exception","reason":"bad field"}}]}"#,
        )
        .unwrap_err();
        assert!(error.to_string().contains("query_shard_exception"));
    }

    #[test]
    fn msearch_response_extracts_oldest_timestamp_and_count() {
        let body = r#"{
            "responses": [{
                "hits": {
                    "total": {"value": 2, "relation": "eq"},
                    "hits": [
                        {"_source": {"@timestamp": 1724054400000, "message": "first"}},
                        {"_source": {"@timestamp": 1724054340000, "message": "second"}}
                    ]
                }
            }]
        }"#;
        let parsed = parse_msearch_response(body).unwrap();
        assert_eq!(parsed.lines.len(), 2);
        assert!(parsed.lines[0].contains("\"message\":\"first\""));
        assert!(parsed.lines[1].contains("\"message\":\"second\""));
        assert_eq!(parsed.oldest_timestamp, Some(1_724_054_340_000));
        assert_eq!(parsed.oldest_count_on_page, 1);
    }

    #[test]
    fn msearch_response_counts_oldest_timestamp_on_multi_timestamp_page() {
        // Descending sort: 600 at T+60s, then 400 at T (the oldest). The oldest
        // timestamp T appears 400 times on this page — this is exactly the case
        // where the old collision-only drain (oldest == newest) missed records.
        let mut hits = Vec::new();
        let newer_ts: i64 = 1_724_054_460_000;
        let oldest_ts: i64 = 1_724_054_400_000;
        for _ in 0..600 {
            hits.push(json!({"_source": {"@timestamp": newer_ts, "message": "newer"}}));
        }
        for _ in 0..400 {
            hits.push(json!({"_source": {"@timestamp": oldest_ts, "message": "oldest"}}));
        }
        let body = json!({"responses": [{"hits": {"total": {"value": 1000, "relation": "eq"}, "hits": hits}}]})
            .to_string();
        let parsed = parse_msearch_response(&body).unwrap();
        assert_eq!(parsed.lines.len(), 1000);
        assert_eq!(parsed.oldest_timestamp, Some(oldest_ts));
        assert_eq!(parsed.oldest_count_on_page, 400);
    }

    #[test]
    fn msearch_response_omits_timestamps_for_an_empty_page() {
        let body = r#"{
            "responses": [{
                "hits": {
                    "total": {"value": 0, "relation": "eq"},
                    "hits": []
                }
            }]
        }"#;
        let parsed = parse_msearch_response(body).unwrap();
        assert!(parsed.lines.is_empty());
        assert!(parsed.oldest_timestamp.is_none());
        assert_eq!(parsed.oldest_count_on_page, 0);
    }

    #[test]
    fn msearch_response_surfaces_elasticsearch_errors() {
        let error = parse_msearch_response(
            r#"{"responses":[{"error":{"type":"query_shard_exception","reason":"bad field"}}]}"#,
        )
        .unwrap_err();
        assert!(error.to_string().contains("query_shard_exception"));
    }
}
