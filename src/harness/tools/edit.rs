//! Edit tool: str_replace one or more edits, with line-number anchored
//! errors, fuzzy near-match suggestions, and read-after-edit verification.
//!
//! [`EditTool`] takes an `edits` array of `{path, old_string, new_string}`
//! objects. Edits to the same file apply in order (an earlier edit may shift
//! text a later edit references); edits to different files run concurrently.
//! Per-edit errors are reported inline and do not block the other edits.
//!
//! When an exact `old_string` is not found, the error includes the best fuzzy
//! near-matches (computed via the `dissimilar` crate) with line numbers, so the
//! model can re-read and retry without guessing.

use std::fs;

use anyhow::Result;
use serde_json::{Value, json};

use super::Tool;

/// Minimum normalized similarity (0.0..=1.0) for a window to qualify as a
/// near-match suggestion. Tuned to surface genuine typos and small drift
/// (whitespace, missing line) while rejecting unrelated regions.
const NEAR_MATCH_THRESHOLD: f64 = 0.6;

/// Maximum number of near-match suggestions to include in an error message.
const MAX_NEAR_MATCHES: usize = 3;

/// Maximum number of multiple-match locations to include in an error message.
const MAX_MULTI_MATCHES: usize = 3;

/// Minimum length of `old_string` for fuzzy matching to kick in. Very short
/// needles produce noisy results, so for those we fall back to word-overlap
/// heuristics only.
const MIN_FUZZY_NEEDLE_LEN: usize = 8;

pub struct EditTool;

impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Edit files by replacing exact strings. Pass an 'edits' array of objects, each with {path, old_string, new_string}. The 'path' goes INSIDE each edit object, not at the top level. Edits to the same file apply in order (an earlier edit may shift text a later edit references); edits to different files run concurrently. Per-edit errors are reported inline and do not block the other edits. The old_string must match uniquely within its file. On failure, the error includes line numbers and similarity scores of fuzzy near-matches so you can retry with a more specific match. After editing, the edited region is read back and included in the result so you can verify the change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "description": "List of edits to apply.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Path to the file to edit (relative to the working directory)."
                                },
                                "old_string": {
                                    "type": "string",
                                    "description": "The exact string to find in the file. Must match uniquely within the file."
                                },
                                "new_string": {
                                    "type": "string",
                                    "description": "The replacement string."
                                }
                            },
                            "required": ["path", "old_string", "new_string"]
                        },
                        "minItems": 1
                    }
                },
                "required": ["edits"]
            }
        })
    }

    fn execute(&self, args: &Value, cwd: &str) -> Result<String> {
        let edits = args["edits"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing or invalid 'edits' array argument"))?;

        if edits.is_empty() {
            anyhow::bail!("'edits' array must contain at least one entry");
        }

        // Parse + validate each entry up front (cheap, no I/O). Group by resolved
        // path so we can apply multiple edits to the same file in order, while
        // independent files run concurrently.
        //
        // Graceful fallback: if none of the edit objects have a `path` but the
        // top-level args has a `path` string, treat it as a default path for all
        // edits. Models sometimes place `path` at the top level instead of
        // inside each edit object; rather than failing, we apply it to every
        // edit so the work proceeds.
        let top_level_path = args["path"].as_str();
        let saw_any_edit_path = edits
            .iter()
            .any(|e| e["path"].as_str().is_some_and(|s| !s.is_empty()));

        let mut groups: Vec<EditGroup> = Vec::new();
        for (idx, entry) in edits.iter().enumerate() {
            let path = entry["path"].as_str();
            let path = if path.is_some_and(|s| !s.is_empty()) {
                path.unwrap()
            } else if !saw_any_edit_path && top_level_path.is_some() {
                top_level_path.unwrap()
            } else {
                anyhow::bail!(
                    "each edit requires a 'path' string inside the edit object \
                     (e.g. {{\"path\": \"src/main.rs\", \"old_string\": \"...\", \"new_string\": \"...\"}}). \
                     Do not put 'path' at the top level of the arguments — it goes inside each edit."
                );
            };
            let old_string = entry["old_string"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("each edit requires an 'old_string' string"))?;
            let new_string = entry["new_string"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("each edit requires a 'new_string' string"))?;

            let full_path = match super::resolve_workspace_path(path, cwd) {
                Ok(p) => p,
                Err(msg) => {
                    // Pre-validation failure: record a synthetic group with a
                    // single error result so it's surfaced inline.
                    groups.push(EditGroup {
                        path_label: path.to_string(),
                        full_path: None,
                        edits: vec![ParsedEdit {
                            original_index: idx,
                            old_string: old_string.to_string(),
                            new_string: new_string.to_string(),
                        }],
                        results: vec![format!("Error: {msg}")],
                    });
                    continue;
                }
            };

            // Find or create the group for this resolved file path.
            let group = groups
                .iter_mut()
                .find(|g| g.full_path.as_deref() == Some(full_path.as_path()));
            match group {
                Some(g) => g.edits.push(ParsedEdit {
                    original_index: idx,
                    old_string: old_string.to_string(),
                    new_string: new_string.to_string(),
                }),
                None => groups.push(EditGroup {
                    path_label: path.to_string(),
                    full_path: Some(full_path),
                    edits: vec![ParsedEdit {
                        original_index: idx,
                        old_string: old_string.to_string(),
                        new_string: new_string.to_string(),
                    }],
                    results: Vec::new(),
                }),
            }
        }

        // Apply groups concurrently. Each group processes its edits sequentially
        // against a single in-memory buffer, so within-file ordering is preserved
        // while different files run in parallel.
        let group_results: Vec<(String, Vec<usize>, Vec<String>)> = std::thread::scope(|s| {
            let handles: Vec<_> = groups
                .iter_mut()
                .map(|g| {
                    let indices: Vec<usize> = g.edits.iter().map(|e| e.original_index).collect();
                    let label = g.path_label.clone();
                    let results = apply_group(g);
                    s.spawn(move || (label, indices, results))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| {
                        (
                            "unknown".to_string(),
                            vec![],
                            vec!["Error: edit group thread panicked".to_string()],
                        )
                    })
                })
                .collect()
        });

        // Re-order results by the original edit index so the output reads in the
        // order the model issued the edits.
        let mut indexed: Vec<(usize, String, String)> = Vec::new();
        for (path_label, indices, results) in group_results {
            for (i, result) in results.into_iter().enumerate() {
                let orig = indices.get(i).copied().unwrap_or(i);
                indexed.push((orig, path_label.clone(), result));
            }
        }
        indexed.sort_by_key(|(i, _, _)| *i);

        let mut output = format!("Applied {} edit(s):\n\n", indexed.len());
        for (i, (orig, path_label, result)) in indexed.iter().enumerate() {
            if i > 0 {
                output.push_str("\n---\n\n");
            }
            output.push_str(&format!("[edit {orig}] {path_label}: {result}\n"));
        }
        Ok(output)
    }
}

/// One parsed edit entry, with its position in the original `edits` array.
struct ParsedEdit {
    original_index: usize,
    old_string: String,
    new_string: String,
}

/// All edits that target the same resolved file path, in the order they were
/// issued. `full_path` is `None` when path resolution failed up front (e.g.
/// absolute path rejected); in that case `results` is pre-filled with the
/// error and `edits` is a single placeholder.
struct EditGroup {
    path_label: String,
    full_path: Option<std::path::PathBuf>,
    edits: Vec<ParsedEdit>,
    results: Vec<String>,
}

/// Apply all edits in a group sequentially to a single in-memory buffer.
/// Writes the file at most once (on success) or not at all if every edit
/// fails. Pushes one result string per edit (same order as `edits`).
fn apply_group(group: &mut EditGroup) -> Vec<String> {
    // If path resolution failed up front, the single error result is already
    // populated — return it as-is (one result per edit).
    if group.full_path.is_none() {
        if group.results.len() == 1 && group.edits.len() > 1 {
            let err = group.results[0].clone();
            group.results = (0..group.edits.len()).map(|_| err.clone()).collect();
        }
        return std::mem::take(&mut group.results);
    }

    let full_path = group.full_path.as_ref().expect("checked above");
    let path_label = &group.path_label;

    let mut content = match fs::read_to_string(full_path) {
        Ok(c) => c,
        Err(e) => {
            // Every edit fails with the read error.
            let err = format!("Error: failed to read {}: {e}", full_path.display());
            return group.edits.iter().map(|_| err.clone()).collect();
        }
    };

    let mut dirty = false;
    let mut results = Vec::with_capacity(group.edits.len());
    for edit in &group.edits {
        let outcome =
            apply_edit_to_buffer(&mut content, path_label, &edit.old_string, &edit.new_string);
        if outcome.starts_with("Successfully edited") {
            dirty = true;
        }
        results.push(outcome);
    }

    if dirty && let Err(e) = fs::write(full_path, &content) {
        // The in-memory edits applied but the write failed. Replace every
        // success with a write-error so the model knows nothing persisted.
        let err = format!("Error: failed to write {}: {e}", full_path.display());
        for r in &mut results {
            if r.starts_with("Successfully edited") {
                *r = err.clone();
            }
        }
    }

    results
}

/// Apply a single `old_string -> new_string` edit to a buffer in place.
/// Returns the user-facing result string (success message with verification,
/// or an error message with near-matches). Does NOT touch the filesystem.
fn apply_edit_to_buffer(
    content: &mut String,
    path_label: &str,
    old_string: &str,
    new_string: &str,
) -> String {
    let match_count = content.matches(old_string).count();
    if match_count == 0 {
        let near_matches = find_near_matches(content, old_string);
        let mut error = format!("old_string not found in {path_label}.\n");
        if !near_matches.is_empty() {
            error.push_str("Near matches:\n");
            for (line_num, snippet, score) in near_matches.iter().take(MAX_NEAR_MATCHES) {
                let percent = (score * 100.0).round() as u32;
                error.push_str(&format!(
                    "  line {line_num} (similarity {percent}%): {snippet}\n"
                ));
            }
        }
        error.push_str("Re-read the file and retry with exact content.");
        return error;
    }
    if match_count > 1 {
        let locations = find_all_match_locations(content, old_string);
        let mut error = format!(
            "old_string matches {match_count} times in {path_label}. Include more context to make it unique.\n"
        );
        for (line_num, snippet) in locations.iter().take(MAX_MULTI_MATCHES) {
            error.push_str(&format!("  line {line_num}: {snippet}\n"));
        }
        return error;
    }

    // Apply the edit to the buffer in place.
    if let Some(idx) = content.find(old_string) {
        let mut rebuilt =
            String::with_capacity(content.len() - old_string.len() + new_string.len());
        rebuilt.push_str(&content[..idx]);
        rebuilt.push_str(new_string);
        rebuilt.push_str(&content[idx + old_string.len()..]);
        *content = rebuilt;
    }

    let verification = verify_edit(content, new_string);
    format!("Successfully edited {path_label}.\n{verification}")
}

/// Find the line number and snippet of each occurrence of `needle`.
fn find_all_match_locations(content: &str, needle: &str) -> Vec<(usize, String)> {
    let mut results = Vec::new();
    let mut search_start = 0;
    while let Some(pos) = content[search_start..].find(needle) {
        let abs_pos = search_start + pos;
        let line_num = content[..abs_pos].lines().count() + 1;
        let line_start = content[..abs_pos].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let line_end = content[abs_pos..]
            .find('\n')
            .map(|i| abs_pos + i)
            .unwrap_or(content.len());
        let snippet = &content[line_start..line_end.min(line_start + 60)];
        results.push((line_num, snippet.to_string()));
        search_start = abs_pos + needle.len();
    }
    results
}

/// Find near-matches for `needle` in `content`. Combines a fuzzy sliding-window
/// search (via the `dissimilar` crate's LCS-based diff) with a word-overlap
/// heuristic, then returns the best candidates by similarity score.
///
/// Returns a list of `(line_number, snippet, similarity_score)` tuples sorted
/// by descending score.
fn find_near_matches(content: &str, needle: &str) -> Vec<(usize, String, f64)> {
    let mut candidates: Vec<(usize, String, f64)> = Vec::new();

    // Fuzzy sliding-window search: for each window of lines whose length is
    // close to the needle's line count, compute a normalized similarity score.
    if needle.len() >= MIN_FUZZY_NEEDLE_LEN {
        candidates.extend(fuzzy_line_window_matches(content, needle));
    }

    // Always also run the word-overlap heuristic — it catches cases where the
    // fuzzy window score is low but the model would still recognize the line.
    for (line_num, snippet) in word_overlap_near_matches(content, needle) {
        // Only add if not already present (dedup by line number).
        if !candidates.iter().any(|(ln, _, _)| ln == &line_num) {
            candidates.push((line_num, snippet, 0.5)); // heuristic, no real score
        }
    }

    // Sort by descending score, then by ascending line number for stability.
    candidates.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    // Dedup by line number, keeping the highest-scoring entry.
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    candidates.retain(|(ln, _, _)| seen.insert(*ln));

    candidates
}

/// Slide a window of lines over `content` and score each window against
/// `needle` using a normalized LCS similarity derived from `dissimilar`'s diff.
/// Only windows scoring above [`NEAR_MATCH_THRESHOLD`] are returned.
fn fuzzy_line_window_matches(content: &str, needle: &str) -> Vec<(usize, String, f64)> {
    let lines: Vec<&str> = content.lines().collect();

    // Window size: match the needle's line count, but allow a little slack
    // (±1 line) to catch off-by-one drift. We try each window size and keep
    // the best score per starting line.
    let needle_lines: Vec<&str> = needle.lines().collect();
    let base_window = needle_lines.len().max(1);
    let window_sizes: Vec<usize> = match base_window {
        1 => vec![1],
        2 => vec![1, 2, 3],
        _ => vec![base_window - 1, base_window, base_window + 1],
    };

    let mut best_by_line: std::collections::HashMap<usize, (String, f64)> =
        std::collections::HashMap::new();

    for &window_size in &window_sizes {
        if window_size == 0 || window_size > lines.len() {
            continue;
        }
        for start in 0..=(lines.len() - window_size) {
            let window: Vec<&str> = lines[start..start + window_size].to_vec();
            let window_text = window.join("\n");
            let score = similarity_score(&window_text, needle);
            if score >= NEAR_MATCH_THRESHOLD {
                let line_num = start + 1;
                let snippet: String = window_text.chars().take(60).collect();
                let entry = best_by_line
                    .entry(line_num)
                    .or_insert((snippet.clone(), 0.0));
                if score > entry.1 {
                    *entry = (snippet, score);
                }
            }
        }
    }

    // Also try character-level sliding windows for single-line needles, since
    // the line-window approach above would otherwise miss intra-line drift.
    if needle_lines.len() <= 1 && needle.len() >= MIN_FUZZY_NEEDLE_LEN {
        for (i, line) in lines.iter().enumerate() {
            let score = similarity_score(line, needle);
            if score >= NEAR_MATCH_THRESHOLD {
                let line_num = i + 1;
                let snippet: String = line.chars().take(60).collect();
                let entry = best_by_line
                    .entry(line_num)
                    .or_insert((snippet.clone(), 0.0));
                if score > entry.1 {
                    *entry = (snippet, score);
                }
            }
        }
    }

    let mut results: Vec<(usize, String, f64)> = best_by_line
        .into_iter()
        .map(|(ln, (snippet, score))| (ln, snippet, score))
        .collect();
    results.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    results
}

/// Normalized similarity score in [0.0, 1.0] between `a` and `b`, based on the
/// longest-common-subsequence length implied by `dissimilar`'s diff chunks.
/// `1.0` means identical; `0.0` means no common subsequence.
fn similarity_score(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    let chunks = dissimilar::diff(a, b);
    // Equal chunks contribute to the LCS length; insert/delete chunks do not.
    let lcs_len: usize = chunks
        .iter()
        .map(|c| match c {
            dissimilar::Chunk::Equal(s) => s.len(),
            _ => 0,
        })
        .sum();
    // 2 * lcs / (len_a + len_b) — standard LCS-based ratio in [0, 1].
    (2.0 * lcs_len as f64) / total as f64
}

/// Word-overlap heuristic near-matches. Lines that share at least half of the
/// needle's "long" words (len > 3) are considered near-matches. Used as a
/// fallback to catch cases the fuzzy window search misses.
fn word_overlap_near_matches(content: &str, needle: &str) -> Vec<(usize, String)> {
    let needle_words: Vec<&str> = needle.split_whitespace().filter(|w| w.len() > 3).collect();
    if needle_words.is_empty() {
        return Vec::new();
    }

    content
        .lines()
        .enumerate()
        .filter(|(_, line)| {
            let line_lower = line.to_lowercase();
            needle_words
                .iter()
                .filter(|w| line_lower.contains(*w))
                .count()
                >= (needle_words.len() / 2).max(1)
        })
        .map(|(i, line)| (i + 1, line.chars().take(60).collect()))
        .take(5)
        .collect()
}

/// Verify the edit was applied and return a brief confirmation.
/// Does NOT echo the full edited content back — the model already knows what
/// it wrote. Just confirms the line range and shows a small preview.
fn verify_edit(new_content: &str, new_string: &str) -> String {
    if let Some(pos) = new_content.find(new_string) {
        let start_line = new_content[..pos].lines().count() + 1;
        let line_count = new_string.lines().count().max(1);
        let end_line = start_line + line_count - 1;

        // Show just the first and last line of the edit as a quick sanity check
        let first_line = new_string.lines().next().unwrap_or("");
        let preview: String = first_line.chars().take(80).collect();

        if line_count == 1 {
            format!("Verified: line {start_line}: {preview}")
        } else {
            let last_line = new_string.lines().last().unwrap_or("");
            let last_preview: String = last_line.chars().take(80).collect();
            format!(
                "Verified: lines {start_line}-{end_line} ({line_count} lines)\n  {start_line}: {preview}\n  {end_line}: {last_preview}"
            )
        }
    } else {
        // new_string not found (e.g. it was a deletion) — confirm the edit applied
        "Verified: edit applied (new_string not in result — likely a deletion)".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_util;
    use super::*;
    use std::io::Write;

    fn make_test_file(dir: &test_util::TestDir, name: &str, content: &str) -> String {
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        write!(f, "{content}").unwrap();
        name.to_string()
    }

    // --- similarity_score unit tests ---

    #[test]
    fn similarity_identical_strings_is_one() {
        assert!((similarity_score("hello world", "hello world") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn similarity_disjoint_strings_is_zero() {
        assert!((similarity_score("abc", "xyz")).abs() < 1e-9);
    }

    #[test]
    fn similarity_partial_overlap_in_between() {
        let s = similarity_score("hello world", "hullo world");
        assert!(s > 0.8 && s < 1.0, "expected ~0.9, got {s}");
    }

    #[test]
    fn similarity_empty_strings_is_one() {
        assert!((similarity_score("", "") - 1.0).abs() < 1e-9);
    }

    // --- fuzzy near-match tests ---

    #[test]
    fn fuzzy_finds_near_match_with_small_typo() {
        let content = "fn hello_world() {\n    println!(\"hi\");\n}\n";
        // Needle has a small typo: hellp vs hello
        let needle = "fn hellp_world() {";
        let matches = find_near_matches(content, needle);
        assert!(!matches.is_empty(), "should find a near match");
        assert!(matches[0].0 == 1, "should match on line 1");
        assert!(
            matches[0].2 > NEAR_MATCH_THRESHOLD,
            "score should exceed threshold"
        );
    }

    #[test]
    fn fuzzy_returns_empty_for_unrelated_content() {
        let content = "fn foo() {}\nfn bar() {}\n";
        let needle = "struct CompletelyUnrelated { x: i32 }";
        let matches = find_near_matches(content, needle);
        // No near-matches should be found for unrelated content.
        assert!(
            matches.is_empty(),
            "expected no near matches, got {matches:?}"
        );
    }

    #[test]
    fn fuzzy_matches_multiline_block_with_drift() {
        let content = "fn old_name() {\n    // does a thing\n    return 42;\n}\n";
        // Needle has slight drift: missing the comment line.
        let needle = "fn old_name() {\n    return 42;\n}\n";
        let matches = find_near_matches(content, needle);
        assert!(!matches.is_empty(), "should find a near match");
        assert!(matches[0].0 == 1, "should match starting at line 1");
    }

    // --- edit tests (all via the 'edits' array form) ---

    #[test]
    fn edits_file_successfully() {
        let dir = test_util::unique_test_dir();
        let name = make_test_file(&dir, "test.txt", "hello world\nfoo bar\n");
        let tool = EditTool;
        let args = json!({
            "edits": [
                {"path": name, "old_string": "hello world", "new_string": "hello universe"}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Successfully edited"));

        let content = std::fs::read_to_string(dir.path().join(&name)).unwrap();
        assert!(content.contains("hello universe"));
        assert!(!content.contains("hello world"));
    }

    #[test]
    fn errors_on_no_match() {
        let dir = test_util::unique_test_dir();
        let name = make_test_file(&dir, "test.txt", "hello world\n");
        let tool = EditTool;
        let args = json!({
            "edits": [
                {"path": name, "old_string": "nonexistent text", "new_string": "replacement"}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("not found"));

        let content = std::fs::read_to_string(dir.path().join(&name)).unwrap();
        assert!(content.contains("hello world"));
    }

    #[test]
    fn errors_on_multiple_matches() {
        let dir = test_util::unique_test_dir();
        let name = make_test_file(&dir, "test.txt", "dup\ndup\ndup\n");
        let tool = EditTool;
        let args = json!({
            "edits": [
                {"path": name, "old_string": "dup", "new_string": "unique"}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("matches 3 times"));
    }

    #[test]
    fn read_after_edit_verification_included() {
        let dir = test_util::unique_test_dir();
        let name = make_test_file(&dir, "test.txt", "fn old_name() {}\n");
        let tool = EditTool;
        let args = json!({
            "edits": [
                {"path": name, "old_string": "fn old_name() {}", "new_string": "fn new_name() {\n    // renamed\n}"}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Verified"));
        assert!(result.contains("new_name"));
    }

    #[test]
    fn rejects_absolute_path() {
        let dir = test_util::unique_test_dir();
        let tool = EditTool;
        let args = json!({
            "edits": [
                {"path": "/etc/passwd", "old_string": "x", "new_string": "y"}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("absolute paths are not allowed"));
    }

    #[test]
    fn no_match_includes_fuzzy_near_match_with_score() {
        let dir = test_util::unique_test_dir();
        let name = make_test_file(
            &dir,
            "test.txt",
            "fn calculate_total() -> i32 {\n    42\n}\n",
        );
        let tool = EditTool;
        // Small typo: calculat_total vs calculate_total
        let args = json!({
            "edits": [
                {"path": name, "old_string": "fn calculat_total() -> i32 {", "new_string": "fn compute_total() -> i32 {"}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("not found"));
        assert!(result.contains("Near matches"));
        assert!(result.contains("similarity"));
    }

    #[test]
    fn edits_single_file_multiple_edits_in_order() {
        let dir = test_util::unique_test_dir();
        let name = make_test_file(&dir, "test.txt", "alpha\nbeta\ngamma\n");
        let tool = EditTool;
        let args = json!({
            "edits": [
                {"path": name, "old_string": "alpha", "new_string": "ALPHA"},
                {"path": name, "old_string": "beta", "new_string": "BETA"}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Applied 2 edit(s)"));
        assert!(result.contains("ALPHA"));
        assert!(result.contains("BETA"));

        let content = std::fs::read_to_string(dir.path().join(&name)).unwrap();
        assert_eq!(content, "ALPHA\nBETA\ngamma\n");
    }

    #[test]
    fn edits_multiple_files_concurrently() {
        let dir = test_util::unique_test_dir();
        let a = make_test_file(&dir, "a.txt", "one\ntwo\n");
        let b = make_test_file(&dir, "b.txt", "three\nfour\n");
        let tool = EditTool;
        let args = json!({
            "edits": [
                {"path": a, "old_string": "one", "new_string": "ONE"},
                {"path": b, "old_string": "three", "new_string": "THREE"}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Applied 2 edit(s)"));

        let ca = std::fs::read_to_string(dir.path().join(&a)).unwrap();
        let cb = std::fs::read_to_string(dir.path().join(&b)).unwrap();
        assert_eq!(ca, "ONE\ntwo\n");
        assert_eq!(cb, "THREE\nfour\n");
    }

    #[test]
    fn reports_per_edit_errors_without_blocking_others() {
        let dir = test_util::unique_test_dir();
        let ok = make_test_file(&dir, "ok.txt", "good content\n");
        let bad = make_test_file(&dir, "bad.txt", "other content\n");
        let tool = EditTool;
        let args = json!({
            "edits": [
                {"path": ok, "old_string": "good content", "new_string": "GREAT"},
                {"path": bad, "old_string": "nonexistent", "new_string": "whatever"}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Applied 2 edit(s)"));
        // The successful edit should be reflected in the file.
        let c = std::fs::read_to_string(dir.path().join(&ok)).unwrap();
        assert_eq!(c, "GREAT\n");
        // The failed edit should report an error inline, and the bad file untouched.
        assert!(result.contains("not found"));
        let cb = std::fs::read_to_string(dir.path().join(&bad)).unwrap();
        assert_eq!(cb, "other content\n");
    }

    #[test]
    fn sequential_edits_within_file_compose_correctly() {
        // An earlier edit can shift text that a later edit references. Verify
        // the second edit still applies correctly because both run against the
        // same in-memory buffer in order.
        let dir = test_util::unique_test_dir();
        let name = make_test_file(&dir, "test.txt", "fn old() {}\n");
        let tool = EditTool;
        let args = json!({
            "edits": [
                {"path": name, "old_string": "fn old() {}", "new_string": "fn renamed() {\n    body\n}"},
                {"path": name, "old_string": "    body", "new_string": "    new_body"}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Applied 2 edit(s)"));
        let content = std::fs::read_to_string(dir.path().join(&name)).unwrap();
        assert!(content.contains("renamed"));
        assert!(content.contains("new_body"));
        assert!(!content.contains("    body\n"));
    }

    #[test]
    fn rejects_empty_edits_array() {
        let dir = test_util::unique_test_dir();
        let tool = EditTool;
        let args = json!({"edits": []});
        let result = tool.execute(&args, dir.as_str());
        assert!(result.is_err());
    }

    #[test]
    fn rejects_absolute_path_inline() {
        let dir = test_util::unique_test_dir();
        let ok = make_test_file(&dir, "ok.txt", "good\n");
        let tool = EditTool;
        let args = json!({
            "edits": [
                {"path": ok, "old_string": "good", "new_string": "GREAT"},
                {"path": "/etc/passwd", "old_string": "x", "new_string": "y"}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        // The good edit still applies.
        assert!(result.contains("Applied 2 edit(s)"));
        assert!(result.contains("GREAT"));
        // The absolute path is rejected inline.
        assert!(result.contains("absolute paths are not allowed"));
        let c = std::fs::read_to_string(dir.path().join(&ok)).unwrap();
        assert_eq!(c, "GREAT\n");
    }

    #[test]
    fn preserves_edit_order_in_output() {
        let dir = test_util::unique_test_dir();
        let a = make_test_file(&dir, "a.txt", "1\n");
        let b = make_test_file(&dir, "b.txt", "2\n");
        let c = make_test_file(&dir, "c.txt", "3\n");
        let tool = EditTool;
        let args = json!({
            "edits": [
                {"path": c, "old_string": "3", "new_string": "C"},
                {"path": a, "old_string": "1", "new_string": "A"},
                {"path": b, "old_string": "2", "new_string": "B"}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        // Output should preserve the original edit order: C, A, B.
        let c_pos = result.find("[edit 0]").unwrap();
        let a_pos = result.find("[edit 1]").unwrap();
        let b_pos = result.find("[edit 2]").unwrap();
        assert!(c_pos < a_pos);
        assert!(a_pos < b_pos);
    }

    #[test]
    fn falls_back_to_top_level_path_when_edits_lack_path() {
        // Models sometimes place `path` at the top level instead of inside each
        // edit object. The tool should apply it as a default to every edit
        // rather than failing, so the work proceeds without a retry round-trip.
        let dir = test_util::unique_test_dir();
        let name = make_test_file(&dir, "test.txt", "alpha\nbeta\n");
        let tool = EditTool;
        let args = json!({
            "path": name,
            "edits": [
                {"old_string": "alpha", "new_string": "ALPHA"},
                {"old_string": "beta", "new_string": "BETA"}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Applied 2 edit(s)"));
        assert!(result.contains("ALPHA"));
        assert!(result.contains("BETA"));

        let content = std::fs::read_to_string(dir.path().join(&name)).unwrap();
        assert_eq!(content, "ALPHA\nBETA\n");
    }

    #[test]
    fn top_level_path_ignored_when_edits_have_their_own_paths() {
        // When edits have their own `path` fields, a top-level `path` is
        // ignored — it should not override the per-edit paths.
        let dir = test_util::unique_test_dir();
        let a = make_test_file(&dir, "a.txt", "one\n");
        let b = make_test_file(&dir, "b.txt", "two\n");
        let tool = EditTool;
        let args = json!({
            "path": "a.txt",
            "edits": [
                {"path": a, "old_string": "one", "new_string": "ONE"},
                {"path": b, "old_string": "two", "new_string": "TWO"}
            ]
        });
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("Applied 2 edit(s)"));
        let ca = std::fs::read_to_string(dir.path().join(&a)).unwrap();
        let cb = std::fs::read_to_string(dir.path().join(&b)).unwrap();
        assert_eq!(ca, "ONE\n");
        assert_eq!(cb, "TWO\n");
    }

    #[test]
    fn errors_with_clear_message_when_path_missing_everywhere() {
        // When no edit has a `path` AND there's no top-level `path`, the error
        // must clearly explain that `path` goes inside each edit object.
        let dir = test_util::unique_test_dir();
        let tool = EditTool;
        let args = json!({
            "edits": [
                {"old_string": "alpha", "new_string": "ALPHA"}
            ]
        });
        let result = tool.execute(&args, dir.as_str());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("inside the edit object"));
        assert!(err.contains("Do not put 'path' at the top level"));
    }
}
