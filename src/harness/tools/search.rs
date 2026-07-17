//! Search tools: grep (powered by ripgrep library) and glob (file pattern matching).

use anyhow::Result;
use grep_regex::RegexMatcherBuilder;
use grep_searcher::{SearcherBuilder, Sink, SinkMatch};
use ignore::WalkBuilder;
use serde_json::{Value, json};
use std::sync::Mutex;

use super::Tool;

pub struct GrepTool;

impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Search file contents using a regular expression (powered by ripgrep). Returns matching lines with file paths and line numbers. Respects .gitignore. Use this to find relevant code before reading entire files. Supports context lines around matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in (default: working directory). May be an absolute path to a location explicitly referenced in the task context."
                    },
                    "include": {
                        "type": "string",
                        "description": "File glob pattern to include (e.g. '*.rs'). Optional."
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines to show before and after each match. Default: 0."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matching lines to return. Default: 100."
                    }
                },
                "required": ["pattern"]
            }
        })
    }

    fn execute(&self, args: &Value, cwd: &str) -> Result<String> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing 'pattern' argument"))?;
        let search_path = args["path"].as_str().unwrap_or(".");
        let include = args["include"].as_str();
        let context_lines = args["context_lines"].as_u64().unwrap_or(0) as usize;
        let max_results = args["max_results"].as_u64().unwrap_or(100) as usize;

        let full_path = match super::resolve_read_path(search_path, cwd) {
            Ok(p) => p,
            Err(msg) => return Ok(format!("Error: {msg}")),
        };

        let matcher = RegexMatcherBuilder::new()
            .case_insensitive(false)
            .build(pattern)
            .map_err(|e| anyhow::anyhow!("invalid regex: {e}"))?;

        let collector = Mutex::new(MatchCollector::new(max_results));

        let mut builder = WalkBuilder::new(&full_path);
        if let Some(glob_pattern) = include {
            let mut overrides = ignore::overrides::OverrideBuilder::new(&full_path);
            overrides
                .add(glob_pattern)
                .map_err(|e| anyhow::anyhow!("invalid include glob: {e}"))?;
            let overrides = overrides
                .build()
                .map_err(|e| anyhow::anyhow!("invalid include glob: {e}"))?;
            builder.overrides(overrides);
        }
        let walker = builder
            .hidden(true)
            .git_ignore(true)
            .git_exclude(true)
            .build();

        let mut searcher = SearcherBuilder::new()
            .line_number(true)
            .before_context(context_lines)
            .after_context(context_lines)
            .build();

        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                continue;
            }

            let path = entry.path();
            let mut sink = CollectorSink {
                path: path.to_string_lossy().to_string(),
                collector: &collector,
            };

            if searcher.search_path(&matcher, path, &mut sink).is_err() {
                continue;
            }

            if collector.lock().unwrap().should_stop() {
                break;
            }
        }

        let results = collector.into_inner().unwrap().results;

        if results.is_empty() {
            return Ok(format!(
                "No matches found for pattern '{pattern}' in {search_path}."
            ));
        }

        let mut output = format!("Found {} match(es):\n\n", results.len());
        for (file, line_num, line) in &results {
            output.push_str(&format!("{file}:{line_num}: {line}\n"));
        }
        Ok(output)
    }
}

struct MatchCollector {
    results: Vec<(String, usize, String)>,
    max_results: usize,
}

impl MatchCollector {
    fn new(max_results: usize) -> Self {
        Self {
            results: Vec::new(),
            max_results,
        }
    }

    fn should_stop(&self) -> bool {
        self.results.len() >= self.max_results
    }
}

struct CollectorSink<'a> {
    path: String,
    collector: &'a Mutex<MatchCollector>,
}

impl<'a> Sink for CollectorSink<'a> {
    type Error = std::io::Error;

    fn matched(
        &mut self,
        _searcher: &grep_searcher::Searcher,
        mat: &SinkMatch<'_>,
    ) -> Result<bool, Self::Error> {
        let mut collector = self.collector.lock().unwrap();
        if collector.should_stop() {
            return Ok(false);
        }

        let line_num = mat.line_number().unwrap_or(0) as usize;
        let line_text = String::from_utf8_lossy(mat.bytes()).trim_end().to_string();

        collector
            .results
            .push((self.path.clone(), line_num, line_text));

        Ok(!collector.should_stop())
    }
}

pub struct GlobTool;

impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn schema(&self) -> Value {
        json!({
            "description": "Find files matching a glob pattern (e.g. '**/*.rs', 'src/**/*.ts'). Returns matching file paths. Respects .gitignore. Use this to discover files by name pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files (e.g. '**/*.rs', 'src/*.py')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: working directory). May be an absolute path to a location explicitly referenced in the task context."
                    }
                },
                "required": ["pattern"]
            }
        })
    }

    fn execute(&self, args: &Value, cwd: &str) -> Result<String> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing 'pattern' argument"))?;
        let search_path = args["path"].as_str().unwrap_or(".");

        let full_path = match super::resolve_read_path(search_path, cwd) {
            Ok(p) => p,
            Err(msg) => return Ok(format!("Error: {msg}")),
        };

        let mut builder = WalkBuilder::new(&full_path);
        let mut overrides = ignore::overrides::OverrideBuilder::new(&full_path);
        overrides
            .add(pattern)
            .map_err(|e| anyhow::anyhow!("invalid glob: {e}"))?;
        let overrides = overrides
            .build()
            .map_err(|e| anyhow::anyhow!("invalid glob: {e}"))?;

        let walker = builder
            .overrides(overrides)
            .hidden(true)
            .git_ignore(true)
            .git_exclude(true)
            .build();

        let mut results = Vec::new();
        for entry in walker.flatten() {
            if entry.file_type().is_some_and(|ft| ft.is_file()) {
                results.push(entry.path().to_string_lossy().to_string());
            }
        }

        if results.is_empty() {
            return Ok(format!(
                "No files matching '{pattern}' found in {search_path}."
            ));
        }

        let mut output = format!("Found {} file(s) matching '{pattern}':\n\n", results.len());
        for path in &results {
            output.push_str(path);
            output.push('\n');
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_util;
    use super::*;

    #[test]
    fn grep_finds_matches_via_ripgrep_library() {
        let dir = test_util::unique_test_dir();
        std::fs::write(
            dir.path().join("test.rs"),
            "fn hello() {}\nfn world() {}\nfn hello_world() {}\n",
        )
        .unwrap();

        let tool = GrepTool;
        let args = json!({"pattern": "hello"});
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("hello"));
        assert!(result.contains("test.rs"));
    }

    #[test]
    fn grep_returns_no_matches() {
        let dir = test_util::unique_test_dir();
        std::fs::write(dir.path().join("empty.txt"), "nothing here\n").unwrap();

        let tool = GrepTool;
        let args = json!({"pattern": "nonexistent_function_xyz"});
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("No matches"));
    }

    #[test]
    fn glob_finds_files() {
        let dir = test_util::unique_test_dir();
        std::fs::File::create(dir.path().join("a.rs")).unwrap();
        std::fs::File::create(dir.path().join("b.rs")).unwrap();
        std::fs::File::create(dir.path().join("c.txt")).unwrap();

        let tool = GlobTool;
        let args = json!({"pattern": "*.rs"});
        let result = tool.execute(&args, dir.as_str()).unwrap();
        assert!(result.contains("a.rs"));
        assert!(result.contains("b.rs"));
        assert!(!result.contains("c.txt"));
    }

    #[test]
    fn grep_searches_absolute_path_outside_workspace() {
        // Searching an absolute path outside the cwd must be allowed when the
        // location is explicitly provided.
        let external = test_util::unique_test_dir();
        std::fs::write(external.path().join("target.rs"), "fn needle() {}\n").unwrap();

        let cwd = test_util::unique_test_dir();
        let tool = GrepTool;
        let args = json!({"pattern": "needle", "path": external.as_str()});
        let result = tool.execute(&args, cwd.as_str()).unwrap();
        assert!(result.contains("needle"));
        assert!(result.contains("target.rs"));
    }

    #[test]
    fn glob_searches_absolute_path_outside_workspace() {
        let external = test_util::unique_test_dir();
        std::fs::File::create(external.path().join("match.rs")).unwrap();

        let cwd = test_util::unique_test_dir();
        let tool = GlobTool;
        let args = json!({"pattern": "*.rs", "path": external.as_str()});
        let result = tool.execute(&args, cwd.as_str()).unwrap();
        assert!(result.contains("match.rs"));
    }
}
