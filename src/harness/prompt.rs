//! Frontier-tier system prompt for the harness agent.
//!
//! The system prompt encodes the agent's behavioral contract only. It does NOT
//! concatenate `AGENTS.md` or any other project-specific file — the agent is
//! expected to read `AGENTS.md` from disk itself when its task prompt instructs
//! it to (this keeps behavior consistent across first-party and third-party ACP
//! servers).

/// The base system prompt that defines the agent's operating principles.
///
/// This is not a placeholder. It encodes the behavioral contract for a frontier
/// coding agent: explore before editing, make targeted changes, verify with
/// tests/build, and stop when done.
const BASE_PROMPT: &str = r#"You are a frontier-tier coding agent. You operate within a workspace directory — the current working directory.

## Workspace

Your workspace is the current working directory. It is the root of the repository you are operating on.

- The `shell` tool runs commands in the working directory by default.
- Use `shell` with `ls` or `find .` to discover the workspace structure if needed.
- **Writes are sandboxed.** `edit` and `write` should only operate on paths within the working directory.
- **Never use `shell` to create or edit files** (e.g. `cat > file`, `echo > file`, `sed -i`, `tee`). Always use `write` to create files and `edit` to modify them.
- **Reads are not sandboxed.** `read`, `grep`, and `glob` may follow absolute paths or paths outside the workspace when a file or directory is explicitly referenced in the task context. Use this only to read context provided by the task — never to pull in unrelated files.
- If a task description references files by absolute path within the repository, prefer stripping the repository prefix and using the relative portion.
- The workspace may be a clone of a repository at a different location than the original. Always work with files as they exist in your workspace, not at some other path.

## Operating Principles

1. **Use `lsp` for code navigation, `grep` for text search.** The `lsp` tool is more precise and faster than `grep` for finding where a symbol is defined (`operation: "definition"`), who calls it (`operation: "references"`), its type signature (`operation: "hover"`), or compile errors (`operation: "diagnostics"`). First discover the language server with `shell` (e.g. `which gopls`, `which rust-analyzer`, `which clangd`), then use `lsp` for all code navigation. Reserve `grep` for searching string literals, comments, or non-symbol text.

2. **Grep before you read, and read targeted sections.** Search for the symbol or concept you need, then read only the specific lines around the match with `read`'s `start_line`/`end_line` (typically 30-80 lines). Never read a whole file when a grep + targeted read will do; reserve whole-file reads for small files (under ~150 lines) or when you genuinely need the full context. A 400-line file costs ~4x more tokens than the 100-line section you actually need.

3. **Make minimal, targeted edits.** Change only what is necessary. Do not refactor unrelated code. Use `edit` with exact string matches for surgical changes. Prefer `edit` over `write` for modifying existing files.

4. **Batch independent operations.** When you need to read multiple files or run independent searches, issue all tool calls in a single response rather than sequentially across turns. `read` takes a `files` array, so reading multiple files is one call. The harness executes independent tool calls concurrently, so batching reduces round-trips and wall-clock time — each round trip costs 1-3 seconds of model time plus your reasoning overhead.

5. **Verify your changes.** After editing, run the build, tests, or linters using `shell` to confirm your changes are correct. Fix any failures before completing.

6. **Stop when the task is done.** Do not over-engineer. When you have completed the task and verified it works, provide your final answer. Do not make additional improvements unless explicitly asked.

7. **Verify unclear concepts before acting.** When a concept, API, library feature, or framework behavior is unclear or you are not certain of the details, use `web_search` to find direct supporting evidence (official docs, source code, authoritative references) before relying on it. Do not guess or assume behavior without proof — a wrong assumption about an API or feature can introduce subtle bugs that are hard to trace. Cite the evidence you found in your reasoning when it materially informed a decision.

## Tool Usage

{tools_section}

- **Prefer rendered web access.** When both `web_fetch` and `fetch` are available, use `web_fetch` for web pages and URLs returned by `web_search`. Use `fetch` only when you specifically need a lightweight raw/static response or browser rendering is unnecessary.

## Important Notes

- Tool results may be compacted to save context. If you need exact current file state, re-read the file rather than relying on memory.
- After a failed edit, re-read the file to get the current content before retrying. Don't re-read files you've already read in this session — their content is already in context (or in the conversation summary after compaction). Re-read only to check the *current* state after an edit, or to fetch a section a previous truncated read missed.
- When running shell commands, check the exit code. Non-zero means failure. If a tool returns an error, analyze it and adjust your approach — don't repeat the same failed action.
- **Act, don't narrate. Keep reasoning concise.** Once you've found the code you need, make the edit immediately — the edit itself is the output. Your internal reasoning is never shown to the user; restate only the key decision and the immediate next step. Skip re-deriving known context, restating the task, or long prose ("Let me analyze...", "I notice that..."). Brief, focused thinking ("function `foo` calls `bar` with no nil check → add guard at line 42") beats verbose chains.

## Output

If the task requires a specific output format (like structured data or a specific response), follow it exactly. Do not narrate every step — focus on the result.
"#;

/// Build the system prompt for the harness agent. Tool descriptions are
/// collected from the registered tools' `prompt_description()` methods and
/// injected dynamically — no tool-specific knowledge here.
pub fn system_prompt(tool_descriptions: &[(String, String)]) -> String {
    let tools_section = tool_descriptions
        .iter()
        .map(|(name, desc)| format!("- **{name}**: {desc}"))
        .collect::<Vec<_>>()
        .join("\n");
    BASE_PROMPT.replace("{tools_section}", &tools_section)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_descriptions() -> Vec<(String, String)> {
        vec![
            (
                "grep".into(),
                "Your primary exploration tool. Always run this before `read`.".into(),
            ),
            (
                "read".into(),
                "Read file contents with line numbers. Always pass `start_line`/`end_line` for large files.".into(),
            ),
            (
                "shell".into(),
                "Run any command — build, test, git, etc.".into(),
            ),
            (
                "lsp".into(),
                "Language server queries — definition, references, hover, symbols, diagnostics."
                    .into(),
            ),
        ]
    }

    fn test_descriptions_no_lsp() -> Vec<(String, String)> {
        vec![
            (
                "grep".into(),
                "Your primary exploration tool. Always run this before `read`.".into(),
            ),
            (
                "read".into(),
                "Read file contents with line numbers. Always pass `start_line`/`end_line` for large files.".into(),
            ),
            (
                "shell".into(),
                "Run any command — build, test, git, etc.".into(),
            ),
        ]
    }

    fn test_descriptions_with_plan() -> Vec<(String, String)> {
        let mut d = test_descriptions();
        d.push((
            "plan".into(),
            "Emit a structured JSON plan as your canonical handoff.".into(),
        ));
        d
    }

    #[test]
    fn system_prompt_covers_key_principles() {
        let p = system_prompt(&test_descriptions());
        assert!(p.contains("Grep before you read"));
        assert!(p.contains("minimal, targeted edits"));
        assert!(p.contains("Verify your changes"));
        assert!(p.contains("Stop when the task is done"));
        assert!(p.contains("compacted"));
    }

    #[test]
    fn system_prompt_prefers_web_fetch_for_pages() {
        let p = system_prompt(&test_descriptions());
        assert!(p.contains("When both `web_fetch` and `fetch` are available"));
        assert!(p.contains("use `web_fetch` for web pages"));
    }

    #[test]
    fn system_prompt_promotes_lsp_for_code_navigation() {
        let p = system_prompt(&test_descriptions());
        assert!(
            p.contains("lsp"),
            "prompt should mention lsp for code navigation"
        );
        assert!(p.contains("definition"), "prompt should mention definition");
        assert!(p.contains("references"), "prompt should mention references");
        // LSP should be the first operating principle — mentioned before
        // "Grep before you read".
        let lsp_principle_pos = p.find("Use `lsp` for code navigation").unwrap();
        let grep_principle_pos = p.find("Grep before you read").unwrap();
        assert!(
            lsp_principle_pos < grep_principle_pos,
            "lsp principle should come before grep principle"
        );
    }

    #[test]
    fn system_prompt_encourages_concise_reasoning() {
        let p = system_prompt(&test_descriptions());
        assert!(
            p.contains("Keep reasoning concise"),
            "prompt should instruct the model to keep reasoning brief"
        );
        assert!(p.contains("never shown to the user"));
    }

    #[test]
    fn system_prompt_encourages_batched_reads() {
        let p = system_prompt(&test_descriptions());
        assert!(p.contains("files"));
        assert!(p.contains("Batch independent operations"));
        assert!(p.contains("Grep before you read"));
    }

    #[test]
    fn system_prompt_prescribes_grep_first_exploration() {
        let p = system_prompt(&test_descriptions());
        // grep is the primary exploration tool — must be mentioned before read.
        let grep_pos = p.find("**grep**").unwrap();
        let read_pos = p.find("**read**").unwrap();
        assert!(grep_pos < read_pos, "grep should be listed before read");
        assert!(p.contains("Always run this before `read`"));
    }

    #[test]
    fn system_prompt_discourages_whole_file_reads_for_large_files() {
        let p = system_prompt(&test_descriptions());
        assert!(p.contains("Always pass `start_line`/`end_line` for large files"));
        assert!(!p.contains("Read whole files by omitting"));
    }

    #[test]
    fn system_prompt_encourages_acting_over_narrating() {
        let p = system_prompt(&test_descriptions());
        assert!(p.contains("Act, don't narrate"));
        assert!(p.contains("make the edit immediately"));
        assert!(p.contains("Brief, focused thinking"));
    }

    #[test]
    fn system_prompt_discourages_re_reading_files() {
        let p = system_prompt(&test_descriptions());
        assert!(p.contains("Don't re-read files you've already read"));
        assert!(p.contains("conversation summary"));
    }

    #[test]
    fn system_prompt_requires_evidence_for_unclear_concepts() {
        let p = system_prompt(&test_descriptions());
        assert!(
            p.contains("Verify unclear concepts before acting"),
            "prompt should instruct the model to verify unclear concepts"
        );
        assert!(
            p.contains("web_search"),
            "prompt should name web_search as the evidence source"
        );
        assert!(
            p.contains("Do not guess or assume behavior without proof"),
            "prompt should forbid guessing without proof"
        );
        assert!(
            p.contains("direct supporting evidence"),
            "prompt should require direct supporting evidence"
        );
    }

    #[test]
    fn system_prompt_includes_plan_tool_when_provided() {
        let p = system_prompt(&test_descriptions_with_plan());
        assert!(p.contains("**plan**"));
        assert!(p.contains("canonical handoff"));
    }

    #[test]
    fn system_prompt_excludes_plan_tool_when_not_provided() {
        let p = system_prompt(&test_descriptions_no_lsp());
        assert!(!p.contains("**plan**"));
        assert!(!p.contains("canonical handoff"));
    }

    #[test]
    fn system_prompt_includes_lsp_tool_when_provided() {
        let p = system_prompt(&test_descriptions());
        assert!(p.contains("**lsp**"));
        assert!(p.contains("definition"));
    }

    #[test]
    fn system_prompt_excludes_lsp_tool_when_not_provided() {
        let p = system_prompt(&test_descriptions_no_lsp());
        assert!(!p.contains("**lsp**"));
    }

    #[test]
    fn system_prompt_does_not_concatenate_agents_md() {
        let p = system_prompt(&test_descriptions());
        assert!(!p.contains("Agent Instructions"));
        assert!(!p.contains("Code Quality"));
        assert!(!p.contains("For Worker Agent"));
        assert!(!p.contains("Project Instructions"));
    }
}
