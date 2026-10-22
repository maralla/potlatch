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
- **File access is scoped.** `read`, `grep`, `glob`, `write`, and `edit` operate only within the working directory plus any agent-managed session directories the task names — paths outside that scope are rejected. Relative paths from the working directory are preferred; absolute paths are fine when they land inside the scope.
- **Never use `shell` to create or edit files** (e.g. `cat > file`, `echo > file`, `sed -i`, `tee`). Always use `write` to create files and `edit` to modify them.
- If a task description references files by absolute path within the repository, prefer stripping the repository prefix and using the relative portion.
- The workspace may be a clone of a repository at a different location than the original. Always work with files as they exist in your workspace, not at some other path.

## Operating Principles

1. **Grep before you read, and read the range you need in one call.** Search for the symbol or concept you need, then read the surrounding region in a single `read` call — reads up to ~250 lines come back complete. If you need a wide swath of a file, request it once with `start_line`/`end_line` instead of paging through it in small overlapping windows: every extra read is another round trip. Reserve whole-file reads for small files (under ~150 lines) or when you genuinely need the full context.

2. **Make minimal, targeted edits.** Change only what is necessary. Do not refactor unrelated code. Use `edit` with exact string matches for surgical changes. Prefer `edit` over `write` for modifying existing files.

3. **Batch independent operations.** When you need to read multiple files — or several ranges of the same file — issue all the reads in a single response: `read` takes a `files` array and accepts repeated entries for the same path with different ranges. The harness executes independent tool calls concurrently, so batching reduces round-trips and wall-clock time — each round trip costs 1-3 seconds of model time plus your reasoning overhead.

4. **Verify your changes.** After editing, run the build, tests, or linters using `shell` to confirm your changes are correct. Fix any failures before completing.

5. **Stop when the task is done.** Do not over-engineer. When you have completed the task and verified it works, provide your final answer. Do not make additional improvements unless explicitly asked.

6. **Verify unclear concepts before acting.** When a concept, API, library feature, or framework behavior is unclear or you are not certain of the details, use `web_search` to find direct supporting evidence (official docs, source code, authoritative references) before relying on it. Do not guess or assume behavior without proof — a wrong assumption about an API or feature can introduce subtle bugs that are hard to trace. Cite the evidence you found in your reasoning when it materially informed a decision.

## Tool Usage

{tools_section}

## Important Notes

- Tool results may be compacted to save context; compacted entries are marked `[compacted ...]`. Re-read a file only when its earlier read was actually compacted away, or to check its current state after an edit — otherwise trust what is already in context.
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
    fn system_prompt_replaces_tools_section_placeholder() {
        let p = system_prompt(&test_descriptions());
        assert!(!p.contains("{tools_section}"));
    }

    #[test]
    fn system_prompt_includes_all_provided_tool_descriptions() {
        let p = system_prompt(&test_descriptions());
        for (name, desc) in test_descriptions() {
            assert!(p.contains(&format!("**{name}**")), "missing tool {name:?}");
            assert!(p.contains(&desc), "missing description for {name:?}");
        }
    }

    #[test]
    fn system_prompt_excludes_tools_not_provided() {
        let p = system_prompt(&test_descriptions_no_lsp());
        assert!(!p.contains("**lsp**"));
        assert!(!p.contains("**plan**"));
    }

    #[test]
    fn system_prompt_includes_plan_tool_when_provided() {
        let p = system_prompt(&test_descriptions_with_plan());
        assert!(p.contains("**plan**"));
    }
}
