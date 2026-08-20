# Potlatch

Potlatch is a fully automatic agentic platform for orchestrating autonomous AI agents across workflows.

## Overview

Potlatch runs several AI agent roles in parallel in a fully automated, non-interactive mode:
- **Worker Agent**: Fetches issues, implements features, and creates merge requests autonomously
- **Reviewer Agent**: Reviews merge requests and either merges them automatically or leaves an approval comment, depending on config
- **PMO Agent**: Triages `action-required` issues (typically after a worker could not finish) and may split work or guide the worker
- **OPS Agent**: Analyzes SSH log files and Grafana Elasticsearch log sources and files actionable issues

All roles operate without requiring any user input, making autonomous decisions based on the code and project documentation.

## Prerequisites

1. **Rust toolchain** (`cargo`, stable) — to build Potlatch from source
2. **Cursor Agent CLI** — The `agent` command on your `PATH` is **Cursor’s agent CLI**. Potlatch keeps **one** long-lived **`agent acp`** subprocess per role (optional **`--model`** first, then **`--print`**, **`--trust`**, **`--force`**, **`--approve-mcps`**). On the first task it runs **`initialize`**, then **`authenticate`** with **`cursor_login`** when Cursor advertises it (per [Cursor ACP](https://cursor.com/docs/cli/acp); use **`agent login`** or **`CURSOR_API_KEY`** / **`CURSOR_AUTH_TOKEN`**), then **`session/new`**; on **each later task** it calls **`session/close`** (best effort) then **`session/new`** again on the **same** stdio link, sends **`session/prompt`**, and leaves the child running so the next task still gets a **clean session** (no prior in-agent chat; workflow continuity stays in Potlatch’s own state files). Model selection follows [ACP Session Config Options](https://agentclientprotocol.com/protocol/session-config-options): if **`configOptions`** includes a model selector and your id is in **`options`**, Potlatch calls **`session/set_config_option`**; otherwise it tries experimental **`session/set_model`**. Completions come from streamed **`session/update`** chunks (including [slash-command](https://agentclientprotocol.com/protocol/slash-commands) and [session mode](https://agentclientprotocol.com/protocol/session-modes) updates). Potlatch tracks **`modes`** / **`configOptions`** and logs at **`RUST_LOG=debug`** (`potlatch::acp_modes`). For unattended runs it also answers **`session/request_permission`** by selecting an **`optionId`** from the agent’s **`options`** list (preferring **`allow_always`**, then **`allow_once`**) and auto-approves **`cursor/create_plan`** [ACP extensions](https://cursor.com/docs/cli/acp). For **`cursor/ask_question`**, worker and reviewer use a simple automatic reply; the **PMO** (when **`cursor_ask_via_gitlab`** is enabled) posts the question on the GitLab issue, sets **`pmo-pending`**, and waits for a **direct thread reply** on that note (`RUST_LOG=debug`: **`potlatch::acp_cursor`**). The **reviewer** may request **`ask`** and the **PMO** **`plan`**. **By default** Potlatch does **not** start its MCP HTTP server and does **not** write `.cursor/mcp.json`. To enable the optional Potlatch MCP endpoint and generated config for extra MCP tools, set **`[mcp] enabled = true`** in `potlatch.toml`.
3. **GitLab CLI** (`glab`) - For GitLab operations
4. **Git** - For repository operations

## Security Note

Potlatch automatically trusts the cloned repository directories (`*-worker`, `*-reviewer`, `*-pmo`, etc.) by passing the `--trust` flag to the agent CLI. This is necessary for non-interactive automation. Only use Potlatch with repositories you trust, as the AI agent will have full access to execute code and modify files in these directories.

## Build

```bash
cargo build --release
```

## Usage

### Basic Usage

```bash
potlatch run
```

Set `gitlab_repo` in `potlatch.toml` (see example below), then start only the agents you configure under `[agent.*]` sections.

Example config:

```toml
gitlab_repo = "https://gitlab.com/username/project"

[agent.worker]
poll_interval_secs = 60

[agent.reviewer]
poll_interval_secs = 120
merge_when_approved = true
```

### With Configuration File

Generate an example config file:

```bash
potlatch init-config
```

This creates `potlatch.toml`. Edit it to configure models, polling intervals, and reviewer merge behavior:

```toml
gitlab_repo = "https://gitlab.com/username/project"

[agent.worker]
# model = "acp://cursor/composer-2"
poll_interval_secs = 60

[agent.reviewer]
# model = "acp://cursor/gpt-5.3-codex"
poll_interval_secs = 120
merge_when_approved = true
```

Then run:

```bash
potlatch run --config potlatch.toml
```

The command will:
1. Clone the repository to `<project-name>-<role>-<n>` directories (worker, reviewer, PMO, …)
2. Start the configured agents with the specified models
3. Run continuously, displaying periodic status updates
4. Never require user input

## How It Works

### Worker Agent

1. Fetches open issues from GitLab
2. Skips issues that are:
   - Marked as `[Draft]`
   - Have `do-not-implement` label
   - Have the `pending` label (human pause — issue stays open; see Labels)
   - Have the `review-only` label (reviewer-only workflow — see Labels)
   - Already have `in-progress` label
3. For each issue:
   - Looks for an open merge request whose description contains `Closes #<issue>` (case-insensitive) before falling back to branch `issue-<number>`
   - If a linked MR is already merged, closes the issue and moves on
   - Checks if an MR already exists for this issue (from session file or GitLab)
   - If MR exists: skips implementation and continues tracking the MR
   - If no MR: proceeds with implementation
   - Checks if a branch `issue-<number>` already exists
   - If branch exists: checks it out and continues previous work
   - If branch doesn't exist: creates new branch from main/master
   - Adds `in-progress` label
   - Analyzes if the issue is clear enough to implement
   - If unclear: adds a comment explaining what's needed (no interactive prompts)
   - If clear: implements the feature autonomously using AI
   - Makes reasonable assumptions for minor unclear details
   - Generates MR title summarizing the changes
   - Creates detailed MR description with Goal, Implementation, and Testing sections
   - Creates merge request with meaningful title and description
4. If the `pending` label is added while the worker holds an issue, it releases its claim and stops watching the MR (no issue close). When `pending` is removed, the worker can claim again and resume from the linked MR if present. If `review-only` is added, the worker also releases its claim/session and stops tracking the issue and related MR, leaving review/merge handling to the reviewer.
5. Tracks active merge requests and monitors for reviewer feedback
   - Checks for new comments on active MRs
   - Automatically addresses reviewer feedback
   - Makes necessary code changes based on comments
   - Pushes updates and adds summary comment
6. When an active MR merges, closes the linked issue (best effort if GitLab did not already).
7. Saves session summaries to `issue_<number>.md` files with MR information

**Key Features**: 
- The worker never asks for user input. It makes all decisions autonomously based on the issue description and project documentation.
- Automatically resumes work on existing branches, preventing rework if the agent was interrupted or needs to continue implementation.
- Tracks MR state across restarts - if an MR is already created for an issue, the worker skips re-implementation and just monitors the MR.
- Responds to reviewer feedback automatically - when the reviewer adds comments, the worker addresses them and pushes updates.

### Reviewer Agent

1. Fetches open merge requests from GitLab
2. For each MR:
   - Checks if there are unresolved comments (skips if yes)
   - Reviews the code changes autonomously
   - Checks if implementation matches requirements
   - Verifies tests and lints pass
   - Makes autonomous decisions:
     - Approves and merges if everything is good, or leaves an approval comment and `reviewer-approved` label when `merge_when_approved = false`
     - Adds clear, actionable feedback if changes are needed (no questions)
3. Re-reviews MRs when they are updated (after worker addresses feedback)
4. Treats each review as fresh (no memory of previous reviews)

**Key Features**: 
- The reviewer never asks questions. It provides direct feedback and makes approval/rejection decisions autonomously.
- Waits for worker to address comments before re-reviewing, creating an efficient feedback loop.

### OPS Agent

OPS accepts SSH and Grafana Elasticsearch entries in the same `logs` list. Grafana sources query the configured datasource through Grafana's `_msearch` proxy, apply the standard two-hour window, and can restrict both matching records and fields sent to the model:

```toml
[agent.ops]
poll_interval_secs = 600
logs = [
  { ssh_user = "deploy", ssh_host = "prod.example.com", log_path = "/var/log/app/app.log" },
  { type = "grafana", url = "https://grafana.example.com", datasource_uid = "elastic-uid", index = "application-logs", org_id = 1, username = "ops", password = "replace-me", filter = '{"term":{"service.name":"api"}}' },
]
```

`filter` accepts either a raw Elasticsearch query object encoded as JSON, such as the `term` filter above, or a Lucene query string such as `level:ERROR`. Query DSL is required when a backend cannot resolve special characters in field names through Lucene syntax. OPS subdivides the same two-hour window used for SSH logs until every matching record is fetched, and stores each complete `_source` object as one JSON line. `org_id` defaults to `1`. Grafana credentials are HTTP Basic Auth credentials stored literally in the local `potlatch.toml`; keep that ignored file private and never commit it.

## Labels

The Worker uses the following labels:
- `in-progress` — the worker is actively implementing or tracking an MR for this issue
- `pending` — pauses the worker: it skips the issue in the queue, releases claim/session, and stops MR watch, without closing the issue. Remove `pending` when work should continue.
- `review-only` — keeps the worker out of the issue: it skips queue pickup and, if already tracking, releases claim/session and stops watching the related MR without closing either.

To prevent implementation, add:
- `do-not-implement` label to issues

## Session Tracking

Each processed issue creates a `issue_<number>.md` file with:
- Issue details
- Associated MR number
- Status and timestamps

These files are git-ignored and used for tracking progress.

## Configuration Options

### Model Selection

You can specify different AI models for worker and reviewer agents:

```toml
[worker]
model = "composer-2"  # Must match an id from `agent --list-models`

[reviewer]
model = "gpt-5.3-codex"      # e.g. a different model for review
merge_when_approved = true   # Set to false to comment approval without merging
```

Use **`agent --list-models`** (or the values listed under the model entry in **`configOptions`** when the agent sends them) for valid ids. Potlatch passes **`--model`** on the CLI and, after **`session/new`**, prefers **`session/set_config_option`** with **`configId`** set to the advertised model option’s **`id`** and **`value`** set to your id—only when that value appears in the option’s **`options`** array, per the [spec](https://agentclientprotocol.com/protocol/session-config-options). If there is no matching advertised option, or the call fails, Potlatch falls back to **`session/set_model`**.

### Polling Intervals

Configure how often agents check for new work:

```toml
[worker]
poll_interval_secs = 60      # Check for new issues every 60 seconds

[reviewer]
poll_interval_secs = 120     # Check for MRs every 120 seconds
merge_when_approved = true   # Auto-merge approved MRs
```

Default values:
- Worker: 60 seconds
- Reviewer: 120 seconds

### Scope label

Optional top-level `scope_label` (default **empty** = no scoping; all open issues and MRs are eligible). When non-empty, worker, reviewer, and PMO only consider items that carry that GitLab label (trimmed, exact match). The worker and PMO filter **issues**; the reviewer filters **merge requests**.

When scoping is enabled, merge requests created by the worker and sub-issues created by the PMO automatically receive `scope_label`.

```toml
scope_label = "potlatch"
```

## License

MIT
