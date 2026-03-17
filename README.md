# Codepair

A CLI-Based AI Agent Pair System that orchestrates AI agents to automatically implement and review GitLab issues.

## Overview

Codepair runs two AI agents in parallel in a fully automated, non-interactive mode:
- **Worker Agent**: Fetches issues, implements features, and creates merge requests autonomously
- **Reviewer Agent**: Reviews merge requests and merges them when approved

Both agents operate without requiring any user input, making autonomous decisions based on the code and project documentation.

## Prerequisites

2. **Cursor Agent CLI** - The `agent` command must be available
3. **GitLab CLI** (`glab`) - For GitLab operations
4. **Git** - For repository operations

## Security Note

Codepair automatically trusts the cloned repository directories (`*-worker` and `*-reviewer`) by passing the `--trust` flag to the agent CLI. This is necessary for non-interactive automation. Only use Codepair with repositories you trust, as the AI agent will have full access to execute code and modify files in these directories.

## Build

```bash
cargo build --release
```

## Usage

### Basic Usage

```bash
codepair <git-repo-address>
```

Example:
```bash
codepair https://gitlab.com/username/project
```

### With Configuration File

First, generate an example config file:

```bash
codepair init-config
```

This creates `codepair.toml`. Edit it to configure models and polling intervals:

```toml
[worker]
model = "claude-3-5-sonnet"
poll_interval_secs = 60

[reviewer]
model = "claude-3-opus"
poll_interval_secs = 120
```

Then run with the config:

```bash
codepair --config codepair.toml https://gitlab.com/username/project
```

The command will:
1. Clone the repository to `<project-name>-worker` and `<project-name>-reviewer` directories
2. Start the Worker and Reviewer agents with specified models
3. Run continuously, displaying periodic status updates
4. Never require user input

## How It Works

### Worker Agent

1. Fetches open issues from GitLab
2. Skips issues that are:
   - Marked as `[Draft]`
   - Have `do-not-implement` label
   - Already have `in-progress` label
3. For each issue:
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
4. Tracks active merge requests and monitors for reviewer feedback
   - Checks for new comments on active MRs
   - Automatically addresses reviewer feedback
   - Makes necessary code changes based on comments
   - Pushes updates and adds summary comment
5. Saves session summaries to `issue_<number>.md` files with MR information

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
     - Approves and merges if everything is good
     - Adds clear, actionable feedback if changes are needed (no questions)
3. Re-reviews MRs when they are updated (after worker addresses feedback)
4. Treats each review as fresh (no memory of previous reviews)

**Key Features**: 
- The reviewer never asks questions. It provides direct feedback and makes approval/rejection decisions autonomously.
- Waits for worker to address comments before re-reviewing, creating an efficient feedback loop.

## Labels

The Worker uses the following label:
- `in-progress` - Indicates an issue is currently being worked on

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
model = "claude-3-5-sonnet"  # Fast, efficient for implementation

[reviewer]
model = "claude-3-opus"      # More thorough for code review
```

Supported models depend on your `agent` CLI configuration. Common options:
- `claude-3-5-sonnet` - Fast and capable
- `claude-3-opus` - Most capable, slower
- `gpt-4` - OpenAI's GPT-4
- Or any model supported by your agent CLI

### Polling Intervals

Configure how often agents check for new work:

```toml
[worker]
poll_interval_secs = 60      # Check for new issues every 60 seconds

[reviewer]
poll_interval_secs = 120     # Check for MRs every 120 seconds
```

Default values:
- Worker: 60 seconds
- Reviewer: 120 seconds

## License

MIT
