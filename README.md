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
2. **ACP server** — An [Agent Client Protocol](https://agentclientprotocol.com/) server. Potlatch communicates with model backends over ACP (JSON-RPC over stdio). Configure the server command and environment under `[acp.*]` in `potlatch.toml`. The built-in harness (`potlatch harness`) is a self-hosted ACP server that talks directly to an OpenAI-compatible LLM endpoint — set `POTLATCH_BASE_URL` and `POTLATCH_API_KEY` via the `env` array in the `[acp.*]` section.
3. **GitLab CLI** (`glab`) - For GitLab operations
4. **Git** - For repository operations
5. **Chrome or Chromium** - The web agent exposes `web_search` and `web_fetch`, opens the system browser visibly, and reuses `~/.potlatch/web-chrome-profile` so cookies and consent state persist. Both tools return only Defuddle Markdown without a JSON wrapper: `web_search` converts the rendered Google results page, while `web_fetch` converts the requested HTTP(S) page. Set `POTLATCH_WEB_PROFILE` to use another non-default profile directory.

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

Generate an example config file:

```bash
potlatch init-config
```

This creates `potlatch.toml`. See [Configuration Options](#configuration-options) for the full format.


## Configuration Options

The config file uses `[acp.<name>]` sections to define ACP backends and `[agent.<name>]` sections to define agents:

```toml
# ACP backend: the built-in harness talks to an OpenAI-compatible endpoint.
[acp.potlatch]
base_url = "https://api.deepseek.com/v1"     # LLM endpoint URL
api_key = "YOUR_DEEPSEEK_API_KEY"            # API key for the endpoint
acp_command = ["potlatch", "harness"]        # command to spawn the ACP server
env = [                                      # env vars passed to the subprocess ({base_url}/{api_key} are interpolated)
  "POTLATCH_BASE_URL={base_url}",
  "POTLATCH_API_KEY={api_key}",
]

# Agent sections: only agents with a section are started.
[agent.worker]
model = "acp://potlatch/deepseek-v4-flash"   # acp://<acp_client>/<model_name>
acp_client = "potlatch"                      # references the [acp.<name>] section
poll_interval = "1m"                         # how often to check for new work
instances = 1                                # number of parallel instances

[agent.reviewer]
model = "acp://potlatch/deepseek-v4-flash"
acp_client = "potlatch"
poll_interval = "2m"
merge_when_approved = true                   # auto-merge approved MRs (default: false)
```

### Fields

- **`model`** — `acp://<vendor>/<model>` URI. The `<vendor>` part references an `[acp.<vendor>]` section; `<model>` is the model name passed to the backend.
- **`acp_client`** — which `[acp.<name>]` profile to use. Must match a defined ACP section.
- **`poll_interval`** — how often the agent polls for new work (e.g. `"1m"`, `"30s"`). Default: 60s for worker, 120s for reviewer.
- **`instances`** — number of parallel agent instances. Default: 1.
- **`merge_when_approved`** (reviewer) — when true, the reviewer merges approved MRs automatically. Default: false.

### Scope label

Optional top-level `scope_label` (default **empty** = no scoping; all open issues and MRs are eligible). When non-empty, worker, reviewer, and PMO only consider items that carry that GitLab label (trimmed, exact match). The worker and PMO filter **issues**; the reviewer filters **merge requests**.

When scoping is enabled, merge requests created by the worker and sub-issues created by the PMO automatically receive `scope_label`.

```toml
scope_label = "potlatch"
```

## License

This project is licensed under the [MIT License](LICENSE).
