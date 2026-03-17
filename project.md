# Codepair

A CLI-Based AI Agent Pair System

## Overview
A system that orchestrates CLI-based AI agents (like `cursor agent`, `claude code`, `gemini cli`) to work as a pair:

- **Worker Agent**: Uses CLI AI to implement features
- **Reviewer Agent**: Uses CLI AI to review code

## Key Points

1. **`agent` command** - Cursor's CLI is just called `agent`, not `cursor agent`
2. **`--print` flag** - Makes it non-interactive, outputs to console (perfect for scripts)
4. **API key** - don't care about API key, we assume it is already handled
5. Use Rust to implement this project


## Design

Here is the interface for interacting with this tool:

```bash
codepair <git-repo-address>
```

Then the command just blocks there with periodic output of the current working issue, how many issues are finished, etc.
It never needs the user to give any input.


Two roles of a pair:

#### Worker

If it does not exist, clone the git repo to directory <git-project-name>-worker. Fetch the latest main or master branch
and rebase to it. This ensures every issue starts with the latest main branch.

Fetch an issue that is not marked [Draft], or has a comment that has the meaning of do not implement or a label
that carries a similar meaning, or a label explicitly indicating it is being worked on.

Add a label to the issue that indicates it is being worked on. Then analyze it. If the issue is clear enough and has
all necessary information (note that the comments of the issue should also be collected for information), implement
it according to requirements stated in AGENTS.md or other doc files. If the issue is ambiguous and not possible
to implement, add a comment to the issue to explicitly explain what is needed and why it can't be implemented, then
remove the working on label, then go to the next issue. If the issue has finished implementation, create a merge request
with necessary descriptions about what is implemented, how it is implemented, etc. And periodically check the status
of the merge request. If there are comments, resolve the comments. If the merge request is closed or merged, go back
to the initial state of fetching the issues again.

If an issue is finished (rejected, MR created), the agent session should be ended with a summary that tracks all 
information about the session, save it to issue_<issue-number>.md (this file pattern should be git ignored). If
the tracking MR has events (comments added, closed, merged), if there are unresolved comments start an agent to handle this
issue again with the previous summary information. If closed or merged, fetch the issue list again to handle the next issue.

#### Reviewer

If it does not exist, clone the git repo to directory <git-project-name>-reviewer. Fetch the latest main or master branch
and rebase to it. This ensures every issue starts with the latest main branch.

The goal of Reviewer is to periodically fetch merge requests and fully review the merge request, adding comments
as feedback to the merge request. Then if there are no issues, and the merge request accurately implements the goal
it states with tests and lints passing, then merge it. Note that Reviewer doesn't wait for the Worker to resolve
all comments, it just keeps reviewing the next merge request. But it should remember if a merge request is already
reviewed. If there are no updates, it goes to the next MR. If it is updated, review it. Every time it reviews an MR, treat
it as a fresh new review.


## Operations


We default to assume based on Gitlab hosted project. So, use the command `glab` for all Gitlab operations.
