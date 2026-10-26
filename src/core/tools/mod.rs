//! Tools core defines but does not register.
//!
//! A tool here is a **catalog definition**: the schema, the config format it
//! consumes, and its execution are owned by core, but nothing registers it
//! on its own. Adoption happens in exactly one place —
//! [`crate::core::agent::CoreAgent::run_from`], the funnel every supervised
//! instance passes through: iff the agent block configures the facility
//! (e.g. `ssh = [...]`), the definition is adopted and registered as an
//! *agent-provided tool* on the cross-agent bus — scoped to that agent's own
//! sessions (see [`crate::core::bus::AgentToolDefinition`]'s `for_agent`)
//! and executed in its process — and the built agent is wrapped in core's
//! facility wrapper, which schedules the inbox drain and answers its cycles.
//! Roles carry no facility code.

pub mod ssh;
