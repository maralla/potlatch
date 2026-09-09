//! Lightweight in-process request/response communication between agents.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak, mpsc};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

struct Route {
    id: u64,
    sender: mpsc::Sender<AgentRequest>,
    tools: Vec<AgentToolDefinition>,
}

/// A named context provider registered by an agent on the bus. Other agents'
/// ACP sessions read these at `session/new` time and inject each channel's
/// content as a system message — a general mechanism for one agent to publish
/// context (e.g. durable project memory) into other agents' sessions.
pub(crate) type ContextProvider = Arc<dyn Fn() -> String + Send + Sync>;

/// A snapshot of a registered context channel: its name and current content.
#[derive(Debug, Clone)]
pub(crate) struct ContextChannel {
    pub name: String,
    pub content: String,
}

/// Guard returned by [`AgentBus::register_context`]. Dropping it unregisters
/// the context channel, mirroring how [`AgentInbox`] unregisters a route.
pub(crate) struct ContextChannelGuard {
    name: String,
    bus: Weak<BusInner>,
}

impl Drop for ContextChannelGuard {
    fn drop(&mut self) {
        if let Some(bus) = self.bus.upgrade()
            && let Ok(mut channels) = bus.context_channels.lock()
        {
            channels.remove(&self.name);
        }
    }
}

struct BusInner {
    routes: Mutex<HashMap<String, Route>>,
    route_changed: Condvar,
    next_route_id: AtomicU64,
    context_channels: Mutex<HashMap<String, ContextProvider>>,
    /// Agents that scope per-caller resources to a task session (e.g. a
    /// subagent agent owns each subagent to its caller's task session)
    /// register a listener here and close those resources when a session
    /// retires.
    lifecycle_listeners: Mutex<Vec<mpsc::Sender<String>>>,
}

/// Reserved payload key the harness injects into every bus tool call,
/// naming the harness session that made the call. Bus-served agents use it
/// to scope per-caller state (e.g. subagent ownership).
pub const CALLER_SESSION_ID_KEY: &str = "__caller_session_id";

#[derive(Clone)]
pub(crate) struct AgentBus {
    inner: Arc<BusInner>,
}

impl std::fmt::Debug for AgentBus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("AgentBus").finish_non_exhaustive()
    }
}

pub(crate) struct AgentInbox {
    name: String,
    route_id: u64,
    receiver: mpsc::Receiver<AgentRequest>,
    bus: Weak<BusInner>,
}

pub(crate) struct AgentRequest {
    pub operation: String,
    pub payload: Value,
    /// The calling harness session id, lifted out of the payload by the bus
    /// (the harness proxy injects it as [`CALLER_SESSION_ID_KEY`]). Handlers
    /// never see it among the tool arguments.
    pub caller_session_id: Option<String>,
    response: mpsc::SyncSender<std::result::Result<Value, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AgentToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub operation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RemoteAgentToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub target: String,
    pub operation: String,
}

impl AgentBus {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(BusInner {
                routes: Mutex::new(HashMap::new()),
                route_changed: Condvar::new(),
                next_route_id: AtomicU64::new(1),
                context_channels: Mutex::new(HashMap::new()),
                lifecycle_listeners: Mutex::new(Vec::new()),
            }),
        }
    }

    pub fn register(
        &self,
        name: impl Into<String>,
        tools: Vec<AgentToolDefinition>,
    ) -> Result<AgentInbox> {
        let name = name.into();
        ensure!(!name.trim().is_empty(), "agent bus route must not be empty");
        let mut tool_names = HashSet::new();
        for tool in &tools {
            ensure!(
                !tool.name.is_empty()
                    && tool.name.len() <= 64
                    && tool
                        .name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')),
                "agent tool name {:?} must match [A-Za-z0-9_-] and be at most 64 bytes",
                tool.name
            );
            ensure!(
                tool_names.insert(tool.name.as_str()),
                "agent tool {:?} is registered more than once by route {name:?}",
                tool.name
            );
            ensure!(
                !tool.description.trim().is_empty(),
                "agent tool {:?} requires a description",
                tool.name
            );
            ensure!(
                tool.parameters.is_object(),
                "agent tool {:?} parameters must be a JSON schema object",
                tool.name
            );
            ensure!(
                !tool.operation.trim().is_empty() && !tool.operation.starts_with("__"),
                "agent tool {:?} operation is empty or reserved",
                tool.name
            );
        }

        let (sender, receiver) = mpsc::channel();
        let route_id = self.inner.next_route_id.fetch_add(1, Ordering::Relaxed);
        let mut routes = self
            .inner
            .routes
            .lock()
            .map_err(|_| anyhow::anyhow!("agent bus route lock poisoned"))?;
        ensure!(
            !routes.contains_key(&name),
            "agent bus route {name:?} is already registered"
        );
        for tool in &tools {
            ensure!(
                !routes
                    .values()
                    .flat_map(|route| &route.tools)
                    .any(|registered| registered.name == tool.name),
                "agent tool {:?} is already registered",
                tool.name
            );
        }
        routes.insert(
            name.clone(),
            Route {
                id: route_id,
                sender,
                tools,
            },
        );
        self.inner.route_changed.notify_all();
        Ok(AgentInbox {
            name,
            route_id,
            receiver,
            bus: Arc::downgrade(&self.inner),
        })
    }

    pub fn request(
        &self,
        target: &str,
        operation: String,
        mut payload: Value,
        timeout: Duration,
    ) -> Result<Value> {
        let deadline = Instant::now() + timeout;
        let sender = self.wait_for_route(target, deadline)?;
        let (response, response_rx) = mpsc::sync_channel(1);
        // Lift the caller identity out of the tool arguments: it is request
        // metadata, and handlers deserialize their payloads strictly.
        let caller_session_id = payload
            .as_object_mut()
            .and_then(|object| object.remove(CALLER_SESSION_ID_KEY))
            .and_then(|value| value.as_str().map(str::to_string));
        sender
            .send(AgentRequest {
                operation,
                payload,
                caller_session_id,
                response,
            })
            .with_context(|| format!("agent bus target {target:?} stopped before receiving"))?;
        response_rx
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .with_context(|| format!("agent bus request to {target:?} timed out"))?
            .map_err(anyhow::Error::msg)
    }

    pub fn registered_tools(&self, wait: Duration) -> Result<Vec<RemoteAgentToolDefinition>> {
        let deadline = Instant::now() + wait;
        let mut routes = self
            .inner
            .routes
            .lock()
            .map_err(|_| anyhow::anyhow!("agent bus route lock poisoned"))?;
        while routes.is_empty() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let (next, timeout) = self
                .inner
                .route_changed
                .wait_timeout(routes, remaining)
                .map_err(|_| anyhow::anyhow!("agent bus route lock poisoned"))?;
            routes = next;
            if timeout.timed_out() {
                break;
            }
        }
        Ok(routes
            .iter()
            .flat_map(|(target, route)| {
                route
                    .tools
                    .iter()
                    .map(move |tool| RemoteAgentToolDefinition {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters: tool.parameters.clone(),
                        target: target.clone(),
                        operation: tool.operation.clone(),
                    })
            })
            .collect())
    }

    /// Register a named context provider on the bus. The provider is called
    /// at `session/new` time by the ACP runtime to collect context that other
    /// agents' sessions inject as system messages. Dropping the returned guard
    /// unregisters the provider.
    pub fn register_context(
        &self,
        name: impl Into<String>,
        provider: ContextProvider,
    ) -> ContextChannelGuard {
        let name = name.into();
        if let Ok(mut channels) = self.inner.context_channels.lock() {
            channels.insert(name.clone(), provider);
        }
        ContextChannelGuard {
            name,
            bus: Arc::downgrade(&self.inner),
        }
    }

    /// Register a listener for caller task-session retirement. Every
    /// retired session id is sent to each live listener; agents close the
    /// resources owned by that session (e.g. its subagents). Dropping the
    /// receiver unregisters it — dead listeners are pruned on the next
    /// notice.
    pub(crate) fn register_session_lifecycle_listener(&self) -> Result<mpsc::Receiver<String>> {
        let (sender, receiver) = mpsc::channel();
        self.inner
            .lifecycle_listeners
            .lock()
            .map_err(|_| anyhow::anyhow!("agent bus lifecycle listener lock poisoned"))?
            .push(sender);
        Ok(receiver)
    }

    /// Fire-and-forget notice that a caller task session retired. Never
    /// blocks and never fails: with no listeners it is a no-op, and
    /// listeners that went away are pruned.
    pub(crate) fn notify_caller_session_closed(&self, session_id: &str) {
        if let Ok(mut listeners) = self.inner.lifecycle_listeners.lock() {
            let session_id = session_id.to_string();
            listeners.retain(|listener| listener.send(session_id.clone()).is_ok());
        }
    }

    /// Snapshot all registered context channels: calls each provider and
    /// returns its name + current content. Used by the ACP runtime at
    /// `session/new` to ship context into other agents' harness sessions.
    pub fn context_channels(&self) -> Result<Vec<ContextChannel>> {
        let channels = self
            .inner
            .context_channels
            .lock()
            .map_err(|_| anyhow::anyhow!("agent bus context channel lock poisoned"))?;
        Ok(channels
            .iter()
            .map(|(name, provider)| ContextChannel {
                name: name.clone(),
                content: provider(),
            })
            .collect())
    }

    fn wait_for_route(
        &self,
        target: &str,
        deadline: Instant,
    ) -> Result<mpsc::Sender<AgentRequest>> {
        let mut routes = self
            .inner
            .routes
            .lock()
            .map_err(|_| anyhow::anyhow!("agent bus route lock poisoned"))?;
        loop {
            if let Some(route) = routes.get(target) {
                return Ok(route.sender.clone());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                bail!("agent bus target {target:?} is not available");
            }
            let (next, timeout) = self
                .inner
                .route_changed
                .wait_timeout(routes, remaining)
                .map_err(|_| anyhow::anyhow!("agent bus route lock poisoned"))?;
            routes = next;
            if timeout.timed_out() && !routes.contains_key(target) {
                bail!("agent bus target {target:?} is not available");
            }
        }
    }
}

impl AgentInbox {
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Option<AgentRequest>> {
        match self.receiver.recv_timeout(timeout) {
            Ok(request) => Ok(Some(request)),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                bail!("agent bus inbox {:?} disconnected", self.name)
            }
        }
    }
}

impl Drop for AgentInbox {
    fn drop(&mut self) {
        let Some(bus) = self.bus.upgrade() else {
            return;
        };
        if let Ok(mut routes) = bus.routes.lock()
            && routes
                .get(&self.name)
                .is_some_and(|route| route.id == self.route_id)
        {
            routes.remove(&self.name);
            bus.route_changed.notify_all();
        }
    }
}

impl AgentRequest {
    pub fn respond(self, result: Result<Value>) {
        let _ = self
            .response
            .send(result.map_err(|error| format!("{error:#}")));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> AgentToolDefinition {
        AgentToolDefinition {
            name: "web_search".to_string(),
            description: "Search.".to_string(),
            parameters: serde_json::json!({"type": "object"}),
            operation: "run".to_string(),
        }
    }

    #[test]
    fn routes_requests_in_process() {
        let bus = AgentBus::new();
        let inbox = bus.register("web", vec![tool()]).unwrap();
        let worker = std::thread::spawn(move || {
            let request = inbox.recv_timeout(Duration::from_secs(1)).unwrap().unwrap();
            let payload = request.payload.clone();
            request.respond(Ok(payload));
        });
        let result = bus
            .request(
                "web",
                "run".to_string(),
                serde_json::json!({"query": "rust"}),
                Duration::from_secs(1),
            )
            .unwrap();
        assert_eq!(result["query"], "rust");
        worker.join().unwrap();
    }

    #[test]
    fn tools_exist_only_while_the_agent_is_registered() {
        let bus = AgentBus::new();
        assert!(bus.registered_tools(Duration::ZERO).unwrap().is_empty());
        let inbox = bus.register("web", vec![tool()]).unwrap();
        assert_eq!(
            bus.registered_tools(Duration::ZERO).unwrap()[0].name,
            "web_search"
        );
        drop(inbox);
        assert!(bus.registered_tools(Duration::ZERO).unwrap().is_empty());
    }

    #[test]
    fn lifecycle_notices_reach_registered_listeners() {
        let bus = AgentBus::new();
        let listener = bus.register_session_lifecycle_listener().unwrap();
        bus.notify_caller_session_closed("session-1");
        assert_eq!(
            listener
                .recv_timeout(std::time::Duration::from_secs(1))
                .unwrap(),
            "session-1"
        );
    }

    #[test]
    fn lifecycle_notices_fan_out_and_prune_dead_listeners() {
        let bus = AgentBus::new();
        let kept = bus.register_session_lifecycle_listener().unwrap();
        let dropped = bus.register_session_lifecycle_listener().unwrap();
        drop(dropped);
        bus.notify_caller_session_closed("session-1");
        assert_eq!(
            kept.recv_timeout(std::time::Duration::from_secs(1))
                .unwrap(),
            "session-1"
        );
        // The bus keeps working after pruning.
        bus.notify_caller_session_closed("session-2");
        assert_eq!(
            kept.recv_timeout(std::time::Duration::from_secs(1))
                .unwrap(),
            "session-2"
        );
    }

    #[test]
    fn lifecycle_notice_without_listeners_is_a_noop() {
        let bus = AgentBus::new();
        bus.notify_caller_session_closed("session-1");
    }

    #[test]
    fn context_channels_snapshot_calls_providers() {
        let bus = AgentBus::new();
        assert!(bus.context_channels().unwrap().is_empty());

        let guard = bus.register_context("memory", Arc::new(|| "project facts here".to_string()));
        let channels = bus.context_channels().unwrap();
        assert_eq!(channels.len(), 1);
        assert_eq!(channels[0].name, "memory");
        assert_eq!(channels[0].content, "project facts here");

        drop(guard);
        assert!(bus.context_channels().unwrap().is_empty());
    }

    #[test]
    fn context_channels_reflect_live_provider_state() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let bus = AgentBus::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_provider = Arc::clone(&counter);
        let _guard = bus.register_context(
            "live",
            Arc::new(move || {
                let n = counter_for_provider.fetch_add(1, Ordering::SeqCst);
                format!("call {n}")
            }),
        );

        // Each snapshot calls the provider fresh.
        let first = bus.context_channels().unwrap();
        let second = bus.context_channels().unwrap();
        assert_eq!(first[0].content, "call 0");
        assert_eq!(second[0].content, "call 1");
    }
}
