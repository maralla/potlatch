use anyhow::{Context, Result};
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use tracing::{debug, info, warn};

pub struct Agent {
    repo_path: String,
    model: Option<String>,
    shutdown: Arc<AtomicBool>,
}

impl Agent {
    pub fn new(repo_path: String, model: Option<String>, shutdown: Arc<AtomicBool>) -> Self {
        Self {
            repo_path,
            model,
            shutdown,
        }
    }

    pub fn run(&self, prompt: &str) -> Result<String> {
        if let Some(model) = &self.model {
            info!(
                "Running agent with model: {}, prompt length: {} chars",
                model,
                prompt.len()
            );
        } else {
            info!("Running agent with prompt length: {} chars", prompt.len());
        }

        debug!("Agent prompt: {}", prompt);

        let mut cmd = Command::new("agent");
        cmd.arg("--print");
        cmd.arg("--trust");
        cmd.arg("--force");

        if let Some(model) = &self.model {
            cmd.arg("--model").arg(model);
        }

        let mut child = cmd
            .current_dir(&self.repo_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn agent process")?;

        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            stdin
                .write_all(prompt.as_bytes())
                .context("Failed to write to agent stdin")?;
        }

        // Drain stdout/stderr in background threads to avoid pipe-buffer
        // deadlocks when the child produces more than ~64KB of output.
        let stdout_handle = child.stdout.take().map(|mut out| {
            thread::spawn(move || {
                let mut buf = Vec::new();
                out.read_to_end(&mut buf).ok();
                buf
            })
        });
        let stderr_handle = child.stderr.take().map(|mut err| {
            thread::spawn(move || {
                let mut buf = Vec::new();
                err.read_to_end(&mut buf).ok();
                buf
            })
        });

        // Poll for child exit while checking the shutdown flag, because the
        // child `agent` process may handle SIGINT gracefully and not die
        // immediately, leaving wait_with_output() blocking indefinitely.
        loop {
            match child.try_wait() {
                Ok(Some(_status)) => break,
                Ok(None) => {
                    if self.shutdown.load(Ordering::SeqCst) {
                        warn!("Shutdown requested, killing agent child process");
                        let _ = child.kill();
                        let _ = child.wait();
                        anyhow::bail!("Agent interrupted by shutdown");
                    }
                    thread::sleep(Duration::from_millis(200));
                }
                Err(e) => {
                    anyhow::bail!("Failed to wait for agent process: {}", e);
                }
            }
        }

        let stdout_buf = stdout_handle
            .map(|h| h.join().unwrap_or_default())
            .unwrap_or_default();
        let stderr_buf = stderr_handle
            .map(|h| h.join().unwrap_or_default())
            .unwrap_or_default();

        let exit_status = child.wait().context("Failed to get agent exit status")?;
        if !exit_status.success() {
            let stderr = String::from_utf8_lossy(&stderr_buf);
            anyhow::bail!("Agent execution failed: {}", stderr);
        }

        let result = String::from_utf8_lossy(&stdout_buf).to_string();
        debug!("Agent output length: {} chars", result.len());
        Ok(result)
    }
}
