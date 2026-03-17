use anyhow::{Context, Result};
use std::process::{Command, Stdio};
use tracing::{debug, info};

pub struct Agent {
    repo_path: String,
    model: Option<String>,
}

impl Agent {
    pub fn new(repo_path: String, model: Option<String>) -> Self {
        Self { repo_path, model }
    }

    pub fn run(&self, prompt: &str) -> Result<String> {
        if let Some(ref model) = self.model {
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

        if let Some(ref model) = self.model {
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

        let output = child
            .wait_with_output()
            .context("Failed to wait for agent")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Agent execution failed: {}", stderr);
        }

        let result = String::from_utf8_lossy(&output.stdout).to_string();
        debug!("Agent output length: {} chars", result.len());

        Ok(result)
    }
}
