pub mod reviewer;
pub mod worker;

use anyhow::Result;
use tracing::error;

use crate::config::Config;

pub async fn run(git_repo_address: String, config: Config) -> Result<()> {
    let worker_handle = tokio::spawn({
        let repo = git_repo_address.clone();
        let worker_config = config.worker.clone();
        async move {
            if let Err(e) = worker::run(repo, worker_config).await {
                error!("Worker error: {}", e);
            }
        }
    });

    let reviewer_handle = tokio::spawn({
        let repo = git_repo_address.clone();
        let reviewer_config = config.reviewer.clone();
        async move {
            if let Err(e) = reviewer::run(repo, reviewer_config).await {
                error!("Reviewer error: {}", e);
            }
        }
    });

    tokio::try_join!(worker_handle, reviewer_handle)?;

    Ok(())
}
