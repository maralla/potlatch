use anyhow::Result;
use clap::Parser;
use tracing::info;

mod agent;
mod agents;
mod config;
mod git;
mod gitlab;

use config::Config;

#[derive(Parser, Debug)]
#[command(name = "codepair")]
#[command(about = "A CLI-Based AI Agent Pair System", long_about = None)]
enum Cli {
    /// Run the codepair agent system
    #[command(name = "run", alias = "start")]
    Run {
        /// GitLab repository address (e.g., https://gitlab.com/user/repo)
        git_repo_address: String,

        /// Path to config file (default: codepair.toml)
        #[arg(short, long)]
        config: Option<String>,
    },

    /// Generate an example config file
    InitConfig {
        /// Output path for the config file
        #[arg(default_value = "codepair.toml")]
        path: String,
    },
}

#[derive(Parser, Debug)]
#[command(name = "codepair")]
#[command(about = "A CLI-Based AI Agent Pair System", long_about = None)]
struct Args {
    /// GitLab repository address (e.g., https://gitlab.com/user/repo)
    git_repo_address: String,

    /// Path to config file (default: codepair.toml)
    #[arg(short, long)]
    config: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::try_parse();

    let (git_repo_address, config_path) = match cli {
        Ok(Cli::InitConfig { path }) => {
            Config::save_example(&path)?;
            println!("Example config file created at: {}", path);
            println!("\nEdit this file to configure worker and reviewer models.");
            println!("\nExample usage:");
            println!("  codepair https://gitlab.com/user/repo");
            println!("  codepair --config {} https://gitlab.com/user/repo", path);
            return Ok(());
        }
        Ok(Cli::Run {
            git_repo_address,
            config,
        }) => (git_repo_address, config),
        Err(_) => {
            let args = Args::parse();
            (args.git_repo_address, args.config)
        }
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let config = Config::load(config_path.as_deref())?;

    info!("Starting Codepair for repository: {}", git_repo_address);

    agents::run(git_repo_address, config)
}
