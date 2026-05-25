use anyhow::Result;
use clap::Parser;
use tracing::info;

mod agents;
mod core;
mod util;

use core::config::Config;

use agents::settings::AgentSettings;

#[derive(Parser, Debug)]
#[command(name = "codepair")]
#[command(about = "A CLI-Based AI Agent Pair System", long_about = None)]
enum Cli {
    /// Run the codepair agent system
    #[command(name = "run", alias = "start")]
    Run {
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
    /// Path to config file (default: codepair.toml)
    #[arg(short, long)]
    config: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::try_parse();

    let config_path = match cli {
        Ok(Cli::InitConfig { path }) => {
            Config::save_example(&path)?;
            println!("Example config file created at: {}", path);
            println!("\nEdit this file to configure agents under [agent.*] sections.");
            println!("\nExample usage:");
            println!("  codepair run");
            println!("  codepair run --config {}", path);
            return Ok(());
        }
        Ok(Cli::Run { config }) => config,
        Err(_) => Args::parse().config,
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let (config, content) = Config::load_with_content(config_path.as_deref())?;
    let agent_settings = AgentSettings::from_toml_str(&content)?;

    if let Some(repo) = agent_settings.gitlab_repo() {
        info!("GitLab repository: {}", repo);
    }

    agents::run(config, agent_settings)
}
