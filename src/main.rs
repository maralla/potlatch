use anyhow::Result;
use clap::Parser;
use std::path::Path;

mod agents;
mod core;
mod ui;
mod util;

use core::config::Config;

use agents::settings::AgentSettings;

#[derive(Parser, Debug)]
#[command(name = "potlatch")]
#[command(about = "Potlatch — fully automatic agentic platform", long_about = None)]
enum Cli {
    /// Run the potlatch agent system
    #[command(name = "run", alias = "start")]
    Run {
        /// Path to config file (default: potlatch.toml)
        #[arg(short, long)]
        config: Option<String>,
    },

    /// Generate an example config file
    InitConfig {
        /// Output path for the config file
        #[arg(default_value = "potlatch.toml")]
        path: String,
    },
}

#[derive(Parser, Debug)]
#[command(name = "potlatch")]
#[command(about = "Potlatch — fully automatic agentic platform", long_about = None)]
struct Args {
    /// Path to config file (default: potlatch.toml)
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
            println!("  potlatch run");
            println!("  potlatch run --config {}", path);
            return Ok(());
        }
        Ok(Cli::Run { config }) => config,
        Err(_) => Args::parse().config,
    };

    ui::init();

    let config_path = config_path
        .as_deref()
        .map(str::to_string)
        .unwrap_or_else(|| "potlatch.toml".to_string());

    let (config, content) = Config::load_with_content(Some(&config_path))?;
    let agent_settings = AgentSettings::from_toml_str(&content)?;

    let agent_names: Vec<String> = config.agent_names().map(str::to_string).collect();
    ui::print_banner(
        Path::new(&config_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(&config_path),
        agent_settings.gitlab_repo(),
        &agent_names,
    );

    agents::run(config, agent_settings)
}
