use anyhow::Result;
use clap::{Parser, Subcommand};
use std::sync::Arc;

mod agents;
mod core;
mod harness;
mod paths;
mod ui;
mod util;

use core::config::Config;
use core::workflow::Workflow;
use paths::{APP_NAME, CONFIG_FILE_NAME};

#[derive(Parser, Debug)]
#[command(about = "Potlatch — fully automatic agentic platform", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
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
        #[arg(default_value = CONFIG_FILE_NAME)]
        path: String,
    },

    /// ACP server with direct LLM backend (self-hosted model, no Cursor cloud)
    Harness,
}

#[derive(Parser, Debug)]
#[command(name = APP_NAME)]
#[command(about = "Potlatch — fully automatic agentic platform", long_about = None)]
struct Args {
    /// Path to config file (default: potlatch.toml)
    #[arg(short, long)]
    config: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::try_parse();

    let command = match cli {
        Ok(Cli { command: Some(cmd) }) => Some(cmd),
        Ok(Cli { command: None }) => None,
        Err(_) => {
            let args = Args::parse();
            return run_workflow(args.config);
        }
    };

    match command {
        Some(Commands::InitConfig { path }) => {
            Config::save_example(&path)?;
            println!("Example config file created at: {}", path);
            println!("\nEdit this file to configure agents under [agent.*] sections.");
            println!("\nExample usage:");
            println!("  potlatch run");
            println!("  potlatch run --config {}", path);
            Ok(())
        }
        Some(Commands::Run { config }) => run_workflow(config),
        Some(Commands::Harness) => harness::run_acp_server(),
        None => run_workflow(None),
    }
}

fn run_workflow(config: Option<String>) -> Result<()> {
    ui::init();

    let config_path = config
        .as_deref()
        .map(str::to_string)
        .unwrap_or_else(|| CONFIG_FILE_NAME.to_string());

    let config = Config::load(Some(&config_path))?;

    let mut workflow =
        Workflow::new(config, config_path).with_activity_reporter(Arc::new(ui::UiActivityReporter));
    agents::register(&mut workflow);
    workflow.run()
}
