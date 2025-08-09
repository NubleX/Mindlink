use clap::{Parser, Subcommand, Args};
mod ai;
mod ai_memory;
use anyhow::Result;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "mindlink", about = "Your persistent CLI AI partner")]
struct Cli {
    /// One-off prompt
    #[arg(short, long)]
    prompt: Option<String>,

    /// Use project-local memory in ./.mindlink (recommended when inside a repo)
    #[arg(long, default_value_t = true)]
    project_memory: bool,

    /// How many recent turns to include
    #[arg(long)]
    memory_turns: Option<usize>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start interactive chat (REPL)
    Chat,
    /// Show memory (last N turns)
    MemoryShow { limit: Option<usize> },
    /// Clear memory
    MemoryClear,
}

fn memory_path(project_mode: bool) -> PathBuf {
    if project_mode {
        let p = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let dir = p.join(".mindlink");
        let _ = std::fs::create_dir_all(&dir);
        return dir.join("memory.db");
    }
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let dir = home.join(".mindlink");
    let _ = std::fs::create_dir_all(&dir);
    dir.join("memory.db")
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(mt) = cli.memory_turns { std::env::set_var("AI_MEMORY_TURNS", mt.to_string()); }

    let mem_path = memory_path(cli.project_memory);
    let agent = ai::AiAgent::new(mem_path.to_string_lossy().as_ref(), cli.project_memory)?;

    if let Some(prompt) = cli.prompt {
        let _ = agent.ask_streaming(&prompt).await?; return Ok(());
    }

    match &cli.command {
        Some(Commands::Chat) => {
            use std::io::{stdin, stdout, Write};
            loop {
                print!("mindlink> "); stdout().flush()?;
                let mut line = String::new(); stdin().read_line(&mut line)?;
                let line = line.trim(); if line.is_empty() { continue; }
                if line == "exit" || line == "quit" { break; }
                let _ = agent.ask_streaming(line).await?;
            }
        }
        Some(Commands::MemoryShow { limit }) => {
            let lim = limit.unwrap_or(50);
            for t in agent.memory_show(lim)? { println!("[{}] {}: {}", t.ts, t.role, t.content); }
        }
        Some(Commands::MemoryClear) => { agent.memory_clear()?; println!("Memory cleared."); }
        None => { println!("mindlink — try: mindlink --prompt 'hello'  |  mindlink chat"); }
    }

    Ok(())
}