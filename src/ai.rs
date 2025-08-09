use crate::ai_memory::Memory;
use serde::{Serialize,Deserialize};
use anyhow::{Result, anyhow};
use std::env;
use reqwest::{Client, header};
use futures_util::StreamExt;

#[derive(Serialize)]
struct OpenAIMessage { role: String, content: String }
#[derive(Serialize)]
struct OpenAIRequest { model: String, messages: Vec<OpenAIMessage>, stream: bool }
#[derive(Deserialize)]
struct StreamChunkChoiceDelta { content: Option<String>, role: Option<String> }
#[derive(Deserialize)]
struct StreamChunkChoice { delta: StreamChunkChoiceDelta, finish_reason: Option<String> }
#[derive(Deserialize)]
struct StreamChunk { choices: Vec<StreamChunkChoice> }

pub struct AiAgent {
    provider: String,
    model: String,
    client: Client,
    mem: Memory,
    memory_turns: usize,
    project_mode: bool,
}

impl AiAgent {
    pub fn new(memory_path: &str, project_mode: bool) -> Result<Self> {
        dotenvy::dotenv().ok();
        let provider = env::var("AI_PROVIDER").unwrap_or_else(|_| "openai".into());
        let model = env::var("AI_MODEL").unwrap_or_else(|_| "gpt-5".into());
        let client = Client::builder().build()?;
        let mem = Memory::open(memory_path)?;
        let memory_turns = env::var("AI_MEMORY_TURNS").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
        Ok(Self { provider, model, client, mem, memory_turns, project_mode })
    }

    fn build_history(&self) -> Result<Vec<OpenAIMessage>> {
        let history = self.mem.last_turns(self.memory_turns)?;
        let mut m = Vec::new();
        for h in history { m.push(OpenAIMessage { role: h.role, content: h.content }); }
        Ok(m)
    }

    pub async fn ask_streaming(&self, user_prompt: &str) -> Result<String> {
        if self.provider != "openai" { return Err(anyhow!("Only 'openai' provider is enabled in this build.")); }
        let api_key = env::var("OPENAI_API_KEY").map_err(|_| anyhow!("OPENAI_API_KEY not set"))?;

        let mut messages = self.build_history()?;
        messages.push(OpenAIMessage { role: "user".into(), content: user_prompt.into() });

        let req = OpenAIRequest { model: self.model.clone(), messages, stream: true };

        let mut res = self.client
            .post("https://api.openai.com/v1/chat/completions")
            .header(header::AUTHORIZATION, format!("Bearer {}", api_key))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&req)
            .send()
            .await?;

        // Stream the SSE-like event payload
        let mut body = res.bytes_stream();
        let mut acc = String::new();

        while let Some(chunk) = body.next().await {
            let chunk = chunk?;
            let s = String::from_utf8_lossy(&chunk);
            for line in s.split('\n') {
                let line = line.trim();
                if !line.starts_with("data:") { continue; }
                let data = line.trim_start_matches("data:").trim();
                if data == "[DONE]" { break; }
                if data.is_empty() { continue; }
                if let Ok(json) = serde_json::from_str::<StreamChunk>(data) {
                    if let Some(choice) = json.choices.get(0) {
                        if let Some(piece) = &choice.delta.content {
                            print!("{}", piece);
                            acc.push_str(piece);
                        }
                    }
                }
            }
            // flush as we go (important on Windows terminals)
            use std::io::Write; let _ = std::io::stdout().flush();
        }

        // Persist after stream completes
        self.mem.append("user", user_prompt)?;
        self.mem.append("assistant", &acc)?;
        println!();
        Ok(acc)
    }

    pub fn memory_show(&self, limit: usize) -> Result<Vec<crate::ai_memory::ChatTurn>> { self.mem.last_turns(limit) }
    pub fn memory_clear(&self) -> Result<()> { self.mem.clear() }
}